"""Development-only H28 inventory and exact controls; no ROW model execution.

See H28_T0_INSTRUMENT_PLAN.md. Standard library only. This utility cannot
produce an accepted scientific result, including when run from clean code.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "artifacts/j1c_curriculum/cells/STAGED_w0/stage3",
    "artifacts/j1c_curriculum/cells/RESET_w0/stage3",
    "artifacts/j1cr_replication/cells/STAGED-R_w0/stage3",
    "artifacts/j1cr_replication/cells/NON-STAGED-R_w0/stage3",
    "artifacts/v4_dev/structured/world_0/lifecycle",
    "artifacts/v5_h20b/r100/world_0/lifecycle",
)
TOL = 1e-12


def entropy_given(rows, conditions):
    """H(y|conditions) for normalized finite joint rows (assignment, mass)."""
    if not rows or any(not math.isfinite(p) or p < 0 for _, p in rows):
        raise ValueError("invalid joint probabilities")
    if abs(math.fsum(p for _, p in rows) - 1) > TOL:
        raise ValueError("joint probabilities must sum to one")
    joint, marginal = defaultdict(float), defaultdict(float)
    for values, mass in rows:
        key = tuple(values[c] for c in conditions)
        joint[key, values['y']] += mass
        marginal[key] += mass
    return -math.fsum(p * math.log(p / marginal[key])
                      for (key, _), p in joint.items() if p > 0)


def bit_rows(kind):
    rows = []
    for z, n, w in product((0, 1), repeat=3):
        if kind == 'closed':
            targets = ((z, 0.9), (z ^ 1, 0.1))
        elif kind == 'micro':
            targets = ((z ^ n, 1.0),)
        elif kind == 'world':
            targets = ((z ^ w, 1.0),)
        elif kind == 'joint':
            targets = ((z ^ n ^ w, 1.0),)
        elif kind == 'constant':
            targets = ((0, 1.0),)
        else:
            raise ValueError(f"unknown control: {kind}")
        for y, p in targets:
            rows.append((dict(z=z, n=n, w=w, y=y), p / 8))
    return rows


def gaps(rows, weak=False):
    macro = entropy_given(rows, ('z',))
    contexts = (('z', 'n'), ('z', 'w'), ('z', 'n', 'w'))
    return {
        'macro_entropy': macro,
        'target_entropy': entropy_given(rows, ()),
        'predictive_gain': entropy_given(rows, ()) - macro,
        'gaps': [macro - (macro if weak else entropy_given(rows, c)) for c in contexts],
    }


def nontrivial_gate(stats, discards_distinctions):
    """Toy-control logic only; not a learned-representation acceptance rule."""
    return (discards_distinctions and stats['target_entropy'] > TOL
            and stats['predictive_gain'] > TOL
            and all(abs(x) <= TOL for x in stats['gaps']))


def gaussian_psi(r, theta):
    if not 0 < abs(r) < 1 or not math.isfinite(theta):
        raise ValueError('require finite angle and 0 < |r| < 1')
    macro = -0.5 * math.log1p(-r * r)
    # Correlations of each rotated coordinate with the transported V_next.
    parts = [-0.5 * math.log1p(-(r * a) ** 2)
             for a in (math.cos(theta), math.sin(theta))]
    return macro - math.fsum(parts)


def controls():
    expected = {'closed': [0, 0, 0], 'micro': [math.log(2), 0, math.log(2)],
                'world': [0, math.log(2), math.log(2)],
                'joint': [0, 0, math.log(2)], 'constant': [0, 0, 0]}
    finite = {}
    for kind, target in expected.items():
        result = gaps(bit_rows(kind))
        if any(abs(a - b) > TOL for a, b in zip(result['gaps'], target)):
            raise ValueError(f'finite control failed: {kind}')
        finite[kind] = result
    if not nontrivial_gate(finite['closed'], True):
        raise ValueError('closed positive failed')
    if nontrivial_gate(finite['constant'], True) or nontrivial_gate(finite['closed'], False):
        raise ValueError('trivial encoder passed')
    weak = gaps(bit_rows('joint'), weak=True)
    if weak['gaps'] != [0, 0, 0] or finite['joint']['gaps'][2] <= TOL:
        raise ValueError('weak detector failure was not exposed')
    gaussian = []
    for r in (0.1, 0.5, 0.8, 0.99):
        identity, rotated, permutation = [gaussian_psi(r, a) for a in (0, math.pi / 4, math.pi / 2)]
        formula = -0.5 * math.log1p(-r * r) + math.log1p(-r * r / 2)
        if abs(identity) > TOL or abs(permutation) > TOL or rotated <= 0 or abs(rotated-formula) > TOL:
            raise ValueError('Gaussian control failed')
        gaussian.append(dict(r=r, identity=identity, rotated=rotated, permutation=permutation))
    return dict(finite=finite, weak_joint=weak, gaussian=gaussian,
                controls_ok=True, learned_probe_calibrated=False)


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def saved_seed(config):
    """Fail-closed reader for this repository's plain block world.seed YAML.

    This is deliberately not a general YAML parser. Reject absent/duplicate
    world blocks or seed fields instead of guessing at unfamiliar YAML.
    """
    blocks = re.findall(r'^world:\s*\n((?:[ \t]+[^\n]*\n|\n)*)', config + '\n', re.M)
    if len(blocks) != 1:
        raise ValueError('expected exactly one plain world YAML block')
    seeds = re.findall(r'^  seed: ([0-9]+)\s*$', blocks[0], re.M)
    if len(seeds) != 1:
        raise ValueError('expected exactly one plain world.seed')
    return int(seeds[0])


def inventory(root=ROOT):
    records = []
    for relative in SOURCES:
        path = root / relative
        if not path.is_dir():
            records.append(dict(path=relative, status='MISSING'))
            continue
        seed = saved_seed((path / 'config.yaml').read_text(encoding='utf-8'))
        if seed != 0:
            raise ValueError(f'non-development-zero seed: {relative}')
        seed_file = path / 'world_seed.txt'
        if seed_file.exists() and int(seed_file.read_text().strip()) != seed:
            raise ValueError(f'seed sidecar mismatch: {relative}')
        names = sorted(p.name for p in path.iterdir() if p.is_file())
        # Never open world_programs, metric arrays, predictions, or tensors.
        # Hash binary model bytes without deserializing them.
        allowed = ('config.yaml', 'model.pt', 'model_state.json', 'fingerprint.json',
                   'git_commit.txt', 'promotion_snapshots.npz', 'promotion_members.json')
        files = {name: dict(bytes=(path/name).stat().st_size, sha256=sha256(path/name))
                 for name in allowed if (path/name).is_file()}
        prior = None
        if '/cells/' in relative:
            durable = path.parent / 'result.json'
            record = json.loads(durable.read_text(encoding='utf-8'))
            if record['stamp']['world'] != seed:
                raise ValueError(f'durable seed mismatch: {relative}')
            expected = record['artifact_sha256'].get('stage3/model.pt')
            if expected is None or files.get('model.pt', {}).get('sha256') != expected:
                raise ValueError(f'durable model hash mismatch: {relative}')
            prior = dict(path=durable.relative_to(root).as_posix(), sha256=sha256(durable),
                         model_hash_matches=True,
                         sidecars_covered_by_prior_hash=False)
        records.append(dict(path=relative, status='METADATA_ONLY', world_seed=seed,
                            file_names=names, files=files, prior_manifest=prior,
                            historical_state_present='history.pt' in names,
                            promotion_snapshots_present='promotion_snapshots.npz' in names,
                            model_reconstructed=False, score_reproduced=False,
                            declared_coarse_interface_validated=False))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-check', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    exact = controls()
    control_seconds = time.perf_counter() - started
    census_start = time.perf_counter()
    records = inventory()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines()
    inputs = ('H28_T0_INSTRUMENT_PLAN.md', 'H28_CLOSURE_RESEARCH_PLAN.md',
              'tools/h28_t0.py', 'tests/test_h28_t0.py')
    out = dict(status='PROVISIONAL_DEVELOPMENT_CHECK', accepted_scientific_result=False,
               git_commit=head, dirty_status=dirty,
               input_sha256={p: sha256(ROOT/p) for p in inputs},
               controls=exact, inventory=records,
               timing_seconds=dict(controls=control_seconds, inventory=time.perf_counter()-census_start),
               limitations=['No model loads, forward passes, or learned probes.',
                            'Metadata availability is not artifact validity or learned closure.',
                            'No causal PROMOTE contrast or H28 economics is scored.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f'preserve prior development output: {args.output}')
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=out['status'], controls_ok=True, sources=len(records),
                          timing_seconds=out['timing_seconds'], output=str(args.output))))


if __name__ == '__main__':
    main()
