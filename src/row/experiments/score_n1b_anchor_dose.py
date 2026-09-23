"""Independent scorer for N1b.

Recomputes both triages from the durable cells and the committed N1 endpoints,
rather than trusting the report's summary, and enforces the construction and
stream invariants the plan registered.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'n1b_anchor_dose.py'
THRESHOLD = 0.05
EXPECTED = {'L1_ONLY': (60, {'1': 60, '2': 0}), 'L2_ONLY': (64, {'1': 0, '2': 64}),
            'DOSE_8': (8, None), 'DOSE_32': (32, None)}


def _passes(values):
    return sum(1 for v in values if v < THRESHOLD) >= 2


def score(root=Path('artifacts/n1b_anchor_dose'), output=Path('reports/n1b_anchor_dose.json')):
    root, output = Path(root), Path(output)
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads(output.read_text())
    protocol, sha = manifest['protocol'], manifest['protocol_sha256']
    if fingerprint(protocol) != sha or report['protocol'] != protocol or report['protocol_sha256'] != sha:
        raise ValueError('protocol mismatch')
    for path, expected in protocol['input_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'input changed since the run: {path}')
    here = Path(__file__).parent
    if digest(here / RUNNER) != protocol['implementation_sha256']:
        raise ValueError('runner changed since the run')
    if digest(here / 'n1_anchor_supply.py') != protocol['n1_implementation_sha256']:
        raise ValueError('N1 module changed since the run')
    mtime = output.stat().st_mtime
    for path in list(protocol['input_sha256']) + [str(here / RUNNER)]:
        if Path(path).stat().st_mtime > mtime:
            raise ValueError(f'report is older than its input: {path}')

    problems, cells = [], {}
    for arm in protocol['arms']:
        for world in protocol['worlds']:
            key = f'{arm}_w{world}'
            saved = json.loads((root / 'cells' / f'{key}.json').read_text())
            if (not saved.get('complete') or saved['stamp'] != {'protocol_sha256': sha}
                    or fingerprint(saved['record']) != saved['record_sha256']):
                raise ValueError(f'cell integrity failure: {key}')
            if report['cells'][key] != saved['record']:
                raise ValueError(f'report disagrees with the durable cell: {key}')
            cells[key] = saved['record']

    for key, r in cells.items():
        count, depths = EXPECTED[r['arm']]
        if not math.isfinite(r['terminal_median']) or r['terminal_median'] < 0:
            problems.append(f'{key}: bad terminal median')
        if r['pool_tasks'] != 188 or r['pool_examples'] != 24064:
            problems.append(f'{key}: pool is not 188 tasks / 24,064 examples')
        if r['scored_tasks'] != 64:
            problems.append(f'{key}: not scored on the canonical 64')
        if r['anchors'] != count:
            problems.append(f'{key}: {r["anchors"]} anchors, registered {count}')
        if depths is not None and r['anchor_depths'] != depths:
            problems.append(f'{key}: anchor depths {r["anchor_depths"]}')

    # Every arm in a world must consume the same minibatch index stream, since
    # the pool size and seed sequence are identical by construction.
    for world in protocol['worlds']:
        draws = {cells[f'{a}_w{world}']['first_64_draws'] == cells[f'{protocol["arms"][0]}_w{world}']['first_64_draws']
                 for a in protocol['arms']}
        if draws != {True}:
            problems.append(f'w{world}: arms drew different minibatch indices')

    n1 = json.loads(Path('reports/n1_anchor_supply.json').read_text())['cells']
    by = {a: {w: cells[f'{a}_w{w}']['terminal_median'] for w in protocol['worlds']}
          for a in protocol['arms']}
    passes = {a: _passes(by[a].values()) for a in protocol['arms']}
    l1, l2 = passes['L1_ONLY'], passes['L2_ONLY']
    length = ('BOTH_SUFFICE' if l1 and l2 else 'L1_SUFFICES_ONLY' if l1
              else 'L2_SUFFICES_ONLY' if l2 else 'NEITHER')
    dose_pass = {0: _passes([n1[f'SHAM_w{w}']['terminal_median'] for w in protocol['worlds']]),
                 8: passes['DOSE_8'], 32: passes['DOSE_32'],
                 124: _passes([n1[f'INTERLEAVED_w{w}']['terminal_median'] for w in protocol['worlds']])}
    doses = sorted(dose_pass)
    k_star = next((k for k in doses if all(dose_pass[j] for j in doses if j >= k)), None)
    non_monotone = any(dose_pass[a] and not dose_pass[b] for a in doses for b in doses if b > a)
    if dose_pass[0] or not dose_pass[124]:
        problems.append('committed N1 endpoints do not bracket the dose curve')

    summary = report['summary']
    if summary.get('length_verdict') != length:
        problems.append(f"length verdict {summary.get('length_verdict')} != recomputed {length}")
    if summary.get('dose_k_star') != k_star:
        problems.append(f"k* {summary.get('dose_k_star')} != recomputed {k_star}")
    if json.loads((root / 'status.json').read_text())['state'] != 'complete':
        problems.append('status is not complete')
    if json.loads((root / 'exit.json').read_text())['exit_code'] != 0:
        problems.append('nonzero exit code')

    return {'valid': not problems, 'problems': problems, 'length_verdict': length,
            'dose_k_star': k_star, 'dose_non_monotone': non_monotone,
            'by_arm': {a: {str(w): v for w, v in by[a].items()} for a in by},
            'dose_pass': {str(k): v for k, v in dose_pass.items()},
            'near_threshold_cells': sorted(k for k, r in cells.items() if r['near_threshold']),
            'interpretation': 'N1b development evidence on worlds 0-2; never confirmatory'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/n1b_anchor_dose'))
    parser.add_argument('--report', type=Path, default=Path('reports/n1b_anchor_dose.json'))
    args = parser.parse_args()
    result = score(args.root, args.report)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
