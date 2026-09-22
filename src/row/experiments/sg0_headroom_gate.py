"""SG0: the synthesis opportunity gate.

Tier 0, frozen artifacts only. Measures COMMITMENT REGRET - the query cost of
choosing the support-optimal route rather than the query-optimal one - on the
J2A staged libraries and their controls. See SYNTHESIS_OPPORTUNITY_GATE_PLAN.md;
this module implements its Revision 2 estimands and nothing beyond them.

No learner, no training, no world generation. Every route is enumerated once per
task as a (routes, examples) squared-error matrix; support sizes, bootstrap
draws and the two query halves are then means over subsets of its columns.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from row.experiments import audit_j2a_staged_library as j2a
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_so1r_route_only import FrozenLibrary, unflatten
from row.experiments.l0d_depth4_execution_gate import DepthLibrary
from row.experiments.preflight_l0d import load_source
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('SYNTHESIS_OPPORTUNITY_GATE_PLAN.md')
ROOT = Path('artifacts/sg0_headroom_gate')
OUTPUT = Path('reports/sg0_headroom_gate.json')

TASKS = 16
SPLIT_SEED = 3101          # registered: splits Q into Q_a / Q_b
BOOTSTRAP_SEED = 3102      # registered: support resampling for the null floor
BOOTSTRAP_DRAWS = 200
NEAR_TIE_EPS = 0.01

STAGED = ('STAGED5000', 'STAGED3001')
CONTROL = ('NONSTAGED3001', 'RESET5000')

# The single most decisive cell, run first as the structural dry run: if regret
# is zero by construction here, the plan's withdrawal clause fires.
DRY_RUN_CELL = ('STAGED5000', 0, 3, 128)


FULL_DEPTHS = (3, 4)
FULL_SUPPORTS = (128, 8, 2)
FULL_WORLDS = (0, 1, 2)


def is_full_grid(depths, supports, worlds):
    """Only the registered grid may be labelled 'full'; everything else is partial."""
    return (tuple(sorted(depths)) == FULL_DEPTHS and tuple(sorted(supports)) == tuple(sorted(FULL_SUPPORTS))
            and tuple(sorted(worlds)) == FULL_WORLDS)


def grid(depths=(3,), supports=(128, 8, 2), worlds=(0, 1, 2)):
    cells = []
    for name in STAGED + CONTROL:
        for world in worlds:
            for depth in depths:
                for support in supports:
                    cells.append((name, world, depth, support))
    return cells


def protocol(cells, label):
    return {'id': 'sg0-headroom-gate-v1', 'git_commit': git_commit(), 'label': label,
            'cells': [list(c) for c in cells], 'tasks': TASKS, 'slots': 12,
            'split_seed': SPLIT_SEED, 'bootstrap_seed': BOOTSTRAP_SEED,
            'bootstrap_draws': BOOTSTRAP_DRAWS, 'near_tie_eps': NEAR_TIE_EPS,
            'tie_rule': 'lowest flattened route index',
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), j2a.OUTPUT)},
            'implementation_sha256': digest(Path(__file__))}


def per_example_squared_error(library, x, y, depth):
    """(routes, examples) squared error, summed over state dimensions.

    Reducing this over every example reproduces `all_route_support_mse_depth`;
    reducing over a subset gives that subset's MSE with no re-execution.
    """
    n, dimension = x.shape
    with torch.no_grad():
        z = x
        route_count = 1
        for _ in range(depth):
            z = library.candidates(z.reshape(-1, dimension))
            z = z.reshape(n, route_count * library.slots, dimension)
            route_count *= library.slots
        return torch.mean((z - y.unsqueeze(1)) ** 2, dim=2).transpose(0, 1).contiguous()


def argmin_route(errors, columns=None):
    """Lowest-index argmin over routes, optionally restricted to some examples."""
    values = errors.mean(dim=1) if columns is None else errors[:, columns].mean(dim=1)
    return int(torch.argmin(values)), values


def nmse_of_route(library, task, route):
    return j2a.query_error(library, route, task['eval_x'], task['eval_y'])


def half_nmse(errors_q, route, columns, denominator):
    """NMSE on a query half, from the precomputed matrix."""
    return float(errors_q[route, columns].mean()) / denominator


def measure_cell(name, world, depth, support):
    started = time.perf_counter()
    cfg, world_obj, model, stage, anchor = load_source(name, world)
    library = DepthLibrary(model)
    before = library_sha(model)
    tasks = j2a.held_out_tasks(cfg, world_obj)[:TASKS]

    split_rng = np.random.default_rng(np.random.SeedSequence([SPLIT_SEED, world, depth]))
    boot_rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, world, depth, support]))

    rows = []
    for index, task in enumerate(tasks):
        if depth == 3 and list(task['program']) != anchor['held_out'][str(index)]['program']:
            raise ValueError('held-out task order differs from J2A')
        x = torch.tensor(task['train_x'], dtype=torch.float32)
        y = torch.tensor(task['train_y'], dtype=torch.float32)
        qx = torch.tensor(task['eval_x'], dtype=torch.float32)
        qy = torch.tensor(task['eval_y'], dtype=torch.float32)
        if len(x) < support:
            raise ValueError('support larger than the canonical support set')

        t0 = time.perf_counter()
        errors_s = per_example_squared_error(library, x, y, depth)
        errors_q = per_example_squared_error(library, qx, qy, depth)
        seconds = time.perf_counter() - t0
        if not (bool(torch.isfinite(errors_s).all()) and bool(torch.isfinite(errors_q).all())):
            raise ValueError('nonfinite squared-error matrix')

        target = np.asarray(task['eval_y'], dtype=np.float64)
        denominator = float(np.mean(np.square(target - np.mean(target, axis=0, keepdims=True))))
        if denominator <= 0.0:
            raise ValueError('NMSE undefined for a constant target')

        n_query = errors_q.shape[1]
        permutation = split_rng.permutation(n_query)
        q_a = torch.tensor(np.sort(permutation[: n_query // 2]), dtype=torch.long)
        q_b = torch.tensor(np.sort(permutation[n_query // 2:]), dtype=torch.long)

        support_columns = torch.arange(support, dtype=torch.long)
        r_hat, support_values = argmin_route(errors_s, support_columns)
        r_star, _ = argmin_route(errors_q, q_a)

        nmse_hat_b = half_nmse(errors_q, r_hat, q_b, denominator)
        nmse_star_b = half_nmse(errors_q, r_star, q_b, denominator)
        regret = nmse_hat_b - nmse_star_b

        ordered = torch.sort(support_values).values
        best = float(ordered[0])
        second = float(ordered[1])
        identifiability = (second - best) / max(best, 1e-12)
        near_tie = torch.nonzero(support_values <= (1.0 + NEAR_TIE_EPS) * best, as_tuple=False).flatten()
        tie_nmse = [half_nmse(errors_q, int(r), q_b, denominator) for r in near_tie]
        near_tie_disagreement = float(max(tie_nmse) - min(tie_nmse))

        deltas = []
        for _ in range(BOOTSTRAP_DRAWS):
            draw = torch.tensor(boot_rng.integers(0, support, size=support), dtype=torch.long)
            resampled, _ = argmin_route(errors_s, draw)
            deltas.append(abs(half_nmse(errors_q, resampled, q_b, denominator) - nmse_hat_b))
        floor = float(np.percentile(deltas, 95))

        row = {'task': index, 'program': list(task['program']),
               'r_hat': unflatten(r_hat, library.slots, depth),
               'r_star': unflatten(r_star, library.slots, depth),
               'r_hat_index': r_hat, 'r_star_index': r_star,
               'routes_differ': r_hat != r_star,
               'nmse_hat_qb': nmse_hat_b, 'nmse_star_qb': nmse_star_b, 'regret': regret,
               'null_floor': floor, 'regret_above_floor': regret > floor,
               'identifiability': identifiability, 'best_support_mse': best,
               'near_tie_size': int(near_tie.numel()),
               'near_tie_disagreement': near_tie_disagreement,
               'enum_seconds': seconds}

        if depth == 3 and support == 128:
            original = anchor['held_out'][str(index)]
            row['anchor_route_matches'] = row['r_hat'] == original['enum_route']
            row['anchor_nmse'] = nmse_of_route(library, task, row['r_hat'])
            row['anchor_nmse_matches'] = row['anchor_nmse'] == original['enum']
            if not (row['anchor_route_matches'] and row['anchor_nmse_matches']):
                raise ValueError(f'J2A ENUM anchor failed: {name} w{world} task {index}')
        rows.append(row)

    if library_sha(model) != before:
        raise ValueError('library changed during the cell')

    regrets = [r['regret'] for r in rows]
    floors = [r['null_floor'] for r in rows]
    return {'name': name, 'world': world, 'depth': depth, 'support': support,
            'staged': name in STAGED, 'library_sha256': before, 'rows': rows,
            'median_regret': float(np.median(regrets)),
            'median_floor': float(np.median(floors)),
            'cell_counts_toward_k': bool(np.median(regrets) > np.median(floors)),
            'routes_differ_count': sum(r['routes_differ'] for r in rows),
            'median_identifiability': float(np.median([r['identifiability'] for r in rows])),
            'median_near_tie_disagreement': float(np.median([r['near_tie_disagreement'] for r in rows])),
            'median_near_tie_size': float(np.median([r['near_tie_size'] for r in rows])),
            'seconds': time.perf_counter() - started}


def validate_cell(record):
    if len(record['rows']) != TASKS:
        raise ValueError('task count mismatch')
    if len({tuple(r['program']) for r in record['rows']}) != TASKS:
        raise ValueError('programs not distinct')
    for row in record['rows']:
        for field in ('nmse_hat_qb', 'nmse_star_qb', 'regret', 'null_floor',
                      'identifiability', 'near_tie_disagreement', 'enum_seconds'):
            if not math.isfinite(row[field]):
                raise ValueError(f'nonfinite {field}')
        for field in ('nmse_hat_qb', 'nmse_star_qb', 'null_floor', 'identifiability',
                      'near_tie_disagreement', 'enum_seconds'):
            if row[field] < 0:
                raise ValueError(f'negative {field}')
        if row['near_tie_size'] < 1:
            raise ValueError('near-tie set must contain the best route')
        if not row['routes_differ'] and row['regret'] != 0.0:
            raise ValueError('identical routes must give exactly zero regret')


def cell_key(cell):
    name, world, depth, support = cell
    return f'{name}_w{world}_d{depth}_s{support}'


def run(cells, label, root=ROOT, output=OUTPUT):
    ROOT_, OUTPUT_ = Path(root), Path(output)
    ROOT_.mkdir(parents=True, exist_ok=True)
    (ROOT_ / 'cells').mkdir(exist_ok=True)
    expected = protocol(cells, label)
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    with writer_lock(ROOT_ / 'launcher.lock'):
        if (ROOT_ / 'manifest.json').exists() and json.loads((ROOT_ / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; preserve the prior run or retire the path')
        atomic_json(ROOT_ / 'manifest.json', manifest)
        atomic_json(ROOT_ / 'run.pid', {'pid': __import__('os').getpid(), 'started_utc': now()})
        records = {}
        try:
            log_line(ROOT_ / 'run.log', f'LAUNCH {sha} label={label} cells={len(cells)}')
            for position, cell in enumerate(cells):
                key = cell_key(cell)
                path = ROOT_ / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(ROOT_ / 'run.log', f'reused validated cell {key}')
                else:
                    log_line(ROOT_ / 'run.log', f'cell start {key}')
                    record = measure_cell(*cell)
                    validate_cell(record)
                    atomic_json(path, {'stamp': {'protocol_sha256': sha}, 'complete': True,
                                       'record': record, 'record_sha256': fingerprint(record),
                                       'finished_utc': now()})
                    records[key] = record
                    log_line(ROOT_ / 'run.log',
                             f'cell finished {key} {record["seconds"]:.3f}s '
                             f'median_regret={record["median_regret"]:.6g} '
                             f'floor={record["median_floor"]:.6g}')
                atomic_json(ROOT_ / 'status.json',
                            {'state': 'running', 'cells_done': position + 1, 'cells_total': len(cells),
                             'running': [], 'updated_utc': now()})
            atomic_json(OUTPUT_, {**manifest, 'complete': True, 'cells': records,
                                 'summary': summarize(records, label), 'finished_utc': now()})
            atomic_json(ROOT_ / 'status.json', {'state': 'complete', 'cells_done': len(cells),
                                               'cells_total': len(cells), 'updated_utc': now()})
            atomic_json(ROOT_ / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            atomic_json(ROOT_ / 'status.json', {'state': 'failed', 'cells_done': len(records),
                                               'cells_total': len(cells), 'updated_utc': now()})
            atomic_json(ROOT_ / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(ROOT_ / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise


def summarize(records, label):
    staged = [r for r in records.values() if r['staged']]
    control = [r for r in records.values() if not r['staged']]
    k = sum(r['cell_counts_toward_k'] for r in staged)
    summary = {'label': label, 'staged_cells': len(staged), 'control_cells': len(control), 'k': k,
               'staged_median_regret': float(np.median([r['median_regret'] for r in staged])) if staged else None,
               'control_median_regret': float(np.median([r['median_regret'] for r in control])) if control else None,
               'staged_routes_differ': sum(r['routes_differ_count'] for r in staged),
               'staged_route_comparisons': len(staged) * TASKS,
               'interpretation': 'SG0 headroom measurement; not a PX7 verdict and not a mechanism comparison'}
    if label != 'full':
        summary['triage'] = 'NOT APPLICABLE'
        summary['triage_reason'] = (
            'the registered triage is defined over the 36 staged cells of the full grid; '
            f'this run is labelled {label!r} and reports descriptive values only')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='run only the single decisive cell (the structural dry run)')
    parser.add_argument('--depths', type=int, nargs='+', default=[3])
    parser.add_argument('--supports', type=int, nargs='+', default=[128, 8, 2])
    parser.add_argument('--worlds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--report', type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    require_clean_code(args.report)
    for check in ('tools/check_prereg.py', 'tools/check_invalid.py'):
        subprocess.run([sys.executable, check], check=True)
    if args.dry_run:
        cells, label = [DRY_RUN_CELL], 'dry-run'
    else:
        depths, supports, worlds = tuple(args.depths), tuple(args.supports), tuple(args.worlds)
        cells = grid(depths, supports, worlds)
        # A partial grid must never present itself as the registered triage.
        label = 'full' if is_full_grid(depths, supports, worlds) else (
            'partial-d' + ''.join(map(str, sorted(depths)))
            + '-s' + '_'.join(map(str, sorted(supports)))
            + '-w' + ''.join(map(str, sorted(worlds))))
    run(cells, label, args.root, args.report)
    print(f'SG0 report: {args.report}')


if __name__ == '__main__':
    main()
