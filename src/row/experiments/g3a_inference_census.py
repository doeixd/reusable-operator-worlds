"""G3a: does route inference BIND on the formed m = 4 vocabularies? (Tier 0 census, no new lifetime.)

Plan: `G3A_INFERENCE_CENSUS_PLAN.md`. For each (m, world) of the G1 probe (m in {0, 4}, worlds 70-72), the slept
vocabulary is rebuilt from G1's saved REROUTE_WAKE terminal with `o3.run_sleep`'s exact calls and must reproduce G1's
recorded per-task terminal bitwise (anchor A1). SG0's estimands are then applied verbatim (`sg0_headroom_gate`'s
per-example squared-error matrix, argmin with lowest-index ties, split and bootstrap seeds, 200 draws, near-tie
eps 0.01) to the 64 held-out depth-3 programs G1 measured (`j2a.held_out_tasks`), at supports 128, 8 and 2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import torch

from row.experiments import audit_j2a_staged_library as j2a
from row.experiments import g1_formation_probe as g1
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import sg0_headroom_gate as sg0
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary, unflatten
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('G3A_INFERENCE_CENSUS_PLAN.md')
G1_REPORT = Path('reports/g1_formation_probe.json')
G1_WORK = Path('artifacts/g1_formation_probe/work')
ROOT = Path('artifacts/g3a_inference_census')
DRY_ROOT = Path('artifacts/g3a_inference_census_dry')
OUTPUT = Path('reports/g3a_inference_census.json')
WORLDS = (70, 71, 72)
SHARED = (0.0, 4.0)
SUPPORTS = (128, 8, 2)
DEPTH = 3
PROGRAMS = 64
FAIL = 0.05            # a selected route whose query NMSE is at or above this fails the task (O-line threshold)
RULE_B_MIN = 0.10      # rule B: pooled bind fraction at one support
RULE_B_MARGIN = 0.05   # ... and its excess over the m = 0 control at that support
HEADROOM_K = 2         # rule A (SG0's cell rule): k >= 2 of the 9 m = 4 cells
JOBS = 3
MIN_FREE_GIB = 2.0


def jobs():
    return [(m, w) for m in SHARED for w in WORLDS]


def key_of(m, w):
    return f'm{m:g}_w{w}'


def terminal_path(m, w):
    return G1_WORK / g1.key_of('REROUTE_WAKE', m, w) / 'lifetime' / 'model.pt'


def rebuild(m, w):
    """G1's RW_SLEEP model, rebuilt exactly as `g1.run_cell` rebuilt it."""
    g1.use_generator(m)
    terminal = terminal_path(m, w)
    cfg3, model, st, canonical = o3.load_shuffled_terminal(terminal, w, g1.STREAM)
    pool = o2d.reservoir(st, w, g1.STREAM, o3.MEMORY)
    o2c.consolidate(cfg3, model, pool, [t.task_id for t in st], [1941, w, g1.STREAM, o3.MEMORY],
                    updates=o3.EXTRA_UPDATES)
    per_task = score(model, SimpleNamespace(tasks=canonical))['per_task']
    cfg3b, world3, _, _, _ = o2.build_stream('SHUFFLED', w, g1.STREAM)
    return model, per_task, j2a.held_out_tasks(cfg3b, world3), digest(terminal)


def measure_task(library, task, world, programs_seen):
    """SG0's per-program estimands at every support, from one pair of error matrices."""
    x = torch.tensor(task['train_x'], dtype=torch.float32)
    y = torch.tensor(task['train_y'], dtype=torch.float32)
    qx = torch.tensor(task['eval_x'], dtype=torch.float32)
    qy = torch.tensor(task['eval_y'], dtype=torch.float32)
    errors_s = sg0.per_example_squared_error(library, x, y, DEPTH)
    errors_q = sg0.per_example_squared_error(library, qx, qy, DEPTH)
    if not (bool(torch.isfinite(errors_s).all()) and bool(torch.isfinite(errors_q).all())):
        raise ValueError('nonfinite squared-error matrix')
    target = np.asarray(task['eval_y'], dtype=np.float64)
    denominator = float(np.mean(np.square(target - np.mean(target, axis=0, keepdims=True))))
    split_rng = np.random.default_rng(np.random.SeedSequence([sg0.SPLIT_SEED, world, DEPTH, programs_seen]))
    permutation = split_rng.permutation(errors_q.shape[1])
    q_a = torch.tensor(np.sort(permutation[: errors_q.shape[1] // 2]), dtype=torch.long)
    q_b = torch.tensor(np.sort(permutation[errors_q.shape[1] // 2:]), dtype=torch.long)
    r_star, _ = sg0.argmin_route(errors_q, q_a)
    nmse_star_b = sg0.half_nmse(errors_q, r_star, q_b, denominator)
    out = {}
    for support in SUPPORTS:
        boot_rng = np.random.default_rng(
            np.random.SeedSequence([sg0.BOOTSTRAP_SEED, world, DEPTH, support, programs_seen]))
        r_hat, values = sg0.argmin_route(errors_s, torch.arange(support, dtype=torch.long))
        nmse_hat_b = sg0.half_nmse(errors_q, r_hat, q_b, denominator)
        ordered = torch.sort(values).values
        best, second = float(ordered[0]), float(ordered[1])
        near = torch.nonzero(values <= (1.0 + sg0.NEAR_TIE_EPS) * best, as_tuple=False).flatten()
        tie_nmse = [sg0.half_nmse(errors_q, int(r), q_b, denominator) for r in near]
        deltas = []
        for _ in range(sg0.BOOTSTRAP_DRAWS):
            draw = torch.tensor(boot_rng.integers(0, support, size=support), dtype=torch.long)
            resampled, _ = sg0.argmin_route(errors_s, draw)
            deltas.append(abs(sg0.half_nmse(errors_q, resampled, q_b, denominator) - nmse_hat_b))
        floor = float(np.percentile(deltas, 95))
        regret = nmse_hat_b - nmse_star_b
        out[str(support)] = {
            'r_hat': unflatten(r_hat, library.slots, DEPTH), 'r_hat_index': r_hat, 'r_star_index': r_star,
            'routes_differ': r_hat != r_star, 'nmse_hat_qb': nmse_hat_b, 'nmse_star_qb': nmse_star_b,
            'regret': regret, 'null_floor': floor, 'binds': bool(regret > floor),
            'inference_failure': bool(nmse_hat_b >= FAIL and nmse_star_b < FAIL),
            'identifiability': (second - best) / max(best, 1e-12), 'best_support_mse': best,
            'near_tie_size': int(near.numel()), 'near_tie_disagreement': float(max(tie_nmse) - min(tie_nmse))}
    return out


def run_job(m, w, programs=PROGRAMS):
    torch.set_num_threads(1)
    started = time.perf_counter()
    g1cells = json.loads(G1_REPORT.read_text())['cells']
    model, per_task, held_out, terminal_sha = rebuild(m, w)
    recorded = g1cells[g1.key_of('RW_SLEEP', m, w)]
    anchor_terminal = per_task == recorded['terminal_per_task']
    library = FrozenLibrary(model)
    library.steps = DEPTH
    before = library_sha(model)
    rows = []
    for index, task in enumerate(held_out[:programs]):
        if len(task['program']) != DEPTH:
            raise ValueError('program length differs from the searched depth')
        rows.append({'task': index, 'program': list(task['program']), 'by_support': measure_task(library, task, w, index)})
    if library_sha(model) != before:
        raise ValueError('library changed during the census')
    # Anchor A2: G1's own formed-vocabulary diagnostics, recomputed by G1's code on the rebuilt model, must equal the
    # recorded ones exactly (it re-runs the 12^3 enumeration on the same 64 programs, independently of SG0's code).
    _, _, st, plan, _ = o2.build_stream('SHUFFLED', w, g1.STREAM)
    anchor_formed = g1.formed_diagnostics(model, st, plan, held_out) == recorded['formed']
    by_support = {}
    for s in map(str, SUPPORTS):
        rs = [r['by_support'][s] for r in rows]
        regrets, floors = [r['regret'] for r in rs], [r['null_floor'] for r in rs]
        by_support[s] = {'median_regret': float(np.median(regrets)), 'median_floor': float(np.median(floors)),
                         'cell_counts': bool(np.median(regrets) > np.median(floors)),
                         'binds': sum(r['binds'] for r in rs), 'inference_failures': sum(r['inference_failure'] for r in rs),
                         'routes_differ': sum(r['routes_differ'] for r in rs),
                         'median_identifiability': float(np.median([r['identifiability'] for r in rs])),
                         'median_nmse_hat_qb': float(np.median([r['nmse_hat_qb'] for r in rs])),
                         'tasks_with_near_ties': sum(r['near_tie_size'] > 1 for r in rs),
                         'median_near_tie_disagreement': float(np.median([r['near_tie_disagreement'] for r in rs]))}
    return {'shared': m, 'world': w, 'programs': len(rows), 'terminal_sha256': terminal_sha,
            'library_sha256': before, 'anchor_terminal': anchor_terminal, 'anchor_formed': anchor_formed, 'by_support': by_support, 'rows': rows,
            'seconds': time.perf_counter() - started}


def validate(rec, programs=PROGRAMS):
    k = key_of(rec['shared'], rec['world'])
    if not rec['anchor_terminal']:
        raise ValueError(f'{k}: rebuilt vocabulary does not reproduce G1 (anchor A1)')
    if not rec['anchor_formed']:
        raise ValueError(f'{k}: G1 diagnostics not reproduced on the rebuilt vocabulary (anchor A2)')
    if rec['programs'] != programs or len({tuple(r['program']) for r in rec['rows']}) != programs:
        raise ValueError(f'{k}: program count')
    for row in rec['rows']:
        for s, v in row['by_support'].items():
            for f in ('nmse_hat_qb', 'nmse_star_qb', 'regret', 'null_floor', 'identifiability'):
                if not math.isfinite(v[f]):
                    raise ValueError(f'{k}: nonfinite {f}')
            if v['null_floor'] < 0 or v['near_tie_size'] < 1:
                raise ValueError(f'{k}: floor/tie')
            if not v['routes_differ'] and v['regret'] != 0.0:
                raise ValueError(f'{k}: identical routes with nonzero regret')


# ------------------------------------------------------------------ reading (registered in the plan)

def reading(records):
    def cells(m):
        return [records[key_of(m, w)] for w in WORLDS]
    k = {f'm{m:g}': sum(r['by_support'][str(s)]['cell_counts'] for r in cells(m) for s in SUPPORTS) for m in SHARED}
    frac = {f'm{m:g}': {str(s): sum(r['by_support'][str(s)]['binds'] for r in cells(m)) / (len(WORLDS) * PROGRAMS)
                        for s in SUPPORTS} for m in SHARED}
    b_supports = [str(s) for s in SUPPORTS
                  if frac['m4'][str(s)] >= RULE_B_MIN and frac['m4'][str(s)] - frac['m0'][str(s)] >= RULE_B_MARGIN]
    if k['m0'] >= HEADROOM_K:
        label = 'CONTROL_FIRES'
    elif k['m4'] >= HEADROOM_K:
        label = 'HEADROOM'
    elif b_supports:
        label = 'PARTIAL'
    else:
        label = 'NO_HEADROOM'
    failures = {f'm{m:g}': {str(s): sum(r['by_support'][str(s)]['inference_failures'] for r in cells(m))
                            for s in SUPPORTS} for m in SHARED}
    return {'label': label, 'k': k, 'cells_per_arm': len(WORLDS) * len(SUPPORTS), 'bind_fraction': frac,
            'rule_b_supports': b_supports, 'inference_failures': failures, 'programs_per_support': len(WORLDS) * PROGRAMS,
            'median_identifiability': {f'm{m:g}': {str(s): [r['by_support'][str(s)]['median_identifiability']
                                                            for r in cells(m)] for s in SUPPORTS} for m in SHARED}}


# ------------------------------------------------------------------ protocol and run

def protocol(root, programs):
    return {'id': 'g3a-inference-census-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 0,
            'worlds': list(WORLDS), 'shared': list(SHARED), 'supports': list(SUPPORTS), 'depth': DEPTH,
            'programs': programs, 'fail': FAIL, 'rule_b': [RULE_B_MIN, RULE_B_MARGIN], 'headroom_k': HEADROOM_K,
            'sg0_seeds': [sg0.SPLIT_SEED, sg0.BOOTSTRAP_SEED, sg0.BOOTSTRAP_DRAWS, sg0.NEAR_TIE_EPS],
            'root': Path(root).as_posix(),
            'terminals_sha256': {key_of(m, w): digest(terminal_path(m, w)) for m, w in jobs()},
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), G1_REPORT)},
            'implementation_sha256': digest(Path(__file__)), 'sg0_sha256': digest(Path(sg0.__file__)),
            'g1_sha256': digest(Path(g1.__file__)), 'o3_sha256': digest(Path(o3.__file__)),
            'o2c_sha256': digest(Path(o2c.__file__)), 'o2d_sha256': digest(Path(o2d.__file__)),
            'j2a_sha256': digest(Path(j2a.__file__))}


def _status(root, state, done, total, running, started):
    elapsed = time.time() - started
    atomic_json(Path(root) / 'status.json', {
        'state': state, 'cells_done': done, 'cells_total': total, 'running': running, 'pid': os.getpid(),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started)), 'updated_utc': now(),
        'eta_seconds': elapsed / done * (total - done) if done else None})


def run(todo, root=ROOT, output=OUTPUT, programs=PROGRAMS, stop_after=None, jobs_n=JOBS):
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root, programs) | {'jobs': [list(j) for j in todo]}
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    started = time.time()
    with writer_lock(root / 'launcher.lock'):
        if (root / 'manifest.json').exists() and json.loads((root / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; retire the path rather than overwrite it')
        atomic_json(root / 'manifest.json', manifest)
        atomic_json(root / 'run.pid', {'pid': os.getpid(), 'started_utc': now()})
        free = psutil.virtual_memory().available / 2 ** 30
        atomic_json(root / 'precondition.json', {'free_gib': free, 'required_gib': MIN_FREE_GIB,
                                                 'passes': free >= MIN_FREE_GIB, 'checked_utc': now()})
        if free < MIN_FREE_GIB:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free')
        records, pending = {}, []
        try:
            for m, w in todo:
                path = root / 'cells' / f'{key_of(m, w)}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key_of(m, w)}')
                    records[key_of(m, w)] = saved['record']
                    log_line(root / 'run.log', f'reused validated cell {key_of(m, w)}')
                else:
                    pending.append((m, w))
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(pending)} reused={len(records)} programs={programs} '
                                       f'free={free:.1f}GiB')
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs_n) as pool:
                futures = {pool.submit(run_job, m, w, programs): (m, w) for m, w in pending}
                _status(root, 'running', len(records), len(todo), [key_of(*j) for j in pending[:jobs_n]], started)
                for future in as_completed(futures):
                    m, w = futures[future]
                    record = future.result()
                    validate(record, programs)
                    atomic_json(root / 'cells' / f'{key_of(m, w)}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key_of(m, w)] = record
                    finished += 1
                    bs = record['by_support']
                    log_line(root / 'run.log', f"cell finished {key_of(m, w)} {record['seconds']:.1f}s " + ' '.join(
                        f"s{s}:regret={bs[s]['median_regret']:.4g}/floor={bs[s]['median_floor']:.4g}"
                        f"/binds={bs[s]['binds']}/fail={bs[s]['inference_failures']}" for s in bs))
                    _status(root, 'running', len(records), len(todo), [], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            report = {**manifest, 'complete': True, 'cells': records, 'finished_utc': now()}
            if programs == PROGRAMS and len(records) == len(jobs()):
                report['reading'] = reading(records)
            atomic_json(output, report)
            log_line(root / 'run.log', f"EXIT 0 {report.get('reading', {}).get('label')}")
            _status(root, 'complete', len(records), len(todo), [], started)
            atomic_json(root / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            log_line(root / 'run.log', 'FAILED ' + traceback.format_exc().replace('\n', ' | '))
            _status(root, 'failed', len(records), len(todo), [], started)
            atomic_json(root / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(root / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after', type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.dry_run:   # two real (m, world) pairs at 4 programs each: every path, both anchors, the restart test
        run([(4.0, 70), (0.0, 70)], root=DRY_ROOT, output=DRY_ROOT / 'report.json', programs=4,
            stop_after=args.stop_after)
        print(f'dry run complete: {DRY_ROOT}')
        return
    require_clean_code(OUTPUT)
    run(jobs())
    print(f'G3a report: {OUTPUT}')


if __name__ == '__main__':
    main()
