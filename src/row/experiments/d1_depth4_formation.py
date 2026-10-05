"""D1 Tier 1 (exploratory, decision 17): does the confirmed fully online protocol form the substrate at depth 4?

Development band 30-49 (allocated 2026-10-04, decision 17). Arms on the depth-4 stream (`d4_stream`):
- SHUFFLED4: wake alone (the O2 construction one level deeper).
- RW_SLEEP4: wake + O8's in-stream Rerouter (exhaustive search over 12^d routes for every earlier task after each
  arrival) + O3's sleep verbatim on the terminal (64 per task, 8,192 updates). O11's confirmed protocol at depth 4.
- SLEEP4: O3's sleep on the SHUFFLED4 terminal (attribution reference).
The cell functions are the depth-3 constructions with the stream builder and learner swapped for `d4_stream`'s;
sleep and the re-route hook are imported verbatim.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import torch

from row.experiments import census_o8_staleness_position as census
from row.experiments import d4_stream as d4
from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o8_instream_reroute as o8
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('D1_DEPTH4_FORMATION_PLAN.md')
ROOT = Path('artifacts/d1_depth4_formation')
DRY_ROOT = Path('artifacts/d1_depth4_formation_dry')
OUTPUT = Path('reports/d1_depth4_formation.json')
WORLDS = tuple(range(30, 37))
STREAMS = (0, 1, 2)
ARMS = ('RW4', 'SHUFFLED4', 'RW_SLEEP4', 'SLEEP4')
PARENT = {'RW_SLEEP4': 'RW4', 'SLEEP4': 'SHUFFLED4'}
THRESHOLD = 0.05
MEMORY = o3.MEMORY
EXTRA_UPDATES = o3.EXTRA_UPDATES
JOBS = 3
MIN_FREE_GIB = 2.0   # PI 2026-10-03 instruction
DRY_WORLD = 49       # band 30-49, held back from D1's worlds 30-36
DRY_SCALE = 16


def cells(worlds=WORLDS):
    return [(a, w, s) for a in ARMS for w in worlds for s in STREAMS]


def terminal_of(root, arm, w, s):
    return Path(root) / 'work' / f'{arm}_w{w}_s{s}' / 'lifetime' / 'model.pt'


def wake(arm, w, s, work: Path, scale=1, enabled=True):
    """o2.run_single / o8.run_arm's construction on the depth-4 stream; arm RW4 adds O8's Rerouter."""
    cfg, world, st, plan, canonical = d4.build_stream(w, s)
    mixed = dataclasses.replace(world, tasks=tuple(st))
    output = work / 'lifetime'
    hook = o8.Rerouter(st, w, s, enabled) if arm == 'RW4' else None
    ran = o2.run_cfg(cfg, scale)
    summary = ll.run(dataclasses.replace(ran, output_directory=output), o2.KIND, world=mixed,
                     model=d4.planned_model(cfg, plan), return_model=True,
                     replay_seed=d4.replay_seed_for(w, s), prospective_hook=hook)
    model = summary.pop('terminal_model')
    terminal = score(model, SimpleNamespace(tasks=canonical))
    rows = o2.task_summaries(output)
    final = {r['task_id']: float(r['final_nmse']) for r in rows}
    last = rows[-1]
    last_task = next(t for t in st if t.task_id == last['task_id'])
    last_terminal = score(model, SimpleNamespace(tasks=[last_task]))['per_task'][last_task.task_id]
    routes = model.hard_routes()
    pool = o2d.reservoir(st, w, s, MEMORY)
    stale = census.staleness(model, st, plan, pool)
    stale.pop('rows')
    canon_ids = {t.task_id for t in canonical}
    return {'arm': arm, 'world': w, 'stream': s, 'scale': scale,
            'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
            'scored_tasks': len(terminal['per_task']), 'stream_tasks': len(st), 'trained_tasks': len(rows),
            'depth_histogram': {str(d): sum(v == d for v in plan.values()) for d in (1, 2, 3, 4)},
            'route_lengths_match_plan': all(t in routes and len(routes[t]) == plan[t] for t in plan),
            'end_of_task_median': float(np.median([final[t] for t in final if t in canon_ids])),
            'anchor_abs_error': abs(last_terminal - float(last['final_nmse'])),
            'reroute_passes': len(hook.changed_per_pass) if hook else 0,
            'routes_changed_total': int(sum(hook.changed_per_pass)) if hook else 0,
            'reroute_seconds': hook.seconds if hook else 0.0,
            'terminal_stale': stale['stale'], 'terminal_stale_by_depth': stale['by_depth'],
            'prequential': summary.get('cumulative_prequential_gaussian_log_loss'),
            'library_sha256': library_sha(model)}


def load_terminal(path: Path, w, s):
    """o3.load_shuffled_terminal's reload, on the depth-4 stream and learner."""
    cfg, _, st, plan, canonical = d4.build_stream(w, s)
    model = d4.planned_model(cfg, plan)
    for t in st:
        model.begin_task(t.task_id)
    state = torch.load(path, weights_only=True)['model_state_dict']
    for k in state:   # probe task codes saved with the terminal (SO2 lesson: reconstruct probe state too)
        if k.startswith('task_codes.') and k not in model.state_dict():
            model.begin_task(k.split('.', 1)[1], d4.DEPTH)
    model.load_state_dict(state, strict=True)
    return cfg, model, st, canonical


def sleep(arm, w, s, terminal: Path):
    """o3.run_sleep's calls verbatim, on a depth-4 terminal."""
    cfg, model, st, canonical = load_terminal(terminal, w, s)
    before = library_sha(model)
    pool = o2d.reservoir(st, w, s, MEMORY)
    o2c.consolidate(cfg, model, pool, [t.task_id for t in st], [1941, w, s, MEMORY], updates=EXTRA_UPDATES)
    terminal_score = score(model, SimpleNamespace(tasks=canonical))
    return {'arm': arm, 'world': w, 'stream': s, 'terminal_median': terminal_score['median'],
            'terminal_per_task': terminal_score['per_task'], 'scored_tasks': len(terminal_score['per_task']),
            'pool_size': len(pool), 'extra_updates': EXTRA_UPDATES,
            'library_sha256_before': before, 'library_sha256': library_sha(model)}


def run_cell(arm, w, s, root, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    if arm in PARENT:
        rec = sleep(arm, w, s, terminal_of(root, PARENT[arm], w, s))
        rec['scale'] = scale
    else:
        rec = wake(arm, w, s, Path(root) / 'work' / f'{arm}_w{w}_s{s}', scale)
    rec['seconds'] = time.perf_counter() - started
    return rec


def validate(rec):
    key = f"{rec['arm']}_w{rec['world']}_s{rec['stream']}"
    if rec['scored_tasks'] != 64 or len(rec['terminal_per_task']) != 64:
        raise ValueError(f'{key}: not scored on the canonical 64 depth-4 tasks')
    if rec['arm'] in PARENT:
        if rec['extra_updates'] != EXTRA_UPDATES or rec['pool_size'] != 252 * MEMORY:
            raise ValueError(f'{key}: sleep construction')
        if rec['library_sha256'] == rec['library_sha256_before']:
            raise ValueError(f'{key}: sleep left the library unchanged')
        return
    if rec['stream_tasks'] != 252 or rec['trained_tasks'] != 252 or not rec['route_lengths_match_plan']:
        raise ValueError(f'{key}: stream/route check')
    if rec['depth_histogram'] != {'1': 60, '2': 64, '3': 64, '4': 64}:
        raise ValueError(f'{key}: depth histogram {rec["depth_histogram"]}')
    if rec['anchor_abs_error'] > o2.ANCHOR_TOLERANCE:
        raise ValueError(f'{key}: last-task anchor {rec["anchor_abs_error"]}')
    if rec['arm'] == 'RW4' and rec['reroute_passes'] != 251:
        raise ValueError(f'{key}: {rec["reroute_passes"]} re-route passes, not 251')


def passing(m):
    return m is not None and math.isfinite(m) and m < THRESHOLD


def label(k, h):
    if h >= 3:
        return 'HARMS'
    if k >= 18:
        return 'TRANSFERS'
    return 'PARTIAL' if k >= 12 else 'DOES_NOT_TRANSFER'


def summarize(records, worlds=WORLDS):
    M = {k: v['terminal_median'] for k, v in records.items()}
    pairs = [(w, s) for w in worlds for s in STREAMS]
    k = {a: sum(passing(M[f'{a}_w{w}_s{s}']) for w, s in pairs) for a in ARMS}
    better = sum(math.isfinite(M[f'RW_SLEEP4_w{w}_s{s}']) and M[f'RW_SLEEP4_w{w}_s{s}'] < M[f'SLEEP4_w{w}_s{s}']
                 for w, s in pairs)
    harm = sum(passing(M[f'SLEEP4_w{w}_s{s}']) and not passing(M[f'RW_SLEEP4_w{w}_s{s}']) for w, s in pairs)
    return {'passes': k, 'denominator': len(pairs), 'n_better_vs_sleep': better, 'h': harm,
            'label': label(k['RW_SLEEP4'], harm),
            'collapses': {a: sum(not (math.isfinite(M[f'{a}_w{w}_s{s}']) and M[f'{a}_w{w}_s{s}'] < 1.0)
                                 for w, s in pairs) for a in ARMS},
            'medians': {a: float(np.median([M[f'{a}_w{w}_s{s}'] for w, s in pairs])) for a in ARMS}}


def protocol(root=ROOT):
    return {'id': 'd1-depth4-formation-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'exploratory': True, 'root': Path(root).as_posix(), 'worlds': list(WORLDS), 'streams': list(STREAMS),
            'arms': list(ARMS), 'threshold': THRESHOLD, 'memory': MEMORY, 'extra_updates': EXTRA_UPDATES,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml')},
            'implementation_sha256': digest(Path(__file__)), 'd4_sha256': digest(Path(d4.__file__)),
            'o8_sha256': digest(Path(o8.__file__)), 'o3_sha256': digest(Path(o3.__file__)),
            'o2_sha256': digest(Path(o2.__file__)), 'o2c_sha256': digest(Path(o2c.__file__)),
            'o2d_sha256': digest(Path(o2d.__file__)), 'census_sha256': digest(Path(census.__file__)),
            'lifetime_sha256': digest(Path(ll.__file__))}


def _status(root, state, done, total, running, started):
    eta = None if not done else (time.time() - started) / done * (total - done)
    atomic_json(Path(root) / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                             'running': running, 'pid': os.getpid(), 'updated_utc': now(),
                                             'eta_seconds': None if eta is None else round(eta)})


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, worlds=WORLDS):
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root) | {'scale': scale}
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
                                                 'passes': free >= MIN_FREE_GIB, 'jobs': jobs, 'checked_utc': now()})
        if free < MIN_FREE_GIB:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free')
        records, todo = {}, []
        try:
            for arm, w, s in todo_cells:
                key = f'{arm}_w{w}_s{s}'
                path = root / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(root / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append((arm, w, s))
            total = len(todo_cells)
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} scale={scale} '
                                       f'free={free:.1f}GiB')
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                def submit(a, w, s):
                    return pool.submit(run_cell, a, w, s, str(root), scale)
                futures, pending = {}, []
                for a, w, s in todo:
                    if a in PARENT and f'{PARENT[a]}_w{w}_s{s}' not in records:
                        pending.append((a, w, s))
                    else:
                        futures[submit(a, w, s)] = (a, w, s)
                while futures:
                    done_future = next(as_completed(futures))
                    a, w, s = futures.pop(done_future)
                    key = f'{a}_w{w}_s{s}'
                    record = done_future.result()
                    validate(record)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    extra = (f" changed={record['routes_changed_total']} stale={record['terminal_stale']} "
                             f"reroute={record['reroute_seconds']:.0f}s") if a == 'RW4' else ''
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"terminal={record['terminal_median']:.6g}{extra}")
                    for dep, parent in PARENT.items():
                        if parent == a and (dep, w, s) in pending:
                            pending.remove((dep, w, s))
                            futures[submit(dep, w, s)] = (dep, w, s)
                    _status(root, 'running', len(records), total,
                            [f'{x}_w{y}_s{z}' for x, y, z in futures.values()][:jobs], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            if pending:
                raise RuntimeError(f'dependent cells never submitted: {pending}')
            report = {**manifest, 'complete': True, 'cells': records, 'finished_utc': now()}
            if scale == 1:
                report['summary'] = summarize(records, worlds)
            atomic_json(output, report)
            _status(root, 'complete', len(records), total, [], started)
            atomic_json(root / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            _status(root, 'failed', len(records), len(todo_cells), [], started)
            atomic_json(root / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(root / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise


def dry_cells():
    return [(a, DRY_WORLD, 0) for a in ARMS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after', type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.dry_run:
        run(dry_cells(), root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=DRY_SCALE,
            stop_after=args.stop_after, worlds=(DRY_WORLD,))
        print('D1 dry run complete')
        return
    require_clean_code(OUTPUT)
    run(cells())
    print(f'D1 report: {OUTPUT}')


if __name__ == '__main__':
    main()
