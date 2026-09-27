"""O6 SEALED confirmation: order-free online anchors + retained memory + re-route + consolidation.

Plan: `O6_SEALED_REROUTE_CONFIRMATION_PLAN.md` (frozen 666f961). Sealed worlds 915-929.
SHUFFLED, SLEEP and PLAIN are O4's constructions (`o3.run_cell`); REROUTE_SLEEP is O5's
`run_cell` construction applied to this run's own SHUFFLED terminal. Gates E1/E2/E3/E3b
reproduce committed development cells bitwise before any sealed world is touched (E3b: with
the swap disabled REROUTE_SLEEP must equal SLEEP, so the swap is the only difference); E4b
runs on sealed streams only after the freeze.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch

from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o5_reroute_sleep as o5
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score as score_model
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O6_SEALED_REROUTE_CONFIRMATION_PLAN.md')
O3_REPORT = Path('reports/o3_online_sleep_v2.json')
O5_REPORT = Path('reports/o5_reroute_sleep.json')
ROOT = Path('artifacts/o6_sealed_reroute')
DRY_ROOT = Path('artifacts/o6_sealed_reroute_dry')
OUTPUT = Path('reports/o6_sealed_reroute.json')
WORLDS = tuple(range(915, 930))
STREAMS = (0, 1, 2)
ARMS = ('SHUFFLED', 'REROUTE_SLEEP', 'SLEEP', 'PLAIN')
ARM_STREAMS = {'SHUFFLED': STREAMS, 'REROUTE_SLEEP': STREAMS, 'SLEEP': STREAMS, 'PLAIN': (0,)}
DEPENDENT = ('REROUTE_SLEEP', 'SLEEP')   # submitted, in this order, once the cell's SHUFFLED terminal exists
THRESHOLD = 0.05
CONFIRM_AT = 41
ATTRIBUTE_AT = 30
JOBS = 3
MIN_FREE_GIB = 8.0
DRY_WORLD = 27   # development band 3, held back; never a sealed world


def cells(worlds=WORLDS):
    return [(a, w, s) for a in ARMS for w in worlds for s in ARM_STREAMS[a]]


def terminal_of(root, w, s):
    return Path(root) / 'work' / f'SHUFFLED_w{w}_s{s}' / 'lifetime' / 'model.pt'


def run_reroute_sleep(w, s, terminal: Path, swap=True):
    """O5's run_cell construction on an explicit terminal path; swap=False is the E3b control."""
    cfg3, model, stream_tasks, canonical = o3.load_shuffled_terminal(terminal, w, s)
    _, _, _, plan, _ = o2.build_stream('SHUFFLED', w, s)
    before = library_sha(model)
    pool = o2d.reservoir(stream_tasks, w, s, o3.MEMORY)
    changed = o5.reroute(model, stream_tasks, plan, pool) if swap else 0
    o2c.consolidate(cfg3, model, pool, [t.task_id for t in stream_tasks], [1941, w, s, o3.MEMORY],
                    updates=o3.EXTRA_UPDATES)
    terminal = score_model(model, SimpleNamespace(tasks=canonical))
    return {'routes_changed': changed, 'swap': swap, 'pool_size': len(pool), 'extra_updates': o3.EXTRA_UPDATES,
            'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
            'scored_tasks': len(terminal['per_task']),
            'library_sha256_before': before, 'library_sha256': library_sha(model)}


def run_cell(arm, w, s, root, scale=1):
    if arm != 'REROUTE_SLEEP':
        return o3.run_cell(arm, w, s, root, scale)
    torch.set_num_threads(1)
    started = time.perf_counter()
    rec = run_reroute_sleep(w, s, terminal_of(root, w, s))
    rec.update({'arm': arm, 'world': w, 'stream': s, 'scale': scale, 'seconds': time.perf_counter() - started})
    return rec


def validate(rec):
    if rec['arm'] != 'REROUTE_SLEEP':
        o3.validate(rec)
        return
    key = f"REROUTE_SLEEP_w{rec['world']}_s{rec['stream']}"
    if rec['scored_tasks'] != 64 or len(rec['terminal_per_task']) != 64:
        raise ValueError(f'{key}: not scored on the canonical 64')
    if rec['extra_updates'] != o3.EXTRA_UPDATES or rec['pool_size'] != 188 * o3.MEMORY or not rec['swap']:
        raise ValueError(f'{key}: construction')
    if rec['library_sha256'] == rec['library_sha256_before']:
        raise ValueError(f'{key}: sleep left the library unchanged')


def protocol(root=ROOT):
    search = {m.__name__: digest(Path(m.__file__)) for m in (
        sys.modules[o5.enum_route.__module__], sys.modules[o5.FrozenLibrary.__module__])}
    return {'id': 'o6-sealed-reroute-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 2,
            'sealed': True, 'worlds': list(WORLDS), 'arm_streams': {a: list(s) for a, s in ARM_STREAMS.items()},
            'threshold': THRESHOLD, 'confirm_at': CONFIRM_AT, 'attribute_at': ATTRIBUTE_AT, 'jobs': JOBS,
            'root': Path(root).as_posix(),
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), O3_REPORT, O5_REPORT)},
            'implementation_sha256': digest(Path(__file__)), 'o3_sha256': digest(Path(o3.__file__)),
            'o5_sha256': digest(Path(o5.__file__)), 'search_sha256': search,
            'o2_sha256': digest(Path(o2.__file__)), 'o2c_sha256': digest(Path(o2c.__file__)),
            'o2d_sha256': digest(Path(o2d.__file__)), 'lifetime_sha256': digest(Path(ll.__file__)),
            'learner_sha256': digest(Path(sys.modules[o2.PlannedDepthRotatedLearner.__module__].__file__))}


def passing(m):
    return math.isfinite(m) and m < THRESHOLD


def attribution(better):
    return 'ATTRIBUTED' if better >= ATTRIBUTE_AT else 'NOT_ATTRIBUTED'


def summarize(records, worlds=WORLDS):
    M = {k: v['terminal_median'] for k, v in records.items()}
    k = {a: sum(passing(M[f'{a}_w{w}_s{s}']) for w in worlds for s in ARM_STREAMS[a]) for a in ARM_STREAMS}
    floor = k['PLAIN'] > 0
    label = 'FLOOR_FAILED' if floor else ('CONFIRMED' if k['REROUTE_SLEEP'] >= CONFIRM_AT else 'NOT_CONFIRMED')
    better = sum(math.isfinite(M[f'REROUTE_SLEEP_w{w}_s{s}'])
                 and M[f'REROUTE_SLEEP_w{w}_s{s}'] < M[f'SLEEP_w{w}_s{s}'] for w in worlds for s in STREAMS)
    return {'passes': k, 'denominator': 3 * len(worlds), 'label': label, 'floor_failed': floor,
            'n_better': better, 'attribution': attribution(better),
            'collapses': {a: sum(not (math.isfinite(M[f'{a}_w{w}_s{s}']) and M[f'{a}_w{w}_s{s}'] < 1.0)
                                 for w in worlds for s in ARM_STREAMS[a]) for a in ARM_STREAMS}}


def gates(root=ROOT):
    g = Path(root) / 'gates'
    g.mkdir(parents=True, exist_ok=True)
    ref = json.loads(O3_REPORT.read_text())['cells']
    ref5 = json.loads(O5_REPORT.read_text())['cells']['O3_w20_s0']

    def same(a, b):
        return a['library_sha256'] == b['library_sha256'] and a['terminal_per_task'] == b['terminal_per_task']

    out = {}
    e1 = o3.run_cell('SHUFFLED', 20, 0, str(g / 'E1'))
    out['E1'] = {'passes': same(e1, ref['SHUFFLED_w20_s0'])}
    log_line(g / 'gates.log', f"E1 {out['E1']}")
    term = terminal_of(g / 'E1', 20, 0)
    out['E2'] = {'passes': same(o3.run_sleep(20, 0, term), ref['SLEEP_w20_s0'])}
    log_line(g / 'gates.log', f"E2 {out['E2']}")
    e3 = run_reroute_sleep(20, 0, term)
    out['E3'] = {'passes': same(e3, ref5) and e3['routes_changed'] == ref5['routes_changed'],
                 'routes_changed': e3['routes_changed']}
    log_line(g / 'gates.log', f"E3 {out['E3']}")
    e3b = run_reroute_sleep(20, 0, term, swap=False)
    out['E3b'] = {'passes': same(e3b, ref['SLEEP_w20_s0']) and not same(e3b, e3)}
    log_line(g / 'gates.log', f"E3b {out['E3b']}")
    e4b = {}
    for w in WORLDS:   # builds sealed streams: allowed only after the freeze (plan)
        for s in STREAMS:
            _, _, st, plan, _ = o2.build_stream('SHUFFLED', w, s)
            e4b[f'w{w}_s{s}'] = len({plan[t.task_id] for t in st[:20]}) >= 2
    out['E4b'] = {'passes': all(e4b.values()), 'failing': [k for k, v in e4b.items() if not v]}
    out['all_pass'] = all(v['passes'] for v in out.values() if isinstance(v, dict))
    out['protocol_sha256'] = fingerprint(protocol())
    atomic_json(g / 'gates.json', out)
    return out


def _status(root, state, done, total, running, started):
    eta = None if not done else (time.time() - started) / done * (total - done)
    atomic_json(Path(root) / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                             'running': running, 'pid': os.getpid(), 'updated_utc': now(),
                                             'eta_seconds': None if eta is None else round(eta)})


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, require_gates=True,
        worlds=WORLDS):
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root) | {'scale': scale}
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    started = time.time()
    with writer_lock(root / 'launcher.lock'):
        if (root / 'manifest.json').exists() and json.loads((root / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; retire the path rather than overwrite it')
        if require_gates:
            g = json.loads((ROOT / 'gates' / 'gates.json').read_text())
            if not g.get('all_pass') or g.get('protocol_sha256') != fingerprint(protocol()):
                raise RuntimeError('gates have not passed at this exact protocol')
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
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} jobs={jobs} '
                                       f'scale={scale} free={free:.1f}GiB')
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                def submit(a, w, s):
                    return pool.submit(run_cell, a, w, s, str(root), scale)
                futures = {}
                for a, w, s in todo:
                    if a in DEPENDENT and f'SHUFFLED_w{w}_s{s}' not in records:
                        continue
                    futures[submit(a, w, s)] = (a, w, s)
                pending = [(a, w, s) for a, w, s in todo if a in DEPENDENT and f'SHUFFLED_w{w}_s{s}' not in records]
                _status(root, 'running', len(records), total,
                        [f'{a}_w{w}_s{s}' for a, w, s in list(futures.values())[:jobs]], started)
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
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"terminal={record['terminal_median']:.6g}"
                                               + (f" changed={record['routes_changed']}"
                                                  if a == 'REROUTE_SLEEP' else ''))
                    if a == 'SHUFFLED':
                        for dep in DEPENDENT:   # REROUTE_SLEEP before SLEEP (plan)
                            if (dep, w, s) in pending:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gates', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after', type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.gates:
        result = gates()
        print(json.dumps(result, indent=1))
        raise SystemExit(0 if result['all_pass'] else 1)
    if args.dry_run:   # development world 27 only: never a sealed world
        dry = [('SHUFFLED', DRY_WORLD, 0), ('SHUFFLED', DRY_WORLD, 1), ('PLAIN', DRY_WORLD, 0),
               ('REROUTE_SLEEP', DRY_WORLD, 0), ('SLEEP', DRY_WORLD, 0)]
        run(dry, root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=o3.DRY_SCALE, stop_after=args.stop_after,
            require_gates=False, worlds=(DRY_WORLD,))
        print('O6 dry run complete')
        return
    if args.stop_after is not None:
        raise SystemExit('--stop-after is for the dry run only')
    require_clean_code(OUTPUT)
    for check in ('tools/check_prereg.py', 'tools/check_invalid.py', 'tools/check_adequacy.py'):
        subprocess.run([sys.executable, check], check=True)
    run(cells())
    print(f'O6 report: {OUTPUT}')


if __name__ == '__main__':
    main()
