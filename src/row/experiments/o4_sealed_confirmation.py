"""O4 SEALED confirmation: order-free online anchors + retained memory + consolidation.

Plan: `O4_SEALED_CONFIRMATION_PLAN.md` (frozen 397aa88). Sealed worlds 900-914.
Constructions are O3's, verbatim: `o3.run_cell` for SHUFFLED, SLEEP and PLAIN, and
`o3.validate`. Gates E1/E2 reproduce O3's committed development cells bitwise before
any sealed world is touched; E4b runs on sealed streams only after the freeze.
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

import psutil
import torch

from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O4_SEALED_CONFIRMATION_PLAN.md')
O3_REPORT = Path('reports/o3_online_sleep_v2.json')
O3_WORK = Path('artifacts/o3_online_sleep_v2/work')
ROOT = Path('artifacts/o4_sealed_confirmation')
DRY_ROOT = Path('artifacts/o4_sealed_confirmation_dry')
OUTPUT = Path('reports/o4_sealed_confirmation.json')
WORLDS = tuple(range(900, 915))
STREAMS = (0, 1, 2)
ARM_STREAMS = {'SHUFFLED': STREAMS, 'SLEEP': STREAMS, 'PLAIN': (0,)}
THRESHOLD = 0.05
CONFIRM_AT = 41
JOBS = 3
MIN_FREE_GIB = 8.0
DRY_WORLD = 27   # development band 3, held back; never a sealed world


def cells(worlds=WORLDS):
    return [(a, w, s) for a in ('SHUFFLED', 'SLEEP', 'PLAIN') for w in worlds for s in ARM_STREAMS[a]]


def protocol(root=ROOT):
    return {'id': 'o4-sealed-confirmation-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 2,
            'sealed': True, 'worlds': list(WORLDS), 'arm_streams': {a: list(s) for a, s in ARM_STREAMS.items()},
            'threshold': THRESHOLD, 'confirm_at': CONFIRM_AT, 'jobs': JOBS, 'root': Path(root).as_posix(),
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), O3_REPORT)},
            'implementation_sha256': digest(Path(__file__)), 'o3_sha256': digest(Path(o3.__file__)),
            'o2_sha256': digest(Path(o2.__file__)), 'o2c_sha256': digest(Path(o2c.__file__)),
            'o2d_sha256': digest(Path(o2d.__file__)), 'lifetime_sha256': digest(Path(ll.__file__)),
            'learner_sha256': digest(Path(sys.modules[o2.PlannedDepthRotatedLearner.__module__].__file__))}


def passing(m):
    return math.isfinite(m) and m < THRESHOLD


def summarize(records, worlds=WORLDS):
    M = {k: v['terminal_median'] for k, v in records.items()}
    k = {a: sum(passing(M[f'{a}_w{w}_s{s}']) for w in worlds for s in ARM_STREAMS[a]) for a in ARM_STREAMS}
    floor = k['PLAIN'] > 0
    label = 'FLOOR_FAILED' if floor else ('CONFIRMED' if k['SLEEP'] >= CONFIRM_AT else 'NOT_CONFIRMED')
    return {'passes': k, 'denominator': 3 * len(worlds), 'label': label, 'floor_failed': floor,
            'collapses': {a: sum(not (math.isfinite(M[f'{a}_w{w}_s{s}']) and M[f'{a}_w{w}_s{s}'] < 1.0)
                                 for w in worlds for s in ARM_STREAMS[a]) for a in ARM_STREAMS}}


def gates(root=ROOT):
    g = Path(root) / 'gates'
    g.mkdir(parents=True, exist_ok=True)
    ref = json.loads(O3_REPORT.read_text())['cells']
    out = {}
    e1 = o3.run_cell('SHUFFLED', 20, 0, str(g / 'E1'))
    out['E1'] = {'passes': e1['library_sha256'] == ref['SHUFFLED_w20_s0']['library_sha256']
                 and e1['terminal_per_task'] == ref['SHUFFLED_w20_s0']['terminal_per_task']}
    log_line(g / 'gates.log', f"E1 {out['E1']}")
    e2 = o3.run_sleep(20, 0, g / 'E1' / 'work' / 'SHUFFLED_w20_s0' / 'lifetime' / 'model.pt')
    out['E2'] = {'passes': e2['library_sha256'] == ref['SLEEP_w20_s0']['library_sha256']
                 and e2['terminal_per_task'] == ref['SLEEP_w20_s0']['terminal_per_task']}
    log_line(g / 'gates.log', f"E2 {out['E2']}")
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
                    return pool.submit(o3.run_cell, a, w, s, str(root), scale)
                futures = {}
                for a, w, s in todo:
                    if a == 'SLEEP' and f'SHUFFLED_w{w}_s{s}' not in records:
                        continue
                    futures[submit(a, w, s)] = (a, w, s)
                pending_sleep = {(w, s) for a, w, s in todo if a == 'SLEEP' and f'SHUFFLED_w{w}_s{s}' not in records}
                _status(root, 'running', len(records), total, [f'{a}_w{w}_s{s}' for a, w, s in list(futures.values())[:jobs]], started)
                while futures:
                    done_future = next(as_completed(futures))
                    a, w, s = futures.pop(done_future)
                    key = f'{a}_w{w}_s{s}'
                    record = done_future.result()
                    o3.validate(record)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"terminal={record['terminal_median']:.6g}")
                    if a == 'SHUFFLED' and (w, s) in pending_sleep:
                        pending_sleep.discard((w, s))
                        futures[submit('SLEEP', w, s)] = ('SLEEP', w, s)
                    _status(root, 'running', len(records), total, [f'{x}_w{y}_s{z}' for x, y, z in futures.values()][:jobs], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
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
        print(json.dumps(gates(), indent=1))
        return
    if args.dry_run:   # development world 27 only: never a sealed world
        dry = [('SHUFFLED', DRY_WORLD, 0), ('SHUFFLED', DRY_WORLD, 1), ('PLAIN', DRY_WORLD, 0), ('SLEEP', DRY_WORLD, 0)]
        run(dry, root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=o3.DRY_SCALE, stop_after=args.stop_after,
            require_gates=False, worlds=(DRY_WORLD,))
        print('O4 dry run complete')
        return
    if args.stop_after is not None:
        raise SystemExit('--stop-after is for the dry run only')
    require_clean_code(OUTPUT)
    for check in ('tools/check_prereg.py', 'tools/check_invalid.py'):
        subprocess.run([sys.executable, check], check=True)
    run(cells())
    print(f'O4 report: {OUTPUT}')


if __name__ == '__main__':
    main()
