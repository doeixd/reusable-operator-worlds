"""O11 SEALED confirmation: wake + in-stream re-routing of earlier tasks + O3's sleep (no end-of-stream search).

Plan: `O11_SEALED_INSTREAM_SLEEP_PLAN.md`. Sealed worlds 945-959.
- REROUTE_WAKE: O8's `run_cell` verbatim.         RW_SLEEP: `o3.run_sleep` verbatim on this run's REROUTE_WAKE terminal.
- SHUFFLED / SLEEP / PLAIN: `o3.run_cell`, exactly O4/O6/O9.
Gates E1/E2/E3/E3b reproduce committed development or opened-world cells bitwise before any sealed world is
touched; E4b builds sealed streams only after the freeze.
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

import numpy as np
import psutil
import torch

from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o8_instream_reroute as o8
from row.experiments import census_o8_staleness_position as census
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O11_SEALED_INSTREAM_SLEEP_PLAN.md')
O3_REPORT = Path('reports/o3_online_sleep_v2.json')
O8_REPORT = Path('reports/o8_instream_reroute.json')
O10_REPORT = Path('reports/o10_rw_sleep.json')
O9_WORK = Path('artifacts/o9_sealed_online/work')
ROOT = Path('artifacts/o11_sealed_instream_sleep')
DRY_ROOT = Path('artifacts/o11_sealed_instream_sleep_dry')
OUTPUT = Path('reports/o11_sealed_instream_sleep.json')
WORLDS = tuple(range(945, 960))
STREAMS = (0, 1, 2)
ARMS = ('REROUTE_WAKE', 'SHUFFLED', 'PLAIN', 'RW_SLEEP', 'SLEEP')   # registered submission order
ARM_STREAMS = {'REROUTE_WAKE': STREAMS, 'SHUFFLED': STREAMS, 'PLAIN': (0,), 'RW_SLEEP': STREAMS, 'SLEEP': STREAMS}
PARENT = {'RW_SLEEP': 'REROUTE_WAKE', 'SLEEP': 'SHUFFLED'}   # dependent arm -> the cell whose terminal it loads
THRESHOLD = 0.05
CONFIRM_AT = 41
ATTRIBUTE_AT = 30
JOBS = 3
MIN_FREE_GIB = 2.0   # PI 2026-10-03: low memory is acceptable
DRY_WORLD = 27       # development band 3, held back; never a sealed world


def cells(worlds=WORLDS):
    return [(a, w, s) for a in ARMS for w in worlds for s in ARM_STREAMS[a]]


def terminal_of(root, arm, w, s):
    return Path(root) / 'work' / f'{arm}_w{w}_s{s}' / 'lifetime' / 'model.pt'


def run_cell(arm, w, s, root, scale=1):
    if arm == 'REROUTE_WAKE':
        return o8.run_cell(arm, w, s, root, scale)
    if arm == 'RW_SLEEP':
        torch.set_num_threads(1)
        started = time.perf_counter()
        rec = o3.run_sleep(w, s, terminal_of(root, 'REROUTE_WAKE', w, s))
        rec.update({'arm': arm, 'world': w, 'stream': s, 'scale': scale, 'seconds': time.perf_counter() - started})
        return rec
    return o3.run_cell(arm, w, s, root, scale)


def validate(rec):
    if rec['arm'] == 'REROUTE_WAKE':
        o8.validate(rec)
        return
    if rec['arm'] == 'RW_SLEEP':
        key = f"RW_SLEEP_w{rec['world']}_s{rec['stream']}"
        if rec['scored_tasks'] != 64 or len(rec['terminal_per_task']) != 64:
            raise ValueError(f'{key}: not scored on the canonical 64')
        if rec['extra_updates'] != o3.EXTRA_UPDATES or rec['pool_size'] != 188 * o3.MEMORY:
            raise ValueError(f'{key}: construction')
        if rec['library_sha256'] == rec['library_sha256_before']:
            raise ValueError(f'{key}: sleep left the library unchanged')
        return
    o3.validate(rec)


def protocol(root=ROOT):
    return {'id': 'o11-sealed-instream-sleep-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 2,
            'sealed': True, 'worlds': list(WORLDS), 'arm_streams': {a: list(v) for a, v in ARM_STREAMS.items()},
            'threshold': THRESHOLD, 'confirm_at': CONFIRM_AT, 'attribute_at': ATTRIBUTE_AT, 'jobs': JOBS,
            'root': Path(root).as_posix(),
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), O3_REPORT, O8_REPORT,
                                                                O10_REPORT)},
            'implementation_sha256': digest(Path(__file__)), 'o8_sha256': digest(Path(o8.__file__)),
            'o3_sha256': digest(Path(o3.__file__)), 'o2_sha256': digest(Path(o2.__file__)),
            'o2c_sha256': digest(Path(o2c.__file__)), 'o2d_sha256': digest(Path(o2d.__file__)),
            'census_sha256': digest(Path(census.__file__)), 'lifetime_sha256': digest(Path(ll.__file__)),
            'learner_sha256': digest(Path(sys.modules[o2.PlannedDepthRotatedLearner.__module__].__file__))}


def passing(m):
    return m is not None and math.isfinite(m) and m < THRESHOLD


def label(k, floor_failed):
    if floor_failed:
        return 'FLOOR_FAILED'
    return 'CONFIRMED' if k >= CONFIRM_AT else 'NOT_CONFIRMED'


def attribution(n_better):
    return 'ATTRIBUTED' if n_better >= ATTRIBUTE_AT else 'NOT_ATTRIBUTED'


def summarize(records, worlds=WORLDS):
    M = {k: v['terminal_median'] for k, v in records.items()}
    k = {a: sum(passing(M[f'{a}_w{w}_s{s}']) for w in worlds for s in ARM_STREAMS[a]) for a in ARM_STREAMS}
    floor = k['PLAIN'] > 0
    pairs = [(w, s) for w in worlds for s in STREAMS]
    better = sum(math.isfinite(M[f'RW_SLEEP_w{w}_s{s}']) and M[f'RW_SLEEP_w{w}_s{s}'] < M[f'SLEEP_w{w}_s{s}']
                 for w, s in pairs)
    sleep_fails = [(w, s) for w, s in pairs if not passing(M[f'SLEEP_w{w}_s{s}'])]
    return {'passes': k, 'denominator': len(pairs), 'label': label(k['RW_SLEEP'], floor), 'floor_failed': floor,
            'n_better': better, 'attribution': attribution(better),
            'descriptive': {
                'rw_sleep_rescues_sleep_fails': sum(passing(M[f'RW_SLEEP_w{w}_s{s}']) for w, s in sleep_fails),
                'sleep_fails': len(sleep_fails),
                'rw_sleep_over_sleep_median': float(np.median([M[f'RW_SLEEP_w{w}_s{s}'] / M[f'SLEEP_w{w}_s{s}']
                                                               for w, s in pairs])),
                'collapses': {a: sum(not (math.isfinite(M[f'{a}_w{w}_s{s}']) and M[f'{a}_w{w}_s{s}'] < 1.0)
                                     for w in worlds for s in ARM_STREAMS[a]) for a in ARM_STREAMS}}}


def gates(root=ROOT):
    g = Path(root) / 'gates'
    g.mkdir(parents=True, exist_ok=True)
    ref3 = json.loads(O3_REPORT.read_text())['cells']
    ref8 = json.loads(O8_REPORT.read_text())['cells']['REROUTE_WAKE_w20_s0']
    ref10 = json.loads(O10_REPORT.read_text())['cells']['w930_s0']

    def same(a, b):
        return a['library_sha256'] == b['library_sha256'] and a['terminal_per_task'] == b['terminal_per_task']

    out = {}
    e1 = o3.run_cell('SHUFFLED', 20, 0, str(g / 'E1'))
    out['E1'] = {'passes': same(e1, ref3['SHUFFLED_w20_s0'])}
    log_line(g / 'gates.log', f"E1 {out['E1']}")
    e2 = o8.run_cell('REROUTE_WAKE', 20, 0, str(g / 'E2'))
    out['E2'] = {'passes': same(e2, ref8) and e2['routes_changed_total'] == ref8['routes_changed_total']}
    log_line(g / 'gates.log', f"E2 {out['E2']}")
    e3 = o3.run_cell('SLEEP', 20, 0, str(g / 'E1'))
    out['E3'] = {'passes': same(e3, ref3['SLEEP_w20_s0'])}
    log_line(g / 'gates.log', f"E3 {out['E3']}")
    e3b = run_cell('RW_SLEEP', 930, 0, str(O9_WORK.parent))   # O9's saved terminal -> must equal O10's committed cell
    out['E3b'] = {'passes': same(e3b, ref10)}
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


def _headline(a, r):
    extra = f" changed={r['routes_changed_total']} stale={r['terminal_stale']}" if a == 'REROUTE_WAKE' else ''
    return f"{r['seconds']:.1f}s terminal={r['terminal_median']:.6g}{extra}"


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
                futures, pending = {}, []
                for a, w, s in todo:
                    if a in PARENT and f'{PARENT[a]}_w{w}_s{s}' not in records:
                        pending.append((a, w, s))
                    else:
                        futures[submit(a, w, s)] = (a, w, s)
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
                    log_line(root / 'run.log', f'cell finished {key} {_headline(a, record)}')
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
    return [('REROUTE_WAKE', DRY_WORLD, 0), ('SHUFFLED', DRY_WORLD, 0), ('PLAIN', DRY_WORLD, 0),
            ('RW_SLEEP', DRY_WORLD, 0), ('SLEEP', DRY_WORLD, 0)]


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
        run(dry_cells(), root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=o3.DRY_SCALE,
            stop_after=args.stop_after, require_gates=False, worlds=(DRY_WORLD,))
        print('O11 dry run complete')
        return
    if args.stop_after is not None:
        raise SystemExit('--stop-after is for the dry run only')
    require_clean_code(OUTPUT)
    for check in ('tools/check_prereg.py', 'tools/check_invalid.py', 'tools/check_adequacy.py'):
        subprocess.run([sys.executable, check], check=True)
    run(cells())
    print(f'O11 report: {OUTPUT}')


if __name__ == '__main__':
    main()
