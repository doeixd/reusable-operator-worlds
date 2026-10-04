"""O10 Tier 1: O3's sleep, verbatim, on O9's saved REROUTE_WAKE terminals (no route search).

Plan: `O10_REROUTE_WAKE_SLEEP_PLAN.md` (frozen 264abd1). EXPLORATORY; worlds 930-944 were opened by O9.
Gates: G0 reloads reproduce O9's recorded per-task terminals; E2 o3.run_sleep on O3's SHUFFLED_w20_s0
terminal reproduces O3's committed SLEEP_w20_s0 bitwise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch

from row.experiments import census_o8_staleness_position as census
from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o9_sealed_online as o9
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O10_REROUTE_WAKE_SLEEP_PLAN.md')
O9_REPORT = Path('reports/o9_sealed_online.json')
O3_REPORT = Path('reports/o3_online_sleep_v2.json')
O9_WORK = Path('artifacts/o9_sealed_online/work')
ROOT = Path('artifacts/o10_rw_sleep')
OUTPUT = Path('reports/o10_rw_sleep.json')
WORLDS = tuple(range(930, 945))
STREAMS = (0, 1, 2)
TARGET = [(931, 0), (931, 2), (932, 2), (937, 1), (937, 2), (938, 1), (938, 2), (944, 2)]
THRESHOLD = 0.05
JOBS = 3
MIN_FREE_GIB = 2.0   # PI 2026-10-03 instruction


def cells():
    return [(w, s) for w in WORLDS for s in STREAMS]


def harm():
    return [c for c in cells() if c not in TARGET]


def terminal_path(w, s):
    return O9_WORK / f'REROUTE_WAKE_w{w}_s{s}' / 'lifetime' / 'model.pt'


def protocol():
    return {'id': 'o10-rw-sleep-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'exploratory': True, 'target': TARGET, 'threshold': THRESHOLD, 'memory': o3.MEMORY,
            'extra_updates': o3.EXTRA_UPDATES,
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), O9_REPORT, O3_REPORT)},
            'implementation_sha256': digest(Path(__file__)), 'o9_sha256': digest(Path(o9.__file__)),
            'o3_sha256': digest(Path(o3.__file__)), 'o2c_sha256': digest(Path(o2c.__file__)),
            'o2d_sha256': digest(Path(o2d.__file__)), 'census_sha256': digest(Path(census.__file__)),
            'lifetime_sha256': digest(Path(ll.__file__))}


def run_cell(w, s):
    torch.set_num_threads(1)
    started = time.perf_counter()
    rec = o3.run_sleep(w, s, terminal_path(w, s))   # the registered construction, verbatim
    # Staleness after sleep: rebuild the slept model with run_sleep's exact calls, require it to reproduce
    # run_sleep's per-task terminal (validate), then measure. Doubles the sleep cost; keeps run_sleep untouched.
    cfg3, slept, st, canonical = o3.load_shuffled_terminal(terminal_path(w, s), w, s)
    pool = o2d.reservoir(st, w, s, o3.MEMORY)
    o2c.consolidate(cfg3, slept, pool, [t.task_id for t in st], [1941, w, s, o3.MEMORY], updates=o3.EXTRA_UPDATES)
    again = score(slept, SimpleNamespace(tasks=canonical))
    _, _, _, plan, _ = o2.build_stream('SHUFFLED', w, s)
    stale = census.staleness(slept, st, plan, pool)
    stale.pop('rows')
    rec.update({'world': w, 'stream': s, 'target': (w, s) in TARGET,
                'rebuild_matches': again['per_task'] == rec['terminal_per_task'],
                'stale_after_sleep': stale['stale'], 'stale_after_sleep_by_depth': stale['by_depth'],
                'seconds': time.perf_counter() - started})
    return rec


def validate(rec):
    key = f"w{rec['world']}_s{rec['stream']}"
    if rec['scored_tasks'] != 64 or rec['pool_size'] != 188 * o3.MEMORY or rec['extra_updates'] != o3.EXTRA_UPDATES:
        raise ValueError(f'{key}: construction')
    if rec['library_sha256'] == rec['library_sha256_before']:
        raise ValueError(f'{key}: sleep left the library unchanged')
    if not rec['rebuild_matches']:
        raise ValueError(f'{key}: staleness rebuild is not the slept model')


def passing(m):
    return m is not None and math.isfinite(m) and m < THRESHOLD


def label(r, b):
    if b >= 3:
        return 'HARMS'
    if r >= 7:
        return 'REPAIRS'
    return 'PARTIAL' if r >= 4 else 'NO_REPAIR'


def summarize(records):
    r = sum(passing(records[f'w{w}_s{s}']['terminal_median']) for w, s in TARGET)
    b = sum(not passing(records[f'w{w}_s{s}']['terminal_median']) for w, s in harm())
    return {'r': r, 'target_cells': len(TARGET), 'b': b, 'harm_cells': len(harm()), 'label': label(r, b),
            'pass_all_45': sum(passing(v['terminal_median']) for v in records.values()),
            'stale_after_sleep_total': sum(v['stale_after_sleep'] for v in records.values())}


def gates():
    o9cells = json.loads(O9_REPORT.read_text())['cells']
    g0 = {}
    for w, s in TARGET + harm()[:3]:
        _, model, _, canonical = o3.load_shuffled_terminal(terminal_path(w, s), w, s)
        g0[f'w{w}_s{s}'] = score(model, SimpleNamespace(tasks=canonical))['per_task'] == \
            o9cells[f'REROUTE_WAKE_w{w}_s{s}']['terminal_per_task']
    ref = json.loads(O3_REPORT.read_text())['cells']['SLEEP_w20_s0']
    e2 = o3.run_sleep(20, 0, Path('artifacts/o3_online_sleep_v2/work/SHUFFLED_w20_s0/lifetime/model.pt'))
    out = {'G0': {'passes': all(g0.values()), 'cells': g0},
           'E2': {'passes': e2['library_sha256'] == ref['library_sha256']
                  and e2['terminal_per_task'] == ref['terminal_per_task']}}
    out['all_pass'] = out['G0']['passes'] and out['E2']['passes']
    return out


def run(jobs=JOBS):
    (ROOT / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol()
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    started = time.time()
    with writer_lock(ROOT / 'launcher.lock'):
        if (ROOT / 'manifest.json').exists() and json.loads((ROOT / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; retire the path rather than overwrite it')
        atomic_json(ROOT / 'manifest.json', manifest)
        free = psutil.virtual_memory().available / 2 ** 30
        atomic_json(ROOT / 'precondition.json', {'free_gib': free, 'required_gib': MIN_FREE_GIB,
                                                 'passes': free >= MIN_FREE_GIB, 'checked_utc': now()})
        if free < MIN_FREE_GIB:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free')
        records = {}
        try:
            gpath = ROOT / 'gates.json'
            g = json.loads(gpath.read_text()) if gpath.exists() else None
            if not g or g.get('protocol_sha256') != sha:
                g = gates() | {'protocol_sha256': sha}
                atomic_json(gpath, g)
            log_line(ROOT / 'run.log', f"GATES G0={g['G0']['passes']} E2={g['E2']['passes']}")
            if not g['all_pass']:
                raise RuntimeError(f'gates failed: {g}')
            todo = []
            for w, s in cells():
                path = ROOT / 'cells' / f'w{w}_s{s}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: w{w}_s{s}')
                    records[f'w{w}_s{s}'] = saved['record']
                else:
                    todo.append((w, s))
            log_line(ROOT / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} free={free:.1f}GiB')
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, w, s): (w, s) for w, s in todo}
                for future in as_completed(futures):
                    w, s = futures[future]
                    record = future.result()
                    validate(record)
                    atomic_json(ROOT / 'cells' / f'w{w}_s{s}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[f'w{w}_s{s}'] = record
                    log_line(ROOT / 'run.log', f"cell finished w{w}_s{s} target={record['target']} "
                                               f"{record['seconds']:.1f}s terminal={record['terminal_median']:.6g} "
                                               f"stale={record['stale_after_sleep']}")
                    atomic_json(ROOT / 'status.json', {'state': 'running', 'cells_done': len(records),
                                                       'cells_total': len(cells()), 'pid': os.getpid(),
                                                       'updated_utc': now()})
            atomic_json(OUTPUT, {**manifest, 'complete': True, 'gates': g, 'cells': records,
                                 'summary': summarize(records), 'finished_utc': now()})
            atomic_json(ROOT / 'status.json', {'state': 'complete', 'cells_done': len(records),
                                               'cells_total': len(cells()), 'pid': os.getpid(), 'updated_utc': now()})
            atomic_json(ROOT / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            atomic_json(ROOT / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(ROOT / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(1)
    require_clean_code(OUTPUT)
    run()
    print(f'O10 report: {OUTPUT}')


if __name__ == '__main__':
    main()
