"""O2D Tier 1: how much must an online learner retain for a sleep phase to close the gap?

Plan: `O2D_SLEEP_MEMORY_DOSE_PLAN.md` (frozen 7902346). EXPLORATORY. Reuses O2C's
`load_terminal` and `consolidate` unchanged; only the pool differs: a sleep
reservoir of M examples per stream task, chosen by SeedSequence([1940, w, s, i]).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import torch

from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O2D_SLEEP_MEMORY_DOSE_PLAN.md')
ROOT = Path('artifacts/o2d_sleep_memory')
OUTPUT = Path('reports/o2d_sleep_memory.json')
MEMORY = {'RES4': 4, 'RES16': 16, 'RES64': 64}
ARMS = ('RES64', 'RES16', 'RES4')   # longest first
UPDATES = 8192
THRESHOLD = 0.05
JOBS = 3
MIN_FREE_GIB = 8.0


def cells():
    return [(a, w, s) for a in ARMS for w in o2.WORLDS for s in o2.STREAMS]


def protocol():
    return {'id': 'o2d-sleep-memory-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(),
            'tier': 1, 'exploratory': True, 'arms': MEMORY, 'updates': UPDATES,
            'reservoir_seed': 'SeedSequence([1940, w, s, i]) per task i; sampling [1941, w, s, M]',
            'input_sha256': {p.as_posix(): digest(p) for p in (
                PLAN, Path('configs/v1.yaml'), o2c.O2_REPORT, o2c.OUTPUT)},
            'implementation_sha256': digest(Path(__file__)),
            'o2c_sha256': digest(Path(o2c.__file__)), 'lifetime_sha256': digest(Path(ll.__file__))}


def reservoir(stream_tasks, world, stream, m):
    pool = []
    for i, task in enumerate(stream_tasks):
        rng = np.random.default_rng(np.random.SeedSequence([1940, world, stream, i]))
        for j in rng.choice(len(task.train_x), size=m, replace=False):
            pool.append((task.train_x[j], task.train_y[j], task.task_id))
    return pool


def run_cell(arm, world, stream):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg3, model, stream_tasks, plan, canonical, _ = o2c.load_terminal(world, stream)
    before = library_sha(model)
    pool = reservoir(stream_tasks, world, stream, MEMORY[arm])
    o2c.consolidate(cfg3, model, pool, [t.task_id for t in stream_tasks], [1941, world, stream, MEMORY[arm]],
                    updates=UPDATES)
    terminal = score(model, SimpleNamespace(tasks=canonical))
    return {'arm': arm, 'world': world, 'stream': stream, 'memory_per_task': MEMORY[arm], 'pool_size': len(pool),
            'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
            'library_sha256_before': before, 'library_sha256': library_sha(model),
            'seconds': time.perf_counter() - started}


def label(k, b):
    if k >= 18 and b <= 1:
        return 'SLEEP_MEMORY_SUFFICES'
    return 'HARMS' if b >= 2 else 'SHORT'


def summarize(records):
    base = json.loads(o2c.O2_REPORT.read_text())['cells']
    passing = [(w, s) for w in o2.WORLDS for s in o2.STREAMS
               if base[f'SHUFFLED_w{w}_s{s}']['terminal_median'] < THRESHOLD]
    out = {}
    for arm in ARMS:
        m = {(w, s): records[f'{arm}_w{w}_s{s}']['terminal_median'] for w in o2.WORLDS for s in o2.STREAMS}
        k = sum(v < THRESHOLD for v in m.values())
        b = sum(not m[c] < THRESHOLD for c in passing)
        out[arm] = {'memory_per_task': MEMORY[arm], 'k_of_21': k, 'b_broken_of_9': b, 'label': label(k, b)}
    suff = [MEMORY[a] for a in ARMS if out[a]['label'] == 'SLEEP_MEMORY_SUFFICES']
    return {'arms': out, 'M_star': min(suff) if suff else None}


def run(jobs=JOBS):
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'cells').mkdir(exist_ok=True)
    expected = protocol()
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    with writer_lock(ROOT / 'launcher.lock'):
        if (ROOT / 'manifest.json').exists() and json.loads((ROOT / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; retire the path rather than overwrite it')
        atomic_json(ROOT / 'manifest.json', manifest)
        free = psutil.virtual_memory().available / 2 ** 30
        atomic_json(ROOT / 'precondition.json', {'free_gib': free, 'required_gib': MIN_FREE_GIB,
                                                 'passes': free >= MIN_FREE_GIB, 'checked_utc': now()})
        if free < MIN_FREE_GIB:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free')
        g0 = o2c.gate_anchor()
        atomic_json(ROOT / 'gates.json', {'G0': g0, 'protocol_sha256': sha})
        log_line(ROOT / 'run.log', f"GATE G0={g0['passes']}")
        if not g0['passes']:
            raise RuntimeError(f'G0 failed: {g0}')
        records, todo = {}, []
        started = time.time()
        try:
            for arm, w, s in cells():
                key = f'{arm}_w{w}_s{s}'
                path = ROOT / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                else:
                    todo.append((arm, w, s))
            total = len(cells())
            log_line(ROOT / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} free={free:.1f}GiB')
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, w, s): (a, w, s) for a, w, s in todo}
                for future in as_completed(futures):
                    a, w, s = futures[future]
                    key = f'{a}_w{w}_s{s}'
                    record = future.result()
                    if record['library_sha256'] == record['library_sha256_before']:
                        raise RuntimeError(f'non-vacuity failed: {key} library unchanged')
                    if record['pool_size'] != 188 * MEMORY[a]:
                        raise RuntimeError(f'{key}: pool {record["pool_size"]}')
                    atomic_json(ROOT / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    log_line(ROOT / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"terminal={record['terminal_median']:.6g}")
                    done_new = len(records) - (total - len(todo))
                    atomic_json(ROOT / 'status.json', {
                        'state': 'running', 'cells_done': len(records), 'cells_total': total, 'pid': os.getpid(),
                        'updated_utc': now(),
                        'eta_seconds': round((time.time() - started) / max(1, done_new) * (total - len(records)))})
            atomic_json(OUTPUT, {**manifest, 'complete': True, 'gates': {'G0': g0}, 'cells': records,
                                 'summary': summarize(records), 'finished_utc': now()})
            atomic_json(ROOT / 'status.json', {'state': 'complete', 'cells_done': total, 'cells_total': total,
                                               'pid': os.getpid(), 'updated_utc': now()})
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
    print(f'O2D report: {OUTPUT}')


if __name__ == '__main__':
    main()
