"""O2E Tier 1: does sleep help STAGED and MIXED_L1, and can it undo a collapse?

Plan: `O2E_SLEEP_OTHER_PROTOCOLS_PLAN.md` (frozen d0e771d). EXPLORATORY. Sleep is
O2D's RES64 setting through O2C's `consolidate`, unchanged.
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

import psutil
import torch

from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O2E_SLEEP_OTHER_PROTOCOLS_PLAN.md')
ROOT = Path('artifacts/o2e_sleep_protocols')
OUTPUT = Path('reports/o2e_sleep_protocols.json')
ARMS = {'STAGED_SLEEP': 'STAGED', 'MIXED_SLEEP': 'MIXED_L1'}
ARM_ID = {'STAGED_SLEEP': 1, 'MIXED_SLEEP': 2}
MEMORY = 64
UPDATES = 8192
THRESHOLD = 0.05
JOBS = 3
MIN_FREE_GIB = 8.0


def cells():
    return [(a, w, s) for a in ARMS for w in o2.WORLDS for s in o2.STREAMS]


def protocol():
    return {'id': 'o2e-sleep-protocols-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(),
            'tier': 1, 'exploratory': True, 'arms': ARMS, 'memory': MEMORY, 'updates': UPDATES,
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), o2c.O2_REPORT)},
            'implementation_sha256': digest(Path(__file__)), 'o2c_sha256': digest(Path(o2c.__file__)),
            'o2d_sha256': digest(Path(o2d.__file__)), 'lifetime_sha256': digest(Path(ll.__file__))}


def _load(model, path, task_ids, probe_depth=None):
    for t in task_ids:
        model.begin_task(t)
    state = torch.load(path, weights_only=True)['model_state_dict']
    for k in state:
        if k.startswith('task_codes.') and k not in model.state_dict():
            model.begin_task(k.split('.', 1)[1]) if probe_depth is None else model.begin_task(k.split('.', 1)[1], probe_depth)
    model.load_state_dict(state, strict=True)
    return model


def load_terminal(arm, world, stream):
    """(cfg, model, tasks the model holds codes for, canonical tasks) for an O2 terminal."""
    base = ARMS[arm]
    work = o2c.O2_WORK / f'{base}_w{world}_s{stream}'
    if base == 'STAGED':
        cfg3, world3, _, _ = stage_setup(world, 3, o2.MODEL_SEED)
        tasks = list(world3.tasks)
        model = _load(build_fast(cfg3), work / 'stage3' / 'model.pt', [t.task_id for t in tasks])
        return cfg3, model, tasks, tasks
    cfg3, _, stream_tasks, plan, canonical = o2.build_stream(base, world, stream)
    model = _load(o2.planned_model(cfg3, plan), work / 'lifetime' / 'model.pt',
                  [t.task_id for t in stream_tasks], probe_depth=3)
    return cfg3, model, stream_tasks, canonical


def gate_anchor():
    report = json.loads(o2c.O2_REPORT.read_text())['cells']
    bad = []
    for arm, base in ARMS.items():
        for w in o2.WORLDS:
            for s in o2.STREAMS:
                _, model, _, canonical = load_terminal(arm, w, s)
                if score(model, SimpleNamespace(tasks=canonical))['per_task'] != \
                        report[f'{base}_w{w}_s{s}']['terminal_per_task']:
                    bad.append(f'{arm}_w{w}_s{s}')
    return {'passes': not bad, 'mismatched': bad}


def run_cell(arm, world, stream):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg3, model, tasks, canonical = load_terminal(arm, world, stream)
    before = library_sha(model)
    pool = o2d.reservoir(tasks, world, stream, MEMORY)
    o2c.consolidate(cfg3, model, pool, [t.task_id for t in tasks], [1942, world, stream, ARM_ID[arm]],
                    updates=UPDATES)
    terminal = score(model, SimpleNamespace(tasks=canonical))
    return {'arm': arm, 'world': world, 'stream': stream, 'pool_size': len(pool), 'tasks': len(tasks),
            'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
            'library_sha256_before': before, 'library_sha256': library_sha(model),
            'seconds': time.perf_counter() - started}


def label(r, b):
    if r >= 4 and b <= 1:
        return 'RESCUES'
    return 'HARMS' if b >= 2 else 'NO_RESCUE'


def summarize(records):
    base = json.loads(o2c.O2_REPORT.read_text())['cells']
    out = {}
    for arm, b_arm in ARMS.items():
        m0 = {(w, s): base[f'{b_arm}_w{w}_s{s}']['terminal_median'] for w in o2.WORLDS for s in o2.STREAMS}
        m1 = {(w, s): records[f'{arm}_w{w}_s{s}']['terminal_median'] for w in o2.WORLDS for s in o2.STREAMS}
        N = [k for k, v in m0.items() if THRESHOLD <= v < 0.2]
        P = [k for k, v in m0.items() if v < THRESHOLD]
        C = [k for k, v in m0.items() if v >= 1.0]
        r = sum(m1[k] < THRESHOLD for k in N)
        b = sum(not m1[k] < THRESHOLD for k in P)
        c = sum(m1[k] < THRESHOLD for k in C)
        out[arm] = {'near': len(N), 'passing': len(P), 'collapsed': len(C), 'r': r, 'b': b, 'c': c,
                    'label': label(r, b), 'k_of_21': sum(v < THRESHOLD for v in m1.values())}
    return out


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
        g0 = gate_anchor()
        atomic_json(ROOT / 'gates.json', {'G0': g0, 'protocol_sha256': sha})
        log_line(ROOT / 'run.log', f"GATE G0={g0['passes']}")
        if not g0['passes']:
            raise RuntimeError(f'G0 failed: {g0}')
        records, todo, started = {}, [], time.time()
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
                        raise RuntimeError(f'non-vacuity failed: {key}')
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
    print(f'O2E report: {OUTPUT}')


if __name__ == '__main__':
    main()
