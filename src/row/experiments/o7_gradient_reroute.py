"""O7 Tier 1: length-linear gradient re-route + sleep, against exhaustive re-route + sleep (O5).

Plan: `O7_GRADIENT_REROUTE_PLAN.md` (frozen f3ec82e). EXPLORATORY; saved SHUFFLED terminals
only. The pipeline is O5's `run_cell` with the route chooser replaced: depth 1 exhaustive (12
evaluations), depth >= 2 SO1R `optimize_route` (500 steps, deterministic), kept only if its
support MSE is strictly below the current route's (SAFE). Gate E1: with the exhaustive chooser
the pipeline reproduces O5's committed `O3_w20_s0` cell bitwise.
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
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o5_reroute_sleep as o5
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_j2a_staged_library import enum_route
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary, optimize_route
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O7_GRADIENT_REROUTE_PLAN.md')
O5_REPORT = Path('reports/o5_reroute_sleep.json')
ROOT = Path('artifacts/o7_gradient_reroute')
OUTPUT = Path('reports/o7_gradient_reroute.json')
TARGET = [('O2', 14, 0), ('O3', 22, 1), ('O4', 901, 0), ('O4', 907, 2), ('O4', 908, 1), ('O4', 909, 2),
          ('O4', 911, 0), ('O4', 911, 2), ('O4', 914, 1), ('O4', 914, 2)]
HARM = [('O2', w, 1) for w in (13, 15, 16, 17, 18, 19)] + [('O3', w, 0) for w in (20, 21, 23, 24, 25, 26)]
OPT_STEPS = 500
THRESHOLD = 0.05
JOBS = 3
MIN_FREE_GIB = 4.5   # PI instruction 2026-09-27 ("Is ok. Just start"); recorded in precondition.json


def cells():
    return TARGET + HARM


def support_mse(library, x, y, route):
    with torch.no_grad():
        return float(torch.mean((library.hard(x, list(route)) - y) ** 2))


def choose(library, xs, ys, d, current, chooser):
    library.steps = d
    if chooser == 'exhaustive' or d == 1:
        return [int(r) for r in enum_route(library, xs, ys)], False
    proposal = [int(r) for r in optimize_route(library, xs, ys, OPT_STEPS)[0]]
    if support_mse(library, xs, ys, proposal) < support_mse(library, xs, ys, current):
        return proposal, True
    return list(current), False


def reroute(model, stream_tasks, plan, pool, chooser):
    """O5's reroute with a pluggable chooser; the logit swap is O5's, verbatim."""
    library = FrozenLibrary(model)
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    current = model.hard_routes()
    changed = accepted = differs = 0
    exhaustive_seconds = 0.0
    for task in stream_tasks:
        d = plan[task.task_id]
        xs = torch.tensor(np.stack([a for a, _ in by[task.task_id]]), dtype=torch.float32)
        ys = torch.tensor(np.stack([b for _, b in by[task.task_id]]), dtype=torch.float32)
        cur = [int(r) for r in current[task.task_id][:d]]
        new, took = choose(library, xs, ys, d, cur, chooser)
        accepted += took
        t0 = time.perf_counter()   # reported only: how the chosen route compares with exhaustive search
        library.steps = d
        differs += [int(r) for r in enum_route(library, xs, ys)] != new
        exhaustive_seconds += time.perf_counter() - t0
        with torch.no_grad():
            code = model.task_codes[task.task_id]
            for step in range(d):
                old = int(torch.argmax(code[step]))
                if old != new[step]:
                    a, b = code[step, old].clone(), code[step, new[step]].clone()
                    code[step, old], code[step, new[step]] = b, a
        if cur != new:
            changed += 1
        if list(model.hard_routes()[task.task_id][:d]) != new:
            raise RuntimeError(f'{task.task_id}: route swap did not take')
    return changed, accepted, differs, exhaustive_seconds


def run_cell(band, w, s, chooser='gradient'):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg3, model, stream_tasks, canonical = o3.load_shuffled_terminal(o5.terminal_path(band, w, s), w, s)
    _, _, _, plan, _ = o2.build_stream('SHUFFLED', w, s)
    before = library_sha(model)
    pool = o2d.reservoir(stream_tasks, w, s, o3.MEMORY)
    t0 = time.perf_counter()
    changed, accepted, differs, exhaustive_seconds = reroute(model, stream_tasks, plan, pool, chooser)
    reroute_seconds = time.perf_counter() - t0 - exhaustive_seconds
    o2c.consolidate(cfg3, model, pool, [t.task_id for t in stream_tasks], [1941, w, s, o3.MEMORY],
                    updates=o3.EXTRA_UPDATES)
    terminal = score(model, SimpleNamespace(tasks=canonical))
    return {'band': band, 'world': w, 'stream': s, 'chooser': chooser, 'routes_changed': changed,
            'gradient_accepted': accepted, 'differs_from_exhaustive': differs,
            'reroute_seconds': reroute_seconds, 'exhaustive_seconds': exhaustive_seconds, 'pool_size': len(pool),
            'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
            'library_sha256_before': before, 'library_sha256': library_sha(model),
            'seconds': time.perf_counter() - started}


def protocol():
    return {'id': 'o7-gradient-reroute-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'exploratory': True, 'target': TARGET, 'harm': HARM, 'opt_steps': OPT_STEPS,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml', O5_REPORT.as_posix())},
            'implementation_sha256': digest(Path(__file__)), 'o5_sha256': digest(Path(o5.__file__)),
            'o3_sha256': digest(Path(o3.__file__)), 'o2c_sha256': digest(Path(o2c.__file__)),
            'o2d_sha256': digest(Path(o2d.__file__)), 'lifetime_sha256': digest(Path(ll.__file__))}


def label(r, b):
    if b >= 2:
        return 'HARMS'
    if r >= 8:
        return 'MATCHES_EXHAUSTIVE'
    return 'PARTIAL' if r >= 4 else 'NO_RESCUE'


def passing(m):
    return np.isfinite(m) and m < THRESHOLD


def summarize(records):
    r = sum(passing(records[f'{b}_w{w}_s{s}']['terminal_median']) for b, w, s in TARGET)
    b = sum(not passing(records[f'{b}_w{w}_s{s}']['terminal_median']) for b, w, s in HARM)
    return {'r': r, 'target_cells': len(TARGET), 'b': b, 'harm_cells': len(HARM), 'label': label(r, b)}


def gate_e1():
    """With the exhaustive chooser, the pipeline reproduces O5's committed O3_w20_s0 bitwise."""
    ref = json.loads(O5_REPORT.read_text())['cells']['O3_w20_s0']
    rec = run_cell('O3', 20, 0, chooser='exhaustive')
    return {'passes': rec['library_sha256'] == ref['library_sha256']
            and rec['terminal_per_task'] == ref['terminal_per_task']
            and rec['routes_changed'] == ref['routes_changed']}


def run(jobs=JOBS):
    (ROOT / 'cells').mkdir(parents=True, exist_ok=True)
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
        gates = {'G0': o5.gate_g0()}
        log_line(ROOT / 'run.log', f"GATE G0={gates['G0']}")
        e1_path = ROOT / 'gate_e1.json'
        if e1_path.exists() and json.loads(e1_path.read_text()).get('protocol_sha256') == sha:
            gates['E1'] = json.loads(e1_path.read_text())['E1']
        else:
            gates['E1'] = gate_e1()
            atomic_json(e1_path, {'E1': gates['E1'], 'protocol_sha256': sha})
        log_line(ROOT / 'run.log', f"GATE E1={gates['E1']}")
        if not (gates['G0']['passes'] and gates['E1']['passes']):
            raise RuntimeError(f'gates failed: {gates}')
        records, todo, started = {}, [], time.time()
        try:
            for band, w, s in cells():
                key = f'{band}_w{w}_s{s}'
                path = ROOT / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(ROOT / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append((band, w, s))
            total = len(cells())
            log_line(ROOT / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} free={free:.1f}GiB')
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, b, w, s): (b, w, s) for b, w, s in todo}
                for future in as_completed(futures):
                    b, w, s = futures[future]
                    key = f'{b}_w{w}_s{s}'
                    record = future.result()
                    if record['pool_size'] != 188 * o3.MEMORY or len(record['terminal_per_task']) != 64:
                        raise RuntimeError(f'{key}: construction')
                    atomic_json(ROOT / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    log_line(ROOT / 'run.log', f"cell finished {key} changed={record['routes_changed']} "
                                               f"accepted={record['gradient_accepted']} "
                                               f"{record['seconds']:.1f}s terminal={record['terminal_median']:.6g}")
                    atomic_json(ROOT / 'status.json', {'state': 'running', 'cells_done': len(records),
                                                       'cells_total': total, 'pid': os.getpid(), 'updated_utc': now()})
            atomic_json(OUTPUT, {**manifest, 'complete': True, 'gates': gates, 'cells': records,
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
    print(f'O7 report: {OUTPUT}')


if __name__ == '__main__':
    main()
