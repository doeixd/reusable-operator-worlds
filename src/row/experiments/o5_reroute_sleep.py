"""O5 Tier 1: re-route every stream task by exhaustive search on its retained examples, then sleep.

Plan: `O5_REROUTE_SLEEP_PLAN.md` (frozen 7ec7a28). EXPLORATORY. Starts from the saved
SHUFFLED terminals of O2 (13-19), O3 (20-26) and O4 (sealed 900-914, design evidence only).
The sleep step is O3's `run_sleep` construction verbatim (reservoir 64/task, 8192 updates,
sampling [1941, w, s, 64]); the only change is a route re-assignment before it. G0 reloads
terminals exactly; G1 requires any cell whose routes all stay unchanged to reproduce the
committed SLEEP cell bitwise.
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
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_j2a_staged_library import enum_route
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O5_REROUTE_SLEEP_PLAN.md')
ROOT = Path('artifacts/o5_reroute_sleep')
OUTPUT = Path('reports/o5_reroute_sleep.json')
BANDS = {
    'O2': {'worlds': range(13, 20), 'work': Path('artifacts/o2_online_reliability/work'),
           'terminal_report': Path('reports/o2_online_reliability.json'), 'terminal_key': 'SHUFFLED_w{w}_s{s}',
           'sleep_report': Path('reports/o2d_sleep_memory.json'), 'sleep_key': 'RES64_w{w}_s{s}'},
    'O3': {'worlds': range(20, 27), 'work': Path('artifacts/o3_online_sleep_v2/work'),
           'terminal_report': Path('reports/o3_online_sleep_v2.json'), 'terminal_key': 'SHUFFLED_w{w}_s{s}',
           'sleep_report': Path('reports/o3_online_sleep_v2.json'), 'sleep_key': 'SLEEP_w{w}_s{s}'},
    'O4': {'worlds': range(900, 915), 'work': Path('artifacts/o4_sealed_confirmation/work'),
           'terminal_report': Path('reports/o4_sealed_confirmation.json'), 'terminal_key': 'SHUFFLED_w{w}_s{s}',
           'sleep_report': Path('reports/o4_sealed_confirmation.json'), 'sleep_key': 'SLEEP_w{w}_s{s}'},
}
STREAMS = (0, 1, 2)
THRESHOLD = 0.05
COLLAPSE = 1.0
JOBS = 3
MIN_FREE_GIB = 8.0


def cells():
    return [(b, w, s) for b, spec in BANDS.items() for w in spec['worlds'] for s in STREAMS]


def protocol():
    reports = sorted({str(spec[k]) for spec in BANDS.values() for k in ('terminal_report', 'sleep_report')})
    return {'id': 'o5-reroute-sleep-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'exploratory': True, 'bands': {b: [min(v['worlds']), max(v['worlds'])] for b, v in BANDS.items()},
            'input_sha256': {p: digest(Path(p)) for p in [PLAN.as_posix(), 'configs/v1.yaml'] + reports},
            'implementation_sha256': digest(Path(__file__)), 'o3_sha256': digest(Path(o3.__file__)),
            'o2c_sha256': digest(Path(o2c.__file__)), 'o2d_sha256': digest(Path(o2d.__file__)),
            'lifetime_sha256': digest(Path(ll.__file__))}


def terminal_path(band, w, s):
    return BANDS[band]['work'] / f'SHUFFLED_w{w}_s{s}' / 'lifetime' / 'model.pt'


def reroute(model, stream_tasks, plan, pool):
    """Exhaustive route search per task on its reservoir examples; minimal logit swap. Returns #changed."""
    library = FrozenLibrary(model)
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    current = model.hard_routes()
    changed = 0
    with torch.no_grad():
        for task in stream_tasks:
            d = plan[task.task_id]
            xs = torch.tensor(np.stack([a for a, _ in by[task.task_id]]), dtype=torch.float32)
            ys = torch.tensor(np.stack([b for _, b in by[task.task_id]]), dtype=torch.float32)
            library.steps = d
            new = list(enum_route(library, xs, ys))
            code = model.task_codes[task.task_id]
            for step in range(d):
                old = int(torch.argmax(code[step]))
                if old != new[step]:
                    a, b = code[step, old].clone(), code[step, new[step]].clone()
                    code[step, old], code[step, new[step]] = b, a
            if list(current[task.task_id][:d]) != new:
                changed += 1
            if list(model.hard_routes()[task.task_id][:d]) != new:
                raise RuntimeError(f'{task.task_id}: route swap did not take')
    return changed


def run_cell(band, w, s):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg3, model, stream_tasks, canonical = o3.load_shuffled_terminal(terminal_path(band, w, s), w, s)
    _, _, _, plan, _ = o2.build_stream('SHUFFLED', w, s)
    before = library_sha(model)
    pool = o2d.reservoir(stream_tasks, w, s, o3.MEMORY)
    changed = reroute(model, stream_tasks, plan, pool)
    o2c.consolidate(cfg3, model, pool, [t.task_id for t in stream_tasks], [1941, w, s, o3.MEMORY],
                    updates=o3.EXTRA_UPDATES)
    terminal = score(model, SimpleNamespace(tasks=canonical))
    return {'band': band, 'world': w, 'stream': s, 'routes_changed': changed, 'pool_size': len(pool),
            'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
            'library_sha256_before': before, 'library_sha256': library_sha(model),
            'seconds': time.perf_counter() - started}


def reference(band, w, s, which):
    spec = BANDS[band]
    rep = json.loads(spec[f'{which}_report'].read_text())['cells']
    return rep[spec[f'{which}_key'].format(w=w, s=s)]


def gate_g0():
    """Reload reproduces the committed SHUFFLED terminal exactly: one cell per band."""
    out = {}
    for band, spec in BANDS.items():
        w = min(spec['worlds'])
        _, model, _, canonical = o3.load_shuffled_terminal(terminal_path(band, w, 0), w, 0)
        out[band] = score(model, SimpleNamespace(tasks=canonical))['per_task'] == \
            reference(band, w, 0, 'terminal')['terminal_per_task']
    return {'passes': all(out.values()), 'by_band': out}


def label(r_c, b):
    if b >= 2:
        return 'HARMS'
    if r_c >= 6:
        return 'COLLAPSE_REPAIRED'
    return 'PARTIAL' if r_c >= 3 else 'NO_REPAIR'


def summarize(records):
    C, P, rows = [], [], {}
    for band, w, s in cells():
        key = f'{band}_w{w}_s{s}'
        term = reference(band, w, s, 'terminal')['terminal_median']
        slp = reference(band, w, s, 'sleep')['terminal_median']
        new = records[key]['terminal_median']
        rows[key] = [term, slp, new]
        if term >= COLLAPSE:
            C.append(key)
        if slp < THRESHOLD:
            P.append(key)
    r_c = sum(rows[k][2] < THRESHOLD for k in C)
    b = sum(not rows[k][2] < THRESHOLD for k in P)
    per_band = {band: {'sleep_pass': sum(rows[f'{band}_w{w}_s{s}'][1] < THRESHOLD for w in spec['worlds'] for s in STREAMS),
                       'reroute_sleep_pass': sum(rows[f'{band}_w{w}_s{s}'][2] < THRESHOLD for w in spec['worlds'] for s in STREAMS),
                       'cells': 3 * len(spec['worlds'])} for band, spec in BANDS.items()}
    return {'collapse_cells': C, 'r_C': r_c, 'sleep_passing': len(P), 'b': b, 'label': label(r_c, b),
            'per_band': per_band, 'rows': rows}


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
        g0 = gate_g0()
        log_line(ROOT / 'run.log', f"GATE G0={g0}")
        if not g0['passes']:
            raise RuntimeError(f'G0 failed: {g0}')
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
                else:
                    todo.append((band, w, s))
            total = len(cells())
            log_line(ROOT / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} free={free:.1f}GiB')
            g1_checked = g1_failed = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, b, w, s): (b, w, s) for b, w, s in todo}
                for future in as_completed(futures):
                    b, w, s = futures[future]
                    key = f'{b}_w{w}_s{s}'
                    record = future.result()
                    if record['routes_changed'] == 0:   # G1: must equal the committed SLEEP cell bitwise
                        ref = reference(b, w, s, 'sleep')
                        record['g1_equals_sleep'] = (record['library_sha256'] == ref['library_sha256']
                                                     and record['terminal_per_task'] == ref['terminal_per_task'])
                        g1_checked += 1
                        if not record['g1_equals_sleep']:
                            raise RuntimeError(f'G1 failed: {key} unchanged routes but differs from SLEEP')
                    atomic_json(ROOT / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    log_line(ROOT / 'run.log', f"cell finished {key} changed={record['routes_changed']} "
                                               f"{record['seconds']:.1f}s terminal={record['terminal_median']:.6g}")
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
    print(f'O5 report: {OUTPUT}')


if __name__ == '__main__':
    main()
