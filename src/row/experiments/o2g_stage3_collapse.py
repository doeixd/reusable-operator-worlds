"""O2G Tier 1: is STAGED's stage-3 collapse systematic or stochastic?

Plan: `O2G_STAGE3_COLLAPSE_PLAN.md` (frozen 9a60528). EXPLORATORY. Re-runs stage 3
from O2's saved stage-2 models with fresh replay seeds, through O2's own
`lifetime` (LEAN) and SO2's `carry_library`. G0: O2's original seed reproduces
O2's stage-3 result bitwise on the three collapsed cells.
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

from row.experiments import audit_so2_online_gate as so2
from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O2G_STAGE3_COLLAPSE_PLAN.md')
O2_REPORT = Path('reports/o2_online_reliability.json')
O2_WORK = Path('artifacts/o2_online_reliability/work')
ROOT = Path('artifacts/o2g_stage3_collapse')
OUTPUT = Path('reports/o2g_stage3_collapse.json')
SEED_ROOT = 7600
RERUNS = (1, 2)
COLLAPSE = 1.0
JOBS = 3
MIN_FREE_GIB = 8.0


def cells():
    return [(w, s, k) for w in o2.WORLDS for s in o2.STREAMS for k in RERUNS]


def collapsed_o2_cells():
    rep = json.loads(O2_REPORT.read_text())['cells']
    return [(w, s) for w in o2.WORLDS for s in o2.STREAMS if rep[f'STAGED_w{w}_s{s}']['terminal_median'] >= COLLAPSE]


def protocol():
    return {'id': 'o2g-stage3-collapse-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(),
            'tier': 1, 'exploratory': True, 'seed_root': SEED_ROOT, 'reruns': list(RERUNS), 'collapse': COLLAPSE,
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), O2_REPORT)},
            'implementation_sha256': digest(Path(__file__)), 'o2_runner_sha256': digest(Path(o2.__file__)),
            'so2_sha256': digest(Path(so2.__file__)), 'lifetime_sha256': digest(Path(ll.__file__))}


def rerun_seed(w, s, k):
    return int(np.random.SeedSequence([SEED_ROOT, w, s, k]).generate_state(1)[0])


def load_stage2(w, s):
    cfg2, world2, _, _ = stage_setup(w, 2, o2.MODEL_SEED)
    model = build_fast(cfg2)
    for t in world2.tasks:
        model.begin_task(t.task_id)
    state = torch.load(O2_WORK / f'STAGED_w{w}_s{s}' / 'stage2' / 'model.pt', weights_only=True)['model_state_dict']
    for key in state:
        if key.startswith('task_codes.') and key not in model.state_dict():
            model.begin_task(key.split('.', 1)[1])
    model.load_state_dict(state, strict=True)
    return model


def stage3(w, s, replay_seed, output: Path):
    cfg3, world3, _, _ = stage_setup(w, 3, o2.MODEL_SEED)
    model = so2.carry_library(load_stage2(w, s), cfg3)
    _, model = o2.lifetime(cfg3, world3, output, model, replay_seed, 1)
    terminal = score(model, SimpleNamespace(tasks=list(world3.tasks)))
    rows = o2.task_summaries(output)
    return {'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
            'library_sha256': library_sha(model),
            'first16_end_of_task_median': float(np.median([r['final_nmse'] for r in rows[:16]])),
            'end_of_task_first8': [r['final_nmse'] for r in rows[:8]]}


def run_cell(w, s, k):
    torch.set_num_threads(1)
    started = time.perf_counter()
    rs = rerun_seed(w, s, k)
    out = stage3(w, s, rs, ROOT / 'work' / f'w{w}_s{s}_k{k}')
    return out | {'world': w, 'stream': s, 'rerun': k, 'replay_seed': rs, 'seconds': time.perf_counter() - started}


def gate_g0():
    rep = json.loads(O2_REPORT.read_text())['cells']
    res = {}
    for w, s in collapsed_o2_cells():
        out = stage3(w, s, o2.replay_seed_for(w, s), ROOT / 'gate' / f'w{w}_s{s}')
        o = rep[f'STAGED_w{w}_s{s}']
        res[f'w{w}_s{s}'] = out['library_sha256'] == o['library_sha256'] and \
            out['terminal_per_task'] == o['terminal_per_task']
    return {'passes': bool(res) and all(res.values()), 'by_cell': res}


def label(c_collapse, h_collapse):
    if c_collapse >= 5 and h_collapse <= 3:
        return 'SYSTEMATIC'
    if c_collapse <= 2:
        return 'STOCHASTIC'
    return 'MIXED'


def summarize(records):
    C = set(collapsed_o2_cells())
    cc = sum(records[f'w{w}_s{s}_k{k}']['terminal_median'] >= COLLAPSE for (w, s) in C for k in RERUNS)
    hc = sum(r['terminal_median'] >= COLLAPSE for r in records.values() if (r['world'], r['stream']) not in C)
    return {'collapsed_o2_cells': sorted(f'w{w}_s{s}' for w, s in C), 'C_reruns_collapsed_of_6': cc,
            'H_reruns_collapsed_of_36': hc, 'label': label(cc, hc)}


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
        gpath = ROOT / 'gates.json'
        if gpath.exists() and json.loads(gpath.read_text()).get('protocol_sha256') == sha:
            g0 = json.loads(gpath.read_text())['G0']
        else:
            g0 = gate_g0()
            atomic_json(gpath, {'G0': g0, 'protocol_sha256': sha})
        log_line(ROOT / 'run.log', f"GATE G0={g0['passes']} {g0['by_cell']}")
        if not g0['passes']:
            raise RuntimeError(f'G0 failed: {g0}')
        records, todo, started = {}, [], time.time()
        try:
            for w, s, k in cells():
                key = f'w{w}_s{s}_k{k}'
                path = ROOT / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                else:
                    todo.append((w, s, k))
            total = len(cells())
            log_line(ROOT / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} free={free:.1f}GiB')
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, w, s, k): (w, s, k) for w, s, k in todo}
                for future in as_completed(futures):
                    w, s, k = futures[future]
                    key = f'w{w}_s{s}_k{k}'
                    record = future.result()
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
    print(f'O2G report: {OUTPUT}')


if __name__ == '__main__':
    main()
