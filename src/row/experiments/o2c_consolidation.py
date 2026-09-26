"""O2C Tier 1: is the order-free online deficit a convergence deficit?

Plan: `O2C_CONSOLIDATION_OPPORTUNITY_PLAN.md` (frozen cee05a3). EXPLORATORY.
Starts from O2's saved SHUFFLED terminal models and applies U = 8192 further
updates on either ALL stream training data (ORACLE_DATA, the opportunity
ceiling) or the reconstructed end-of-stream replay buffer (REPLAY_ONLY, what an
online learner retains). Gates: G0 anchor (reload reproduces O2's terminal),
G1 buffer reconstruction against a spy on a real scaled lifetime, G2 non-vacuity.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O2C_CONSOLIDATION_OPPORTUNITY_PLAN.md')
O2_REPORT = Path('reports/o2_online_reliability.json')
O2_WORK = Path('artifacts/o2_online_reliability/work')
ROOT = Path('artifacts/o2c_consolidation')
OUTPUT = Path('reports/o2c_consolidation.json')
ARMS = ('ORACLE_DATA', 'REPLAY_ONLY')
ARM_ID = {'ORACLE_DATA': 1, 'REPLAY_ONLY': 2}
UPDATES = 8192
BATCH = 2
SEED_ROOT = 1930
THRESHOLD = 0.05
NEAR_HIGH = 0.2
JOBS = 3
MIN_FREE_GIB = 8.0
REPLAY_PER_TASK = 4


def cells():
    return [(a, w, s) for a in ARMS for w in o2.WORLDS for s in o2.STREAMS]


def protocol():
    return {'id': 'o2c-consolidation-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(),
            'tier': 1, 'exploratory': True, 'arms': list(ARMS), 'updates': UPDATES, 'batch': BATCH,
            'seed_root': SEED_ROOT, 'threshold': THRESHOLD, 'jobs': JOBS,
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), O2_REPORT)},
            'implementation_sha256': digest(Path(__file__)),
            'o2_runner_sha256': digest(Path(o2.__file__)),
            'lifetime_sha256': digest(Path(ll.__file__))}


# ------------------------------------------------------------------ construction

def replay_seed(world: int, stream: int, model_seed: int) -> int:
    """The seed learned_lifetime gives TaskReplayBuffer: model seed + 1 at stream 0."""
    rs = o2.replay_seed_for(world, stream)
    return model_seed + 1 if rs is None else rs


def reconstruct_buffer(stream_tasks, seed: int, examples_per_task: int, replay_per_task: int = REPLAY_PER_TASK):
    """Replay the lifetime's exact buffer call sequence: sample(1) per arriving example, add_task after each task."""
    buffer = ll.TaskReplayBuffer(seed)
    for task in stream_tasks:
        for _ in range(examples_per_task):
            buffer.sample(1)
        buffer.add_task(task, replay_per_task)
    return buffer.items


def load_terminal(world: int, stream: int):
    cfg3, _, stream_tasks, plan, canonical = o2.build_stream('SHUFFLED', world, stream)
    model = o2.planned_model(cfg3, plan)
    for t in stream_tasks:
        model.begin_task(t.task_id)
    state = torch.load(O2_WORK / f'SHUFFLED_w{world}_s{stream}' / 'lifetime' / 'model.pt',
                       weights_only=True)['model_state_dict']
    extra = [k.split('.', 1)[1] for k in state if k.startswith('task_codes.') and k not in model.state_dict()]
    for probe in extra:
        model.begin_task(probe, 3)
    model.load_state_dict(state, strict=True)
    return cfg3, model, stream_tasks, plan, canonical, extra


def consolidate(cfg, model, pool, task_ids, seed_entropy, updates=UPDATES):
    global_lr, task_lr, weight_decay, _, _, _, _ = ll._training_values(cfg, o2.KIND)
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = ll._shared_optimizer(model, global_lr, weight_decay)
    optimizer.add_param_group({'params': [model.task_codes[t] for t in task_ids], 'lr': task_lr,
                               'weight_decay': 0.0})
    model.set_training_progress(1.0)
    model.train()
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    for _ in range(updates):
        idx = rng.choice(len(pool), size=BATCH, replace=False)
        x = torch.as_tensor(np.stack([pool[i][0] for i in idx]), dtype=torch.float32)
        y = torch.as_tensor(np.stack([pool[i][1] for i in idx]), dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model.forward_tasks(x, [pool[i][2] for i in idx]), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def run_cell(arm, world, stream):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg3, model, stream_tasks, plan, canonical, _ = load_terminal(world, stream)
    before = library_sha(model)
    if arm == 'ORACLE_DATA':
        pool = [(t.train_x[i], t.train_y[i], t.task_id) for t in stream_tasks for i in range(len(t.train_x))]
    else:
        pool = reconstruct_buffer(stream_tasks, replay_seed(world, stream, cfg3.discrete_model.seed),
                                  cfg3.world.examples_per_task)
    consolidate(cfg3, model, pool, [t.task_id for t in stream_tasks], [SEED_ROOT, world, stream, ARM_ID[arm]],
                updates=UPDATES)
    terminal = score(model, SimpleNamespace(tasks=canonical))
    return {'arm': arm, 'world': world, 'stream': stream, 'pool_size': len(pool),
            'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
            'library_sha256_before': before, 'library_sha256': library_sha(model),
            'seconds': time.perf_counter() - started}


# ------------------------------------------------------------------ gates

def gate_anchor():
    """G0: reloading every O2 SHUFFLED terminal reproduces its recorded per-task terminal exactly."""
    report = json.loads(O2_REPORT.read_text())['cells']
    bad = []
    for w in o2.WORLDS:
        for s in o2.STREAMS:
            _, model, _, _, canonical, _ = load_terminal(w, s)
            per = score(model, SimpleNamespace(tasks=canonical))['per_task']
            if per != report[f'SHUFFLED_w{w}_s{s}']['terminal_per_task']:
                bad.append(f'w{w}_s{s}')
    return {'passes': not bad, 'mismatched': bad}


def gate_buffer(tmp: Path):
    """G1: a spy on a real scale-16 lifetime sees exactly the reconstructed buffer."""
    captured = {}

    class Spy(ll.TaskReplayBuffer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured['buffer'] = self

    results = {}
    original = ll.TaskReplayBuffer
    try:
        ll.TaskReplayBuffer = Spy
        for stream in (0, 1):
            o2.run_single('SHUFFLED', 13, stream, tmp / f'G1_s{stream}', o2.DRY_SCALE)
            real = captured['buffer'].items
            cfg3, _, stream_tasks, _, _ = o2.build_stream('SHUFFLED', 13, stream)
            scaled = o2.run_cfg(cfg3, o2.DRY_SCALE)
            rebuilt = reconstruct_buffer(stream_tasks, replay_seed(13, stream, cfg3.discrete_model.seed),
                                         scaled.world.examples_per_task)
            same = len(real) == len(rebuilt) and all(
                a[2] == b[2] and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                for a, b in zip(real, rebuilt))
            results[f's{stream}'] = {'items': len(real), 'identical': bool(same)}
    finally:
        ll.TaskReplayBuffer = original
    return {'passes': all(v['identical'] and v['items'] == 752 for v in results.values()), 'by_stream': results}


# ------------------------------------------------------------------ pool

def label(r, b):
    if r >= 5 and b <= 1:
        return 'RESCUES'
    if b >= 2:
        return 'HARMS'
    return 'NO_RESCUE'


def summarize(records):
    base = json.loads(O2_REPORT.read_text())['cells']
    m0 = {(w, s): base[f'SHUFFLED_w{w}_s{s}']['terminal_median'] for w in o2.WORLDS for s in o2.STREAMS}
    near = [k for k, v in m0.items() if THRESHOLD <= v < NEAR_HIGH]
    passing = [k for k, v in m0.items() if v < THRESHOLD]
    out = {'near_miss_cells': len(near), 'passing_cells': len(passing), 'arms': {}}
    for arm in ARMS:
        m1 = {(w, s): records[f'{arm}_w{w}_s{s}']['terminal_median'] for w in o2.WORLDS for s in o2.STREAMS}
        r = sum(m1[k] < THRESHOLD for k in near)
        b = sum(not m1[k] < THRESHOLD for k in passing)
        out['arms'][arm] = {'r_rescued_of_8': r, 'b_broken_of_9': b, 'label': label(r, b),
                            'pass_of_21': sum(v < THRESHOLD for v in m1.values()),
                            'median_ratio_to_terminal': float(np.median([m1[k] / m0[k] for k in m0]))}
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
        gates_path = ROOT / 'gates.json'
        if gates_path.exists() and json.loads(gates_path.read_text()).get('protocol_sha256') == sha:
            gates = json.loads(gates_path.read_text())
        else:
            gates = {'G0': gate_anchor(), 'G1': gate_buffer(ROOT / 'gate_work'), 'protocol_sha256': sha}
            atomic_json(gates_path, gates)
        log_line(ROOT / 'run.log', f"GATES G0={gates['G0']['passes']} G1={gates['G1']['passes']}")
        if not (gates['G0']['passes'] and gates['G1']['passes']):
            raise RuntimeError(f'gates failed: {gates}')
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
            log_line(ROOT / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} free={free:.1f}GiB')
            atomic_json(ROOT / 'status.json', {'state': 'running', 'cells_done': len(records), 'cells_total': 42,
                                               'pid': os.getpid(), 'updated_utc': now()})
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, w, s): (a, w, s) for a, w, s in todo}
                for future in as_completed(futures):
                    a, w, s = futures[future]
                    key = f'{a}_w{w}_s{s}'
                    record = future.result()
                    if record['library_sha256'] == record['library_sha256_before']:
                        raise RuntimeError(f'G2 non-vacuity failed: {key} library unchanged')
                    atomic_json(ROOT / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    log_line(ROOT / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"terminal={record['terminal_median']:.6g}")
                    elapsed = time.time() - started
                    done_new = len(records) - (42 - len(todo))
                    atomic_json(ROOT / 'status.json', {
                        'state': 'running', 'cells_done': len(records), 'cells_total': 42, 'pid': os.getpid(),
                        'updated_utc': now(),
                        'eta_seconds': round(elapsed / max(1, done_new) * (42 - len(records)))})
            atomic_json(OUTPUT, {**manifest, 'complete': True, 'gates': gates, 'cells': records,
                                 'summary': summarize(records), 'finished_utc': now()})
            atomic_json(ROOT / 'status.json', {'state': 'complete', 'cells_done': 42, 'cells_total': 42,
                                               'pid': os.getpid(), 'updated_utc': now()})
            atomic_json(ROOT / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            atomic_json(ROOT / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(ROOT / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time-one', action='store_true', help='time one REPLAY_ONLY cell at 512 updates; no record')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.time_one:
        global UPDATES
        UPDATES = 512
        t0 = time.perf_counter()
        r = run_cell('REPLAY_ONLY', 13, 0)
        print(json.dumps({'seconds_512_updates': time.perf_counter() - t0, 'pool': r['pool_size']}))
        return
    require_clean_code(OUTPUT)
    run()
    print(f'O2C report: {OUTPUT}')


if __name__ == '__main__':
    main()
