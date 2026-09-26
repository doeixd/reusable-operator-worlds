"""O3 Tier 2: online order-free anchors + a sleep phase, against the same compute interleaved.

Plan: `O3_ONLINE_SLEEP_CONFIRMATION_PLAN.md` (frozen 5120482). Worlds 20-26 of
development band 3, streams 0-2, model seed 5000.

- SHUFFLED: `o2.run_single` verbatim (LEAN).
- SLEEP: the same cell's saved SHUFFLED terminal, then O2D's RES64 sleep verbatim
  (`o2c.consolidate`, 8192 updates, `o2d.reservoir` at 64 per task, sampling [1941, w, s, 64]).
- INTERLEAVED: the same SHUFFLED lifetime plus a `prospective_hook` that, after each
  completed task, adds that task's reservoir examples and runs its share of exactly 8192
  consolidation updates on the pool so far (persistent AdamW, sampling [1950, w, s]).
- PLAIN: `o2.run_staged('PLAIN', w, 0)`.

Modes: default = registered run (needs passing gates at this protocol); `--gates`;
`--dry-run [--stop-after N]` = scale-16 restart test on world 20.
"""
from __future__ import annotations

import argparse
import dataclasses
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

import numpy as np
import psutil
import torch

from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import ANCHOR_TOLERANCE, score
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O3_ONLINE_SLEEP_CONFIRMATION_PLAN.md')
O2_REPORT = Path('reports/o2_online_reliability.json')
O2D_REPORT = Path('reports/o2d_sleep_memory.json')
# v2 paths: the first launch (v1, 2026-09-26) failed on an invalid anchor check before writing any cell;
# its directory is kept, archived to reports/o3_failed_launch_20260926/, and never reused.
ROOT = Path('artifacts/o3_online_sleep_v2')
DRY_ROOT = Path('artifacts/o3_online_sleep_v2_dry')
OUTPUT = Path('reports/o3_online_sleep_v2.json')
WORLDS = (20, 21, 22, 23, 24, 25, 26)
STREAMS = (0, 1, 2)
ARM_STREAMS = {'INTERLEAVED': STREAMS, 'SHUFFLED': STREAMS, 'SLEEP': STREAMS, 'PLAIN': (0,)}
EXTRA_UPDATES = 8192
MEMORY = 64
THRESHOLD = 0.05
JOBS = 3
MIN_FREE_GIB = 8.0
DRY_SCALE = 16


def cells(worlds=WORLDS):
    return [(a, w, s) for a in ('INTERLEAVED', 'SHUFFLED', 'SLEEP', 'PLAIN') for w in worlds for s in ARM_STREAMS[a]]


def protocol(root=ROOT):
    return {'id': 'o3-online-sleep-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 2,
            'worlds': list(WORLDS), 'arm_streams': {a: list(s) for a, s in ARM_STREAMS.items()},
            'extra_updates': EXTRA_UPDATES, 'memory_per_task': MEMORY, 'threshold': THRESHOLD, 'jobs': JOBS,
            'root': Path(root).as_posix(),
            'seeds': {'order': '[1920, w] / [1920, w, s]', 'replay': 'None / [7500, w, s]',
                      'reservoir': '[1940, w, s, i]', 'sleep_sampling': '[1941, w, s, 64]',
                      'interleaved_sampling': '[1950, w, s]'},
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), O2_REPORT, O2D_REPORT)},
            'implementation_sha256': digest(Path(__file__)), 'o2_sha256': digest(Path(o2.__file__)),
            'o2c_sha256': digest(Path(o2c.__file__)), 'o2d_sha256': digest(Path(o2d.__file__)),
            'lifetime_sha256': digest(Path(ll.__file__)),
            'learner_sha256': digest(Path(sys.modules[o2.PlannedDepthRotatedLearner.__module__].__file__))}


# ------------------------------------------------------------------ construction

def schedule(total, n_tasks):
    """n_t = round(total (t+1)/n) - round(total t/n): exactly `total` updates over the stream."""
    return [int(round(total * (t + 1) / n_tasks)) - int(round(total * t / n_tasks)) for t in range(n_tasks)]


class Interleaver:
    """prospective_hook: after task t, add its reservoir examples, then run n_t consolidation updates."""

    def __init__(self, cfg, stream_tasks, world, stream, total=EXTRA_UPDATES):
        self.cfg, self.stream, self.world, self.s = cfg, stream_tasks, world, stream
        self.n = schedule(total, len(stream_tasks))
        self.pool, self.optimizer, self.code_group, self.updates = [], None, None, 0
        self.rng = np.random.default_rng(np.random.SeedSequence([1950, world, stream]))

    def __call__(self, model, lifetime_index, world_task_index):
        task = self.stream[world_task_index]
        pick = np.random.default_rng(np.random.SeedSequence([1940, self.world, self.s, world_task_index]))
        for j in pick.choice(len(task.train_x), size=MEMORY, replace=False):
            self.pool.append((task.train_x[j], task.train_y[j], task.task_id))
        n = self.n[lifetime_index]
        if n == 0:
            return None
        codes = [model.task_codes[t.task_id] for t in self.stream[: world_task_index + 1]]
        if self.optimizer is None:
            global_lr, task_lr, weight_decay, _, _, _, _ = ll._training_values(self.cfg, o2.KIND)
            self.optimizer = ll._shared_optimizer(model, global_lr, weight_decay)
            self.optimizer.add_param_group({'params': codes, 'lr': task_lr, 'weight_decay': 0.0})
            self.code_group = self.optimizer.param_groups[-1]
        else:
            known = {id(p) for p in self.code_group['params']}
            self.code_group['params'].extend(p for p in codes if id(p) not in known)
        model.train()
        for _ in range(n):
            idx = self.rng.choice(len(self.pool), size=2, replace=False)
            x = torch.as_tensor(np.stack([self.pool[i][0] for i in idx]), dtype=torch.float32)
            y = torch.as_tensor(np.stack([self.pool[i][1] for i in idx]), dtype=torch.float32)
            self.optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(model.forward_tasks(x, [self.pool[i][2] for i in idx]), y).backward()
            self.optimizer.step()
        self.updates += n
        return None


def first16_canonical_eot(output: Path, canonical_ids: set) -> float:
    rows = [r for r in o2.task_summaries(output) if r['task_id'] in canonical_ids]
    return float(np.median([float(r['final_nmse']) for r in rows[:16]]))


def run_interleaved(world, stream, work: Path, scale=1, total=EXTRA_UPDATES):
    """o2.run_single's construction, plus the Interleaver hook."""
    cfg3, world3, stream_tasks, plan, canonical = o2.build_stream('SHUFFLED', world, stream)
    mixed = dataclasses.replace(world3, tasks=tuple(stream_tasks))
    output = work / 'lifetime'
    hook = Interleaver(cfg3, stream_tasks, world, stream, total)
    ran = o2.run_cfg(cfg3, scale)
    summary = ll.run(dataclasses.replace(ran, output_directory=output), o2.KIND, world=mixed,
                     model=o2.planned_model(cfg3, plan), return_model=True,
                     replay_seed=o2.replay_seed_for(world, stream), prospective_hook=hook)
    model = summary.pop('terminal_model')
    canon_ids = {t.task_id for t in canonical}
    terminal = score(model, SimpleNamespace(tasks=[t for t in canonical if t.task_id in model.task_codes]))
    rows = o2.task_summaries(output)
    last = rows[-1]
    last_task = next(t for t in stream_tasks if t.task_id == last['task_id'])
    last_terminal = score(model, SimpleNamespace(tasks=[last_task]))['per_task'][last_task.task_id]
    routes = model.hard_routes()
    return {'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
            'scored_tasks': len(terminal['per_task']), 'stream_tasks': len(stream_tasks), 'trained_tasks': len(rows),
            'route_lengths_match_plan': all(t in routes and len(routes[t]) == plan[t] for t in plan),
            'anchor_abs_error': abs(last_terminal - float(last['final_nmse'])),
            'end_of_task_median': float(np.median([float(r['final_nmse']) for r in rows if r['task_id'] in canon_ids])),
            'first16_canonical_eot': first16_canonical_eot(output, canon_ids),
            'extra_updates': hook.updates, 'pool_size': len(hook.pool), 'library_sha256': library_sha(model)}


def load_shuffled_terminal(path: Path, world, stream):
    cfg3, _, stream_tasks, plan, canonical = o2.build_stream('SHUFFLED', world, stream)
    model = o2.planned_model(cfg3, plan)
    for t in stream_tasks:
        model.begin_task(t.task_id)
    state = torch.load(path, weights_only=True)['model_state_dict']
    for k in state:
        if k.startswith('task_codes.') and k not in model.state_dict():
            model.begin_task(k.split('.', 1)[1], 3)
    model.load_state_dict(state, strict=True)
    return cfg3, model, stream_tasks, canonical


def run_sleep(world, stream, terminal_path: Path, updates=EXTRA_UPDATES):
    """O2D's RES64 construction verbatim, applied to a saved SHUFFLED terminal."""
    cfg3, model, stream_tasks, canonical = load_shuffled_terminal(terminal_path, world, stream)
    before = library_sha(model)
    pool = o2d.reservoir(stream_tasks, world, stream, MEMORY)
    o2c.consolidate(cfg3, model, pool, [t.task_id for t in stream_tasks], [1941, world, stream, MEMORY],
                    updates=updates)
    terminal = score(model, SimpleNamespace(tasks=canonical))
    return {'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
            'scored_tasks': len(terminal['per_task']), 'pool_size': len(pool), 'extra_updates': updates,
            'library_sha256_before': before, 'library_sha256': library_sha(model)}


def run_cell(arm, world, stream, root, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    root = Path(root)
    work = root / 'work' / f'{arm}_w{world}_s{stream}'
    if arm == 'SHUFFLED':
        rec = o2.run_single('SHUFFLED', world, stream, work, scale)
        canon = set(rec['terminal_per_task'])
        rec['first16_canonical_eot'] = first16_canonical_eot(work / 'lifetime', canon)
    elif arm == 'SLEEP':
        rec = run_sleep(world, stream, root / 'work' / f'SHUFFLED_w{world}_s{stream}' / 'lifetime' / 'model.pt')
    elif arm == 'INTERLEAVED':
        rec = run_interleaved(world, stream, work, scale)
    else:
        rec = o2.run_staged('PLAIN', world, stream, work, scale)
    rec.update({'arm': arm, 'world': world, 'stream': stream, 'scale': scale,
                'seconds': time.perf_counter() - started})
    return rec


def validate(rec):
    key = f"{rec['arm']}_w{rec['world']}_s{rec['stream']}"
    if rec['scored_tasks'] != 64 or len(rec['terminal_per_task']) != 64:
        raise ValueError(f'{key}: not scored on the canonical 64')
    if rec['arm'] in ('SHUFFLED', 'INTERLEAVED'):
        if rec['stream_tasks'] != 188 or rec['trained_tasks'] != 188 or not rec['route_lengths_match_plan']:
            raise ValueError(f'{key}: stream/route check')
    # The last-task anchor (terminal == end-of-task on the last stream task) holds only when nothing trains
    # after the last task. INTERLEAVED consolidates after every task, including the last, so its anchor
    # error is recorded but not required to vanish (the v1 launch failed on exactly this).
    if rec['arm'] == 'SHUFFLED' and rec['anchor_abs_error'] > ANCHOR_TOLERANCE:
        raise ValueError(f'{key}: last-task anchor {rec["anchor_abs_error"]}')
    if rec['arm'] in ('SLEEP', 'INTERLEAVED') and rec['extra_updates'] != EXTRA_UPDATES:
        raise ValueError(f'{key}: extra updates {rec["extra_updates"]} != {EXTRA_UPDATES}')
    if rec['arm'] == 'SLEEP' and rec['library_sha256'] == rec['library_sha256_before']:
        raise ValueError(f'{key}: sleep left the library unchanged')


# ------------------------------------------------------------------ rules

def label(k):
    return 'RELIABLE' if k >= 19 else ('INTERMEDIATE' if k >= 17 else 'UNRELIABLE')


def contrast_r(ms, mi):
    fs, fi = math.isfinite(ms), math.isfinite(mi)
    if not fs and not fi:
        return 0.0
    if not fs:
        return math.inf
    if not fi:
        return -math.inf
    return math.log10(max(ms, 1e-12) / max(mi, 1e-12))


def contrast_label(r_by_world):
    """r_by_world: 7 lists of 3 r values (stream order)."""
    r = [x for ws in r_by_world for x in ws]
    neg, pos = sum(x < 0 for x in r), sum(x > 0 for x in r)
    wneg = sum(sum(x < 0 for x in ws) >= 2 for ws in r_by_world)
    wpos = sum(sum(x > 0 for x in ws) >= 2 for ws in r_by_world)
    med = float(np.median(r))
    if neg >= 16 and wneg >= 6 and med <= -0.15:
        return 'PHASE_MATTERS'
    if pos >= 16 and wpos >= 6 and med >= 0.15:
        return 'INTERLEAVED_BETTER'
    if 7 <= neg <= 14 and abs(med) <= 0.10:
        return 'EQUIVALENT'
    return 'INDETERMINATE'


def passing(m):
    return math.isfinite(m) and m < THRESHOLD


def summarize(records, worlds=WORLDS):
    M = {k: v['terminal_median'] for k, v in records.items()}
    k = {a: sum(passing(M[f'{a}_w{w}_s{s}']) for w in worlds for s in ARM_STREAMS[a]) for a in ARM_STREAMS}
    floor = k['PLAIN'] > 0
    r = [[contrast_r(M[f'SLEEP_w{w}_s{s}'], M[f'INTERLEAVED_w{w}_s{s}']) for s in STREAMS] for w in worlds]
    out = {'passes': k, 'floor_failed': floor,
           'primary': 'FLOOR_FAILED' if floor else 'WAKE_SLEEP_' + label(k['SLEEP']),
           'contrast': 'FLOOR_FAILED' if floor else contrast_label(r),
           'interleaved_label': 'FLOOR_FAILED' if floor else label(k['INTERLEAVED']),
           'r_by_world': {str(w): v for w, v in zip(worlds, r)},
           'collapses': {a: sum(not (math.isfinite(M[f'{a}_w{w}_s{s}']) and M[f'{a}_w{w}_s{s}'] < 1.0)
                                for w in worlds for s in ARM_STREAMS[a]) for a in ARM_STREAMS}}
    return out


# ------------------------------------------------------------------ gates

def gates(root=ROOT):
    g = Path(root) / 'gates'
    g.mkdir(parents=True, exist_ok=True)
    out = {}
    o2cells = json.loads(O2_REPORT.read_text())['cells']
    o2dcells = json.loads(O2D_REPORT.read_text())['cells']
    e1 = o2.run_single('SHUFFLED', 13, 0, g / 'E1')
    ref = o2cells['SHUFFLED_w13_s0']
    out['E1'] = {'passes': e1['library_sha256'] == ref['library_sha256']
                 and e1['terminal_per_task'] == ref['terminal_per_task']}
    log_line(g / 'gates.log', f"E1 {out['E1']}")
    e2 = run_sleep(13, 0, o2c.O2_WORK / 'SHUFFLED_w13_s0' / 'lifetime' / 'model.pt')
    ref2 = o2dcells['RES64_w13_s0']
    out['E2'] = {'passes': e2['library_sha256'] == ref2['library_sha256']
                 and e2['terminal_per_task'] == ref2['terminal_per_task']}
    log_line(g / 'gates.log', f"E2 {out['E2']}")
    base = o2.run_single('SHUFFLED', 20, 0, g / 'E3_base', DRY_SCALE)
    zero = run_interleaved(20, 0, g / 'E3_zero', DRY_SCALE, total=0)
    full = run_interleaved(20, 0, g / 'E3_full', DRY_SCALE)
    out['E3'] = {'passes': zero['library_sha256'] == base['library_sha256']
                 and zero['terminal_per_task'] == base['terminal_per_task']
                 and full['library_sha256'] != base['library_sha256'] and full['extra_updates'] == EXTRA_UPDATES,
                 'zero_matches_base': zero['library_sha256'] == base['library_sha256'],
                 'full_updates': full['extra_updates']}
    log_line(g / 'gates.log', f"E3 {out['E3']}")
    e4b = {}
    for w in WORLDS:
        for s in STREAMS:
            _, _, st, plan, _ = o2.build_stream('SHUFFLED', w, s)
            e4b[f'w{w}_s{s}'] = len({plan[t.task_id] for t in st[:20]}) >= 2
    out['E4b'] = {'passes': all(e4b.values()), 'failing': [k for k, v in e4b.items() if not v]}
    out['all_pass'] = all(v['passes'] for v in out.values() if isinstance(v, dict))
    out['protocol_sha256'] = fingerprint(protocol())
    atomic_json(g / 'gates.json', out)
    return out


# ------------------------------------------------------------------ pool

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
                    if a == 'SLEEP' and f'SHUFFLED_w{w}_s{s}' not in records:
                        continue   # submitted when its SHUFFLED cell completes
                    futures[submit(a, w, s)] = (a, w, s)
                pending_sleep = {(w, s) for a, w, s in todo if a == 'SLEEP' and f'SHUFFLED_w{w}_s{s}' not in records}
                _status(root, 'running', len(records), total, [f'{a}_w{w}_s{s}' for a, w, s in list(futures.values())[:jobs]], started)
                while futures:
                    done_future = next(as_completed(futures))
                    a, w, s = futures.pop(done_future)
                    key = f'{a}_w{w}_s{s}'
                    record = done_future.result()
                    validate(record)   # at every scale, so the dry run exercises it (v1 lesson)
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
    if args.dry_run:
        dry = [('INTERLEAVED', 20, 0), ('SHUFFLED', 20, 0), ('SHUFFLED', 20, 1), ('PLAIN', 20, 0), ('SLEEP', 20, 0)]
        run(dry, root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=DRY_SCALE, stop_after=args.stop_after,
            require_gates=False, worlds=(20,))
        print('O3 dry run complete')
        return
    if args.stop_after is not None:
        raise SystemExit('--stop-after is for the dry run only')
    require_clean_code(OUTPUT)
    for check in ('tools/check_prereg.py', 'tools/check_invalid.py'):
        subprocess.run([sys.executable, check], check=True)
    run(cells())
    print(f'O3 report: {OUTPUT}')


if __name__ == '__main__':
    main()
