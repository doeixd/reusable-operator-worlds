"""G1S2 (Tier 1, EXPLORATORY, sizing): sleep length and wake updates for the `tanh` learner, on one world.

Plan: `G1S2_TANH_BUDGET_PLAN.md`. G1S (`G1S_TANH_SIZING_PLAN.md`, `reports/g1s_tanh_sizing.json`) labelled
`IMPROVES`: at lr 0.003, initial gain 1 the slept depth-1 error is 0.025 but depth 3 is 0.184, and the gain over G1T came
from sleep, not wake. Its plan's next step: a second sizing axis at the chosen setting, wake updates per example and
sleep length. Construction, world 70, stream 0, lr 0.003, gain 1, everything else G1S's:

- WAKE_u{u}: `o2.run_single('SHUFFLED')` verbatim with `discrete_model.updates_per_example = u`, u in {1, 2};
- SLEEP_u{u}: on that wake terminal, `o2c.consolidate`'s exact loop (same optimizer, groups, batch, seed
  `[1941, w, stream, 64]`, 64-per-task reservoir) run for 65,536 updates, scored at 8,192 / 16,384 / 32,768 / 65,536.
  One trajectory, so every checkpoint is a longer sleep from the same start with the same minibatch stream prefix.

Anchor A0: WAKE_u1 and SLEEP_u1 at 8,192 must equal G1S's `WAKE_lr0.003_g1` and `SLEEP_lr0.003_g1` per-task terminals
exactly; that licenses the inlined loop and the updates rebinding. At exit the runner copies its operational logs and
cell records into `reports/g1s2_tanh_budget_<date>/` (G1S learning: `artifacts/` is untracked and was emptied once
between a run's exit and its record)."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import torch
import yaml

from row.experiments import g1_formation_probe as g1
from row.experiments import g1s_tanh_sizing as g1s
from row.experiments import g1t_tanh_formation as g1t
from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock
from row.models import tanh_form
from row.models.tanh_form import TanhFormLearner

PLAN = Path('G1S2_TANH_BUDGET_PLAN.md')
G1S_REPORT = Path('reports/g1s_tanh_sizing.json')
ROOT = Path('artifacts/g1s2_tanh_budget')
DRY_ROOT = Path('artifacts/g1s2_tanh_budget_dry')
OUTPUT = Path('reports/g1s2_tanh_budget.json')
ARCHIVE_PREFIX = 'reports/g1s2_tanh_budget_'
WORLD, DRY_WORLD, STREAM = 70, 99, 0
SETTING = (0.003, 1.0)
WAKE_UPDATES = (1, 2)
CHECKPOINTS = (8192, 16384, 32768, 65536)
DRY_CHECKPOINTS = (64, 128)
ARMS = ('WAKE', 'SLEEP')
FITS, IMPROVES, IDENTITY_ARI = 0.05, 0.1, 0.8
JOBS = 2
MIN_FREE_GIB = 2.0
DRY_SCALE = o3.DRY_SCALE


def key_of(arm, u):
    return f'{arm}_u{u}'


def cells():
    return [(a, u) for a in ARMS for u in WAKE_UPDATES]


def work_of(root, u):
    return Path(root) / 'work' / f'u{u}'


def apply_budget(u):
    """G1S's setting (lr 0.003, gain 1) plus `updates_per_example = u`, in every stage config."""
    g1s.apply_setting(SETTING)
    sized = g1t.o2.stage_setup

    def budget_stage_setup(world_seed, stage, model_seed=5000):
        cfg, world, updates, seeds = sized(world_seed, stage, model_seed)
        return (replace(cfg, discrete_model=replace(cfg.discrete_model, updates_per_example=int(u))),
                world, updates, seeds)

    o2.stage_setup = budget_stage_setup


def restore():
    g1s.restore()


def sleep_curve(cfg, model, pool, task_ids, seed_entropy, checkpoints, at_checkpoint):
    """`o2c.consolidate`'s loop verbatim, with `at_checkpoint(n)` called after update n for every n in checkpoints.
    Scoring is `@torch.no_grad` in eval mode and touches no parameter, optimizer or RNG state."""
    global_lr, task_lr, weight_decay, _, _, _, _ = ll._training_values(cfg, o2.KIND)
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = ll._shared_optimizer(model, global_lr, weight_decay)
    optimizer.add_param_group({'params': [model.task_codes[t] for t in task_ids], 'lr': task_lr,
                               'weight_decay': 0.0})
    model.set_training_progress(1.0)
    model.train()
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    for n in range(1, max(checkpoints) + 1):
        idx = rng.choice(len(pool), size=o2c.BATCH, replace=False)
        x = torch.as_tensor(np.stack([pool[i][0] for i in idx]), dtype=torch.float32)
        y = torch.as_tensor(np.stack([pool[i][1] for i in idx]), dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model.forward_tasks(x, [pool[i][2] for i in idx]), y)
        loss.backward()
        optimizer.step()
        if n in checkpoints:
            at_checkpoint(n)
            model.train()
    model.eval()
    return global_lr


def run_cell(arm, u, w, root, scale=1, checkpoints=CHECKPOINTS):
    torch.set_num_threads(1)
    apply_budget(u)
    started = time.perf_counter()
    work = work_of(root, u)
    terminal_path = work / 'lifetime' / 'model.pt'
    if arm == 'WAKE':
        rec = o2.run_single('SHUFFLED', w, STREAM, work, scale)
        rec['d1_ari'] = g1.formed_diagnostics_d1(terminal_path, w)
        resolved = yaml.safe_load((work / 'lifetime' / 'config.yaml').read_text(encoding='utf-8'))
        rec['lifetime_lr'] = resolved[f'{o2.KIND}_model']['global_learning_rate']
        rec['lifetime_updates_per_example'] = resolved[f'{o2.KIND}_model']['updates_per_example']
    elif arm == 'SLEEP':
        cfg3, model, st, canonical = o3.load_shuffled_terminal(terminal_path, w, STREAM)
        before = o3.library_sha(model)
        pool = o2d.reservoir(st, w, STREAM, o3.MEMORY)
        _, _, _, plan, _ = o2.build_stream('SHUFFLED', w, STREAM)
        curve = {}

        def at_checkpoint(n):
            terminal = score(model, SimpleNamespace(tasks=canonical))
            curve[str(n)] = {'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
                             'depth': g1s.depth_profile(model, st, plan), 'd1_ari': g1s.d1_ari(model, st, plan),
                             'library_sha256': o3.library_sha(model), 'seconds': time.perf_counter() - started}

        sleep_lr = sleep_curve(cfg3, model, pool, [t.task_id for t in st], [1941, w, STREAM, o3.MEMORY],
                               checkpoints, at_checkpoint)
        last = curve[str(max(checkpoints))]
        rec = {'curve': curve, 'checkpoints': list(checkpoints), 'terminal_median': last['terminal_median'],
               'terminal_per_task': last['terminal_per_task'], 'scored_tasks': len(last['terminal_per_task']),
               'pool_size': len(pool), 'library_sha256_before': before, 'sleep_lr': sleep_lr,
               'learner_class': type(model).__name__}
    else:
        raise ValueError(arm)
    rec.update({'arm': arm, 'wake_updates': u, 'lr': SETTING[0], 'init_gain': SETTING[1], 'world': w,
                'stream': STREAM, 'scale': scale, 'learner_implementation': TanhFormLearner.implementation,
                'seconds': time.perf_counter() - started})
    return rec


def validate(rec, scale=1):
    key = key_of(rec['arm'], rec['wake_updates'])
    if rec['scored_tasks'] != 64 or len(rec['terminal_per_task']) != 64:
        raise ValueError(f'{key}: not scored on the canonical 64')
    if rec['arm'] == 'WAKE':
        o2.validate_cell({**rec, 'arm': 'SHUFFLED'}, scale)
        if rec['lifetime_lr'] != SETTING[0] or rec['lifetime_updates_per_example'] != rec['wake_updates']:
            raise ValueError(f'{key}: lifetime ran at lr {rec["lifetime_lr"]}, '
                             f'{rec["lifetime_updates_per_example"]} updates per example')
    else:
        if rec['pool_size'] != 188 * o3.MEMORY:
            raise ValueError(f'{key}: sleep pool')
        if rec['sleep_lr'] != SETTING[0] or rec['learner_class'] != 'TanhFormLearner':
            raise ValueError(f'{key}: sleep lr or learner')
        shas = [rec['library_sha256_before']] + [rec['curve'][str(n)]['library_sha256'] for n in rec['checkpoints']]
        if len(set(shas)) != len(shas):
            raise ValueError(f'{key}: the library did not change between checkpoints')
        for n in rec['checkpoints']:
            c = rec['curve'][str(n)]
            if len(c['terminal_per_task']) != 64 or not all(math.isfinite(v) for v in c['terminal_per_task'].values()):
                raise ValueError(f'{key}: checkpoint {n} terminal')
            if scale == 1 and [c['depth'][d]['n'] for d in '123'] != [60, 64, 64]:
                raise ValueError(f'{key}: checkpoint {n} depth denominators')


# ------------------------------------------------------------------ reading (exploratory; chooses, never a verdict)

def g1s_anchor():
    cells_ = json.loads(G1S_REPORT.read_text())['cells']
    return {'WAKE': cells_['WAKE_lr0.003_g1']['terminal_per_task'],
            'SLEEP': cells_['SLEEP_lr0.003_g1']['terminal_per_task']}


def point_tag(u, n):
    return f'u{u}_s{n}'


def reading(records, anchor):
    """A0, then choose the lowest slept terminal median over every (wake updates, sleep checkpoint) point whose slept
    vocabulary keeps identity, and label it. Compounding ratio = depth-3 / depth-1 median, per point."""
    a0 = (records[key_of('WAKE', 1)]['terminal_per_task'] == anchor['WAKE']
          and records[key_of('SLEEP', 1)]['curve'][str(CHECKPOINTS[0])]['terminal_per_task'] == anchor['SLEEP'])
    points = {}
    for u in WAKE_UPDATES:
        for n in CHECKPOINTS:
            c = records[key_of('SLEEP', u)]['curve'][str(n)]
            points[point_tag(u, n)] = {'terminal': c['terminal_median'], 'd1_ari': c['d1_ari'],
                                       'depth': {d: c['depth'][d]['median'] for d in '123'},
                                       'compounding': c['depth']['3']['median'] / c['depth']['1']['median']}
    eligible = {t: p['terminal'] for t, p in points.items() if p['d1_ari'] >= IDENTITY_ARI}
    chosen = min(eligible, key=eligible.get) if eligible else None
    best = eligible[chosen] if chosen else None
    if not a0:
        label = 'ANCHOR_MISS'
    elif chosen is None:
        label = 'IDENTITY_LOST'
    elif best < FITS:
        label = 'FITS'
    elif best < IMPROVES:
        label = 'IMPROVES'
    else:
        label = 'NO_GAIN'
    last = {f'u{u}': points[point_tag(u, CHECKPOINTS[-1])]['terminal'] / points[point_tag(u, CHECKPOINTS[-2])]['terminal']
            for u in WAKE_UPDATES}   # < 0.9 means the last sleep doubling still lowered the terminal by > 10%
    return {'label': label, 'anchor_A0': a0, 'chosen': chosen, 'chosen_terminal': best, 'points': points,
            'last_doubling_ratio': last, 'still_falling': {k: v < 0.9 for k, v in last.items()},
            'wake_terminal': {f'u{u}': records[key_of('WAKE', u)]['terminal_median'] for u in WAKE_UPDATES},
            'wake_d1_ari': {f'u{u}': records[key_of('WAKE', u)]['d1_ari'] for u in WAKE_UPDATES}}


# ------------------------------------------------------------------ gates and protocol

def gates():
    """B1: under each budget the stage configs carry lr 0.003 and `updates_per_example = u`, a fresh learner starts at
    gain 1, and the u = 1 learner is bitwise G1S's chosen setting's. B2: `restore` returns the defaults. B3: the
    inlined loop reproduces `o2c.consolidate` bitwise on a small pool (a few updates on a fresh learner)."""
    from row.config import load_config
    cfg = load_config('configs/v1.yaml')
    out = {'B1': {}}
    restore()
    g1s.apply_setting(SETTING)
    reference = {k: v.clone() for k, v in o2.planned_model(cfg, {}).state_dict().items()}
    for u in WAKE_UPDATES:
        apply_budget(u)
        stage = [o2.stage_setup(WORLD, k, 5000)[0].discrete_model for k in (1, 2, 3)]
        model = o2.planned_model(cfg, {})
        ok = (all(s.global_learning_rate == SETTING[0] and s.updates_per_example == u for s in stage)
              and tanh_form.INIT_GAIN == SETTING[1])
        if u == 1:
            ok = ok and all(torch.equal(v, model.state_dict()[k]) for k, v in reference.items())
        out['B1'][f'u{u}'] = bool(ok)
    apply_budget(1)
    cfg3 = o2.stage_setup(WORLD, 3, 5000)[0]
    tasks = ['a', 'b']
    pool = [(np.random.default_rng(i).normal(size=16).astype(np.float32),
             np.random.default_rng(100 + i).normal(size=16).astype(np.float32), tasks[i % 2]) for i in range(32)]
    m1, m2 = o2.planned_model(cfg3, {}), o2.planned_model(cfg3, {})
    for m in (m1, m2):
        for t in tasks:
            m.begin_task(t)
    o2c.consolidate(cfg3, m1, pool, tasks, [1, 2, 3], updates=24)
    seen = []
    sleep_curve(cfg3, m2, pool, tasks, [1, 2, 3], (8, 24), lambda n: seen.append(n))
    out['B3'] = (seen == [8, 24]
                 and all(torch.equal(v, m2.state_dict()[k]) for k, v in m1.state_dict().items()))
    restore()
    out['B2'] = tanh_form.INIT_GAIN == 1.0 and o2.stage_setup is g1t._ORIGINAL['stage_setup']
    out['all_pass'] = all(out['B1'].values()) and out['B2'] and out['B3']
    return out


def protocol(root=ROOT, world=WORLD, checkpoints=CHECKPOINTS):
    modules = (g1s, g1t, g1, o2, o2c, o2d, o3, ll, tanh_form, sys.modules[score.__module__])
    return {'id': 'g1s2-tanh-budget-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'exploratory': True, 'world': world, 'stream': STREAM, 'setting': list(SETTING),
            'wake_updates': list(WAKE_UPDATES), 'checkpoints': list(checkpoints), 'fits': FITS,
            'improves': IMPROVES, 'identity_ari': IDENTITY_ARI, 'jobs': JOBS, 'root': Path(root).as_posix(),
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), G1S_REPORT)},
            'implementation_sha256': digest(Path(__file__)),
            'module_sha256': {m.__name__: digest(Path(m.__file__)) for m in modules}}


def _status(root, state, done, total, running, started):
    elapsed = time.time() - started
    eta = elapsed / done * (total - done) if done else None
    atomic_json(Path(root) / 'status.json', {
        'state': state, 'cells_done': done, 'cells_total': total, 'running': running, 'pid': os.getpid(),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started)), 'updated_utc': now(),
        'eta_seconds': eta})


def _headline(rec):
    if rec['arm'] == 'WAKE':
        return f"{rec['seconds']:.1f}s terminal={rec['terminal_median']:.6g} d1_ari={rec['d1_ari']:.3f}"
    parts = [f"{n}:{rec['curve'][str(n)]['terminal_median']:.4g}" for n in rec['checkpoints']]
    return f"{rec['seconds']:.1f}s curve " + ' '.join(parts)


def archive(root, dest):
    """Copy the small operational records and every cell record into the tracked reports tree."""
    dest = Path(dest)
    (dest / 'cells').mkdir(parents=True, exist_ok=True)
    for name in ('run.log', 'status.json', 'exit.json', 'gates.json', 'precondition.json', 'manifest.json',
                 'run.pid', 'error.json', 'stdout.txt', 'stderr.txt'):
        if (Path(root) / name).exists():
            shutil.copy2(Path(root) / name, dest / name)
    for cell in sorted((Path(root) / 'cells').glob('*.json')):
        shutil.copy2(cell, dest / 'cells' / cell.name)


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, world=WORLD,
        checkpoints=CHECKPOINTS, archive_to=None):
    """The parent writes every log line (G0F learning)."""
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root, world, checkpoints) | {'scale': scale, 'cells': [list(c) for c in todo_cells]}
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    started = time.time()
    with writer_lock(root / 'launcher.lock'):
        if (root / 'manifest.json').exists() and json.loads((root / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; retire the path rather than overwrite it')
        atomic_json(root / 'manifest.json', manifest)
        atomic_json(root / 'run.pid', {'pid': os.getpid(), 'started_utc': now()})
        free = psutil.virtual_memory().available / 2 ** 30
        atomic_json(root / 'precondition.json', {'free_gib': free, 'required_gib': MIN_FREE_GIB,
                                                 'passes': free >= MIN_FREE_GIB, 'jobs': jobs, 'checked_utc': now()})
        if free < MIN_FREE_GIB:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free')
        records, todo = {}, []
        try:
            gpath = root / 'gates.json'
            g = json.loads(gpath.read_text()) if gpath.exists() else None
            if not g or g.get('protocol_sha256') != sha:
                g = gates() | {'protocol_sha256': sha}
                atomic_json(gpath, g)
            log_line(root / 'run.log', f"GATES B1={g['B1']} B2={g['B2']} B3={g['B3']}")
            if not g['all_pass']:
                raise RuntimeError(f'gates failed: {g}')
            for arm, u in todo_cells:
                key = key_of(arm, u)
                path = root / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(root / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append((arm, u))
            total = len(todo_cells)
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} jobs={jobs} '
                                       f'scale={scale} free={free:.1f}GiB')
            # longest first: the u = 2 wake, then u = 1
            ready = sorted([c for c in todo if c[0] == 'WAKE' or key_of('WAKE', c[1]) in records],
                           key=lambda c: (c[0] != 'WAKE', -c[1]))
            waiting = [c for c in todo if c not in ready]
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, u, world, str(root), scale, checkpoints): (a, u) for a, u in ready}
                for a, u in ready[:jobs]:
                    log_line(root / 'run.log', f'submitted {key_of(a, u)}')
                _status(root, 'running', len(records), total, [key_of(*c) for c in ready[:jobs]], started)
                while futures:
                    done_future = next(as_completed(futures))
                    a, u = futures.pop(done_future)
                    key = key_of(a, u)
                    record = done_future.result()
                    validate(record, scale)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    log_line(root / 'run.log', f'cell finished {key} {_headline(record)}')
                    for dep in [c for c in waiting if c[1] == u and a == 'WAKE']:
                        waiting.remove(dep)
                        futures[pool.submit(run_cell, dep[0], u, world, str(root), scale, checkpoints)] = dep
                        log_line(root / 'run.log', f'submitted {key_of(*dep)}')
                    _status(root, 'running', len(records), total, [key_of(*c) for c in futures.values()], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            if waiting:
                raise RuntimeError(f'dependent cells never submitted: {waiting}')
            report = {**manifest, 'complete': True, 'gates': g, 'cells': records, 'finished_utc': now()}
            if scale == 1:
                report['reading'] = reading(records, g1s_anchor())
            atomic_json(output, report)
            log_line(root / 'run.log', f"EXIT 0 {report.get('reading', {}).get('label')}")
            _status(root, 'complete', len(records), total, [], started)
            atomic_json(root / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            log_line(root / 'run.log', 'FAILED ' + traceback.format_exc().replace('\n', ' | '))
            _status(root, 'failed', len(records), len(todo_cells), [], started)
            atomic_json(root / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(root / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise
        finally:
            if archive_to is not None:
                archive(root, archive_to)


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
        run(cells(), root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=DRY_SCALE, stop_after=args.stop_after,
            world=DRY_WORLD, checkpoints=DRY_CHECKPOINTS, archive_to=DRY_ROOT / 'archive')
        print(f'dry run complete: {DRY_ROOT}')
        return
    require_clean_code(OUTPUT)
    run(cells(), archive_to=ARCHIVE_PREFIX + time.strftime('%Y%m%d', time.gmtime()))
    print(f'G1S2 report: {OUTPUT}')


if __name__ == '__main__':
    main()
