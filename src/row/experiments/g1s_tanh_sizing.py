"""G1S (Tier 1, EXPLORATORY, sizing): learning rate and initial gain for the `tanh` learner, on one world.

Plan: `G1S_TANH_SIZING_PLAN.md`. The pre-stated `FAILS` branch of G1T (`G1T_TANH_FORMATION_PLAN.md`): G1T's learner,
at the rotated form's hyperparameters, routed every operation to its own slot but fitted none of them (depth-1 task
error 0.13-0.16 after sleep). This rung varies only the two learner-side settings that govern per-operation fit:

- `global_learning_rate` of the shared parameters, in wake AND sleep (both read `config.discrete_model`);
- `tanh_form.INIT_GAIN`, the gain of the slots' initial Haar-orthogonal `A` (and with it the initial output scale).

Construction per setting, world 70, stream 0, everything else G1T's:
- WAKE: `o2.run_single('SHUFFLED')` verbatim (wake alone on the order-free stream);
- SLEEP: `o3.run_sleep`'s exact call sequence on that wake terminal (load, 64-per-task reservoir, consolidate, score),
  inlined so the slept model can also be measured (per-depth error, depth-1 identity ARI) without sleeping twice.

The baseline setting (0.001, 1.0) is G1T's construction, so its two cells must reproduce G1T's `SHUFFLED_w70` and
`SLEEP_w70` per-task terminals exactly (anchor A0); that anchor is also what licenses the inlined sleep.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

from row.config import load_config
from row.experiments import g1_formation_probe as g1
from row.experiments import g1t_tanh_formation as g1t
from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock
from row.form_world import tanh_gain_scale
from row.models import tanh_form
from row.models.tanh_form import TanhFormLearner

PLAN = Path('G1S_TANH_SIZING_PLAN.md')
G1T_REPORT = Path('reports/g1t_tanh_formation.json')
ROOT = Path('artifacts/g1s_tanh_sizing')
DRY_ROOT = Path('artifacts/g1s_tanh_sizing_dry')
OUTPUT = Path('reports/g1s_tanh_sizing.json')
WORLD = 70
DRY_WORLD = 99
STREAM = 0
BASELINE = (0.001, 1.0)
SETTINGS = ((0.001, 1.0), (0.003, 1.0), (0.01, 1.0), (0.001, 2.0), (0.003, 2.0), (0.01, 2.0))
DRY_SETTINGS = ((0.001, 1.0), (0.01, 2.0))
ARMS = ('WAKE', 'SLEEP')
FITS, IMPROVES = 0.05, 0.2
IDENTITY_ARI = 0.8
JOBS = 3
MIN_FREE_GIB = 2.0
DRY_SCALE = o3.DRY_SCALE


# ------------------------------------------------------------------ settings

def setting_tag(setting):
    lr, gain = setting
    return f'lr{lr:g}_g{gain:g}'


def key_of(arm, setting):
    return f'{arm}_{setting_tag(setting)}'


def cells(settings=SETTINGS):
    """Every WAKE first (they are the long cells), then every SLEEP; a SLEEP is submitted when its WAKE exists."""
    return [(a, s) for a in ARMS for s in settings]


def work_of(root, setting):
    return Path(root) / 'work' / setting_tag(setting)


def apply_setting(setting):
    """G1T's name rebinding, then the two sized values. Module globals, so it holds in a spawned worker too."""
    lr, gain = setting
    g1t.use_form()
    tanh_form.INIT_GAIN = float(gain)

    def sized_stage_setup(world_seed, stage, model_seed=5000):
        cfg, world, updates, seeds = g1t.form_stage_setup(world_seed, stage, model_seed)
        return (replace(cfg, discrete_model=replace(cfg.discrete_model, global_learning_rate=float(lr))),
                world, updates, seeds)

    o2.stage_setup = sized_stage_setup


def restore():
    tanh_form.INIT_GAIN = 1.0
    g1t.restore()


# ------------------------------------------------------------------ cells

def depth_profile(model, stream_tasks, plan):
    per = score(model, SimpleNamespace(tasks=stream_tasks))['per_task']
    out = {}
    for d in (1, 2, 3):
        v = [per[t.task_id] for t in stream_tasks if plan[t.task_id] == d]
        out[str(d)] = {'median': float(np.median(v)), 'n': len(v)}
    return out


def d1_ari(model, stream_tasks, plan):
    routes = model.hard_routes()
    d1 = [t for t in stream_tasks if plan[t.task_id] == 1]
    return float(g1.ari_nmi([t.program.primitive_ids[0] for t in d1], [int(routes[t.task_id][0]) for t in d1])[0])


def run_cell(arm, setting, w, root, scale=1):
    torch.set_num_threads(1)
    apply_setting(setting)
    started = time.perf_counter()
    work = work_of(root, setting)
    terminal_path = work / 'lifetime' / 'model.pt'
    if arm == 'WAKE':
        rec = o2.run_single('SHUFFLED', w, STREAM, work, scale)
        rec['d1_ari'] = g1.formed_diagnostics_d1(terminal_path, w)
        resolved = yaml.safe_load((work / 'lifetime' / 'config.yaml').read_text(encoding='utf-8'))
        rec['lifetime_lr'] = resolved[f'{o2.KIND}_model']['global_learning_rate']
    elif arm == 'SLEEP':
        # o3.run_sleep's exact sequence (anchor A0 proves it on the baseline), keeping the slept model.
        cfg3, model, st, canonical = o3.load_shuffled_terminal(terminal_path, w, STREAM)
        before = o3.library_sha(model)
        pool = o2d.reservoir(st, w, STREAM, o3.MEMORY)
        o2c.consolidate(cfg3, model, pool, [t.task_id for t in st], [1941, w, STREAM, o3.MEMORY],
                        updates=o3.EXTRA_UPDATES)
        terminal = score(model, SimpleNamespace(tasks=canonical))
        _, _, _, plan, _ = o2.build_stream('SHUFFLED', w, STREAM)
        rec = {'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
               'scored_tasks': len(terminal['per_task']), 'pool_size': len(pool), 'extra_updates': o3.EXTRA_UPDATES,
               'library_sha256_before': before, 'library_sha256': o3.library_sha(model),
               'sleep_lr': ll._training_values(cfg3, o2.KIND)[0],
               'depth': depth_profile(model, st, plan), 'd1_ari': d1_ari(model, st, plan),
               'learner_class': type(model).__name__}
    else:
        raise ValueError(arm)
    rec.update({'arm': arm, 'lr': setting[0], 'init_gain': setting[1], 'world': w, 'stream': STREAM,
                'scale': scale, 'learner_implementation': TanhFormLearner.implementation,
                'seconds': time.perf_counter() - started})
    return rec


def validate(rec, scale=1):
    key = key_of(rec['arm'], (rec['lr'], rec['init_gain']))
    if rec['scored_tasks'] != 64 or len(rec['terminal_per_task']) != 64:
        raise ValueError(f'{key}: not scored on the canonical 64')
    if not all(math.isfinite(v) for v in rec['terminal_per_task'].values()):
        raise ValueError(f'{key}: non-finite terminal')
    if rec['arm'] == 'WAKE':
        o2.validate_cell({**rec, 'arm': 'SHUFFLED'}, scale)   # the construction G1T's validator knows
        if rec['lifetime_lr'] != rec['lr']:
            raise ValueError(f'{key}: lifetime ran at lr {rec["lifetime_lr"]}')
    else:
        if rec['pool_size'] != 188 * o3.MEMORY or rec['extra_updates'] != o3.EXTRA_UPDATES:
            raise ValueError(f'{key}: sleep construction')
        if rec['library_sha256'] == rec['library_sha256_before']:
            raise ValueError(f'{key}: sleep left the library unchanged')
        if rec['sleep_lr'] != rec['lr']:
            raise ValueError(f'{key}: sleep ran at lr {rec["sleep_lr"]}')
        if rec['learner_class'] != 'TanhFormLearner':
            raise ValueError(f'{key}: learner is {rec["learner_class"]}')
        if scale == 1 and [rec['depth'][d]['n'] for d in '123'] != [60, 64, 64]:
            raise ValueError(f'{key}: depth denominators')


# ------------------------------------------------------------------ reading (exploratory; chooses, never a verdict)

def g1t_anchor(world=WORLD):
    cells_ = json.loads(G1T_REPORT.read_text())['cells']
    return {'WAKE': cells_[f'SHUFFLED_w{world}']['terminal_per_task'],
            'SLEEP': cells_[f'SLEEP_w{world}']['terminal_per_task']}


def reading(records, anchor, settings=SETTINGS):
    """A0: the baseline reproduces G1T exactly. Then choose, among settings whose slept vocabulary keeps operation
    identity (`d1_ari >= 0.8`), the lowest slept terminal median; label by where it lands."""
    a0 = all(records[key_of(arm, BASELINE)]['terminal_per_task'] == anchor[arm] for arm in ARMS)
    sleep = {setting_tag(s): records[key_of('SLEEP', s)] for s in settings}
    eligible = {t: r for t, r in sleep.items() if r['d1_ari'] >= IDENTITY_ARI}
    chosen = min(eligible, key=lambda t: eligible[t]['terminal_median']) if eligible else None
    best = eligible[chosen]['terminal_median'] if chosen else None
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
    return {'label': label, 'anchor_A0': a0, 'chosen': chosen, 'chosen_terminal': best,
            'slept_terminal': {t: r['terminal_median'] for t, r in sleep.items()},
            'wake_terminal': {setting_tag(s): records[key_of('WAKE', s)]['terminal_median'] for s in settings},
            'slept_depth_median': {t: {d: r['depth'][d]['median'] for d in '123'} for t, r in sleep.items()},
            'slept_d1_ari': {t: r['d1_ari'] for t, r in sleep.items()},
            'wake_d1_ari': {setting_tag(s): records[key_of('WAKE', s)]['d1_ari'] for s in settings}}


# ------------------------------------------------------------------ gates and protocol

def gates(settings=SETTINGS):
    """S1: under every setting, a fresh learner's slots start at the set gain (every singular value of `A` equals it,
    the output scale is `tanh_gain_scale(gain)`), and every stage config carries the set learning rate; the baseline
    builds G1T's learner bitwise. S2: `restore` returns the defaults."""
    cfg = load_config('configs/v1.yaml')
    out = {'S1': {}}
    restore()
    g1t.use_form()
    reference = {k: v.clone() for k, v in o2.planned_model(cfg, {}).state_dict().items()}
    for s in settings:
        apply_setting(s)
        model = o2.planned_model(cfg, {})
        sv = torch.stack([torch.linalg.svdvals(op.A.detach().double()) for op in model.library])
        scale_ok = all(abs(float(op.alpha.detach()) - tanh_gain_scale(s[1])) < 1e-6 for op in model.library)
        lrs = [o2.stage_setup(WORLD, k, 5000)[0].discrete_model.global_learning_rate for k in (1, 2, 3)]
        ok = bool(torch.allclose(sv, torch.full_like(sv, s[1]), atol=1e-5)) and scale_ok and lrs == [s[0]] * 3
        if s == BASELINE:
            ok = ok and all(torch.equal(v, model.state_dict()[k]) for k, v in reference.items())
        out['S1'][setting_tag(s)] = ok
    restore()
    out['S2'] = tanh_form.INIT_GAIN == 1.0 and o2.stage_setup is g1t._ORIGINAL['stage_setup']
    out['all_pass'] = all(out['S1'].values()) and out['S2']
    return out


def protocol(root=ROOT, settings=SETTINGS, world=WORLD):
    modules = (g1t, g1, o2, o2c, o2d, o3, ll, tanh_form, sys.modules[score.__module__])
    return {'id': 'g1s-tanh-sizing-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'exploratory': True, 'world': world, 'stream': STREAM, 'settings': [list(s) for s in settings],
            'arms': list(ARMS), 'fits': FITS, 'improves': IMPROVES, 'identity_ari': IDENTITY_ARI, 'jobs': JOBS,
            'root': Path(root).as_posix(),
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), G1T_REPORT)},
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
    extra = f" d1_ari={rec['d1_ari']:.3f}"
    if rec['arm'] == 'SLEEP':
        extra += ' depth=' + '/'.join(f"{rec['depth'][d]['median']:.3g}" for d in '123')
    return f"{rec['seconds']:.1f}s terminal={rec['terminal_median']:.6g}{extra}"


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, world=WORLD,
        settings=SETTINGS):
    """The parent writes every log line (G0F learning)."""
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root, settings, world) | {'scale': scale, 'cells': [[a, list(s)] for a, s in todo_cells]}
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
                g = gates(settings) | {'protocol_sha256': sha}
                atomic_json(gpath, g)
            log_line(root / 'run.log', f"GATES S1={g['S1']} S2={g['S2']}")
            if not g['all_pass']:
                raise RuntimeError(f'gates failed: {g}')
            for arm, s in todo_cells:
                key = key_of(arm, s)
                path = root / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(root / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append((arm, s))
            total = len(todo_cells)
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} jobs={jobs} '
                                       f'scale={scale} free={free:.1f}GiB')
            ready = [c for c in todo if c[0] == 'WAKE' or key_of('WAKE', c[1]) in records]
            waiting = [c for c in todo if c not in ready]
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, s, world, str(root), scale): (a, s) for a, s in ready}
                for a, s in ready[:jobs]:
                    log_line(root / 'run.log', f'submitted {key_of(a, s)}')
                _status(root, 'running', len(records), total, [key_of(*c) for c in ready[:jobs]], started)
                while futures:
                    done_future = next(as_completed(futures))
                    a, s = futures.pop(done_future)
                    key = key_of(a, s)
                    record = done_future.result()
                    validate(record, scale)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    log_line(root / 'run.log', f'cell finished {key} {_headline(record)}')
                    for dep in [c for c in waiting if c[0] == 'SLEEP' and c[1] == s and a == 'WAKE']:
                        waiting.remove(dep)
                        futures[pool.submit(run_cell, dep[0], dep[1], world, str(root), scale)] = dep
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
                report['reading'] = reading(records, g1t_anchor(world), settings)
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
        run(cells(DRY_SETTINGS), root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=DRY_SCALE,
            stop_after=args.stop_after, world=DRY_WORLD, settings=DRY_SETTINGS)
        print(f'dry run complete: {DRY_ROOT}')
        return
    require_clean_code(OUTPUT)
    run(cells())
    print(f'G1S report: {OUTPUT}')


if __name__ == '__main__':
    main()
