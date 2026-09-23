"""O1 Tier 1: does an order-free anchor supply make ONLINE formation work?

Plan: `O1_ONLINE_ANCHOR_TIER1_PLAN.md`. EXPLORATORY, no verdict. Worlds 10-12 of
development band 2, model seed 5000, replay stream 0.

`STAGED` and `PLAIN` call SO2's `run_arm` verbatim, with an O1-owned artifact
path so SO2's own directory is never written. `SHUFFLED` and `MIXED_L1` are one
online lifetime each through the unchanged `learned_lifetime.run`, on the
canonical length-3 world with its task list replaced by a registered shuffle,
using `PlannedDepthRotatedLearner` (bitwise the committed learner at uniform
depth). Every arm is scored as the TERMINAL model on the canonical 64 length-3
tasks only.
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

from row.experiments import audit_so2_online_gate as so2
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import _last_task_end_of_task, score
from row.experiments.learned_lifetime import run as lifetime_run
from row.models.online_variable_depth import PlannedDepthRotatedLearner
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O1_ONLINE_ANCHOR_TIER1_PLAN.md')
ROOT = Path('artifacts/o1_online_anchor')
OUTPUT = Path('reports/o1_online_anchor.json')
KIND = 'rotated_discrete_fast'

WORLDS = (10, 11, 12)
ARMS = ('STAGED', 'SHUFFLED', 'MIXED_L1', 'PLAIN')   # longest first
SEEDS = {'SHUFFLED': 1920, 'MIXED_L1': 1921}
MODEL_SEED = 5000
THRESHOLD = 0.05
JOBS = 3
MIN_FREE_GIB = 8.0


def protocol():
    return {'id': 'o1-online-anchor-tier1-v1', 'git_commit': git_commit(), 'plan': PLAN.as_posix(),
            'tier': 1, 'exploratory': True, 'worlds': list(WORLDS), 'arms': list(ARMS),
            'model_seed': MODEL_SEED, 'replay_stream': 0, 'shuffle_seeds': SEEDS,
            'threshold': THRESHOLD, 'jobs': JOBS, 'kind': KIND,
            'scored_on': 'terminal model, canonical 64 length-3 tasks only',
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'))},
            'implementation_sha256': digest(Path(__file__)),
            'so2_implementation_sha256': digest(Path(so2.__file__)),
            'learner_sha256': digest(Path(sys.modules[PlannedDepthRotatedLearner.__module__].__file__))}


def planned_model(cfg, plan):
    sel = cfg.discrete_model
    return PlannedDepthRotatedLearner(
        d=cfg.world.state_dim, operator_slots=sel.operator_slots, operator_rank=sel.operator_rank,
        task_steps=3, alpha=sel.operator_alpha_init, initial_temperature=sel.initial_temperature,
        final_temperature=sel.final_temperature, seed=sel.seed, learnable_alpha=sel.learnable_alpha,
        activation=sel.operator_activation, depth_plan=plan)


def build_stream(arm, world_seed):
    """The registered single-lifetime stream: canonical length-3 plus anchors, shuffled."""
    cfg3, world3, _, _ = stage_setup(world_seed, 3, MODEL_SEED)
    _, world1, _, _ = stage_setup(world_seed, 1, MODEL_SEED)
    tasks = list(world3.tasks) + list(world1.tasks)
    if arm == 'SHUFFLED':
        _, world2, _, _ = stage_setup(world_seed, 2, MODEL_SEED)
        tasks += list(world2.tasks)
    elif arm != 'MIXED_L1':
        raise ValueError(f'{arm} is not a single-lifetime arm')
    order = np.random.default_rng(np.random.SeedSequence([SEEDS[arm], world_seed])).permutation(len(tasks))
    stream = [tasks[int(i)] for i in order]
    if len({t.task_id for t in stream}) != len(stream):
        raise ValueError('task ids collide across stages')
    plan = {t.task_id: len(t.program.primitive_ids) for t in stream}
    canonical = [t for t in world3.tasks]
    return cfg3, world3, stream, plan, canonical


def run_single(arm, world_seed, work):
    cfg3, world3, stream, plan, canonical = build_stream(arm, world_seed)
    mixed = dataclasses.replace(world3, tasks=tuple(stream))
    model = planned_model(cfg3, plan)
    output = work / 'lifetime'
    summary = lifetime_run(dataclasses.replace(cfg3, output_directory=output), KIND, world=mixed,
                           model=model, return_model=True)
    model = summary.pop('terminal_model')
    scored = SimpleNamespace(tasks=[t for t in canonical if t.task_id in model.task_codes])
    terminal = score(model, scored)
    last_id, last_end, end_median = _last_task_end_of_task(output)
    routes = model.hard_routes()
    return {'terminal_median': terminal['median'], 'terminal_below': terminal['below_0.05'],
            'terminal_per_task': terminal['per_task'], 'scored_tasks': len(scored.tasks),
            'stream_tasks': len(stream),
            'depth_histogram': {str(d): sum(1 for v in plan.values() if v == d) for d in (1, 2, 3)},
            'route_lengths_match_plan': all(len(routes[t]) == plan[t] for t in plan if t in routes),
            'first_20_stream_depths': [plan[t.task_id] for t in stream[:20]],
            'end_of_task_median': end_median, 'last_task_id': last_id,
            'prequential': summary.get('cumulative_prequential_gaussian_log_loss'),
            'library_sha256': library_sha(model)}


def run_cell(arm, world_seed, root):
    torch.set_num_threads(1)
    started = time.perf_counter()
    work = Path(root) / 'work' / f'{arm}_w{world_seed}'
    if arm in ('STAGED', 'PLAIN'):
        r = so2.run_arm(arm, world_seed, artifact=work)
        stage3 = r['stages']['3']
        record = {'terminal_median': r['terminal_median'], 'terminal_below': stage3['terminal_below_threshold'],
                  'terminal_per_task': stage3['terminal_per_task'], 'scored_tasks': len(stage3['terminal_per_task']),
                  'stream_tasks': sum(v['tasks'] for v in r['stages'].values()),
                  'depth_histogram': {str(v['length']): v['tasks'] for v in r['stages'].values()},
                  'route_lengths_match_plan': True, 'first_20_stream_depths': None,
                  'end_of_task_median': r['end_of_task_median'], 'last_task_id': stage3['anchor_task_id'],
                  'prequential': r['prequential_total'], 'library_sha256': stage3['library_sha256']}
    else:
        record = run_single(arm, world_seed, work)
    record.update({'arm': arm, 'world': world_seed, 'seconds': time.perf_counter() - started})
    return record


EXPECTED_STREAM = {'STAGED': 188, 'SHUFFLED': 188, 'MIXED_L1': 124, 'PLAIN': 64}


def validate_cell(r):
    if r['arm'] not in ARMS or r['world'] not in WORLDS:
        raise ValueError('cell identity')
    if not math.isfinite(r['terminal_median']) or r['terminal_median'] < 0:
        raise ValueError('bad terminal median')
    if r['scored_tasks'] != 64:
        raise ValueError(f"{r['arm']}_w{r['world']} scored {r['scored_tasks']} tasks, not the canonical 64")
    if r['stream_tasks'] != EXPECTED_STREAM[r['arm']]:
        raise ValueError(f"{r['arm']} stream has {r['stream_tasks']} tasks")
    if not r['route_lengths_match_plan']:
        raise ValueError('a route length disagrees with its planned depth')


def summarize(records):
    by = {}
    for r in records.values():
        by.setdefault(r['arm'], {})[r['world']] = r['terminal_median']
    passes = {a: sum(1 for v in by.get(a, {}).values() if v < THRESHOLD) for a in ARMS}
    live = passes['SHUFFLED'] >= 2 or passes['MIXED_L1'] >= 2
    return {'tier': 1, 'exploratory': True,
            'decision': 'LIVE' if live else 'NOT_LIVE',
            'worlds_passing': passes,
            'by_arm': {a: {str(w): v for w, v in by.get(a, {}).items()} for a in ARMS},
            'note': 'Tier 1: an indication that sizes a Tier 2, never a verdict'}


def host_ok():
    free = psutil.virtual_memory().available / 2 ** 30
    return free, free >= MIN_FREE_GIB


def _status(state, done, total, running):
    atomic_json(ROOT / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                       'running': running, 'pid': os.getpid(), 'updated_utc': now()})


def run(cells, jobs=JOBS):
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'cells').mkdir(exist_ok=True)
    expected = protocol()
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    with writer_lock(ROOT / 'launcher.lock'):
        if (ROOT / 'manifest.json').exists() and json.loads((ROOT / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; retire the path rather than overwrite it')
        atomic_json(ROOT / 'manifest.json', manifest)
        atomic_json(ROOT / 'run.pid', {'pid': os.getpid(), 'started_utc': now()})
        free, ok = host_ok()
        atomic_json(ROOT / 'precondition.json', {'free_gib': free, 'required_gib': MIN_FREE_GIB,
                                                 'passes': ok, 'jobs': jobs, 'checked_utc': now()})
        if not ok:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free, {MIN_FREE_GIB} required')
        records, todo = {}, []
        try:
            for arm, world in cells:
                key = f'{arm}_w{world}'
                path = ROOT / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(ROOT / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append((arm, world))
            log_line(ROOT / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} '
                                       f'jobs={jobs} free={free:.1f}GiB')
            _status('running', len(records), len(cells), [f'{a}_w{w}' for a, w in todo[:jobs]])
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, arm, world, str(ROOT)): (arm, world) for arm, world in todo}
                for future in as_completed(futures):
                    arm, world = futures[future]
                    key = f'{arm}_w{world}'
                    record = future.result()
                    validate_cell(record)
                    atomic_json(ROOT / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    log_line(ROOT / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"terminal={record['terminal_median']:.6g}")
                    pending = [f'{a}_w{w}' for a, w in todo if f'{a}_w{w}' not in records]
                    _status('running', len(records), len(cells), pending[:jobs])
            atomic_json(OUTPUT, {**manifest, 'complete': True, 'cells': records,
                                 'summary': summarize(records), 'finished_utc': now()})
            _status('complete', len(records), len(cells), [])
            atomic_json(ROOT / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            _status('failed', len(records), len(cells), [])
            atomic_json(ROOT / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(ROOT / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=JOBS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    require_clean_code(OUTPUT)
    for check in ('tools/check_prereg.py', 'tools/check_invalid.py'):
        subprocess.run([sys.executable, check], check=True)
    run([(a, w) for a in ARMS for w in WORLDS], jobs=args.jobs)
    print(f'O1 Tier 1 report: {OUTPUT}')


if __name__ == '__main__':
    main()
