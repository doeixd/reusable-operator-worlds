"""O2 Tier 2: is order-free ONLINE formation reliable across worlds and replay streams?

Plan: `O2_ONLINE_ANCHOR_RELIABILITY_PLAN.md` (FROZEN b3c1c85). Worlds 13-19 of
development band 2, model seed 5000, streams 0-2 (PLAIN: stream 0 only).

Arms, as constructions:
- SHUFFLED / MIXED_L1: O1's single-lifetime streams (canonical length-3 tasks,
  then length-1, then for SHUFFLED length-2, concatenated in that order and
  permuted by `SeedSequence([base, w])` at stream 0 and `[base, w, s]` after),
  one online lifetime through the unchanged `learned_lifetime.run`, learner
  `PlannedDepthRotatedLearner`.
- STAGED: the SO2 stage sequence (`stage_setup` / `so2.carry_library` /
  `learned_lifetime.run`), LIFETIMES ONLY - no export margin, no export
  diagnostic (gate E1 proves this is bitwise O1's construction).
- PLAIN: one canonical length-3 lifetime from a fresh model, stream 0.
Stream `s` sets the replay seed: None at s = 0 (the runner's canonical
`model seed + 1`), else `SeedSequence([7500, w, s]).generate_state(1)[0]`.

Every arm is scored as the TERMINAL model on the canonical 64 length-3 tasks.

Modes: default = the registered run (requires a passing gates record at this
exact protocol); `--gates` = E1-E4b; `--dry-run` = scale-16 restart test root.
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
from row.experiments.audit_rotated_g5r_interference import ANCHOR_TOLERANCE, score
from row.experiments.learned_lifetime import run as lifetime_run
from row.models.online_variable_depth import PlannedDepthRotatedLearner
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O2_ONLINE_ANCHOR_RELIABILITY_PLAN.md')
O1_CELLS = Path('reports/o1_online_anchor_20260924/cells')
ROOT = Path('artifacts/o2_online_reliability')
DRY_ROOT = Path('artifacts/o2_online_reliability_dry')
OUTPUT = Path('reports/o2_online_reliability.json')
KIND = 'rotated_discrete_fast'

WORLDS = (13, 14, 15, 16, 17, 18, 19)
STREAMS = (0, 1, 2)
ARMS = ('SHUFFLED', 'STAGED', 'MIXED_L1', 'PLAIN')   # registered order: decisive arm first
ARM_STREAMS = {'SHUFFLED': STREAMS, 'STAGED': STREAMS, 'MIXED_L1': STREAMS, 'PLAIN': (0,)}
ORDER_BASE = {'SHUFFLED': 1920, 'MIXED_L1': 1921}
STREAM_ROOT = 7500
MODEL_SEED = 5000
THRESHOLD = 0.05
JOBS = 3
MIN_FREE_GIB = 8.0
EXPECTED_STREAM = {'STAGED': 188, 'SHUFFLED': 188, 'MIXED_L1': 124, 'PLAIN': 64}
EXPECTED_DEPTHS = {'SHUFFLED': {'1': 60, '2': 64, '3': 64}, 'MIXED_L1': {'1': 60, '2': 0, '3': 64},
                   'STAGED': {'1': 60, '2': 64, '3': 64}, 'PLAIN': {'1': 0, '2': 0, '3': 64}}
GATE_WORLD = 12   # O1's world, reused only for the equivalence gates E1/E2
DRY_SCALE = 16
# Performance-pass switch (plan: "Performance-pass candidates"). LEAN turns off the
# deep-copied checkpoint probes and extended teacher diagnostics, which nothing reads.
# Admissible only because gates E1/E2 must still reproduce O1's cells BITWISE with it on.
LEAN = True


def cells():
    return [(a, w, s) for a in ARMS for w in WORLDS for s in ARM_STREAMS[a]]


def protocol(root=ROOT):
    return {'id': 'o2-online-reliability-v1', 'git_commit': git_commit(), 'plan': PLAN.as_posix(),
            'tier': 2, 'worlds': list(WORLDS), 'arms': list(ARMS),
            'arm_streams': {a: list(s) for a, s in ARM_STREAMS.items()},
            'order_seeds': 'SeedSequence([base, w]) at s=0, [base, w, s] after; base ' + json.dumps(ORDER_BASE),
            'replay_seeds': f'None at s=0 (model seed + 1); SeedSequence([{STREAM_ROOT}, w, s]).generate_state(1)[0]',
            'model_seed': MODEL_SEED, 'threshold': THRESHOLD, 'jobs': JOBS, 'kind': KIND, 'lean': LEAN,
            'root': Path(root).as_posix(),
            'scored_on': 'terminal model, canonical 64 length-3 tasks only',
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'))},
            'implementation_sha256': digest(Path(__file__)),
            'so2_implementation_sha256': digest(Path(so2.__file__)),
            'learner_sha256': digest(Path(sys.modules[PlannedDepthRotatedLearner.__module__].__file__)),
            'lifetime_sha256': digest(Path(sys.modules[lifetime_run.__module__].__file__))}


# ------------------------------------------------------------------ construction

def replay_seed_for(world: int, stream: int) -> int | None:
    if stream == 0:
        return None
    return int(np.random.SeedSequence([STREAM_ROOT, world, stream]).generate_state(1)[0])


def order_entropy(arm: str, world: int, stream: int) -> list[int]:
    return [ORDER_BASE[arm], world] if stream == 0 else [ORDER_BASE[arm], world, stream]


def run_cfg(cfg, scale: int = 1):
    """The config actually run: LEAN diagnostics, optional dry-run scaling."""
    if LEAN:
        cfg = dataclasses.replace(cfg, evaluation=dataclasses.replace(
            cfg.evaluation, lifetime_checkpoints=(), extended_diagnostics=False))
    if scale != 1:
        cfg = dataclasses.replace(cfg, world=dataclasses.replace(
            cfg.world, tasks=max(2, cfg.world.tasks // scale),
            examples_per_task=max(4, cfg.world.examples_per_task // scale),
            evaluation_examples=max(4, cfg.world.evaluation_examples // scale)))
    return cfg


def build_stream(arm: str, world_seed: int, stream: int):
    """O1's stream construction with the stream-indexed order seed (identical at s = 0)."""
    cfg3, world3, _, _ = stage_setup(world_seed, 3, MODEL_SEED)
    _, world1, _, _ = stage_setup(world_seed, 1, MODEL_SEED)
    tasks = list(world3.tasks) + list(world1.tasks)
    if arm == 'SHUFFLED':
        _, world2, _, _ = stage_setup(world_seed, 2, MODEL_SEED)
        tasks += list(world2.tasks)
    elif arm != 'MIXED_L1':
        raise ValueError(f'{arm} is not a single-lifetime arm')
    order = np.random.default_rng(np.random.SeedSequence(order_entropy(arm, world_seed, stream))).permutation(len(tasks))
    stream_tasks = [tasks[int(i)] for i in order]
    if len({t.task_id for t in stream_tasks}) != len(stream_tasks):
        raise ValueError('task ids collide across stages')
    plan = {t.task_id: len(t.program.primitive_ids) for t in stream_tasks}
    return cfg3, world3, stream_tasks, plan, list(world3.tasks)


def planned_model(cfg, plan):
    sel = cfg.discrete_model
    return PlannedDepthRotatedLearner(
        d=cfg.world.state_dim, operator_slots=sel.operator_slots, operator_rank=sel.operator_rank,
        task_steps=3, alpha=sel.operator_alpha_init, initial_temperature=sel.initial_temperature,
        final_temperature=sel.final_temperature, seed=sel.seed, learnable_alpha=sel.learnable_alpha,
        activation=sel.operator_activation, depth_plan=plan)


def task_summaries(output: Path) -> list[dict]:
    rows = []
    with (output / 'metrics.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('record_type') == 'task_summary':
                rows.append(row)
    if not rows:
        raise RuntimeError(f'{output} has no task_summary rows')
    return rows


def _median(values):
    return float(np.median(values)) if values else None


def lifetime(cfg, world, output: Path, model, replay_seed, scale: int):
    ran = run_cfg(cfg, scale)
    summary = lifetime_run(dataclasses.replace(ran, output_directory=output), KIND, world=world, model=model,
                           return_model=True, replay_seed=replay_seed)
    return summary, summary.pop('terminal_model')


_STREAM_SEED = object()


def run_single(arm, world_seed, stream, work: Path, scale=1, replay_seed=_STREAM_SEED):
    """`replay_seed` defaults to the stream's registered seed; gate E3 alone overrides it."""
    if replay_seed is _STREAM_SEED:
        replay_seed = replay_seed_for(world_seed, stream)
    cfg3, world3, stream_tasks, plan, canonical = build_stream(arm, world_seed, stream)
    mixed = dataclasses.replace(world3, tasks=tuple(stream_tasks))
    output = work / 'lifetime'
    summary, model = lifetime(cfg3, mixed, output, planned_model(cfg3, plan), replay_seed, scale)
    canon_ids = {t.task_id for t in canonical}
    scored = SimpleNamespace(tasks=[t for t in canonical if t.task_id in model.task_codes])
    terminal = score(model, scored)
    rows = task_summaries(output)
    final = {r['task_id']: float(r['final_nmse']) for r in rows}
    last = rows[-1]
    last_task = next(t for t in stream_tasks if t.task_id == last['task_id'])
    last_terminal = score(model, SimpleNamespace(tasks=[last_task]))['per_task'][last_task.task_id]
    routes = model.hard_routes()
    return {'terminal_median': terminal['median'], 'terminal_below': terminal['below_0.05'],
            'terminal_per_task': terminal['per_task'], 'scored_tasks': len(scored.tasks),
            'end_of_task_median': _median([final[t] for t in final if t in canon_ids]),
            'end_of_task_median_by_depth': {str(d): _median([v for t, v in final.items() if plan[t] == d])
                                            for d in (1, 2, 3)},
            'anchor_task_id': last['task_id'], 'anchor_depth': plan[last['task_id']],
            'anchor_abs_error': abs(last_terminal - float(last['final_nmse'])),
            'stream_tasks': len(stream_tasks), 'trained_tasks': len(rows),
            'depth_histogram': {str(d): sum(1 for v in plan.values() if v == d) for d in (1, 2, 3)},
            'routes_checked': sum(1 for t in plan if t in routes),
            'route_lengths_match_plan': all(t in routes and len(routes[t]) == plan[t] for t in plan),
            'first_20_stream_depths': [plan[t.task_id] for t in stream_tasks[:20]],
            'length1_positions': [i for i, t in enumerate(stream_tasks) if plan[t.task_id] == 1],
            'order_seed': order_entropy(arm, world_seed, stream),
            'prequential': summary.get('cumulative_prequential_gaussian_log_loss'),
            'library_sha256': library_sha(model)}


def run_staged(arm, world_seed, stream, work: Path, scale=1):
    """STAGED (stages 1-3, library carried) or PLAIN (stage 3 only), lifetimes only."""
    stages = (1, 2, 3) if arm == 'STAGED' else (3,)
    rs = replay_seed_for(world_seed, stream)
    model, by_stage, routes_ok, routes_checked, depth_hist = None, {}, True, 0, {'1': 0, '2': 0, '3': 0}
    prequential = 0.0
    for stage in stages:
        cfg, world, _, _ = stage_setup(world_seed, stage, MODEL_SEED)
        if model is not None:
            model = so2.carry_library(model, cfg)
        output = work / f'stage{stage}'
        summary, model = lifetime(cfg, world, output, model, rs, scale)
        rows = task_summaries(output)
        routes = model.hard_routes()
        ids = [t.task_id for t in world.tasks]
        routes_checked += sum(1 for t in ids if t in routes)
        routes_ok = routes_ok and all(t in routes and len(routes[t]) == stage for t in ids)
        depth_hist[str(stage)] = len(ids)
        by_stage[str(stage)] = _median([float(r['final_nmse']) for r in rows])
        prequential += summary.get('cumulative_prequential_gaussian_log_loss') or 0.0
        if stage == 3:
            canonical = list(world.tasks)
            scored = SimpleNamespace(tasks=[t for t in canonical if t.task_id in model.task_codes])
            terminal = score(model, scored)
            last = rows[-1]
            end3 = {r['task_id']: float(r['final_nmse']) for r in rows}
    return {'terminal_median': terminal['median'], 'terminal_below': terminal['below_0.05'],
            'terminal_per_task': terminal['per_task'], 'scored_tasks': len(scored.tasks),
            'end_of_task_median': _median([end3[t.task_id] for t in canonical]),
            'end_of_task_median_by_depth': {d: by_stage.get(d) for d in ('1', '2', '3')},
            'anchor_task_id': last['task_id'], 'anchor_depth': 3,
            'anchor_abs_error': abs(terminal['per_task'][last['task_id']] - float(last['final_nmse'])),
            'stream_tasks': sum(depth_hist.values()), 'trained_tasks': sum(depth_hist.values()),
            'depth_histogram': depth_hist, 'routes_checked': routes_checked,
            'route_lengths_match_plan': routes_ok, 'first_20_stream_depths': None,
            'length1_positions': None, 'order_seed': None,
            'prequential': prequential, 'library_sha256': library_sha(model)}


def run_cell(arm, world_seed, stream, root, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    work = Path(root) / 'work' / f'{arm}_w{world_seed}_s{stream}'
    if arm in ('SHUFFLED', 'MIXED_L1'):
        record = run_single(arm, world_seed, stream, work, scale)
    else:
        record = run_staged(arm, world_seed, stream, work, scale)
    record.update({'arm': arm, 'world': world_seed, 'stream': stream, 'scale': scale,
                   'replay_seed': replay_seed_for(world_seed, stream),
                   'seconds': time.perf_counter() - started})
    return record


def validate_cell(r, scale=1):
    key = f"{r['arm']}_w{r['world']}_s{r['stream']}"
    if r['arm'] not in ARMS or r['stream'] not in ARM_STREAMS[r['arm']]:
        raise ValueError(f'{key}: cell identity')
    if r['scored_tasks'] != 64 or len(r['terminal_per_task']) != 64:
        raise ValueError(f'{key}: scored {r["scored_tasks"]} tasks, not the canonical 64')
    if r['stream_tasks'] != EXPECTED_STREAM[r['arm']] or r['trained_tasks'] != EXPECTED_STREAM[r['arm']]:
        raise ValueError(f'{key}: stream {r["stream_tasks"]}, trained {r["trained_tasks"]}')
    if r['depth_histogram'] != EXPECTED_DEPTHS[r['arm']]:
        raise ValueError(f'{key}: depth histogram {r["depth_histogram"]}')
    if not r['route_lengths_match_plan'] or r['routes_checked'] != EXPECTED_STREAM[r['arm']]:
        raise ValueError(f'{key}: routes {r["routes_checked"]} checked, match {r["route_lengths_match_plan"]}')
    if r['anchor_abs_error'] > ANCHOR_TOLERANCE:
        raise ValueError(f'{key}: last-task anchor error {r["anchor_abs_error"]}')
    if r['first_20_stream_depths'] is not None and len(set(r['first_20_stream_depths'])) < 2:
        raise ValueError(f'{key}: first 20 stream tasks are one depth (E4b)')


def label(k: int) -> str:
    return 'RELIABLE' if k >= 19 else ('INTERMEDIATE' if k >= 17 else 'UNRELIABLE')


def summarize(records):
    passes, flagged = {}, []
    for a in ARMS:
        n = 0
        for w in WORLDS:
            for s in ARM_STREAMS[a]:
                m = records[f'{a}_w{w}_s{s}']['terminal_median']
                if not math.isfinite(m):
                    flagged.append(f'{a}_w{w}_s{s}')
                elif m < THRESHOLD:
                    n += 1
        passes[a] = n
    floor_failed = passes['PLAIN'] > 0
    labels = {a: label(passes[a]) for a in ('SHUFFLED', 'STAGED', 'MIXED_L1')}
    per_world = {str(w): {a: sum(records[f'{a}_w{w}_s{s}']['terminal_median'] < THRESHOLD for s in STREAMS)
                          for a in ('SHUFFLED', 'STAGED', 'MIXED_L1')} for w in WORLDS}
    d_w = {w: v['SHUFFLED'] - v['STAGED'] for w, v in per_world.items()}
    return {'tier': 2, 'passes': passes, 'denominator': 21,
            'primary': ('ORDER_FREE_' + labels['SHUFFLED']) if not floor_failed else 'FLOOR_FAILED',
            'labels': labels if not floor_failed else {a: 'FLOOR_FAILED' for a in labels},
            'floor_failed': floor_failed, 'non_finite_cells': flagged,
            'per_world_passing_streams': per_world, 'd_w_shuffled_minus_staged': d_w,
            'd_w_sum': sum(d_w.values())}


# ------------------------------------------------------------------ gates

def gates(root=ROOT):
    """E1/E2 bitwise against O1's committed world-12 cells; E3 seed neutrality; E4/E4b on scaled cells."""
    root = Path(root) / 'gates'
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, arm in (('E1', 'STAGED'), ('E2', 'SHUFFLED')):
        t0 = time.perf_counter()
        r = run_cell(arm, GATE_WORLD, 0, root / name)
        o1 = json.loads((O1_CELLS / f'{arm}_w{GATE_WORLD}.json').read_text())['record']
        out[name] = {'passes': r['library_sha256'] == o1['library_sha256']
                     and r['terminal_per_task'] == o1['terminal_per_task'],
                     'library_match': r['library_sha256'] == o1['library_sha256'],
                     'terminal_median': r['terminal_median'], 'o1_terminal_median': o1['terminal_median'],
                     'worst_abs_diff': max(abs(r['terminal_per_task'][k] - o1['terminal_per_task'][k])
                                           for k in o1['terminal_per_task']),
                     'seconds': time.perf_counter() - t0}
        log_line(root / 'gates.log', f"{name} {out[name]}")
    # E3: explicit canonical replay seed == None path (scaled SHUFFLED cell, world 13)
    cfg3 = stage_setup(13, 3, MODEL_SEED)[0]
    explicit = cfg3.discrete_model.seed + 1
    a = run_single('SHUFFLED', 13, 0, root / 'E3_none', DRY_SCALE)
    b = run_single('SHUFFLED', 13, 0, root / 'E3_explicit', DRY_SCALE, replay_seed=explicit)
    out['E3'] = {'passes': a['library_sha256'] == b['library_sha256'] and a['terminal_per_task'] == b['terminal_per_task'],
                 'explicit_seed': explicit}
    # E4: streams distinct (scaled, world 13, every three-stream arm); E4b: interleaving
    e4 = {}
    for arm in ('SHUFFLED', 'STAGED', 'MIXED_L1'):
        shas = [run_cell(arm, 13, s, root / 'E4', DRY_SCALE)['library_sha256'] for s in STREAMS]
        e4[arm] = len(set(shas)) == len(shas)
    out['E4'] = {'passes': all(e4.values()), 'by_arm': e4}
    e4b = {}
    for arm in ('SHUFFLED', 'MIXED_L1'):
        for w in WORLDS:
            for s in STREAMS:
                _, _, st, plan, _ = build_stream(arm, w, s)
                e4b[f'{arm}_w{w}_s{s}'] = len({plan[t.task_id] for t in st[:20]}) >= 2
    out['E4b'] = {'passes': all(e4b.values()), 'failing': [k for k, v in e4b.items() if not v]}
    out['all_pass'] = all(v['passes'] for k, v in out.items() if isinstance(v, dict) and 'passes' in v)
    out['protocol_sha256'] = fingerprint(protocol())
    atomic_json(Path(root) / 'gates.json', out)
    return out


# ------------------------------------------------------------------ pool

def host_ok():
    free = psutil.virtual_memory().available / 2 ** 30
    return free, free >= MIN_FREE_GIB


def _status(root, state, done, total, running, started):
    elapsed = time.time() - started
    eta = None if not done else elapsed / done * (total - done)
    atomic_json(Path(root) / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                             'running': running, 'pid': os.getpid(), 'updated_utc': now(),
                                             'eta_seconds': None if eta is None else round(eta)})


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, require_gates=True):
    root, output = Path(root), Path(output)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'cells').mkdir(exist_ok=True)
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
                raise RuntimeError('gates E1-E4b have not passed at this exact protocol')
        atomic_json(root / 'manifest.json', manifest)
        atomic_json(root / 'run.pid', {'pid': os.getpid(), 'started_utc': now()})
        free, ok = host_ok()
        atomic_json(root / 'precondition.json', {'free_gib': free, 'required_gib': MIN_FREE_GIB,
                                                 'passes': ok, 'jobs': jobs, 'checked_utc': now()})
        if not ok:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free, {MIN_FREE_GIB} required')
        records, todo = {}, []
        try:
            for arm, world, stream in todo_cells:
                key = f'{arm}_w{world}_s{stream}'
                path = root / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(root / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append((arm, world, stream))
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} '
                                       f'jobs={jobs} scale={scale} free={free:.1f}GiB')
            total = len(todo_cells)
            _status(root, 'running', len(records), total, [f'{a}_w{w}_s{s}' for a, w, s in todo[:jobs]], started)
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, w, s, str(root), scale): (a, w, s) for a, w, s in todo}
                for future in as_completed(futures):
                    a, w, s = futures[future]
                    key = f'{a}_w{w}_s{s}'
                    record = future.result()
                    validate_cell(record, scale)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"terminal={record['terminal_median']:.6g}")
                    pending = [f'{x}_w{y}_s{z}' for x, y, z in todo if f'{x}_w{y}_s{z}' not in records]
                    _status(root, 'running', len(records), total, pending[:jobs], started)
                    if stop_after is not None and finished >= stop_after:
                        for f in futures:
                            f.cancel()
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            report = {**manifest, 'complete': True, 'cells': records, 'finished_utc': now()}
            if scale == 1:
                report['summary'] = summarize(records)
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
    parser.add_argument('--dry-run', action='store_true', help='scale 16, world 13, all arms/streams, separate root')
    parser.add_argument('--stop-after', type=int, default=None, help='dry run only: exit after N cells')
    parser.add_argument('--jobs', type=int, default=JOBS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.gates:
        print(json.dumps(gates(), indent=1))
        return
    if args.dry_run:
        dry = [(a, 13, s) for a in ARMS for s in ARM_STREAMS[a]]
        run(dry, root=DRY_ROOT, output=DRY_ROOT / 'report.json', jobs=args.jobs, scale=DRY_SCALE,
            stop_after=args.stop_after, require_gates=False)
        print(f'O2 dry run complete: {DRY_ROOT}')
        return
    if args.stop_after is not None:
        raise SystemExit('--stop-after is for the dry run only')
    require_clean_code(OUTPUT)
    for check in ('tools/check_prereg.py', 'tools/check_invalid.py'):
        subprocess.run([sys.executable, check], check=True)
    run(cells(), jobs=args.jobs)
    print(f'O2 report: {OUTPUT}')


if __name__ == '__main__':
    main()
