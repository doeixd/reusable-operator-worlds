"""G1 formation probe (Tier 1, EXPLORATORY): the confirmed fully online pipeline on the g1 generator at shared rotation m = 4.

Plan: `G1_FORMATION_PROBE_PLAN.md`. Development band 6, worlds 70-72 (G0's worlds), stream 0, m in {0, 4}.
- REROUTE_WAKE: `o8.run_arm` verbatim (wake + in-stream re-routing of earlier tasks).
- RW_SLEEP: `o3.run_sleep` verbatim on this run's REROUTE_WAKE terminal (O11's confirmed construction), then the
  slept model is rebuilt with the same calls (required to reproduce `run_sleep`'s per-task terminal) and measured.
- PLAIN: `o2.run_staged('PLAIN')` verbatim (canonical depth-3 tasks only, no anchors): the refusing arm.

Every stream in these modules is built by `o2.build_stream` / `o2.run_staged`, which call the name `stage_setup` bound
in `o2`'s namespace. `use_generator(m)` rebinds that one name to `general_stage_setup`, which is J1c's `stage_setup`
with `row.general_world`'s generators; at m = 0 it reproduces the committed streams bitwise (gate E1, and
`tests/test_g1_formation_probe.py`). No committed module is edited.
"""
from __future__ import annotations

import argparse
import functools
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

from row import general_world
from row.config import load_config
from row.curriculum_world import CurriculumWorldConfig
from row.experiments import audit_j1c_curriculum as j1c
from row.experiments import audit_j2a_staged_library as j2a
from row.experiments import census_o8_staleness_position as census
from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o8_instream_reroute as o8
from row.experiments.audit_h47_baselines import ari_nmi
from row.experiments.audit_so1r_route_only import unflatten
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score, world_config
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('G1_FORMATION_PROBE_PLAN.md')
G0_REPORT = Path('reports/g0_generator_census.json')
ROOT = Path('artifacts/g1_formation_probe')
DRY_ROOT = Path('artifacts/g1_formation_probe_dry')
OUTPUT = Path('reports/g1_formation_probe.json')
WORLDS = (70, 71, 72)
SHARED = (0.0, 4.0)
GAIN = 1.0
STREAM = 0
ARMS = ('REROUTE_WAKE', 'PLAIN', 'RW_SLEEP')   # RW_SLEEP is submitted when its REROUTE_WAKE terminal exists
THRESHOLD = 0.05
NEAR_TIE = 0.01   # SG0: a route within 1% of the winner
JOBS = 3
MIN_FREE_GIB = 2.0   # PI 2026-10-03: low memory is acceptable
DRY_WORLD = 99       # development band 6, never a probe world
DRY_SCALE = o3.DRY_SCALE


# ------------------------------------------------------------------ generator plumbing

def general_stage_setup(world_seed: int, stage: int, model_seed: int = 5000, shared: float = 0.0):
    """`audit_j1c_curriculum.stage_setup` with the g1 generators (gain 1, shared rotation `shared`)."""
    base = load_config("configs/v1.yaml")
    length, tasks, updates, root, tail = j1c.STAGES[stage - 1]
    if stage == 3:
        cfg = world_config(base, world_seed)
        world = general_world.generate_general_world(cfg.world, GAIN, shared)
    else:
        cfg = replace(base, world=CurriculumWorldConfig.from_world(
            replace(base.world, seed=world_seed), program_length=length, tasks=tasks, stage=f"length-{length}"))
        world = general_world.generate_general_curriculum_world(cfg.world, GAIN, shared)
    cfg = replace(cfg, discrete_model=replace(cfg.discrete_model, task_steps=length, seed=model_seed))
    return cfg, world, updates, np.random.SeedSequence([root, world_seed, tail])


def use_generator(shared: float) -> None:
    o2.stage_setup = functools.partial(general_stage_setup, shared=float(shared))


def restore_generator() -> None:
    o2.stage_setup = j1c.stage_setup


def key_of(arm, m, w):
    return f'{arm}_m{m:g}_w{w}'


def cells(worlds=WORLDS):
    return [(a, m, w) for a in ARMS for m in SHARED for w in worlds]


def work_of(root, arm, m, w):
    return Path(root) / 'work' / key_of(arm, m, w)


# ------------------------------------------------------------------ diagnostics on a formed library

def formed_diagnostics(model, stream_tasks, plan, held_out):
    """Depth-1 identity ARI of the formed routes, and SG0's identifiability on 64 UNSEEN length-3 programs.

    `held_out` is `j2a.held_out_tasks` on this cell's world (the world's own teacher library), exactly SG0's sample;
    identifiability is SG0's `(second - best) / best` over all `slots^3` routes on the 128 support examples, with the
    library frozen, and a near tie is identifiability < 0.01."""
    routes = model.hard_routes()
    d1 = [t for t in stream_tasks if plan[t.task_id] == 1]
    ari1, _ = ari_nmi([t.program.primitive_ids[0] for t in d1], [int(routes[t.task_id][0]) for t in d1])
    library = FrozenLibrary(model)
    library.steps = 3
    ident, runner_nmse, winner_query = [], [], []
    for t in held_out:
        y = torch.tensor(t['train_y'], dtype=torch.float32)
        mse = library.all_route_support_mse(torch.tensor(t['train_x'], dtype=torch.float32), y)
        order = torch.argsort(mse)
        best, second = float(mse[order[0]]), float(mse[order[1]])
        ident.append((second - best) / max(best, 1e-12))
        runner_nmse.append(second / float(torch.var(y, unbiased=False)))
        win = [int(r) for r in unflatten(int(order[0]), library.slots, 3)]
        with torch.no_grad():
            pred = library.hard(torch.tensor(t['eval_x'], dtype=torch.float32), win).numpy()
        winner_query.append(float(np.mean((pred - t['eval_y']) ** 2) / np.var(t['eval_y'])))
    return {'d1_ari': float(ari1), 'd1_tasks': len(d1),
            'identifiability_median': float(np.median(ident)),
            'near_tie_tasks': int(sum(i < NEAR_TIE for i in ident)),
            'formed_runner_up_nmse_median': float(np.median(runner_nmse)),
            'winner_query_nmse_median': float(np.median(winner_query)),
            'winner_query_below_0.05': int(sum(q < THRESHOLD for q in winner_query)),
            'identifiability_tasks': len(held_out)}


# ------------------------------------------------------------------ cells

def run_cell(arm, m, w, root, scale=1):
    torch.set_num_threads(1)
    use_generator(m)
    started = time.perf_counter()
    work = work_of(root, arm, m, w)
    if arm == 'REROUTE_WAKE':
        rec = o8.run_arm(arm, w, STREAM, work, scale)
        rec['d1_ari_wake'] = formed_diagnostics_d1(work / 'lifetime' / 'model.pt', w)
    elif arm == 'PLAIN':
        rec = o2.run_staged('PLAIN', w, STREAM, work, scale)
    elif arm == 'RW_SLEEP':
        terminal = work_of(root, 'REROUTE_WAKE', m, w) / 'lifetime' / 'model.pt'
        rec = o3.run_sleep(w, STREAM, terminal)   # the registered construction, verbatim
        cfg3, slept, st, canonical = o3.load_shuffled_terminal(terminal, w, STREAM)
        pool = o2d.reservoir(st, w, STREAM, o3.MEMORY)
        o2c.consolidate(cfg3, slept, pool, [t.task_id for t in st], [1941, w, STREAM, o3.MEMORY],
                        updates=o3.EXTRA_UPDATES)
        rec['rebuild_matches'] = score(slept, SimpleNamespace(tasks=canonical))['per_task'] == rec['terminal_per_task']
        cfg3b, world3, _, plan, _ = o2.build_stream('SHUFFLED', w, STREAM)
        rec['formed'] = formed_diagnostics(slept, st, plan, j2a.held_out_tasks(cfg3b, world3))
    else:
        raise ValueError(arm)
    rec.update({'arm': arm, 'shared': m, 'gain': GAIN, 'world': w, 'stream': STREAM, 'scale': scale,
                'generator': general_world.VERSION, 'seconds': time.perf_counter() - started})
    return rec


def formed_diagnostics_d1(terminal, w):
    """Depth-1 identity ARI of the wake terminal's routes (before sleep)."""
    _, model, st, _ = o3.load_shuffled_terminal(terminal, w, STREAM)
    _, _, _, plan, _ = o2.build_stream('SHUFFLED', w, STREAM)
    routes = model.hard_routes()
    d1 = [t for t in st if plan[t.task_id] == 1]
    return float(ari_nmi([t.program.primitive_ids[0] for t in d1], [int(routes[t.task_id][0]) for t in d1])[0])


def validate(rec):
    key = key_of(rec['arm'], rec['shared'], rec['world'])
    if rec['scored_tasks'] != 64 or len(rec['terminal_per_task']) != 64:
        raise ValueError(f'{key}: not scored on the canonical 64')
    if rec['arm'] == 'REROUTE_WAKE':
        o8.validate(rec)
    elif rec['arm'] == 'PLAIN':
        if rec['depth_histogram'] != {'1': 0, '2': 0, '3': 64} or not rec['route_lengths_match_plan']:
            raise ValueError(f'{key}: PLAIN construction')
        if rec['anchor_abs_error'] > o2.ANCHOR_TOLERANCE:
            raise ValueError(f'{key}: last-task anchor {rec["anchor_abs_error"]}')
    else:
        if rec['pool_size'] != 188 * o3.MEMORY or rec['extra_updates'] != o3.EXTRA_UPDATES:
            raise ValueError(f'{key}: sleep construction')
        if rec['library_sha256'] == rec['library_sha256_before']:
            raise ValueError(f'{key}: sleep left the library unchanged')
        if not rec['rebuild_matches']:
            raise ValueError(f'{key}: diagnostic rebuild is not the slept model')
        if rec['formed']['d1_tasks'] != 60 or rec['formed']['identifiability_tasks'] != 64:
            raise ValueError(f'{key}: diagnostic denominators')


# ------------------------------------------------------------------ reading (exploratory; no verdict)

def passing(v):
    return v is not None and math.isfinite(v) and v < THRESHOLD


def reading(records, worlds=WORLDS):
    M = {k: v['terminal_median'] for k, v in records.items()}
    anchored = [w for w in worlds if passing(M[key_of('RW_SLEEP', 0.0, w)])]
    k4 = sum(passing(M[key_of('RW_SLEEP', 4.0, w)]) for w in anchored)
    if not anchored:
        label = 'ANCHOR_MISS'
    elif k4 == len(anchored):
        label = 'FORMS'
    elif k4 == 0:
        label = 'FAILS'
    else:
        label = 'MIXED'
    plain4 = sum(passing(M[key_of('PLAIN', 4.0, w)]) for w in worlds)
    return {'label': label, 'anchored_worlds': anchored, 'm4_passes': k4, 'denominator': len(anchored),
            'plain_m4_passes': plain4, 'plain_m0_passes': sum(passing(M[key_of('PLAIN', 0.0, w)]) for w in worlds),
            'anchors_needed_at_m4': plain4 == 0,
            'terminal': {k: M[k] for k in sorted(M)},
            'formed': {key_of('RW_SLEEP', m, w): records[key_of('RW_SLEEP', m, w)]['formed']
                       for m in SHARED for w in worlds}}


# ------------------------------------------------------------------ gates and protocol

def gates():
    """E1: at m = 0 the rebound stage_setup reproduces the committed streams bitwise (worlds 70 and 20, stream 0).
    E2: at m = 4 the canonical world's library differs from m = 0 (the knob reaches the pipeline)."""
    def snapshot(w):
        cfg3, world3, st, plan, canonical = o2.build_stream('SHUFFLED', w, STREAM)
        cfgp = o2.stage_setup(w, 3, o2.MODEL_SEED)[0]
        return (cfg3, cfgp, [(t.task_id, t.program.primitive_ids, t.train_x.tobytes(), t.train_y.tobytes(),
                              t.eval_x.tobytes(), t.eval_y.tobytes()) for t in st], plan,
                [t.task_id for t in canonical])
    e1 = {}
    for w in (70, 20):
        restore_generator()
        ref = snapshot(w)
        use_generator(0.0)
        e1[str(w)] = snapshot(w) == ref
    use_generator(4.0)
    y4 = o2.build_stream('SHUFFLED', 70, STREAM)[4][0].train_y
    use_generator(0.0)
    y0 = o2.build_stream('SHUFFLED', 70, STREAM)[4][0].train_y
    restore_generator()
    out = {'E1': {'passes': all(e1.values()), 'worlds': e1}, 'E2': {'passes': not np.array_equal(y4, y0)}}
    out['all_pass'] = out['E1']['passes'] and out['E2']['passes']
    return out


def protocol(root=ROOT):
    return {'id': 'g1-formation-probe-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'exploratory': True, 'worlds': list(WORLDS), 'shared': list(SHARED), 'gain': GAIN, 'stream': STREAM,
            'arms': list(ARMS), 'threshold': THRESHOLD, 'near_tie': NEAR_TIE, 'jobs': JOBS,
            'root': Path(root).as_posix(), 'generator': general_world.VERSION,
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), G0_REPORT)},
            'implementation_sha256': digest(Path(__file__)), 'general_world_sha256': digest(Path(general_world.__file__)),
            'o8_sha256': digest(Path(o8.__file__)), 'o3_sha256': digest(Path(o3.__file__)),
            'o2_sha256': digest(Path(o2.__file__)), 'o2c_sha256': digest(Path(o2c.__file__)),
            'o2d_sha256': digest(Path(o2d.__file__)), 'j1c_sha256': digest(Path(j1c.__file__)),
            'census_sha256': digest(Path(census.__file__)), 'j2a_sha256': digest(Path(j2a.__file__)), 'lifetime_sha256': digest(Path(ll.__file__)),
            'learner_sha256': digest(Path(sys.modules[o2.PlannedDepthRotatedLearner.__module__].__file__))}


def _status(root, state, done, total, running, started):
    elapsed = time.time() - started
    eta = elapsed / done * (total - done) if done else None
    atomic_json(Path(root) / 'status.json', {
        'state': state, 'cells_done': done, 'cells_total': total, 'running': running, 'pid': os.getpid(),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started)), 'updated_utc': now(),
        'eta_seconds': eta})


def _headline(rec):
    extra = ''
    if rec['arm'] == 'RW_SLEEP':
        f = rec['formed']
        extra = (f" d1_ari={f['d1_ari']:.3f} ident={f['identifiability_median']:.3g} "
                 f"ties={f['near_tie_tasks']} runner={f['formed_runner_up_nmse_median']:.3g}")
    elif rec['arm'] == 'REROUTE_WAKE':
        extra = f" d1_ari_wake={rec['d1_ari_wake']:.3f} changed={rec['routes_changed_total']}"
    return f"{rec['seconds']:.1f}s terminal={rec['terminal_median']:.6g}{extra}"


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, require_gates=True,
        worlds=WORLDS):
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root) | {'scale': scale, 'cells': [list(c) for c in todo_cells]}
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
            log_line(root / 'run.log', f"GATES E1={g['E1']['passes']} E2={g['E2']['passes']}")
            if require_gates and not g['all_pass']:
                raise RuntimeError(f'gates failed: {g}')
            for arm, m, w in todo_cells:
                key = key_of(arm, m, w)
                path = root / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(root / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append((arm, m, w))
            total = len(todo_cells)
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} jobs={jobs} '
                                       f'scale={scale} free={free:.1f}GiB')
            ready = [c for c in todo if c[0] != 'RW_SLEEP'
                     or key_of('REROUTE_WAKE', c[1], c[2]) in records]
            waiting = [c for c in todo if c not in ready]
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, m, w, str(root), scale): (a, m, w) for a, m, w in ready}
                _status(root, 'running', len(records), total, [key_of(*c) for c in ready[:jobs]], started)
                while futures:
                    done_future = next(as_completed(futures))
                    a, m, w = futures.pop(done_future)
                    key = key_of(a, m, w)
                    record = done_future.result()
                    validate(record)   # at every scale, so the dry run exercises it
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    log_line(root / 'run.log', f'cell finished {key} {_headline(record)}')
                    if a == 'REROUTE_WAKE' and ('RW_SLEEP', m, w) in waiting:
                        waiting.remove(('RW_SLEEP', m, w))
                        futures[pool.submit(run_cell, 'RW_SLEEP', m, w, str(root), scale)] = ('RW_SLEEP', m, w)
                    _status(root, 'running', len(records), total, [key_of(*c) for c in futures.values()], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            if waiting:
                raise RuntimeError(f'dependent cells never submitted: {waiting}')
            report = {**manifest, 'complete': True, 'gates': g, 'cells': records, 'finished_utc': now()}
            if scale == 1:
                report['reading'] = reading(records, worlds)
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
        run(cells((DRY_WORLD,)), root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=DRY_SCALE,
            stop_after=args.stop_after, worlds=(DRY_WORLD,))
        print(f'dry run complete: {DRY_ROOT}')
        return
    require_clean_code(OUTPUT)
    run(cells())
    print(f'G1 probe report: {OUTPUT}')


if __name__ == '__main__':
    main()
