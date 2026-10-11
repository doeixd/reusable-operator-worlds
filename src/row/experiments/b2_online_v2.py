"""B2-online v2 Tier 1 (exploratory): with the halting predicate reading the DETACHED state, does a learner form the
vocabulary AND learn data-dependent iteration online?

Plan: `B2_ONLINE_V2_PLAN.md`. Development band 5 (fresh): worlds 50, 51, 52, streams 0-2; dry-run world 69. Stream:
`loop_stream` unchanged (O2's SHUFFLED depth-3 stream plus 48 loop tasks, 236 tasks). End of stream: B2-online's
construction unchanged (re-route of the straight-line tasks, support-only loop re-fit, O3 sleep).

Arms (constructions):
- LOOP: `LoopLearner(state_halt=True, detach_halt_input=True)`: the predicate reads the detached state in wake and
  sleep, so its gradient trains the halting parameters but never the library through the states
  (`reports/b2o_localize_census.json`).
- LOOP_ATTACHED: B2-online v1's LOOP exactly (`detach_halt_input=False`): the mechanism control.
- CONSTCOUNT: as in v1 (`state_halt=False`; halting cannot read the state, so detaching is moot).
REFUSAL is not re-run: its refusal cost is measured (census 1.02; v1 0.90) and CONSTCOUNT carries the formation
control.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
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

from row.experiments import b1_online as b1o
from row.experiments import b2_online as b2
from row.experiments import census_b2_iteration_necessity as c2
from row.experiments import deep_reroute as dr
from row.experiments import learned_lifetime as ll
from row.experiments import loop_stream as ls
from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock
from row.models.loop_gated import LoopLearner

PLAN = Path('B2_ONLINE_V2_PLAN.md')
ROOT = Path('artifacts/b2_online_v2')
DRY_ROOT = Path('artifacts/b2_online_v2_dry')
OUTPUT = Path('reports/b2_online_v2.json')
WORLDS = (50, 51, 52)
STREAMS = (0, 1, 2)
ARMS = ('LOOP', 'LOOP_ATTACHED', 'CONSTCOUNT')
N_LOOP = 48
MEMORY = b1o.MEMORY
SLEEP_UPDATES = b1o.SLEEP_UPDATES
K = ls.K
THRESHOLD = 0.05
LOOP_BAR = 0.38   # sqrt(0.214 x 0.675), reports/b2_support64_census.json
WAKE_RATIO = 0.5
JOBS = 3
MIN_FREE_GIB = 2.0   # PI 2026-10-03 instruction
DRY_WORLD = 69
DRY_SCALE = 16


def cells(worlds=WORLDS):
    return [(a, w, s) for a in ARMS for w in worlds for s in STREAMS]


count_accuracy = b2.count_accuracy


def run_arm(arm, w, s, work: Path, scale=1):
    cfg, mixed, st, plan, canonical, ltasks, meta = ls.stream(w, s, N_LOOP)
    loop_ids = {t.task_id for t in ltasks}
    model = b1o.build(LoopLearner, cfg, plan, loop_plan={t: 'state' for t in loop_ids}, max_iterations=K,
                      state_halt=(arm != 'CONSTCOUNT'), detach_halt_input=(arm == 'LOOP'))
    summary = ll.run(dataclasses.replace(o2.run_cfg(cfg, scale), output_directory=work / 'lifetime'), o2.KIND,
                     world=mixed, model=model, return_model=True, replay_seed=o2.replay_seed_for(w, s))
    model = summary.pop('terminal_model')
    rec = {'arm': arm, 'world': w, 'stream': s, 'scale': scale, 'stream_tasks': len(st),
           'wake_canonical_median': score(model, SimpleNamespace(tasks=canonical))['median'],
           'wake_loop_median': score(model, SimpleNamespace(tasks=ltasks))['median']}
    if arm != 'CONSTCOUNT':
        rec['wake_count_accuracy_median'] = float(np.median(count_accuracy(model, ltasks, meta)))
    # non-vacuity: wake must move the halting parameters off their zero init
    rec['wake_halt_norm_median'] = float(np.median([
        float(torch.cat([p.detach().flatten() for p in model.loop_parameters([t.task_id])]).norm()) for t in ltasks]))
    pool = o2d.reservoir(st, w, s, MEMORY)
    straight = [t for t in st if t.task_id not in loop_ids]
    rec['routes_changed'] = dr.reroute(model, straight, plan, pool)
    rec['loop_refits_installed'] = b2.refit_loops(model, ltasks, pool, w, 'CONSTCOUNT' if arm == 'CONSTCOUNT' else 'LOOP')
    extra = model.loop_parameters(loop_ids)
    before = library_sha(model)
    b1o.sleep(cfg, model, pool, [t.task_id for t in st], [1941, w, s, MEMORY], extra)
    rec['library_sha256_before_sleep'], rec['library_sha256'] = before, library_sha(model)
    canon = score(model, SimpleNamespace(tasks=canonical))
    lp = score(model, SimpleNamespace(tasks=ltasks))
    rec.update({'canonical_median': canon['median'], 'canonical_per_task': canon['per_task'],
                'scored_canonical': len(canon['per_task']), 'loop_median': lp['median'], 'loop_per_task': lp['per_task'],
                'loop_below_threshold': sum(v < THRESHOLD for v in lp['per_task'].values())})
    if arm != 'CONSTCOUNT':
        acc = count_accuracy(model, ltasks, meta)
        rec['count_accuracy_median'] = float(np.median(acc))
        rec['count_accuracy_per_task'] = dict(zip([t.task_id for t in ltasks], acc))
    return rec


def run_cell(arm, w, s, root, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    rec = run_arm(arm, w, s, Path(root) / 'work' / f'{arm}_w{w}_s{s}', scale)
    rec['seconds'] = time.perf_counter() - started
    return rec


def validate(rec):
    key = f"{rec['arm']}_w{rec['world']}_s{rec['stream']}"
    if rec['scored_canonical'] != 64 or rec['stream_tasks'] != 188 + N_LOOP or len(rec['loop_per_task']) != N_LOOP:
        raise ValueError(f'{key}: construction')
    if rec['library_sha256'] == rec['library_sha256_before_sleep']:
        raise ValueError(f'{key}: sleep left the library unchanged')
    if not rec['wake_halt_norm_median'] > 0:
        raise ValueError(f'{key}: wake never moved the halting parameters')


def passing(m, bar=THRESHOLD):
    return m is not None and math.isfinite(m) and m < bar


def label(k_formation, n_wake, n_terminal):
    if k_formation <= 6:
        return 'FORMATION_BROKEN'
    if n_terminal >= 8 and n_wake >= 8:
        return 'LOOPS_ONLINE'
    if n_terminal >= 8:
        return 'LOOPS_AFTER_SLEEP'
    return 'NO_LOOPS'


def mechanism_label(n_mechanism):
    return 'DETACH_PROTECTS' if n_mechanism >= 8 else 'DETACH_NOT_SHOWN'


def summarize(records, worlds=WORLDS):
    pairs = [(w, s) for w in worlds for s in STREAMS]
    L = {(w, s): records[f'LOOP_w{w}_s{s}'] for w, s in pairs}
    A = {(w, s): records[f'LOOP_ATTACHED_w{w}_s{s}'] for w, s in pairs}
    C = {(w, s): records[f'CONSTCOUNT_w{w}_s{s}'] for w, s in pairs}
    k_formation = sum(passing(L[p]['canonical_median']) for p in pairs)
    n_wake = sum(L[p]['wake_loop_median'] < WAKE_RATIO * C[p]['wake_loop_median'] for p in pairs)
    n_terminal = sum(passing(L[p]['loop_median'], LOOP_BAR) for p in pairs)
    n_mechanism = sum(L[p]['canonical_median'] < A[p]['canonical_median'] for p in pairs)
    n_control = sum(L[p]['loop_median'] < C[p]['loop_median'] for p in pairs)
    med = lambda arm, key: float(np.median([records[f'{arm}_w{w}_s{s}'][key] for w, s in pairs]))   # noqa: E731
    return {'cells': len(pairs), 'k_formation': k_formation, 'n_wake': n_wake, 'n_terminal': n_terminal,
            'n_mechanism': n_mechanism, 'n_control': n_control, 'label': label(k_formation, n_wake, n_terminal),
            'mechanism_label': mechanism_label(n_mechanism),
            'canonical_passes': {a: sum(passing(records[f'{a}_w{w}_s{s}']['canonical_median']) for w, s in pairs)
                                 for a in ARMS},
            'canonical_median': {a: med(a, 'canonical_median') for a in ARMS},
            'loop_median': {a: med(a, 'loop_median') for a in ARMS},
            'wake_loop_median': {a: med(a, 'wake_loop_median') for a in ARMS},
            'wake_canonical_median': {a: med(a, 'wake_canonical_median') for a in ARMS},
            'loop_below_threshold_total': {a: sum(records[f'{a}_w{w}_s{s}']['loop_below_threshold'] for w, s in pairs)
                                           for a in ARMS},
            'count_accuracy_median': {a: med(a, 'count_accuracy_median') for a in ('LOOP', 'LOOP_ATTACHED')},
            'wake_count_accuracy_median': {a: med(a, 'wake_count_accuracy_median') for a in ('LOOP', 'LOOP_ATTACHED')}}


def protocol(root=ROOT):
    return {'id': 'b2-online-v2', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1, 'exploratory': True,
            'root': Path(root).as_posix(), 'worlds': list(WORLDS), 'streams': list(STREAMS), 'arms': list(ARMS),
            'n_loop': N_LOOP, 'k': K, 'memory': MEMORY, 'sleep_updates': SLEEP_UPDATES, 'threshold': THRESHOLD,
            'loop_bar': LOOP_BAR, 'wake_ratio': WAKE_RATIO,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml')},
            'implementation_sha256': digest(Path(__file__)), 'b2_online_v1_sha256': digest(Path(b2.__file__)), 'loop_stream_sha256': digest(Path(ls.__file__)),
            'loop_learner_sha256': digest(Path(sys.modules[LoopLearner.__module__].__file__)),
            'b2_census_sha256': digest(Path(c2.__file__)), 'b1_online_sha256': digest(Path(b1o.__file__)),
            'deep_reroute_sha256': digest(Path(dr.__file__)), 'o2_sha256': digest(Path(o2.__file__)),
            'o2d_sha256': digest(Path(o2d.__file__)), 'lifetime_sha256': digest(Path(ll.__file__))}


def _status(root, state, done, total, running, started):
    eta = None if not done else (time.time() - started) / done * (total - done)
    atomic_json(Path(root) / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                             'running': running, 'pid': os.getpid(), 'updated_utc': now(),
                                             'eta_seconds': None if eta is None else round(eta)})


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, worlds=WORLDS):
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root) | {'scale': scale}
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    started = time.time()
    with writer_lock(root / 'launcher.lock'):
        if (root / 'manifest.json').exists() and json.loads((root / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; retire the path rather than overwrite it')
        atomic_json(root / 'manifest.json', manifest)
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
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} scale={scale} free={free:.1f}GiB')
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, w, s, str(root), scale): (a, w, s) for a, w, s in todo}
                while futures:
                    done = next(as_completed(futures))
                    a, w, s = futures.pop(done)
                    key = f'{a}_w{w}_s{s}'
                    record = done.result()
                    validate(record)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"canonical={record['canonical_median']:.4g} loop={record['loop_median']:.4g} "
                                               f"wake_loop={record['wake_loop_median']:.4g}")
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
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after', type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.dry_run:
        run([(a, DRY_WORLD, 0) for a in ARMS], root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=DRY_SCALE,
            stop_after=args.stop_after, worlds=(DRY_WORLD,))
        print('B2-online v2 dry run complete')
        return
    require_clean_code(OUTPUT)
    run(cells())
    print(f'B2-online v2 report: {OUTPUT}')


if __name__ == '__main__':
    main()
