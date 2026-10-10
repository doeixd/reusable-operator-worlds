"""B2-online Tier 1 (exploratory): does a learner form the vocabulary AND learn data-dependent iteration online?

Plan: `B2_ONLINE_PLAN.md`. Development worlds 44, 45, 46 (opened by B1-online and B1-hard; design evidence only),
streams 0-2. Stream (`loop_stream`): O2's SHUFFLED depth-3 stream (188 tasks) plus 48 loop tasks
`z = x; repeat at most 6 times: stop if w.z <= 0, else z = P(z)`, one random order over 236 tasks.

Arms (constructions):
- LOOP: `LoopLearner` with the loop tasks in `loop_plan` (loop identity GIVEN; body = the task's depth-1 route code,
  halting = a linear predicate on the state, zero init); wake on the stream; end of stream: (1)
  `deep_reroute.reroute` of the 188 straight-line tasks; (2) loop re-fit, support only, on the frozen current library:
  for every slot s, infer each retained example's count as argmin_L ||[s]^L x - y||, fit a linear go/stop predicate
  on the learned states (the B2 census construction, seed [5820, w, i]), keep the (s, predicate) with the lowest
  hardened support MSE, install it only if it beats the task's current hardened support MSE; (3) sleep: O3's
  construction (8,192 updates, sampling [1941, w, s, 64], 64-per-task reservoir of all 236 tasks) on the library,
  all task codes and the halting parameters.
- CONSTCOUNT (capacity control): `LoopLearner(state_halt=False)`: the same body and iteration machinery, halting a
  learned bias per step (state-independent; in evaluation a learned fixed count, the route [s]^L); the same end of
  stream with the re-fit restricted to the best (s, L) on support; the same sleep.
- REFUSAL: `PlannedDepthRotatedLearner`, loop tasks as ordinary single-route tasks of planned depth 3; re-route of all
  236 tasks; the same sleep.
Scored on the terminal and after wake: canonical 64 straight-line tasks and the 48 loop tasks, query NMSE with hard
routes and hard halting; for LOOP, the fraction of query inputs whose hard iteration count equals the teacher's.
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
from row.experiments import census_b2_iteration_necessity as c2
from row.experiments import deep_reroute as dr
from row.experiments import learned_lifetime as ll
from row.experiments import loop_stream as ls
from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock
from row.models.loop_gated import LoopLearner
from row.models.online_variable_depth import PlannedDepthRotatedLearner

PLAN = Path('B2_ONLINE_PLAN.md')
ROOT = Path('artifacts/b2_online')
DRY_ROOT = Path('artifacts/b2_online_dry')
OUTPUT = Path('reports/b2_online.json')
WORLDS = (44, 45, 46)
STREAMS = (0, 1, 2)
ARMS = ('LOOP', 'CONSTCOUNT', 'REFUSAL')
N_LOOP = 48
MEMORY = b1o.MEMORY
SLEEP_UPDATES = b1o.SLEEP_UPDATES
K = ls.K
THRESHOLD = 0.05
LOOP_BAR = 0.25
WAKE_RATIO = 0.5
JOBS = 3
MIN_FREE_GIB = 2.0   # PI 2026-10-03 instruction
DRY_WORLD = 49
DRY_SCALE = 16
BIG = 10.0


def cells(worlds=WORLDS):
    return [(a, w, s) for a in ARMS for w in worlds for s in STREAMS]


def set_body(model, task_id, slot):
    with torch.no_grad():
        code = model.task_codes[task_id]
        old = int(torch.argmax(code[0]))
        if old != slot:
            a, b = code[0, old].clone(), code[0, slot].clone()
            code[0, old], code[0, slot] = b, a
        if int(torch.argmax(code[0])) != slot:   # tied row (e.g. untrained): make the slot the strict argmax
            code[0, slot] = code[0].max() + 1.0


def hard_support(model, task_id, xs, ys):
    model.eval()
    with torch.no_grad():
        return float(torch.mean((model(xs, task_id) - ys) ** 2))


def refit_loops(model, ltasks, pool, w, arm):
    """Support-only loop re-fit on the frozen current library (see module docstring). Returns installs."""
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    lib = FrozenLibrary(model)
    installed = 0
    for i, t in enumerate(ltasks):
        X = np.stack([a for a, _ in by[t.task_id]])
        Y = np.stack([b for _, b in by[t.task_id]])
        xs, ys = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
        current = hard_support(model, t.task_id, xs, ys)
        best = None
        for s in range(lib.slots):
            U = c2.unrolled(lib, s, X)
            if arm == 'CONSTCOUNT':
                for L in range(K + 1):
                    loss = float(np.mean((U[L] - Y) ** 2))
                    if best is None or loss < best[0]:
                        best = (loss, s, ('count', L))
                continue
            inferred = np.argmin(((U - Y[None]) ** 2).sum(-1), axis=0)
            Xh, yh = [], []
            for j in range(K):
                at = inferred >= j
                Xh.append(U[j][at])
                yh.append((inferred[at] > j).astype(int))
            labels = np.concatenate(yh)
            if len(set(labels.tolist())) < 2:
                go = bool(labels[0]) if len(labels) else False
                Wd = np.zeros(X.shape[1] + 1)
                Wd[-1] = BIG if go else -BIG
            else:
                Wl = c2.fit_linear(np.concatenate(Xh), labels, 2, seed=int(np.random.SeedSequence([5820, w, i]).generate_state(1)[0]) % 2**31)
                Wd = Wl[:, 1] - Wl[:, 0]
            pred, _ = c2.loop_with(lambda Z: (np.c_[Z, np.ones(len(Z))] @ Wd) > 0, U)
            loss = float(np.mean((pred - Y) ** 2))
            if best is None or loss < best[0]:
                best = (loss, s, ('predicate', Wd))
        if best[0] < current:
            set_body(model, t.task_id, best[1])
            with torch.no_grad():
                if best[2][0] == 'count':
                    L = best[2][1]
                    model.halt_b[t.task_id].copy_(torch.tensor([BIG if j < L else -BIG for j in range(K)]))
                else:
                    Wd = torch.tensor(best[2][1], dtype=torch.float32)
                    model.halt_w[t.task_id].copy_(Wd[:-1])
                    model.halt_b[t.task_id].copy_(Wd[-1:])
            installed += 1
    return installed


def count_accuracy(model, ltasks, meta):
    accs = []
    model.eval()
    for t in ltasks:
        _, k = ls.teacher_loop(t.teacher_library[meta[t.task_id]['P']], np.array(meta[t.task_id]['w']), t.eval_x)
        kh = model.iteration_counts(torch.tensor(t.eval_x, dtype=torch.float32), t.task_id).numpy()
        accs.append(float(np.mean(kh == k)))
    return accs


def run_arm(arm, w, s, work: Path, scale=1):
    cfg, mixed, st, plan, canonical, ltasks, meta = ls.stream(w, s, N_LOOP)
    loop_ids = {t.task_id for t in ltasks}
    if arm == 'REFUSAL':
        plan = {k: (3 if k in loop_ids else v) for k, v in plan.items()}
        model = b1o.build(PlannedDepthRotatedLearner, cfg, plan)
    else:
        model = b1o.build(LoopLearner, cfg, plan, loop_plan={t: 'state' for t in loop_ids},
                          max_iterations=K, state_halt=(arm == 'LOOP'))
    summary = ll.run(dataclasses.replace(o2.run_cfg(cfg, scale), output_directory=work / 'lifetime'), o2.KIND,
                     world=mixed, model=model, return_model=True, replay_seed=o2.replay_seed_for(w, s))
    model = summary.pop('terminal_model')
    rec = {'arm': arm, 'world': w, 'stream': s, 'scale': scale, 'stream_tasks': len(st),
           'wake_canonical_median': score(model, SimpleNamespace(tasks=canonical))['median'],
           'wake_loop_median': score(model, SimpleNamespace(tasks=ltasks))['median']}
    if arm == 'LOOP':
        rec['wake_count_accuracy_median'] = float(np.median(count_accuracy(model, ltasks, meta)))
    if arm != 'REFUSAL':   # non-vacuity: wake must move the halting parameters off their zero init
        rec['wake_halt_norm_median'] = float(np.median([
            float(torch.cat([p.detach().flatten() for p in model.loop_parameters([t.task_id])]).norm()) for t in ltasks]))
    pool = o2d.reservoir(st, w, s, MEMORY)
    straight = [t for t in st if not (arm != 'REFUSAL' and t.task_id in loop_ids)]
    rec['routes_changed'] = dr.reroute(model, straight, plan, pool)
    extra = []
    if arm != 'REFUSAL':
        rec['loop_refits_installed'] = refit_loops(model, ltasks, pool, w, arm)
        extra = model.loop_parameters(loop_ids)
    before = library_sha(model)
    b1o.sleep(cfg, model, pool, [t.task_id for t in st], [1941, w, s, MEMORY], extra)
    rec['library_sha256_before_sleep'], rec['library_sha256'] = before, library_sha(model)
    canon = score(model, SimpleNamespace(tasks=canonical))
    lp = score(model, SimpleNamespace(tasks=ltasks))
    rec.update({'canonical_median': canon['median'], 'canonical_per_task': canon['per_task'],
                'scored_canonical': len(canon['per_task']), 'loop_median': lp['median'], 'loop_per_task': lp['per_task'],
                'loop_below_threshold': sum(v < THRESHOLD for v in lp['per_task'].values())})
    if arm == 'LOOP':
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
    if rec['arm'] != 'REFUSAL' and not rec['wake_halt_norm_median'] > 0:
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


def summarize(records, worlds=WORLDS):
    pairs = [(w, s) for w in worlds for s in STREAMS]
    L = {(w, s): records[f'LOOP_w{w}_s{s}'] for w, s in pairs}
    C = {(w, s): records[f'CONSTCOUNT_w{w}_s{s}'] for w, s in pairs}
    k_formation = sum(passing(L[p]['canonical_median']) for p in pairs)
    n_wake = sum(L[p]['wake_loop_median'] < WAKE_RATIO * C[p]['wake_loop_median'] for p in pairs)
    n_terminal = sum(passing(L[p]['loop_median'], LOOP_BAR) for p in pairs)
    n_control = sum(L[p]['loop_median'] < C[p]['loop_median'] for p in pairs)
    med = lambda arm, key: float(np.median([records[f'{arm}_w{w}_s{s}'][key] for w, s in pairs]))   # noqa: E731
    return {'cells': len(pairs), 'k_formation': k_formation, 'n_wake': n_wake, 'n_terminal': n_terminal,
            'n_control': n_control, 'label': label(k_formation, n_wake, n_terminal),
            'canonical_passes': {a: sum(passing(records[f'{a}_w{w}_s{s}']['canonical_median']) for w, s in pairs)
                                 for a in ARMS},
            'canonical_median': {a: med(a, 'canonical_median') for a in ARMS},
            'loop_median': {a: med(a, 'loop_median') for a in ARMS},
            'wake_loop_median': {a: med(a, 'wake_loop_median') for a in ARMS},
            'wake_canonical_median': {a: med(a, 'wake_canonical_median') for a in ARMS},
            'loop_below_threshold_total': {a: sum(records[f'{a}_w{w}_s{s}']['loop_below_threshold'] for w, s in pairs)
                                           for a in ARMS},
            'count_accuracy_median': med('LOOP', 'count_accuracy_median'),
            'wake_count_accuracy_median': med('LOOP', 'wake_count_accuracy_median')}


def protocol(root=ROOT):
    return {'id': 'b2-online-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1, 'exploratory': True,
            'root': Path(root).as_posix(), 'worlds': list(WORLDS), 'streams': list(STREAMS), 'arms': list(ARMS),
            'n_loop': N_LOOP, 'k': K, 'memory': MEMORY, 'sleep_updates': SLEEP_UPDATES, 'threshold': THRESHOLD,
            'loop_bar': LOOP_BAR, 'wake_ratio': WAKE_RATIO,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml')},
            'implementation_sha256': digest(Path(__file__)), 'loop_stream_sha256': digest(Path(ls.__file__)),
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
        print('B2-online dry run complete')
        return
    require_clean_code(OUTPUT)
    run(cells())
    print(f'B2-online report: {OUTPUT}')


if __name__ == '__main__':
    main()
