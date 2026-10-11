"""B2-hard Tier 1 (exploratory): does the learner DISCOVER which tasks iterate, without being told, while forming the
vocabulary?

Plan: `B2_HARD_PLAN.md`. Development band 5 (fresh): worlds 53, 54, 55, streams 0-2; dry-run world 68. Stream:
`loop_stream` unchanged (O2's SHUFFLED depth-3 stream plus 48 loop tasks, 236 tasks). Loop bodies have planned
depth 1, so the CANDIDATES are every depth-1 task: the 48 loop tasks and the 60 straight length-1 tasks (108). The
learner is not told which candidates loop.

Arms (constructions):
- ALLLOOP: `LoopLearner` with EVERY candidate in `loop_plan`, `state_halt=True`, `step_bias=True` (predicate
  h.z_j + b_j, which nests every fixed count, so a straight task stays expressible), `detach_halt_input=True`
  (B2-online v2), `halt_decay=HALT_DECAY` on h in wake and sleep (a usage cost on reading the state). End of
  stream: re-route the depth-2/3 tasks (`deep_reroute.reroute`); per candidate, on its 64 retained examples with the
  library frozen, the best COUNT form (body slot, fixed count 0..K) and the best LOOP form (body slot, state
  predicate fitted as `b2_online.refit_loops` fits it); the loop is kept iff LOOP support MSE < LOOP_RATIO x COUNT
  support MSE, otherwise the count form is installed (h = 0, biases +/-BIG). Then sleep (O3's construction, 8,192
  updates, sampling [1941, w, s, 64]) on the library, all task codes, the biases, and h with the decay.
- ALLCOUNT: the same candidates in `loop_plan` with `state_halt=False` (per-step biases only: halting cannot read
  the state; the capacity control). End of stream: re-route; every candidate gets its best COUNT form; then
  B2-online v2's sleep (`b1_online.sleep` with the halting biases).
- GIVEN: B2-online v2's LOOP exactly (loop identity GIVEN; the 48 loop tasks only), the paired reference for what
  discovery costs.
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
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock
from row.models.loop_gated import LoopLearner

PLAN = Path('B2_HARD_PLAN.md')
ROOT = Path('artifacts/b2_hard')
DRY_ROOT = Path('artifacts/b2_hard_dry')
OUTPUT = Path('reports/b2_hard.json')
WORLDS = (53, 54, 55)
STREAMS = (0, 1, 2)
ARMS = ('ALLLOOP', 'ALLCOUNT', 'GIVEN')
N_LOOP = 48
N_CAND = 108
MEMORY = b1o.MEMORY
SLEEP_UPDATES = b1o.SLEEP_UPDATES
K = ls.K
BIG = b2.BIG
HALT_DECAY = 0.1
LOOP_RATIO = 0.5
THRESHOLD = 0.05
LOOP_BAR = 0.38   # B2-online v2's bar (sqrt(0.214 x 0.675), reports/b2_support64_census.json); reported only
JOBS = 3
MIN_FREE_GIB = 2.0   # PI 2026-10-03 instruction
DRY_WORLD = 68
DRY_SCALE = 16


def cells(worlds=WORLDS):
    return [(a, w, s) for a in ARMS for w in worlds for s in STREAMS]


def auc(pos, neg):
    """Probability a random positive outscores a random negative (ties count half)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def fit_forms(lib, X, Y, seed):
    """Best COUNT form (loss, slot, L) and best LOOP form (loss, slot, Wd) on support, the census construction
    (`census_b2h_data_test.forms`) returning the fitted predicate."""
    best_c = best_l = None
    for s in range(lib.slots):
        U = c2.unrolled(lib, s, X)
        for L in range(K + 1):
            loss = float(np.mean((U[L] - Y) ** 2))
            if best_c is None or loss < best_c[0]:
                best_c = (loss, s, L)
        inferred = np.argmin(((U - Y[None]) ** 2).sum(-1), axis=0)
        Xh, yh = [], []
        for j in range(K):
            at = inferred >= j
            Xh.append(U[j][at])
            yh.append((inferred[at] > j).astype(int))
        labels = np.concatenate(yh)
        if len(set(labels.tolist())) < 2:
            Wd = np.zeros(X.shape[1] + 1)
            Wd[-1] = BIG if (bool(labels[0]) if len(labels) else False) else -BIG
        else:
            Wl = c2.fit_linear(np.concatenate(Xh), labels, 2, seed=seed)
            Wd = Wl[:, 1] - Wl[:, 0]
        p, _ = c2.loop_with(lambda Z: (np.c_[Z, np.ones(len(Z))] @ Wd) > 0, U)
        loss = float(np.mean((p - Y) ** 2))
        if best_l is None or loss < best_l[0]:
            best_l = (loss, s, Wd)
    return best_c, best_l


def install_count(model, task_id, slot, L):
    b2.set_body(model, task_id, slot)
    with torch.no_grad():
        if task_id in model.halt_w:
            model.halt_w[task_id].zero_()
        model.halt_b[task_id].copy_(torch.tensor([BIG if j < L else -BIG for j in range(K)]))


def install_loop(model, task_id, slot, Wd):
    b2.set_body(model, task_id, slot)
    with torch.no_grad():
        model.halt_w[task_id].copy_(torch.tensor(Wd[:-1], dtype=torch.float32))
        model.halt_b[task_id].fill_(float(Wd[-1]))


def end_of_stream(model, cands, pool, w, arm):
    """Per candidate: the data test (ALLLOOP) or the count form (ALLCOUNT). Returns per-task decisions."""
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    lib = FrozenLibrary(model)
    out = {}
    for i, t in cands:
        X = np.stack([a for a, _ in by[t.task_id]])
        Y = np.stack([b for _, b in by[t.task_id]])
        seed = int(np.random.SeedSequence([5820, w, i]).generate_state(1)[0]) % 2**31
        bc, bl = fit_forms(lib, X, Y, seed)
        keep = arm == 'ALLLOOP' and bl[0] < LOOP_RATIO * bc[0]
        if keep:
            install_loop(model, t.task_id, bl[1], bl[2])
        else:
            install_count(model, t.task_id, bc[1], bc[2])
        out[t.task_id] = {'loop': bool(keep), 'count_support': bc[0], 'count_L': bc[2], 'loop_support': bl[0]}
    return out


def sleep_decay(cfg, model, pool, task_ids, seed_entropy):
    """`b1_online.sleep`'s construction with h in its own weight-decay group (HALT_DECAY)."""
    global_lr, task_lr, weight_decay, _, _, _, _ = ll._training_values(cfg, o2.KIND)
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = ll._shared_optimizer(model, global_lr, weight_decay)
    optimizer.add_param_group({'params': [model.task_codes[t] for t in task_ids] +
                               [model.halt_b[t] for t in task_ids if t in model.halt_b],
                               'lr': task_lr, 'weight_decay': 0.0})
    optimizer.add_param_group({'params': [model.halt_w[t] for t in task_ids if t in model.halt_w],
                               'lr': task_lr, 'weight_decay': HALT_DECAY})
    model.set_training_progress(1.0)
    model.train()
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    for _ in range(SLEEP_UPDATES):
        idx = rng.choice(len(pool), size=2, replace=False)
        x = torch.as_tensor(np.stack([pool[i][0] for i in idx]), dtype=torch.float32)
        y = torch.as_tensor(np.stack([pool[i][1] for i in idx]), dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model.forward_tasks(x, [pool[i][2] for i in idx]), y).backward()
        optimizer.step()
    model.eval()


def build(arm, cfg, plan, cand_ids, loop_ids):
    if arm == 'ALLLOOP':
        return b1o.build(LoopLearner, cfg, plan, loop_plan={t: 'state' for t in cand_ids}, max_iterations=K,
                         state_halt=True, step_bias=True, detach_halt_input=True, halt_decay=HALT_DECAY)
    if arm == 'ALLCOUNT':
        return b1o.build(LoopLearner, cfg, plan, loop_plan={t: 'state' for t in cand_ids}, max_iterations=K,
                         state_halt=False)
    return b1o.build(LoopLearner, cfg, plan, loop_plan={t: 'state' for t in loop_ids}, max_iterations=K,
                     state_halt=True, detach_halt_input=True)


def run_arm(arm, w, s, work: Path, scale=1):
    cfg, mixed, st, plan, canonical, ltasks, meta = ls.stream(w, s, N_LOOP)
    loop_ids = {t.task_id for t in ltasks}
    cands = [(i, t) for i, t in enumerate(st) if plan[t.task_id] == 1]
    cand_ids = {t.task_id for _, t in cands}
    anchors = [t for _, t in cands if t.task_id not in loop_ids]
    model = build(arm, cfg, plan, cand_ids, loop_ids)
    summary = ll.run(dataclasses.replace(o2.run_cfg(cfg, scale), output_directory=work / 'lifetime'), o2.KIND,
                     world=mixed, model=model, return_model=True, replay_seed=o2.replay_seed_for(w, s))
    model = summary.pop('terminal_model')
    rec = {'arm': arm, 'world': w, 'stream': s, 'scale': scale, 'stream_tasks': len(st), 'candidates': len(cands),
           'loop_task_ids': sorted(loop_ids),
           'wake_canonical_median': score(model, SimpleNamespace(tasks=canonical))['median'],
           'wake_loop_median': score(model, SimpleNamespace(tasks=ltasks))['median'],
           'wake_anchor_median': score(model, SimpleNamespace(tasks=anchors))['median']}
    halted = [t.task_id for t in st if t.task_id in model.halt_b]
    # non-vacuity: wake must move the halting parameters off their zero init
    rec['wake_halt_norm_median'] = float(np.median([
        float(torch.cat([p.detach().flatten() for p in model.loop_parameters([t])]).norm()) for t in halted]))
    if arm == 'ALLLOOP':
        norms = {t: float(model.halt_w[t].detach().norm()) for t in cand_ids}
        rec['wake_state_norms'] = norms
        rec['wake_state_auc'] = auc([norms[t] for t in loop_ids], [v for t, v in norms.items() if t not in loop_ids])
        rec['wake_state_norm_median'] = {'loop': float(np.median([norms[t] for t in loop_ids])),
                                         'straight': float(np.median([v for t, v in norms.items() if t not in loop_ids]))}
    if arm != 'ALLCOUNT':
        rec['wake_count_accuracy_median'] = float(np.median(b2.count_accuracy(model, ltasks, meta)))
    pool = o2d.reservoir(st, w, s, MEMORY)
    rec['routes_changed'] = dr.reroute(model, [t for t in st if t.task_id not in model.loop_plan], plan, pool)
    before = library_sha(model)
    if arm == 'GIVEN':
        rec['loop_refits_installed'] = b2.refit_loops(model, ltasks, pool, w, 'LOOP')
        b1o.sleep(cfg, model, pool, [t.task_id for t in st], [1941, w, s, MEMORY], model.loop_parameters(loop_ids))
    else:
        dec = end_of_stream(model, cands, pool, w, arm)
        rec['decisions'] = dec
        rec['loops_kept'] = sum(dec[t]['loop'] for t in loop_ids)
        rec['straight_kept_count'] = sum(not v['loop'] for t, v in dec.items() if t not in loop_ids)
        if arm == 'ALLLOOP':
            sleep_decay(cfg, model, pool, [t.task_id for t in st], [1941, w, s, MEMORY])
        else:
            b1o.sleep(cfg, model, pool, [t.task_id for t in st], [1941, w, s, MEMORY], model.loop_parameters(cand_ids))
    rec['library_sha256_before_sleep'], rec['library_sha256'] = before, library_sha(model)
    canon = score(model, SimpleNamespace(tasks=canonical))
    lp = score(model, SimpleNamespace(tasks=ltasks))
    an = score(model, SimpleNamespace(tasks=anchors))
    rec.update({'canonical_median': canon['median'], 'canonical_per_task': canon['per_task'],
                'scored_canonical': len(canon['per_task']), 'loop_median': lp['median'], 'loop_per_task': lp['per_task'],
                'loop_below_threshold': sum(v < THRESHOLD for v in lp['per_task'].values()),
                'anchor_median': an['median'], 'anchor_per_task': an['per_task']})
    if arm != 'ALLCOUNT':
        acc = b2.count_accuracy(model, ltasks, meta)
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
    if (rec['scored_canonical'] != 64 or rec['stream_tasks'] != 188 + N_LOOP or len(rec['loop_per_task']) != N_LOOP
            or rec['candidates'] != N_CAND or len(rec['anchor_per_task']) != N_CAND - N_LOOP):
        raise ValueError(f'{key}: construction')
    if rec['library_sha256'] == rec['library_sha256_before_sleep']:
        raise ValueError(f'{key}: sleep left the library unchanged')
    if not rec['wake_halt_norm_median'] > 0:
        raise ValueError(f'{key}: wake never moved the halting parameters')
    if rec['arm'] != 'GIVEN' and len(rec['decisions']) != N_CAND:
        raise ValueError(f'{key}: decisions')
    if rec['arm'] == 'ALLCOUNT' and rec['loops_kept'] != 0:
        raise ValueError(f'{key}: the control kept a loop')


def passing(m, bar=THRESHOLD):
    return m is not None and math.isfinite(m) and m < bar


def label(k_formation, auc_median, recall, specificity, n_control):
    if k_formation <= 6:
        return 'FORMATION_BROKEN'
    if auc_median >= 0.9 and recall >= 0.8 and specificity >= 0.95 and n_control >= 8:
        return 'DISCOVERS'
    return 'NOT_DISCOVERED'


def summarize(records, worlds=WORLDS):
    pairs = [(w, s) for w in worlds for s in STREAMS]
    A = {p: records[f'ALLLOOP_w{p[0]}_s{p[1]}'] for p in pairs}
    C = {p: records[f'ALLCOUNT_w{p[0]}_s{p[1]}'] for p in pairs}
    G = {p: records[f'GIVEN_w{p[0]}_s{p[1]}'] for p in pairs}
    k_formation = sum(passing(A[p]['canonical_median']) for p in pairs)
    auc_median = float(np.median([A[p]['wake_state_auc'] for p in pairs]))
    recall = sum(A[p]['loops_kept'] for p in pairs) / (N_LOOP * len(pairs))
    specificity = sum(A[p]['straight_kept_count'] for p in pairs) / ((N_CAND - N_LOOP) * len(pairs))
    n_control = sum(A[p]['loop_median'] < C[p]['loop_median'] for p in pairs)
    n_terminal = sum(passing(A[p]['loop_median'], LOOP_BAR) for p in pairs)
    n_vs_given = sum(A[p]['loop_median'] <= G[p]['loop_median'] for p in pairs)
    med = lambda D, key: float(np.median([D[p][key] for p in pairs]))   # noqa: E731
    arms = {'ALLLOOP': A, 'ALLCOUNT': C, 'GIVEN': G}
    return {'cells': len(pairs), 'k_formation': k_formation, 'wake_state_auc_median': auc_median,
            'loop_recall': recall, 'straight_specificity': specificity, 'n_control': n_control,
            'label': label(k_formation, auc_median, recall, specificity, n_control),
            'n_terminal': n_terminal, 'n_loop_at_or_below_given': n_vs_given,
            'canonical_passes': {a: sum(passing(D[p]['canonical_median']) for p in pairs) for a, D in arms.items()},
            'canonical_median': {a: med(D, 'canonical_median') for a, D in arms.items()},
            'loop_median': {a: med(D, 'loop_median') for a, D in arms.items()},
            'anchor_median': {a: med(D, 'anchor_median') for a, D in arms.items()},
            'wake_canonical_median': {a: med(D, 'wake_canonical_median') for a, D in arms.items()},
            'wake_loop_median': {a: med(D, 'wake_loop_median') for a, D in arms.items()},
            'wake_anchor_median': {a: med(D, 'wake_anchor_median') for a, D in arms.items()},
            'count_accuracy_median': {a: med(arms[a], 'count_accuracy_median') for a in ('ALLLOOP', 'GIVEN')},
            'wake_count_accuracy_median': {a: med(arms[a], 'wake_count_accuracy_median') for a in ('ALLLOOP', 'GIVEN')},
            'wake_state_norm_median': {k: float(np.median([A[p]['wake_state_norm_median'][k] for p in pairs]))
                                       for k in ('loop', 'straight')}}


def protocol(root=ROOT):
    return {'id': 'b2-hard-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1, 'exploratory': True,
            'root': Path(root).as_posix(), 'worlds': list(WORLDS), 'streams': list(STREAMS), 'arms': list(ARMS),
            'n_loop': N_LOOP, 'n_candidates': N_CAND, 'k': K, 'memory': MEMORY, 'sleep_updates': SLEEP_UPDATES,
            'halt_decay': HALT_DECAY, 'loop_ratio': LOOP_RATIO, 'threshold': THRESHOLD, 'loop_bar': LOOP_BAR,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml')},
            'implementation_sha256': digest(Path(__file__)), 'b2_online_v1_sha256': digest(Path(b2.__file__)),
            'loop_stream_sha256': digest(Path(ls.__file__)),
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
                    extra = (f" auc={record['wake_state_auc']:.3f} kept={record['loops_kept']}"
                             f" count={record['straight_kept_count']}" if 'wake_state_auc' in record else '')
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"canonical={record['canonical_median']:.4g} loop={record['loop_median']:.4g} "
                                               f"anchor={record['anchor_median']:.4g} wake_loop={record['wake_loop_median']:.4g}{extra}")
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
        print('B2-hard dry run complete')
        return
    require_clean_code(OUTPUT)
    run(cells())
    print(f'B2-hard report: {OUTPUT}')


if __name__ == '__main__':
    main()
