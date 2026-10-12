"""B2-depth2 sizing (Tier 1, EXPLORATORY, no verdict): with loop tasks whose BODY is two steps in the stream, does
the vocabulary still form, and does the B2-hard v2 construction still discover which tasks iterate?

Census (`reports/b2d2_necessity_census.json`): on formed vocabularies a two-step-body loop is necessary (108/108 against
one route, an input-picked count and the best one-slot loop), its body is trivially found by search, and execution
drift sets its ceiling (true-predicate loop 0.083). So the live question is FORMATION with these tasks in the stream.

Stream: `loop_stream.stream(w, s, 48, body_depth=2)` (O2's SHUFFLED depth-3 stream plus 48 loop tasks with body
P2(P1(z)), 236 tasks). Candidates: every depth-2 task (the 48 loops and the 64 straight length-2 tasks, 112). One
development world (band 5, world 60, stream 0), one cell per arm:
- ONESTEP2: `LoopLearner(body_depth=2)` with every candidate in `loop_plan`, B2-hard v2's ONESTEP halting
  (state_halt, step_bias, detached input, h decay 0.1, halt_init 4). End of stream: re-route the other tasks
  (`deep_reroute.reroute`); per candidate the ONE_STEP data test over two-step bodies (count forms L in {0, 1}; loop
  form = the body with the lowest inferred-count support error, then a linear predicate; keep the loop iff its support
  MSE < 0.5 x the count form's); then `b2_hard.sleep_decay`.
- GIVEN2: loop identity given (the 48 loop tasks only, B2-online v2's halting with `body_depth=2`); end of stream:
  re-route the other tasks, re-fit each loop's body and predicate (installed iff better on support), `b1_online.sleep`.
Records per arm: wake and terminal medians (canonical, loop, length-2 anchors, length-1 anchors), wake state-weight AUC
(ONESTEP2), decisions, count accuracy, and the length-2 anchors' best fixed count over all two-step bodies at the end
of wake (a ROOT analogue: best L >= 2; teacher programs used only in this record). Body choice by inferred-count error
is a cheaper construction than B2-hard's per-slot predicate fit (the census found it picks the matched body 108/108).
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import b1_online as b1o
from row.experiments import b2_hard as bh
from row.experiments import b2_online as b2
from row.experiments import census_b2_iteration_necessity as c2
from row.experiments import census_b2d2_necessity as cd
from row.experiments import deep_reroute as dr
from row.experiments import learned_lifetime as ll
from row.experiments import loop_stream as ls
from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now
from row.models.loop_gated import LoopLearner

ROOT = Path('artifacts/b2d2_sizing')
OUTPUT = Path('reports/b2d2_sizing.json')
WORLD, STREAM = 60, 0
ARMS = ('ONESTEP2', 'GIVEN2')
N_LOOP = 48
BODY = 2
K = ls.K
HALT_INIT = 4.0
HALT_DECAY = bh.HALT_DECAY
LOOP_RATIO = bh.LOOP_RATIO
BIG = b2.BIG
ONE_STEP = (0, 1)


def build(arm, cfg, plan, cand_ids, loop_ids):
    if arm == 'ONESTEP2':
        return b1o.build(LoopLearner, cfg, plan, loop_plan={t: 'state' for t in cand_ids}, max_iterations=K,
                         state_halt=True, step_bias=True, detach_halt_input=True, halt_decay=HALT_DECAY,
                         halt_init=HALT_INIT, body_depth=BODY)
    return b1o.build(LoopLearner, cfg, plan, loop_plan={t: 'state' for t in loop_ids}, max_iterations=K,
                     state_halt=True, detach_halt_input=True, body_depth=BODY)


def bodies(slots, depth=BODY):
    return list(itertools.product(range(slots), repeat=depth))


def predicate(U, Y, seed):
    """B2-hard's support-only linear predicate on a body's unrolled states (constant rule when labels agree)."""
    inferred = np.argmin(((U - Y[None]) ** 2).sum(-1), axis=0)
    Xh, yh = [], []
    for j in range(K):
        at = inferred >= j
        Xh.append(U[j][at])
        yh.append((inferred[at] > j).astype(int))
    labels = np.concatenate(yh)
    if len(set(labels.tolist())) < 2:
        Wd = np.zeros(U.shape[2] + 1)
        Wd[-1] = BIG if (bool(labels[0]) if len(labels) else False) else -BIG
        return Wd
    Wl = c2.fit_linear(np.concatenate(Xh), labels, 2, seed=seed)
    return Wl[:, 1] - Wl[:, 0]


def forms(lib, X, Y, seed, count_Ls=ONE_STEP):
    """Best COUNT form (loss, body, L) over bodies and L in count_Ls; best LOOP form (loss, body, Wd) on the body with
    the lowest inferred-count support error."""
    best_c, best_oc = None, None
    for body in bodies(lib.slots):
        U = cd.unrolled(lib, body, X, K)
        for L in count_Ls:
            loss = float(np.mean((U[L] - Y) ** 2))
            if best_c is None or loss < best_c[0]:
                best_c = (loss, body, L)
        oc = cd.oracle_count_error(U, Y)
        if best_oc is None or oc < best_oc[0]:
            best_oc = (oc, body, U)
    _, body, U = best_oc
    Wd = predicate(U, Y, seed)
    p, _ = c2.loop_with(lambda Z: (np.c_[Z, np.ones(len(Z))] @ Wd) > 0, U)
    return best_c, (float(np.mean((p - Y) ** 2)), body, Wd)


def set_route(model, task_id, body):
    with torch.no_grad():
        code = model.task_codes[task_id]
        for step, slot in enumerate(body):
            old = int(torch.argmax(code[step]))
            if old != slot:
                a, b = code[step, old].clone(), code[step, slot].clone()
                code[step, old], code[step, slot] = b, a
            if int(torch.argmax(code[step])) != slot:
                code[step, slot] = code[step].max() + 1.0


def install_count(model, task_id, body, L):
    set_route(model, task_id, body)
    with torch.no_grad():
        if task_id in model.halt_w:
            model.halt_w[task_id].zero_()
        model.halt_b[task_id].copy_(torch.tensor([BIG if j < L else -BIG for j in range(K)]))


def install_loop(model, task_id, body, Wd):
    set_route(model, task_id, body)
    with torch.no_grad():
        model.halt_w[task_id].copy_(torch.tensor(Wd[:-1], dtype=torch.float32))
        model.halt_b[task_id].fill_(float(Wd[-1]))


def by_task(pool):
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    return {t: (np.stack([a for a, _ in v]), np.stack([b for _, b in v])) for t, v in by.items()}


def seed_for(w, i):
    return int(np.random.SeedSequence([5830, w, i]).generate_state(1)[0]) % 2**31


def end_of_stream_onestep(model, cands, data, w):
    lib = FrozenLibrary(model)
    out = {}
    for i, t in cands:
        X, Y = data[t.task_id]
        bc, bl = forms(lib, X, Y, seed_for(w, i))
        keep = bl[0] < LOOP_RATIO * bc[0]
        if keep:
            install_loop(model, t.task_id, bl[1], bl[2])
        else:
            install_count(model, t.task_id, bc[1], bc[2])
        out[t.task_id] = {'loop': bool(keep), 'count_support': bc[0], 'count_L': bc[2], 'loop_support': bl[0]}
    return out


def refit_given(model, ltasks, data, w):
    lib = FrozenLibrary(model)
    installed = 0
    for i, t in enumerate(ltasks):
        X, Y = data[t.task_id]
        current = b2.hard_support(model, t.task_id, torch.tensor(X, dtype=torch.float32),
                                  torch.tensor(Y, dtype=torch.float32))
        _, bl = forms(lib, X, Y, seed_for(w, i))
        if bl[0] < current:
            install_loop(model, t.task_id, bl[1], bl[2])
            installed += 1
    return installed


def count_accuracy(model, ltasks, meta):
    accs = []
    model.eval()
    for t in ltasks:
        _, k = ls.teacher_loop(ls.body_fn(t.teacher_library, t.program.primitive_ids), np.array(meta[t.task_id]['w']),
                               t.eval_x)
        kh = model.iteration_counts(torch.tensor(t.eval_x, dtype=torch.float32), t.task_id).numpy()
        accs.append(float(np.mean(kh == k)))
    return accs


def anchor_best_counts(model, anchors, data):
    """Each straight length-2 anchor's best fixed count over all two-step bodies and L in 0..K, on the wake library."""
    lib = FrozenLibrary(model)
    out = {}
    for t in anchors:
        X, Y = data[t.task_id]
        best = None
        for body in bodies(lib.slots):
            U = cd.unrolled(lib, body, X, K)
            for L in range(K + 1):
                loss = float(np.mean((U[L] - Y) ** 2))
                if best is None or loss < best[0]:
                    best = (loss, body, L)
        out[t.task_id] = {'L': best[2], 'loss': best[0], 'program': list(t.program.primitive_ids)}
    return out


def run_arm(arm, w=WORLD, s=STREAM, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, mixed, st, plan, canonical, ltasks, meta = ls.stream(w, s, N_LOOP, BODY)
    loop_ids = {t.task_id for t in ltasks}
    cands = [(i, t) for i, t in enumerate(st) if plan[t.task_id] == BODY]
    cand_ids = {t.task_id for _, t in cands}
    anchors2 = [t for _, t in cands if t.task_id not in loop_ids]
    anchors1 = [t for t in st if plan[t.task_id] == 1]
    model = build(arm, cfg, plan, cand_ids, loop_ids)
    work = ROOT / ('work' if scale == 1 else f'dry{scale}') / f'{arm}_w{w}_s{s}'
    summary = ll.run(dataclasses.replace(o2.run_cfg(cfg, scale), output_directory=work / 'lifetime'), o2.KIND,
                     world=mixed, model=model, return_model=True, replay_seed=o2.replay_seed_for(w, s))
    model = summary.pop('terminal_model')

    def med(ts):
        return score(model, SimpleNamespace(tasks=ts))['median']

    def medians(prefix):
        return {f'{prefix}canonical_median': med(canonical), f'{prefix}loop_median': med(ltasks),
                f'{prefix}anchor2_median': med(anchors2), f'{prefix}anchor1_median': med(anchors1),
                f'{prefix}count_accuracy_median': float(np.median(count_accuracy(model, ltasks, meta)))}

    rec = {'arm': arm, 'world': w, 'stream': s, 'scale': scale, 'stream_tasks': len(st), 'candidates': len(cands),
           **medians('wake_')}
    if arm == 'ONESTEP2':
        norms = {t: float(model.halt_w[t].detach().norm()) for t in cand_ids}
        rec['wake_state_auc'] = bh.auc([norms[t] for t in loop_ids], [v for t, v in norms.items() if t not in loop_ids])
        rec['wake_state_norm_median'] = {'loop': float(np.median([norms[t] for t in loop_ids])),
                                         'straight': float(np.median([v for t, v in norms.items() if t not in loop_ids]))}
    pool = o2d.reservoir(st, w, s, bh.MEMORY)
    data = by_task(pool)
    best = anchor_best_counts(model, anchors2, data)
    rec['wake_anchor2_best_L_hist'] = np.bincount([v['L'] for v in best.values()], minlength=K + 1).tolist()
    rec['wake_anchor2_rooted'] = sum(v['L'] >= 2 for v in best.values())
    rec['routes_changed'] = dr.reroute(model, [t for t in st if t.task_id not in model.loop_plan], plan, pool)
    rec['reroute_canonical_median'] = med(canonical)
    before = library_sha(model)
    if arm == 'ONESTEP2':
        dec = end_of_stream_onestep(model, cands, data, w)
        rec['loops_kept'] = sum(dec[t]['loop'] for t in loop_ids)
        rec['straight_kept_count'] = sum(not v['loop'] for t, v in dec.items() if t not in loop_ids)
        bh.sleep_decay(cfg, model, pool, [t.task_id for t in st], [1941, w, s, bh.MEMORY])
    else:
        rec['loop_refits_installed'] = refit_given(model, ltasks, data, w)
        b1o.sleep(cfg, model, pool, [t.task_id for t in st], [1941, w, s, bh.MEMORY], model.loop_parameters(loop_ids))
    rec['library_changed_in_sleep'] = library_sha(model) != before
    rec.update(medians(''))
    rec['seconds'] = time.perf_counter() - started
    return rec


def main(scale=1):
    """Restartable: each finished cell is written atomically to `cells/`, stamped with the commit and scale; a
    relaunch reuses stamped cells and reruns the rest from initialization; a mismatched stamp fails closed."""
    out = OUTPUT if scale == 1 else ROOT / f'dry{scale}.json'
    if scale == 1:
        require_clean_code(OUTPUT)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    stamp = {'commit': commit, 'scale': scale, 'world': WORLD, 'stream': STREAM}
    cells = ROOT / ('cells' if scale == 1 else f'cells_dry{scale}')
    cells.mkdir(parents=True, exist_ok=True)
    done, todo = {}, []
    for a in ARMS:
        f = cells / f'{a}.json'
        if f.exists():
            saved = json.loads(f.read_text())
            if saved['stamp'] != stamp:
                raise ValueError(f'{f}: stamp {saved["stamp"]} != {stamp}')
            done[a] = saved['record']
        else:
            todo.append(a)
    print(now(), 'LAUNCH', json.dumps(stamp), 'reused', sorted(done), flush=True)
    with ProcessPoolExecutor(max_workers=max(1, len(todo))) as ex:
        futs = {ex.submit(run_arm, a, WORLD, STREAM, scale): a for a in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            atomic_json(cells / f'{futs[fut]}.json', {'stamp': stamp, 'record': rec})
            done[futs[fut]] = rec
            print(now(), json.dumps(rec), flush=True)
    atomic_json(out, {'tier': 1, 'exploratory': True, 'stamp': stamp, 'cells': [done[a] for a in ARMS],
                      'finished_utc': now()})
    print(now(), 'EXIT 0', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(*(int(a) for a in sys.argv[1:2])))
