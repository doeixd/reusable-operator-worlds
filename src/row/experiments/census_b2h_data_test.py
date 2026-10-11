"""B2-hard Tier 0 census (descriptive): can an end-of-stream DATA TEST find which tasks iterate, without being told?

B2-online v2 (`reports/b2_online_v2.json`, LOOPS_ONLINE) GAVE the learner loop identity. B2-hard would not. Loop
tasks have planned depth 1 (the body), so the candidates are every depth-1 task: 48 loop tasks and the 60 straight
length-1 anchors. Per candidate, on its 64 RETAINED examples (`o2d.reservoir`) and the frozen wake library:
- COUNT form: the best (body slot s, fixed count L in 0..K) by support MSE. L = 1 is the single route, so this
  form contains every straight depth-1 solution (B2-online's CONSTCOUNT re-fit);
- LOOP form: the best (s, state predicate) by support MSE, the predicate fitted exactly as `b2_online.refit_loops`
  fits it for LOOP (inferred counts, linear fit, seed [5820, w, i]).
The data test keeps a loop iff LOOP support MSE < RATIO x COUNT support MSE. Both forms are also scored on each
task's held-out query set, so in-sample overfitting by the predicate shows up as a query ratio near or above 1.

Questions, written before it ran:
- separation: AUC of the support ratio (loop tasks lower), recall and specificity at RATIO 0.25/0.5/0.75/0.9;
- the cost of a FALSE loop: straight tasks' query error under the LOOP form against the COUNT form;
- whether the test is near-guaranteed on a formed library (as B1-hard's was): if so, a B2-hard plan's
  question is carried by formation and by wake, not by this test.

Inputs: B2-online v2's 9 LOOP wake models (`artifacts/b2_online_v2/work/LOOP_w{w}_s{s}/lifetime/model.pt`), worlds
50-52, behind gate G0 (reload reproduces the recorded wake canonical and loop medians exactly). These libraries
were formed WITH loop identity; a B2-hard wake library may differ. Output `reports/b2h_data_test_census.json`.
Tier 0 on reused development cells; descriptive, no verdict.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import b1_hard as bh
from row.experiments import b1_online as b1o
from row.experiments import b2_online as b2
from row.experiments import b2_online_v2 as b2v
from row.experiments import census_b2_iteration_necessity as c2
from row.experiments import loop_stream as ls
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now
from row.models.loop_gated import LoopLearner

OUTPUT = Path('reports/b2h_data_test_census.json')
RATIOS = (0.25, 0.5, 0.75, 0.9)
K = b2.K


def forms(lib, X, Y, Xq, Yq, seed):
    """Best COUNT and best LOOP form on support (X, Y); both also evaluated on the query (Xq, Yq)."""
    best_c = best_l = None
    for s in range(lib.slots):
        U, Uq = c2.unrolled(lib, s, X), c2.unrolled(lib, s, Xq)
        for L in range(K + 1):
            loss = float(np.mean((U[L] - Y) ** 2))
            if best_c is None or loss < best_c[0]:
                best_c = (loss, float(np.mean((Uq[L] - Yq) ** 2)), s, L)
        inferred = np.argmin(((U - Y[None]) ** 2).sum(-1), axis=0)
        Xh, yh = [], []
        for j in range(K):
            at = inferred >= j
            Xh.append(U[j][at])
            yh.append((inferred[at] > j).astype(int))
        labels = np.concatenate(yh)
        if len(set(labels.tolist())) < 2:
            Wd = np.zeros(X.shape[1] + 1)
            Wd[-1] = b2.BIG if (bool(labels[0]) if len(labels) else False) else -b2.BIG
        else:
            Wl = c2.fit_linear(np.concatenate(Xh), labels, 2, seed=seed)
            Wd = Wl[:, 1] - Wl[:, 0]
        pred = lambda Z: (np.c_[Z, np.ones(len(Z))] @ Wd) > 0   # noqa: E731
        p, _ = c2.loop_with(pred, U)
        pq, _ = c2.loop_with(pred, Uq)
        loss = float(np.mean((p - Y) ** 2))
        if best_l is None or loss < best_l[0]:
            best_l = (loss, float(np.mean((pq - Yq) ** 2)), s)
    return best_c, best_l


def one(ws):
    w, s = ws
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, mixed, st, plan, canonical, ltasks, meta = ls.stream(w, s, b2.N_LOOP)
    loop_ids = {t.task_id for t in ltasks}
    model = b1o.build(LoopLearner, cfg, plan, loop_plan={t: 'state' for t in loop_ids}, max_iterations=K,
                      state_halt=True, detach_halt_input=True)
    for t in st:
        model.begin_task(t.task_id)
    state = torch.load(f'artifacts/b2_online_v2/work/LOOP_w{w}_s{s}/lifetime/model.pt',
                       weights_only=True)['model_state_dict']
    model.load_state_dict(state, strict=False)   # only the unused novel-composition probe code is unexpected
    model.eval()
    ref = json.loads(Path(f'artifacts/b2_online_v2/cells/LOOP_w{w}_s{s}.json').read_text())['record']
    g0 = (score(model, SimpleNamespace(tasks=canonical))['median'] == ref['wake_canonical_median']
          and score(model, SimpleNamespace(tasks=ltasks))['median'] == ref['wake_loop_median'])
    pool = o2d.reservoir(st, w, s, b2.MEMORY)
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    lib = FrozenLibrary(model)
    rows = []
    for i, t in enumerate(st):
        if plan[t.task_id] != 1:
            continue
        X = np.stack([a for a, _ in by[t.task_id]])
        Y = np.stack([b for _, b in by[t.task_id]])
        seed = int(np.random.SeedSequence([5820, w, i]).generate_state(1)[0]) % 2**31
        bc, bl = forms(lib, X, Y, np.asarray(t.eval_x, dtype=np.float64), np.asarray(t.eval_y, dtype=np.float64), seed)
        rows.append({'task': t.task_id, 'loop': t.task_id in loop_ids, 'count_support': bc[0], 'count_query': bc[1],
                     'count_slot': bc[2], 'count_L': bc[3], 'loop_support': bl[0], 'loop_query': bl[1],
                     'loop_slot': bl[2]})
    return {'cell': f'LOOP_w{w}_s{s}', 'G0_reload_reproduces': g0, 'rows': rows,
            'seconds': time.perf_counter() - started}


def summarize(rows):
    lp = [r for r in rows if r['loop']]
    st = [r for r in rows if not r['loop']]
    ratio = lambda r, k: r[f'loop_{k}'] / r[f'count_{k}']   # noqa: E731
    out = {'n_loop': len(lp), 'n_straight': len(st)}
    for k in ('support', 'query'):
        out[f'{k}_ratio_median'] = {'loop': float(np.median([ratio(r, k) for r in lp])),
                                    'straight': float(np.median([ratio(r, k) for r in st]))}
        out[f'{k}_ratio_auc_loop_lower'] = bh.auc([-ratio(r, k) for r in lp], [-ratio(r, k) for r in st])
        for thr in RATIOS:
            out[f'{k}_thr{thr}'] = {'recall': sum(ratio(r, k) < thr for r in lp) / len(lp),
                                    'specificity': sum(ratio(r, k) >= thr for r in st) / len(st)}
    out['straight_count_L_hist'] = np.bincount([r['count_L'] for r in st], minlength=K + 1).tolist()
    out['loop_count_L_hist'] = np.bincount([r['count_L'] for r in lp], minlength=K + 1).tolist()
    out['query_median'] = {f'{a}_{f}': float(np.median([r[f'{f}_query'] for r in (lp if a == 'loop' else st)]))
                           for a in ('loop', 'straight') for f in ('count', 'loop')}
    return out


def main():
    cells = [(w, s) for w in b2v.WORLDS for s in b2v.STREAMS]
    with ProcessPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(one, cells))
    rows = [r for c in res for r in c['rows']]
    out = {'tier': 0, 'descriptive': True, 'for': 'B2-hard design', 'G0': [c['G0_reload_reproduces'] for c in res],
           'summary': summarize(rows), 'per_cell': {c['cell']: summarize(c['rows']) for c in res}, 'cells': res,
           'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps({'G0': out['G0'], **out['summary']}, indent=1))
    return 0 if all(out['G0']) else 1


if __name__ == '__main__':
    sys.exit(main())
