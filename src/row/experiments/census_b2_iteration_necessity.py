"""B2 Tier 0 necessity/opportunity census for data-dependent iteration (descriptive; written before any B2 plan).

Task: z = x; repeat at most K times: if w.z <= 0 stop, else z = P(z); y = z. P one teacher primitive, w a random
unit hyperplane, K = 6 (inside C0's clean ~8 applications). The repeat count k(x) in 0..K varies per input.

On B1-online's saved GATED wake vocabularies (worlds 44-46, stream 0; reload gate G0), teacher primitive -> learned
slot by functional matching (one application, 1,024 random inputs; the teacher is used here only to build the
oracle structure, as in B0). 36 tasks per world (draws [5680, world, i]), support n in (128, 512), query 1,024.
Arms (query NMSE):
- REFUSAL: best single straight-line route (exhaustive depth <= 3 over all slots, plus [s]^L for L = 0..K), chosen
  on support: what one route per task can do;
- KWAY_INPUT (impostor): the K+1 unrolled routes [s]^L, a multinomial linear classifier on the INPUT picks L;
- LOOP_LEARNED: the loop itself, run with the learned slot and a linear halting predicate learned on the learned
  states;
- LOOP_ORACLE: the loop with the true hyperplane w applied to the learned states (ceiling of the loop form).
LIBRARY = 'final' (default) rebuilds B1-online's FINAL library from the saved wake terminal with its exact
end-of-stream steps (re-route of straight tasks, branch re-fit, sleep) and requires the committed GATED per-task
canonical and branch scores to be reproduced exactly (gate G1); LIBRARY = 'wake' uses the wake terminal as saved
(first run, `reports/b2_iteration_necessity_census_wake.json`).
Both learned arms get the same support-only labels: each support example's count is inferred as argmin_L of
||[s]^L x - y||, so neither sees the true count. Also recorded: the true count distribution and the inferred-count
accuracy. Output `reports/b2_iteration_necessity_census.json`.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import b1_online as b1o
from row.experiments import branch_stream as bs
from row.experiments import deep_reroute as dr
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now
from row.models.branch_gated import BranchGatedLearner

OUTPUT = Path('reports/b2_iteration_necessity_census.json')
LIBRARY = 'final'
WORLDS = (44, 45, 46)
TASKS = 36
K = 6
SIZES = (128, 512)
QUERY = 1024


def nmse(pred, target):
    return float(np.mean((pred - target) ** 2) / max(np.var(target), 1e-12))


def teacher_loop(P, w, x):
    z = np.array(x, dtype=np.float64)
    k = np.zeros(len(z), dtype=int)
    for _ in range(K):
        go = z @ w > 0
        if not go.any():
            break
        z[go] = P(z[go])
        k += go
    return z, k


def unrolled(lib, s, x):
    """[s]^L x for L = 0..K, as an array (K+1, n, d)."""
    out = [np.asarray(x, dtype=np.float64)]
    z = torch.tensor(x, dtype=torch.float32)
    lib.steps = 1
    with torch.no_grad():
        for _ in range(K):
            z = lib.hard(z, [s])
            out.append(z.numpy().astype(np.float64))
    return np.stack(out)


def fit_linear(X, labels, classes, steps=400, seed=0):
    torch.manual_seed(seed)
    Xt = torch.tensor(np.c_[X, np.ones(len(X))], dtype=torch.float32)
    yt = torch.tensor(labels, dtype=torch.long)
    W = torch.zeros(Xt.shape[1], classes, requires_grad=True)
    opt = torch.optim.Adam([W], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(Xt @ W, yt).backward()
        opt.step()
    return W.detach().numpy()


def predict_linear(W, X):
    return np.argmax(np.c_[X, np.ones(len(X))] @ W, axis=1)


def loop_with(predicate, U):
    """Run the loop on precomputed unrolled learned states U (K+1, n, d) with a go/stop predicate on states."""
    n = U.shape[1]
    k = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    for j in range(K):
        go = alive & predicate(U[j])
        k += go
        alive = go
    return U[k, np.arange(n)], k


def one(w):
    torch.set_num_threads(1)
    cfg, mixed, st, plan, canonical, btasks, meta = bs.stream(w, 0, b1o.N_BRANCH)
    model = b1o.build(BranchGatedLearner, cfg, plan, branch_plan={t.task_id: 'input' for t in btasks},
                      branch_seed=b1o.BRANCH_SEED)
    for t in st:
        model.begin_task(t.task_id)
    state = torch.load(f'artifacts/b1_online/work/GATED_w{w}_s0/lifetime/model.pt', weights_only=True)['model_state_dict']
    model.load_state_dict(state, strict=False)   # only the unused novel-composition probe code is unexpected
    model.eval()
    ref = json.loads(Path(f'artifacts/b1_online/cells/GATED_w{w}_s0.json').read_text())['record']
    g0 = score(model, SimpleNamespace(tasks=canonical))['median'] == ref['wake_canonical_median']
    g1 = None
    if LIBRARY == 'final':
        pool = o2d.reservoir(st, w, 0, b1o.MEMORY)
        branch_ids = {t.task_id for t in btasks}
        dr.reroute(model, [t for t in st if t.task_id not in branch_ids], plan, pool)
        b1o.refit_branches(model, btasks, pool, w)
        b1o.sleep(cfg, model, pool, [t.task_id for t in st], [1941, w, 0, b1o.MEMORY],
                  model.branch_parameters(branch_ids))
        g1 = (score(model, SimpleNamespace(tasks=canonical))['per_task'] == ref['canonical_per_task']
              and score(model, SimpleNamespace(tasks=btasks))['per_task'] == ref['branch_per_task'])
    lib = FrozenLibrary(model)
    teacher = mixed.library
    d = cfg.world.state_dim
    xm = np.random.default_rng(np.random.SeedSequence([5679, w])).normal(size=(1024, d))
    smap, match = {}, {}
    for k in range(len(teacher)):
        y = teacher[k](xm)
        errs = [nmse(unrolled(lib, s, xm)[1], y) for s in range(lib.slots)]
        smap[k], match[k] = int(np.argmin(errs)), float(min(errs))
    rows = []
    for i in range(TASKS):
        rng = np.random.default_rng(np.random.SeedSequence([5680, w, i]))
        wv = rng.normal(size=d)
        wv /= np.linalg.norm(wv)
        kp = int(rng.integers(0, len(teacher)))
        P, s = teacher[kp], smap[kp]
        xs_all, xq = rng.normal(size=(max(SIZES), d)), rng.normal(size=(QUERY, d))
        ys_all, ks_all = teacher_loop(P, wv, xs_all)
        yq, kq = teacher_loop(P, wv, xq)
        Uq = unrolled(lib, s, xq)
        row = {'task': i, 'P': kp, 'slot': s, 'count_hist_query': np.bincount(kq, minlength=K + 1).tolist(),
               'sizes': {}}
        oracle_pred, _ = loop_with(lambda Z: Z @ wv > 0, Uq)
        row['LOOP_ORACLE'] = nmse(oracle_pred, yq)
        for n in SIZES:
            xs, ys, ks = xs_all[:n], ys_all[:n], ks_all[:n]
            Us = unrolled(lib, s, xs)
            inferred = np.argmin(((Us - ys[None]) ** 2).sum(-1), axis=0)
            res = {'inferred_count_accuracy': float(np.mean(inferred == ks))}
            # REFUSAL: best single straight-line route, chosen on support
            best_L = int(np.argmin([np.mean((Us[L] - ys) ** 2) for L in range(K + 1)]))
            best_rep = float(np.mean((Us[best_L] - ys) ** 2))
            xs_t, ys_t = torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)
            r3 = dr.exhaustive_route(lib, xs_t, ys_t, 3)
            lib.steps = 3
            with torch.no_grad():
                sup3 = float(torch.mean((lib.hard(xs_t, r3) - ys_t) ** 2))
                q3 = lib.hard(torch.tensor(xq, dtype=torch.float32), r3).numpy().astype(np.float64)
            res['REFUSAL'] = nmse(q3, yq) if sup3 < best_rep else nmse(Uq[best_L], yq)
            # KWAY_INPUT impostor: a linear classifier on the input picks the unrolled length
            Wk = fit_linear(xs, inferred, K + 1, seed=i)
            Lq = predict_linear(Wk, xq)
            res['KWAY_INPUT'] = nmse(Uq[Lq, np.arange(QUERY)], yq)
            # LOOP_LEARNED: binary go/stop on learned states, from the inferred counts
            Xh, yh = [], []
            for j in range(K):
                at = inferred >= j
                Xh.append(Us[j][at])
                yh.append((inferred[at] > j).astype(int))
            Wh = fit_linear(np.concatenate(Xh), np.concatenate(yh), 2, seed=i)
            pred, _ = loop_with(lambda Z: predict_linear(Wh, Z) == 1, Uq)
            res['LOOP_LEARNED'] = nmse(pred, yq)
            row['sizes'][str(n)] = res
        rows.append(row)
    return {'world': w, 'library': LIBRARY, 'G0_reload_reproduces': g0, 'G1_final_reproduces': g1, 'slot_match_nmse': match, 'slot_map': smap, 'rows': rows}


def main():
    with ProcessPoolExecutor(max_workers=3) as ex:
        worlds = list(ex.map(one, WORLDS))
    rows = [r for wd in worlds for r in wd['rows']]
    hist = np.sum([r['count_hist_query'] for r in rows], axis=0)
    summ = {'count_distribution': (hist / hist.sum()).round(4).tolist(),
            'LOOP_ORACLE_median': float(np.median([r['LOOP_ORACLE'] for r in rows]))}
    for n in SIZES:
        k = str(n)
        arms = ('REFUSAL', 'KWAY_INPUT', 'LOOP_LEARNED')
        summ[f'n{n}'] = {'median_query_nmse': {a: float(np.median([r['sizes'][k][a] for r in rows])) for a in arms},
                         'LOOP_below_KWAY': sum(r['sizes'][k]['LOOP_LEARNED'] < r['sizes'][k]['KWAY_INPUT'] for r in rows),
                         'LOOP_below_0.05': sum(r['sizes'][k]['LOOP_LEARNED'] < 0.05 for r in rows),
                         'KWAY_below_0.05': sum(r['sizes'][k]['KWAY_INPUT'] < 0.05 for r in rows),
                         'inferred_count_accuracy_median': float(np.median([r['sizes'][k]['inferred_count_accuracy']
                                                                            for r in rows])),
                         'tasks': len(rows)}
    out = {'tier': 0, 'descriptive': True, 'K': K, 'library': LIBRARY, 'summary': summ, 'worlds': worlds, 'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps({'G0': [wd['G0_reload_reproduces'] for wd in worlds],
                      'G1': [wd['G1_final_reproduces'] for wd in worlds],
                      'slot_match_max': [max(wd['slot_match_nmse'].values()) for wd in worlds], **summ}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
