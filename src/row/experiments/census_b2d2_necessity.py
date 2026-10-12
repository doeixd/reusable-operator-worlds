"""B2-depth2 Tier 0 necessity/opportunity census for loops with a DEPTH-2 BODY (descriptive; written before any plan).

Task: z = x; repeat at most K times: if w.z <= 0 stop, else z = P2(P1(z)); y = z. P1 != P2 two teacher primitives,
w a random unit hyperplane, K = 6 (so up to 12 operator applications, past C0's clean ~8: the ceiling arm measures
what that costs). The repeat count k(x) in 0..K varies per input.

Vocabularies: B2-hard v2's ONESTEP FINAL libraries (worlds 57-59, stream 0), rebuilt from the saved wake terminal with
the run's exact end of stream (re-route, ONE_STEP data test, sleep). Gates: G0 the reload reproduces the recorded wake
canonical / loop / anchor medians; G1 the rebuilt library's sha256 equals the recorded final `library_sha256`.
Teacher primitive -> learned slot by functional matching (one application, 1,024 inputs; teacher used only to build
the oracle structures). 36 tasks per world (draws [5900, world, i]), support n in (128, 512), query 1,024.
Arms (query NMSE; every learned arm sees support only, counts inferred as argmin_L ||body^L x - y||):
- REFUSAL: best single straight route on support: exhaustive depth <= 3, or [s1 s2]^L for L = 0..K on the matched body;
- KWAY_INPUT (unrolled impostor): [s1 s2]^L, a linear classifier on the INPUT picks L;
- LOOP1_BEST (depth-1-body impostor): the best ONE-slot loop, body s chosen over all slots by support error with
  inferred counts up to 2K, then a linear halting predicate on its states;
- LOOP2_LEARNED: the depth-2 loop on the matched body with a learned linear halting predicate;
- LOOP2_SEARCH: the depth-2 body chosen over all slots^2 bodies by support error with inferred counts (findability),
  then halting learned as LOOP2_LEARNED; records agreement with the matched body;
- LOOP2_ORACLE: matched body with the true hyperplane (ceiling of the loop form on this vocabulary).
Also: count distribution, inferred-count accuracy, and the oracle error split by true count (execution drift).
Output `reports/b2d2_necessity_census.json`.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import b2_hard as bh
from row.experiments import b2_hard_v2 as bv
from row.experiments import census_b2_iteration_necessity as c2
from row.experiments import census_b2h_root as cr
from row.experiments import deep_reroute as dr
from row.experiments import loop_stream as ls
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now

OUTPUT = Path('reports/b2d2_necessity_census.json')
WORLDS = (57, 58, 59)
TASKS = 36
K = 6
SIZES = (128, 512)
QUERY = 1024
nmse, fit_linear, predict_linear = c2.nmse, c2.fit_linear, c2.predict_linear


def teacher_loop2(P1, P2, w, x):
    z = np.array(x, dtype=np.float64)
    k = np.zeros(len(z), dtype=int)
    for _ in range(K):
        go = z @ w > 0
        if not go.any():
            break
        z[go] = P2(P1(z[go]))
        k += go
    return z, k


def unrolled(lib, body, x, reps):
    """body^L x for L = 0..reps, as an array (reps+1, n, d)."""
    out = [np.asarray(x, dtype=np.float64)]
    z = torch.tensor(x, dtype=torch.float32)
    lib.steps = len(body)
    with torch.no_grad():
        for _ in range(reps):
            z = lib.hard(z, list(body))
            out.append(z.numpy().astype(np.float64))
    return np.stack(out)


def run_loop(predicate, U, reps):
    n = U.shape[1]
    k = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    for j in range(reps):
        go = alive & predicate(U[j])
        k += go
        alive = go
    return U[k, np.arange(n)], k


def inferred(U, y):
    return np.argmin(((U - y[None]) ** 2).sum(-1), axis=0)


def oracle_count_error(U, y):
    """Support MSE if every example stopped at its best count: a lower bound for any halting rule on this body."""
    return float(np.mean(np.min(((U - y[None]) ** 2).mean(-1), axis=0)))


def learned_loop(Us, Uq, ys, reps, seed):
    inf = inferred(Us, ys)
    Xh, yh = [], []
    for j in range(reps):
        at = inf >= j
        Xh.append(Us[j][at])
        yh.append((inf[at] > j).astype(int))
    yh = np.concatenate(yh)
    if yh.min() == yh.max():   # every label the same: a constant rule (always continue or always stop)
        return (Uq[reps] if yh[0] else Uq[0]), inf
    Wh = fit_linear(np.concatenate(Xh), yh, 2, seed=seed)
    pred, _ = run_loop(lambda Z: predict_linear(Wh, Z) == 1, Uq, reps)
    return pred, inf


def rebuild(w, s=0):
    cfg, mixed, st, plan, canonical, ltasks, meta = ls.stream(w, s, bv.N_LOOP)
    loop_ids = {t.task_id for t in ltasks}
    cands = [(i, t) for i, t in enumerate(st) if plan[t.task_id] == 1]
    cand_ids = {t.task_id for _, t in cands}
    anchors = [t for _, t in cands if t.task_id not in loop_ids]
    model = bv.build('ONESTEP', cfg, plan, cand_ids, loop_ids)
    for t in st:
        model.begin_task(t.task_id)
    state = torch.load(bv.ROOT / 'work' / f'ONESTEP_w{w}_s{s}' / 'lifetime' / 'model.pt', weights_only=True)
    model.load_state_dict(state['model_state_dict'], strict=False)   # only the unused probe code is unexpected
    model.eval()
    ref = json.loads((bv.ROOT / 'cells' / f'ONESTEP_w{w}_s{s}.json').read_text())['record']

    def med(ts):
        return score(model, SimpleNamespace(tasks=ts))['median']

    g0 = (med(canonical), med(ltasks), med(anchors)) == (ref['wake_canonical_median'], ref['wake_loop_median'],
                                                        ref['wake_anchor_median'])
    pool = o2d.reservoir(st, w, s, bh.MEMORY)
    dr.reroute(model, [t for t in st if t.task_id not in model.loop_plan], plan, pool)
    cr.one_step_end_of_stream(model, cands, pool, w, 'ALLLOOP')
    bh.sleep_decay(cfg, model, pool, [t.task_id for t in st], [1941, w, s, bh.MEMORY])
    g1 = library_sha(model) == ref['library_sha256']
    return cfg, mixed, model, {'G0_wake_reproduces': g0, 'G1_final_library_reproduces': g1,
                               'final_canonical_median': med(canonical), 'recorded': ref['canonical_median']}


def one(w):
    torch.set_num_threads(1)
    cfg, mixed, model, gates = rebuild(w)
    lib = FrozenLibrary(model)
    teacher = mixed.library
    d = cfg.world.state_dim
    xm = np.random.default_rng(np.random.SeedSequence([5679, w])).normal(size=(1024, d))
    smap, match = {}, {}
    for k in range(len(teacher)):
        y = teacher[k](xm)
        errs = [nmse(unrolled(lib, [s], xm, 1)[1], y) for s in range(lib.slots)]
        smap[k], match[k] = int(np.argmin(errs)), float(min(errs))
    bodies = [(a, b) for a in range(lib.slots) for b in range(lib.slots)]
    rows = []
    for i in range(TASKS):
        rng = np.random.default_rng(np.random.SeedSequence([5900, w, i]))
        wv = rng.normal(size=d)
        wv /= np.linalg.norm(wv)
        p1, p2 = (int(v) for v in rng.choice(len(teacher), size=2, replace=False))
        body = (smap[p1], smap[p2])
        xs_all, xq = rng.normal(size=(max(SIZES), d)), rng.normal(size=(QUERY, d))
        ys_all, ks_all = teacher_loop2(teacher[p1], teacher[p2], wv, xs_all)
        yq, kq = teacher_loop2(teacher[p1], teacher[p2], wv, xq)
        Uq = unrolled(lib, body, xq, K)
        oracle_pred, _ = run_loop(lambda Z: Z @ wv > 0, Uq, K)
        row = {'task': i, 'P': [p1, p2], 'body': list(body),
               'count_hist_query': np.bincount(kq, minlength=K + 1).tolist(),
               'LOOP2_ORACLE': nmse(oracle_pred, yq),
               'oracle_by_count': {str(c): nmse(oracle_pred[kq == c], yq[kq == c]) for c in range(K + 1)
                                   if (kq == c).sum() >= 8},
               'sizes': {}}
        for n in SIZES:
            xs, ys, ks = xs_all[:n], ys_all[:n], ks_all[:n]
            xs_t, ys_t = torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)
            Us = unrolled(lib, body, xs, K)
            res = {'inferred_count_accuracy': float(np.mean(inferred(Us, ys) == ks))}
            # REFUSAL: one straight route
            best_L = int(np.argmin([np.mean((Us[L] - ys) ** 2) for L in range(K + 1)]))
            rep = float(np.mean((Us[best_L] - ys) ** 2))
            r3 = dr.exhaustive_route(lib, xs_t, ys_t, 3)
            lib.steps = 3
            with torch.no_grad():
                sup3 = float(torch.mean((lib.hard(xs_t, r3) - ys_t) ** 2))
                q3 = lib.hard(torch.tensor(xq, dtype=torch.float32), r3).numpy().astype(np.float64)
            res['REFUSAL'] = nmse(q3, yq) if sup3 < rep else nmse(Uq[best_L], yq)
            # KWAY_INPUT: the input picks the unrolled length
            Wk = fit_linear(xs, inferred(Us, ys), K + 1, seed=i)
            res['KWAY_INPUT'] = nmse(Uq[predict_linear(Wk, xq), np.arange(QUERY)], yq)
            # LOOP2_LEARNED
            pred, _ = learned_loop(Us, Uq, ys, K, i)
            res['LOOP2_LEARNED'] = nmse(pred, yq)
            # LOOP2_SEARCH: body by inferred-count support error over all bodies
            errs2 = [oracle_count_error(unrolled(lib, b, xs, K), ys) for b in bodies]
            order = np.argsort(errs2)
            sb = bodies[int(order[0])]
            res['search_body'] = list(sb)
            res['search_body_is_matched'] = sb == body
            res['search_margin'] = float(errs2[order[1]] / max(errs2[order[0]], 1e-12))
            if sb == body:
                res['LOOP2_SEARCH'] = res['LOOP2_LEARNED']
            else:
                pred, _ = learned_loop(unrolled(lib, sb, xs, K), unrolled(lib, sb, xq, K), ys, K, i)
                res['LOOP2_SEARCH'] = nmse(pred, yq)
            # LOOP1_BEST: best one-slot loop with up to 2K iterations
            errs1 = [oracle_count_error(unrolled(lib, [s], xs, 2 * K), ys) for s in range(lib.slots)]
            s1 = int(np.argmin(errs1))
            pred, _ = learned_loop(unrolled(lib, [s1], xs, 2 * K), unrolled(lib, [s1], xq, 2 * K), ys, 2 * K, i)
            res['LOOP1_BEST'] = nmse(pred, yq)
            res['loop1_slot'] = s1
            row['sizes'][str(n)] = res
        rows.append(row)
    return {'world': w, **gates, 'slot_match_nmse': match, 'slot_map': smap,
            'slot_map_injective': len(set(smap.values())) == len(smap), 'rows': rows}


def main():
    with ProcessPoolExecutor(max_workers=3) as ex:
        worlds = list(ex.map(one, WORLDS))
    rows = [r for wd in worlds for r in wd['rows']]
    hist = np.sum([r['count_hist_query'] for r in rows], axis=0)
    by_count = {str(c): float(np.median([r['oracle_by_count'][str(c)] for r in rows if str(c) in r['oracle_by_count']]))
                for c in range(K + 1) if any(str(c) in r['oracle_by_count'] for r in rows)}
    summ = {'count_distribution': (hist / hist.sum()).round(4).tolist(),
            'LOOP2_ORACLE_median': float(np.median([r['LOOP2_ORACLE'] for r in rows])),
            'LOOP2_ORACLE_by_true_count_median': by_count}
    arms = ('REFUSAL', 'KWAY_INPUT', 'LOOP1_BEST', 'LOOP2_LEARNED', 'LOOP2_SEARCH')
    for n in SIZES:
        k = str(n)

        def get(a):
            return [r['sizes'][k][a] for r in rows]

        def below(a, b):
            return sum(r['sizes'][k][a] < r['sizes'][k][b] for r in rows)

        summ[f'n{n}'] = {'median_query_nmse': {a: float(np.median(get(a))) for a in arms},
                         'LOOP2_below_LOOP1': below('LOOP2_LEARNED', 'LOOP1_BEST'),
                         'LOOP2_below_KWAY': below('LOOP2_LEARNED', 'KWAY_INPUT'),
                         'LOOP2_below_REFUSAL': below('LOOP2_LEARNED', 'REFUSAL'),
                         'search_body_matched': sum(get('search_body_is_matched')),
                         'search_margin_median': float(np.median(get('search_margin'))),
                         'inferred_count_accuracy_median': float(np.median(get('inferred_count_accuracy'))),
                         'tasks': len(rows)}
    out = {'tier': 0, 'descriptive': True, 'K': K, 'summary': summ, 'worlds': worlds, 'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps({'gates': [(wd['G0_wake_reproduces'], wd['G1_final_library_reproduces']) for wd in worlds],
                      'slot_match_max': [max(wd['slot_match_nmse'].values()) for wd in worlds],
                      'injective': [wd['slot_map_injective'] for wd in worlds], **summ}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
