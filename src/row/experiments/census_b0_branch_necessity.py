"""B0 Tier 0 census (descriptive; SUCCESSOR_LADDER.md Track C, first control-flow rung): does a branch task
REQUIRE state-conditional routing, and is the branch decision recoverable from support data?

Library: rebuilt depth-4 libraries from D1 worlds 30, 31, 32 (C0's construction: end-of-stream re-route + sleep on
the SHUFFLED4 terminal); teacher operation -> learned slot from the single-operation anchors (C0's map).

Tasks (32 per world, seed [5300, w, i]), on 128 support and 256 query inputs drawn N(0, I):
- INPUT branch (i < 16): y = A(x) if w.x > 0 else B(x);
- MID branch (i >= 16): z = C(x) for one teacher operation C; y = A(z) if w.z > 0 else B(z).
A and B are different random length-2 teacher programs; w is a random unit vector. Teacher executes the targets.

Measured per task (query NMSE unless stated):
- BEST_SINGLE: the best single straight-line route over lengths 1-4, chosen exhaustively on support. This is the
  most the current learner can reach: it must use one route per task (the REFUSAL arm).
- ORACLE_BRANCH: the mapped learned routes, branch chosen per example by the TRUE predicate (ceiling).
- LEARNED_PREDICATE: branch structure given (the two mapped routes), predicate learned from support only: each
  support example is labelled by which branch route fits it better, and a linear logistic classifier on the state
  the decision is made from (x, or the learned prefix output C(x)) is fitted; query examples are routed by it.
  Also its query accuracy against the true predicate (evaluated on teacher states).

Reading, stated before running: NECESSITY if median BEST_SINGLE >= 0.05 and >= 5x median ORACLE_BRANCH while
median ORACLE_BRANCH < 0.05; OPPORTUNITY (for B1, structure known) if median LEARNED_PREDICATE < 0.05. Reported
separately for INPUT and MID tasks. Descriptive only. Output `reports/b0_branch_necessity_census.json`.
"""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from row.experiments import census_c0_repeated_circuit as c0
from row.experiments import deep_reroute as dr
from row.experiments import dn_stream as dn
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now
from row.rotated_world import generate_rotated_world

OUTPUT = Path('reports/b0_branch_necessity_census.json')
WORLDS = (30, 31, 32)
TASKS = 32
N_SUPPORT, N_QUERY = 128, 256
MAX_SINGLE = 4


def run_route(lib, route, x):
    lib.steps = len(route)
    with torch.no_grad():
        return lib.hard(torch.as_tensor(x, dtype=torch.float32), list(route)).numpy().astype(np.float64)


def nmse(pred, y):
    return float(np.mean((pred - y) ** 2) / max(np.var(y), 1e-12))


def fit_logistic(z, labels, steps=500):
    zt = torch.tensor(z, dtype=torch.float32)
    yt = torch.tensor(labels, dtype=torch.float32)
    w = torch.zeros(z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(zt @ w + b, yt) + 1e-4 * (w ** 2).sum()
        loss.backward()
        opt.step()
    return w.detach().numpy(), float(b.detach())


def task(teacher, smap, lib, w, i, d):
    rng = np.random.default_rng(np.random.SeedSequence([5300, w, i]))
    ops = sorted(smap)
    wvec = rng.normal(size=d)
    wvec /= np.linalg.norm(wvec)
    A = [int(o) for o in rng.choice(ops, size=2)]
    B = list(A)
    while B == A:
        B = [int(o) for o in rng.choice(ops, size=2)]
    mid = i >= TASKS // 2
    C = [int(rng.choice(ops))] if mid else []
    xs, xq = rng.normal(size=(N_SUPPORT, d)), rng.normal(size=(N_QUERY, d))

    def teach(x):
        z = c0.teacher_apply(teacher, C, x) if C else np.array(x, dtype=np.float64)
        p = z @ wvec > 0
        return np.where(p[:, None], c0.teacher_apply(teacher, A, z), c0.teacher_apply(teacher, B, z)), p

    ys, ps = teach(xs)
    yq, pq = teach(xq)
    slot = lambda prog: [smap[k]['slot'] for k in prog]   # noqa: E731
    rA, rB = slot(C + A), slot(C + B)
    # refusal arm: the best single straight-line route, any length 1..4, chosen on support
    best = None
    for L in range(1, MAX_SINGLE + 1):
        route = dr.exhaustive_route(lib, torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32), L)
        s_mse = float(np.mean((run_route(lib, route, xs) - ys) ** 2))
        if best is None or s_mse < best[0]:
            best = (s_mse, route)
    best_single = nmse(run_route(lib, best[1], xq), yq)
    oracle = nmse(np.where(pq[:, None], run_route(lib, rA, xq), run_route(lib, rB, xq)), yq)
    # learned predicate, structure given
    eA = np.sum((run_route(lib, rA, xs) - ys) ** 2, axis=1)
    eB = np.sum((run_route(lib, rB, xs) - ys) ** 2, axis=1)
    labels = (eA < eB).astype(np.float64)
    zs = run_route(lib, slot(C), xs) if C else xs
    zq = run_route(lib, slot(C), xq) if C else xq
    wl, bl = fit_logistic(zs, labels)
    pred = zq @ wl + bl > 0
    learned = nmse(np.where(pred[:, None], run_route(lib, rA, xq), run_route(lib, rB, xq)), yq)
    return {'variant': 'MID' if mid else 'INPUT', 'A': A, 'B': B, 'C': C, 'best_single_route': best[1],
            'best_single_nmse': best_single, 'oracle_branch_nmse': oracle, 'learned_predicate_nmse': learned,
            'predicate_accuracy': float(np.mean(pred == pq)), 'support_label_agreement': float(np.mean(labels == ps))}


def world(w):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, model, st, plan = c0.rebuilt_library(4, w)
    smap = c0.slot_map(model, st, plan)
    teacher = generate_rotated_world(dn.config(4, w).world).library
    lib = FrozenLibrary(model)
    tasks = [task(teacher, smap, lib, w, i, cfg.world.state_dim) for i in range(TASKS)]
    return w, {'tasks': tasks, 'slot_map': {str(k): v for k, v in smap.items()}, 'seconds': time.perf_counter() - started}


def main():
    started = time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        worlds = dict(pool.map(world, WORLDS))
    summary = {}
    for variant in ('INPUT', 'MID'):
        ts = [t for w in worlds.values() for t in w['tasks'] if t['variant'] == variant]
        med = {k: statistics.median(t[k] for t in ts) for k in
               ('best_single_nmse', 'oracle_branch_nmse', 'learned_predicate_nmse', 'predicate_accuracy',
                'support_label_agreement')}
        necessity = med['oracle_branch_nmse'] < 0.05 and med['best_single_nmse'] >= max(0.05, 5 * med['oracle_branch_nmse'])
        summary[variant] = med | {'tasks': len(ts), 'NECESSITY': necessity,
                                  'OPPORTUNITY': med['learned_predicate_nmse'] < 0.05,
                                  'best_single_below_0.05': sum(t['best_single_nmse'] < 0.05 for t in ts),
                                  'learned_predicate_below_0.05': sum(t['learned_predicate_nmse'] < 0.05 for t in ts)}
    out = {'tier': 0, 'descriptive': True, 'summary': summary, 'worlds': {str(w): v for w, v in worlds.items()},
           'seconds': time.time() - started, 'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
