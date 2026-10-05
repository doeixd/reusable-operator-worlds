"""B1 Tier 1 (exploratory): the branch 2x2 (issue #2 section 4) on a frozen formed vocabulary.

Plan: `B1_BRANCH_2X2_PLAN.md`. Libraries: the rebuilt depth-4 libraries of D1 worlds 30, 31, 32 (C0's
construction); teacher operation -> learned slot from the anchors (C0's map) is used ONLY by the oracle-structure
arms. Tasks `IF(p(state), A, B)` as in B0 (hyperplane predicate; A != B length-2 teacher programs; INPUT tasks branch
on x, MID tasks on the state after a shared first operation C), new seeds [5400, w, i], 64 tasks per world, 2,048
support and 256 query examples.

Arms (query NMSE; a task passes below 0.05):
- REFUSAL: the best single straight-line route of lengths 1-4 on support (B0's construction).
- OS_OP: oracle structure, oracle predicate (B0's oracle branch; the ceiling).
- OS_LP: oracle structure, learned predicate (B0b's construction at 2,048 support).
- LS_OP: learned structure, oracle predicate: support split by the TRUE predicate; for each side, the best route of
  the task's program length (2 for INPUT, 3 for MID) by exhaustive search on that side's examples; query routed by
  the true predicate.
- LS_LP: both learned. Per-example errors E[i, r] of every route of the task's program length on support; candidate
  routes = those that are the per-example argmin on >= 2% of support examples; the pair (r1, r2) minimizing
  sum_i min(E[i, r1], E[i, r2]) over all candidate pairs; labels = which of the pair fits each example better; the
  predicate is a linear logistic classifier on the decision state (x for INPUT; for MID, the output of the shared
  learned first step, taken as the first slot of r1 when r1 and r2 share it, otherwise x); query routed by it.
  Program length is given (an assumption stated in the plan); identities, predicate and split are not.
"""
from __future__ import annotations

import itertools
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from row.experiments import census_b0_branch_necessity as b0
from row.experiments import census_c0_repeated_circuit as c0
from row.experiments import deep_reroute as dr
from row.experiments import dn_stream as dn
from row.experiments.audit_so1r_route_only import FrozenLibrary, unflatten
from row.experiments.so1_storage import atomic_json, digest, fingerprint, now
from row.rotated_world import generate_rotated_world

PLAN = Path('B1_BRANCH_2X2_PLAN.md')
OUTPUT = Path('reports/b1_branch_2x2.json')
WORLDS = (30, 31, 32)
TASKS = 64
N_SUPPORT, N_QUERY = 2048, 256
THRESHOLD = 0.05
CANDIDATE_SHARE = 0.02
CHUNK_BYTES = 64 * 2 ** 20   # last-step chunk budget for the refusal search (memory-bounded at 2,048 examples)
ARMS = ('REFUSAL', 'OS_OP', 'OS_LP', 'LS_OP', 'LS_LP')


def per_example_errors(lib, x, y, length):
    """(n, slots**length) squared error of every hard route, lexicographic order (as all_route_support_mse)."""
    lib.steps = length
    with torch.no_grad():
        n, d = x.shape
        z = x
        for _ in range(length):
            z = lib.candidates(z.reshape(-1, d)).reshape(n, -1, d)
        return torch.sum((z - y.unsqueeze(1)) ** 2, dim=2)


def best_pair(E):
    winners = torch.argmin(E, dim=1)
    counts = torch.bincount(winners, minlength=E.shape[1])
    cand = torch.nonzero(counts >= max(1, int(CANDIDATE_SHARE * E.shape[0]))).flatten().tolist()
    if len(cand) < 2:
        cand = torch.topk(counts, 2).indices.tolist()
    best = None
    for a, b in itertools.combinations(cand, 2):
        cost = float(torch.minimum(E[:, a], E[:, b]).sum())
        if best is None or cost < best[0]:
            best = (cost, a, b)
    return best[1], best[2], len(cand)


def task(teacher, smap, lib, w, i, d):
    rng = np.random.default_rng(np.random.SeedSequence([5400, w, i]))
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
    xs_t, ys_t = torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)
    run, nm = b0.run_route, b0.nmse
    slot = lambda prog: [smap[k]['slot'] for k in prog]   # noqa: E731
    length = len(C) + 2
    out = {'variant': 'MID' if mid else 'INPUT'}
    # REFUSAL
    best = None
    for L in range(1, 5):
        # prefix-split search (bounded memory at 2,048 examples; equal to enum_route's argmin, tests/test_deep_reroute)
        lib.steps = L
        if L == 1:   # 12 routes; the split search needs at least one prefix step
            r = dr.exhaustive_route(lib, xs_t, ys_t, 1)
        else:
            chunk = max(1, CHUNK_BYTES // (xs_t.shape[0] * lib.slots * xs_t.shape[1] * 4))
            r = unflatten(int(torch.argmin(dr.split_route_mse(lib, xs_t, ys_t, L, chunk=chunk))), lib.slots, L)
        m = float(np.mean((run(lib, r, xs) - ys) ** 2))
        if best is None or m < best[0]:
            best = (m, r)
    out['REFUSAL'] = nm(run(lib, best[1], xq), yq)
    # oracle structure
    rA, rB = slot(C + A), slot(C + B)
    out['OS_OP'] = nm(np.where(pq[:, None], run(lib, rA, xq), run(lib, rB, xq)), yq)
    eA = np.sum((run(lib, rA, xs) - ys) ** 2, axis=1)
    eB = np.sum((run(lib, rB, xs) - ys) ** 2, axis=1)
    zs = run(lib, slot(C), xs) if C else xs
    zq = run(lib, slot(C), xq) if C else xq
    wl, bl = b0.fit_logistic(zs, (eA < eB).astype(np.float64))
    out['OS_LP'] = nm(np.where((zq @ wl + bl > 0)[:, None], run(lib, rA, xq), run(lib, rB, xq)), yq)
    # learned structure, oracle predicate
    sides = []
    for mask in (ps, ~ps):
        sides.append(dr.exhaustive_route(lib, xs_t[torch.tensor(mask)], ys_t[torch.tensor(mask)], length))
    out['LS_OP'] = nm(np.where(pq[:, None], run(lib, sides[0], xq), run(lib, sides[1], xq)), yq)
    out['LS_OP_structure_exact'] = sides[0] == rA and sides[1] == rB
    # both learned
    E = per_example_errors(lib, xs_t, ys_t, length)
    a, b, ncand = best_pair(E)
    r1, r2 = unflatten(a, lib.slots, length), unflatten(b, lib.slots, length)
    labels = (E[:, a] < E[:, b]).numpy().astype(np.float64)
    shared = r1[:len(C)] == r2[:len(C)] if C else True
    zs2 = run(lib, r1[:len(C)], xs) if (C and shared) else xs
    zq2 = run(lib, r1[:len(C)], xq) if (C and shared) else xq
    w2, b2 = b0.fit_logistic(zs2, labels)
    out['LS_LP'] = nm(np.where((zq2 @ w2 + b2 > 0)[:, None], run(lib, r1, xq), run(lib, r2, xq)), yq)
    out['LS_LP_pair_matches_oracle'] = {tuple(r1), tuple(r2)} == {tuple(rA), tuple(rB)}
    out['LS_LP_candidates'] = ncand
    return out


def world(w):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, model, st, plan = c0.rebuilt_library(4, w)
    smap = c0.slot_map(model, st, plan)
    teacher = generate_rotated_world(dn.config(4, w).world).library
    lib = FrozenLibrary(model)
    return w, {'tasks': [task(teacher, smap, lib, w, i, cfg.world.state_dim) for i in range(TASKS)],
               'seconds': time.perf_counter() - started}


def label(k, n):
    if k >= 0.8 * n:
        return 'LEARNABLE'
    return 'PARTIAL' if k >= 0.5 * n else 'NOT_LEARNABLE'


def summarize(worlds):
    ts = [t for w in worlds.values() for t in w['tasks']]
    out = {}
    for variant in ('ALL', 'INPUT', 'MID'):
        sub = [t for t in ts if variant == 'ALL' or t['variant'] == variant]
        out[variant] = {'tasks': len(sub)} | {
            arm: {'passes': sum(t[arm] < THRESHOLD for t in sub), 'median': statistics.median(t[arm] for t in sub)}
            for arm in ARMS}
        out[variant]['LS_LP_pair_matches_oracle'] = sum(t['LS_LP_pair_matches_oracle'] for t in sub)
        out[variant]['LS_OP_structure_exact'] = sum(t['LS_OP_structure_exact'] for t in sub)
    n = out['ALL']['tasks']
    out['label'] = label(out['ALL']['LS_LP']['passes'], n)
    return out


def protocol():
    return {'id': 'b1-branch-2x2-v1', 'tier': 1, 'exploratory': True, 'worlds': list(WORLDS), 'tasks': TASKS,
            'support': N_SUPPORT, 'query': N_QUERY, 'threshold': THRESHOLD, 'candidate_share': CANDIDATE_SHARE,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml')},
            'implementation_sha256': digest(Path(__file__)), 'b0_sha256': digest(Path(b0.__file__)),
            'c0_sha256': digest(Path(c0.__file__)), 'deep_reroute_sha256': digest(Path(dr.__file__))}


def main():
    started = time.time()
    proto = protocol()
    with ProcessPoolExecutor(max_workers=3) as pool:
        worlds = dict(pool.map(world, WORLDS))
    out = {'protocol': proto, 'protocol_sha256': fingerprint(proto), 'complete': True,
           'summary': summarize(worlds), 'worlds': {str(w): v for w, v in worlds.items()},
           'seconds': time.time() - started, 'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps(out['summary'], indent=1))


if __name__ == '__main__':
    main()
