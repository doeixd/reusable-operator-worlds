"""B0b Tier 0 diagnostic (POST HOC, written after B0's result; it changes no B0 reading).

B0 found NECESSITY (no single route fits a branch task) but missed its OPPORTUNITY threshold: the learned-predicate
arm reached median NMSE 0.11-0.12 with support labels 100% correct and query predicate accuracy ~95%. Hypothesis:
the miss is predicate SAMPLE SIZE (a linear boundary in 16 dimensions from 128 labelled points), not learnability.
Test: B0's tasks 0..31 on D1 world 30's rebuilt library (B0's construction, identical seeds), the learned-predicate
arm re-run with N_SUPPORT in (128, 512, 2048); 128 must reproduce B0's world-30 values exactly (equivalence check).
Output `reports/b0b_predicate_sample_size.json`.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from row.experiments import census_b0_branch_necessity as b0
from row.experiments import census_c0_repeated_circuit as c0
from row.experiments import dn_stream as dn
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now
from row.rotated_world import generate_rotated_world

OUTPUT = Path('reports/b0b_predicate_sample_size.json')
W = 30
SIZES = (128, 512, 2048)


def learned_only(teacher, smap, lib, w, i, d, n_support):
    """B0's task construction and learned-predicate arm, with the support size as the only change. The extra
    support examples are drawn AFTER B0's draws from the same generator, so n_support = 128 is B0 exactly."""
    rng = np.random.default_rng(np.random.SeedSequence([5300, w, i]))
    ops = sorted(smap)
    wvec = rng.normal(size=d)
    wvec /= np.linalg.norm(wvec)
    A = [int(o) for o in rng.choice(ops, size=2)]
    B = list(A)
    while B == A:
        B = [int(o) for o in rng.choice(ops, size=2)]
    mid = i >= b0.TASKS // 2
    C = [int(rng.choice(ops))] if mid else []
    xs, xq = rng.normal(size=(b0.N_SUPPORT, d)), rng.normal(size=(b0.N_QUERY, d))
    if n_support > b0.N_SUPPORT:
        xs = np.concatenate([xs, rng.normal(size=(n_support - b0.N_SUPPORT, d))])

    def teach(x):
        z = c0.teacher_apply(teacher, C, x) if C else np.array(x, dtype=np.float64)
        p = z @ wvec > 0
        return np.where(p[:, None], c0.teacher_apply(teacher, A, z), c0.teacher_apply(teacher, B, z)), p

    ys, _ = teach(xs)
    yq, pq = teach(xq)
    slot = lambda prog: [smap[k]['slot'] for k in prog]   # noqa: E731
    rA, rB = slot(C + A), slot(C + B)
    eA = np.sum((b0.run_route(lib, rA, xs) - ys) ** 2, axis=1)
    eB = np.sum((b0.run_route(lib, rB, xs) - ys) ** 2, axis=1)
    labels = (eA < eB).astype(np.float64)
    zs = b0.run_route(lib, slot(C), xs) if C else xs
    zq = b0.run_route(lib, slot(C), xq) if C else xq
    wl, bl = b0.fit_logistic(zs, labels)
    pred = zq @ wl + bl > 0
    return {'variant': 'MID' if mid else 'INPUT',
            'learned_predicate_nmse': b0.nmse(np.where(pred[:, None], b0.run_route(lib, rA, xq), b0.run_route(lib, rB, xq)), yq),
            'predicate_accuracy': float(np.mean(pred == pq))}


def main():
    torch.set_num_threads(1)
    started = time.time()
    cfg, model, st, plan = c0.rebuilt_library(4, W)
    smap = c0.slot_map(model, st, plan)
    teacher = generate_rotated_world(dn.config(4, W).world).library
    lib = FrozenLibrary(model)
    res = {n: [learned_only(teacher, smap, lib, W, i, cfg.world.state_dim, n) for i in range(b0.TASKS)] for n in SIZES}
    ref = json.loads(Path(b0.OUTPUT).read_text())['worlds'][str(W)]['tasks']
    equal = all(r['learned_predicate_nmse'] == t['learned_predicate_nmse'] and r['predicate_accuracy'] == t['predicate_accuracy']
                for r, t in zip(res[128], ref))
    summary = {str(n): {v: {'learned_predicate_median': statistics.median(r['learned_predicate_nmse'] for r in res[n] if r['variant'] == v),
                            'predicate_accuracy_median': statistics.median(r['predicate_accuracy'] for r in res[n] if r['variant'] == v),
                            'below_0.05': sum(r['learned_predicate_nmse'] < 0.05 for r in res[n] if r['variant'] == v)}
                        for v in ('INPUT', 'MID')} for n in SIZES}
    out = {'tier': 0, 'post_hoc': True, 'world': W, 'n128_reproduces_b0': equal, 'summary': summary,
           'tasks': {str(n): r for n, r in res.items()}, 'seconds': time.time() - started, 'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps({k: v for k, v in out.items() if k != 'tasks'}, indent=1))


if __name__ == '__main__':
    main()
