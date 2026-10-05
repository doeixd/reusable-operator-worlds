"""B1g Tier 1 (exploratory): can GRADIENT training, as wake uses, find a branch's structure and decision on a frozen
formed vocabulary? B1 found them by search; this replaces the search with gradient descent on soft codes.

Plan: `B1G_GRADIENT_BRANCH_PLAN.md`. Same libraries, tasks and seeds as B1 (`b1_branch_2x2`: rebuilt depth-4
vocabularies of D1 worlds 30-32; 192 tasks seed [5400, w, i]; 2,048 support, 256 query), so every task is paired
with B1's search result.

Gated branch model per task (on `FrozenLibrary`, library frozen): two soft route codes c1, c2 of shape
(length, slots) and a linear gate g(s) = sigmoid(w.s + b) on the decision state s; prediction
g(s) * F(x; softmax(c1 / T)) + (1 - g(s)) * F(x; softmax(c2 / T)). For INPUT tasks s = x; for MID tasks
s = F_1(x; softmax(c1[0] / T)), the output of c1's first step (the decision is made after the shared first step).
Training: support MSE, Adam lr 0.05 on codes and gate (SO1R's OPT_LR), temperature 1.0 -> 0.1 (SO1R's schedule),
STEPS updates, full batch. Symmetry breaking: codes initialized N(0, 0.1) from seed [5500, w, i, r]; gate zero.
RESTARTS restarts; the restart with the lowest final support loss is kept (support only).
Hardening: argmax routes, gate sign; query NMSE. Program length is given (as in B1).
Switch check (test): with the gate pinned to route 1 everywhere, the model equals the single-route forward bitwise.
"""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from row.experiments import b1_branch_2x2 as b1
from row.experiments import census_b0_branch_necessity as b0
from row.experiments import census_c0_repeated_circuit as c0
from row.experiments import dn_stream as dn
from row.experiments.audit_so1r_route_only import OPT_LR, T_END, T_START, FrozenLibrary
from row.experiments.so1_storage import atomic_json, digest, fingerprint, now
from row.rotated_world import generate_rotated_world

PLAN = Path('B1G_GRADIENT_BRANCH_PLAN.md')
OUTPUT = Path('reports/b1g_gradient_branch.json')
STEPS = 600
RESTARTS = 4
THRESHOLD = 0.05


def soft_forward(lib, x, coeffs):
    lib.steps = coeffs.shape[0]
    return lib.forward(x, coeffs)


def gated_predict(lib, x, c1, c2, w, b, T, mid, hard=False):
    if hard:
        p1 = torch.nn.functional.one_hot(torch.argmax(c1, -1), lib.slots).float()
        p2 = torch.nn.functional.one_hot(torch.argmax(c2, -1), lib.slots).float()
    else:
        p1, p2 = torch.softmax(c1 / T, -1), torch.softmax(c2 / T, -1)
    s = soft_forward(lib, x, p1[:1]) if mid else x
    logit = s @ w + b
    g = (logit > 0).float() if hard else torch.sigmoid(logit)
    return g.unsqueeze(1) * soft_forward(lib, x, p1) + (1 - g.unsqueeze(1)) * soft_forward(lib, x, p2)


def fit(lib, xs, ys, length, mid, seed):
    g = torch.Generator().manual_seed(int(np.random.SeedSequence(seed).generate_state(1)[0]))
    c1 = (0.1 * torch.randn(length, lib.slots, generator=g)).requires_grad_(True)
    c2 = (0.1 * torch.randn(length, lib.slots, generator=g)).requires_grad_(True)
    w = torch.zeros(xs.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([c1, c2, w, b], lr=OPT_LR)
    for step in range(STEPS):
        T = T_START * (T_END / T_START) ** (step / max(1, STEPS - 1))
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((gated_predict(lib, xs, c1, c2, w, b, T, mid) - ys) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        hard_support = float(torch.mean((gated_predict(lib, xs, c1, c2, w, b, T_END, mid, hard=True) - ys) ** 2))
    return hard_support, c1.detach(), c2.detach(), w.detach(), b.detach()


def task(teacher, smap, lib, w_seed, i, d, b1_record):
    rng = np.random.default_rng(np.random.SeedSequence([5400, w_seed, i]))   # B1's construction, verbatim draws
    ops = sorted(smap)
    wvec = rng.normal(size=d)
    wvec /= np.linalg.norm(wvec)
    A = [int(o) for o in rng.choice(ops, size=2)]
    B = list(A)
    while B == A:
        B = [int(o) for o in rng.choice(ops, size=2)]
    mid = i >= b1.TASKS // 2
    C = [int(rng.choice(ops))] if mid else []
    xs, xq = rng.normal(size=(b1.N_SUPPORT, d)), rng.normal(size=(b1.N_QUERY, d))

    def teach(x):
        z = c0.teacher_apply(teacher, C, x) if C else np.array(x, dtype=np.float64)
        p = z @ wvec > 0
        return np.where(p[:, None], c0.teacher_apply(teacher, A, z), c0.teacher_apply(teacher, B, z)), p

    ys, _ = teach(xs)
    yq, _ = teach(xq)
    xs_t, ys_t = torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)
    xq_t = torch.tensor(xq, dtype=torch.float32)
    length = len(C) + 2
    fits = [fit(lib, xs_t, ys_t, length, mid, [5500, w_seed, i, r]) for r in range(RESTARTS)]
    best = min(range(RESTARTS), key=lambda r: fits[r][0])
    _, c1, c2, w, b = fits[best]
    with torch.no_grad():
        pred = gated_predict(lib, xq_t, c1, c2, w, b, T_END, mid, hard=True).numpy().astype(np.float64)
    slot = lambda prog: [smap[k]['slot'] for k in prog]   # noqa: E731
    routes = {tuple(torch.argmax(c1, -1).tolist()), tuple(torch.argmax(c2, -1).tolist())}
    return {'variant': 'MID' if mid else 'INPUT', 'GRAD': b0.nmse(pred, yq), 'B1_LS_LP': b1_record['LS_LP'],
            'restart_support_losses': [f[0] for f in fits], 'chosen_restart': best,
            'routes_match_oracle': routes == {tuple(slot(C + A)), tuple(slot(C + B))},
            'routes_distinct': len(routes) == 2}


def world(w):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, model, st, plan = c0.rebuilt_library(4, w)
    smap = c0.slot_map(model, st, plan)
    teacher = generate_rotated_world(dn.config(4, w).world).library
    lib = FrozenLibrary(model)
    ref = json.loads(b1.OUTPUT.read_text())['worlds'][str(w)]['tasks']
    return w, {'tasks': [task(teacher, smap, lib, w, i, cfg.world.state_dim, ref[i]) for i in range(b1.TASKS)],
               'seconds': time.perf_counter() - started}


def label(k, n):
    if k >= 0.8 * n:
        return 'GRADIENT_FINDS'
    return 'PARTIAL' if k >= 0.5 * n else 'GRADIENT_FAILS'


def summarize(worlds):
    ts = [t for w in worlds.values() for t in w['tasks']]
    out = {}
    for v in ('ALL', 'INPUT', 'MID'):
        sub = [t for t in ts if v == 'ALL' or t['variant'] == v]
        out[v] = {'tasks': len(sub),
                  'GRAD_passes': sum(t['GRAD'] < THRESHOLD for t in sub),
                  'GRAD_median': statistics.median(t['GRAD'] for t in sub),
                  'B1_passes': sum(t['B1_LS_LP'] < THRESHOLD for t in sub),
                  'both_pass': sum(t['GRAD'] < THRESHOLD and t['B1_LS_LP'] < THRESHOLD for t in sub),
                  'grad_only': sum(t['GRAD'] < THRESHOLD <= t['B1_LS_LP'] for t in sub),
                  'search_only': sum(t['B1_LS_LP'] < THRESHOLD <= t['GRAD'] for t in sub),
                  'routes_match_oracle': sum(t['routes_match_oracle'] for t in sub),
                  'routes_distinct': sum(t['routes_distinct'] for t in sub)}
    out['label'] = label(out['ALL']['GRAD_passes'], out['ALL']['tasks'])
    return out


def protocol():
    return {'id': 'b1g-gradient-branch-v1', 'tier': 1, 'exploratory': True, 'steps': STEPS, 'restarts': RESTARTS,
            'lr': OPT_LR, 'T': [T_START, T_END], 'threshold': THRESHOLD,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml', b1.OUTPUT.as_posix())},
            'implementation_sha256': digest(Path(__file__)), 'b1_sha256': digest(Path(b1.__file__))}


def main():
    started = time.time()
    proto = protocol()
    with ProcessPoolExecutor(max_workers=3) as pool:
        worlds = dict(pool.map(world, b1.WORLDS))
    out = {'protocol': proto, 'protocol_sha256': fingerprint(proto), 'complete': True,
           'summary': summarize(worlds), 'worlds': {str(w): v for w, v in worlds.items()},
           'seconds': time.time() - started, 'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps(out['summary'], indent=1))


if __name__ == '__main__':
    main()
