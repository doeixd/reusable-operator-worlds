"""Quick Tier 0 check (scratch): is the B2 loop body identifiable from support alone, and do loops win on straight tasks?"""
import json
from types import SimpleNamespace

import numpy as np
import torch

import row.experiments.census_b2_iteration_necessity as c
from row.experiments import b1_online as b1o, branch_stream as bs, deep_reroute as dr, o2d_sleep_memory as o2d
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.models.branch_gated import BranchGatedLearner
from row.world import Program


def loop_fit(lib, s, xs, ys, seed):
    Us = c.unrolled(lib, s, xs)
    inferred = np.argmin(((Us - ys[None]) ** 2).sum(-1), axis=0)
    Xh, yh = [], []
    for j in range(c.K):
        at = inferred >= j
        Xh.append(Us[j][at]); yh.append((inferred[at] > j).astype(int))
    X, Y = np.concatenate(Xh), np.concatenate(yh)
    if len(set(Y.tolist())) < 2:
        const = int(Y[0]) if len(Y) else 0
        pred = lambda Z: np.full(len(Z), const == 1)
    else:
        Wh = c.fit_linear(X, Y, 2, seed=seed)
        pred = lambda Z: c.predict_linear(Wh, Z) == 1
    out, _ = c.loop_with(pred, Us)
    return float(np.mean((out - ys) ** 2)), pred


def main(w):
    torch.set_num_threads(1)
    cfg, mixed, st, plan, canonical, btasks, meta = bs.stream(w, 0, b1o.N_BRANCH)
    model = b1o.build(BranchGatedLearner, cfg, plan, branch_plan={t.task_id: 'input' for t in btasks},
                      branch_seed=b1o.BRANCH_SEED)
    for t in st:
        model.begin_task(t.task_id)
    sd = torch.load(f'artifacts/b1_online/work/GATED_w{w}_s0/lifetime/model.pt', weights_only=True)['model_state_dict']
    model.load_state_dict(sd, strict=False)
    model.eval()
    pool = o2d.reservoir(st, w, 0, b1o.MEMORY)
    bids = {t.task_id for t in btasks}
    dr.reroute(model, [t for t in st if t.task_id not in bids], plan, pool)
    b1o.refit_branches(model, btasks, pool, w)
    b1o.sleep(cfg, model, pool, [t.task_id for t in st], [1941, w, 0, b1o.MEMORY], model.branch_parameters(bids))
    ref = json.loads(open(f'artifacts/b1_online/cells/GATED_w{w}_s0.json').read())['record']
    assert score(model, SimpleNamespace(tasks=canonical))['per_task'] == ref['canonical_per_task']
    lib = FrozenLibrary(model)
    T = mixed.library
    d = cfg.world.state_dim
    rep = json.load(open('reports/b2_iteration_necessity_census.json'))
    smap = {int(k): v for k, v in [wd for wd in rep['worlds'] if wd['world'] == w][0]['slot_map'].items()}
    body_ok, loop_chosen_loop, straight_loopwins = 0, 0, 0
    for i in range(12):
        rng = np.random.default_rng(np.random.SeedSequence([5680, w, i]))
        wv = rng.normal(size=d); wv /= np.linalg.norm(wv)
        kp = int(rng.integers(0, len(T)))
        xs = rng.normal(size=(512, d))[:128]
        ys, _ = c.teacher_loop(T[kp], wv, xs)
        losses = [loop_fit(lib, s, xs, ys, i)[0] for s in range(lib.slots)]
        best = int(np.argmin(losses))
        body_ok += best == smap[kp]
        xt, yt = torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)
        r3 = dr.exhaustive_route(lib, xt, yt, 3); lib.steps = 3
        with torch.no_grad():
            straight = float(torch.mean((lib.hard(xt, r3) - yt) ** 2))
        loop_chosen_loop += min(losses) < straight
    for i in range(24):
        rng = np.random.default_rng(np.random.SeedSequence([5690, w, i]))
        L = int(rng.integers(1, 4))
        prog = tuple(int(o) for o in rng.integers(0, len(T), size=L))
        xs = rng.normal(size=(128, d)); ys = Program(prog).execute(T, xs)
        xt, yt = torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)
        r = dr.exhaustive_route(lib, xt, yt, L); lib.steps = L
        with torch.no_grad():
            straight = float(torch.mean((lib.hard(xt, r) - yt) ** 2))
        best_loop = min(loop_fit(lib, s, xs, ys, i)[0] for s in range(lib.slots))
        straight_loopwins += best_loop < straight
    print(w, 'body found', body_ok, '/12; loop chosen on loop tasks', loop_chosen_loop, '/12; loop beats straight on straight tasks',
          straight_loopwins, '/24')


if __name__ == '__main__':
    import sys
    main(int(sys.argv[1]))
