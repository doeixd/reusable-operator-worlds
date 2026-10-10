"""B1-mid Tier 0 necessity/opportunity census (descriptive; written before any B1-mid plan).

Question: for branch tasks whose decision is on an INTERMEDIATE state, y = IF(w.C(x) > 0, A(C(x)), B(C(x))) with C one
teacher primitive and A != B length-2 teacher programs, does a gate that reads the learner's state after its first
routed step beat the cheaper impostor, a linear gate on the raw input? Each operator is a near-identity residual
followed by a rotation, so w.C(x) may be close to linear in x, in which case an intermediate-state rung has no
necessity. On B1-online's saved GATED wake vocabularies (worlds 44-46, stream 0; reload gate G0 as in
`census_b1h_data_test`), 24 new tasks per world (draws [5650, world, i]), support sizes 128 (the stream's examples per
task) and 512, query 512. Per task and size: SINGLE = best single depth-3 route (exhaustive); INPUT = B1g fit with the
gate on x; MID = B1g fit with the gate on route 1's state after step 1 (4 restarts each, chosen on support); query
NMSE for each. Output `reports/b1mid_necessity_census.json`.
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
from row.experiments import b1g_gradient_branch as b1g
from row.experiments import branch_stream as bs
from row.experiments import deep_reroute as dr
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now
from row.metrics import nmse
from row.models.branch_gated import BranchGatedLearner
from row.world import Program

OUTPUT = Path('reports/b1mid_necessity_census.json')
WORLDS = (44, 45, 46)
TASKS = 24
SIZES = (128, 512)
QUERY = 512


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
    lib = FrozenLibrary(model)
    teacher = mixed.library
    n_ops, d = len(teacher), cfg.world.state_dim
    rows = []
    for i in range(TASKS):
        rng = np.random.default_rng(np.random.SeedSequence([5650, w, i]))
        wv = rng.normal(size=d)
        wv /= np.linalg.norm(wv)
        C = (int(rng.integers(0, n_ops)),)
        A = tuple(int(o) for o in rng.integers(0, n_ops, size=2))
        B = A
        while B == A:
            B = tuple(int(o) for o in rng.integers(0, n_ops, size=2))
        pc, pa, pb = Program(C), Program(C + A), Program(C + B)

        def target(x):
            z = pc.execute(teacher, x)
            return np.where((z @ wv > 0)[:, None], pa.execute(teacher, x), pb.execute(teacher, x))

        xs_all, xq = rng.normal(size=(max(SIZES), d)), rng.normal(size=(QUERY, d))
        ys_all, yq = target(xs_all), target(xq)
        # how linear in x is the true decision? best linear separator of the label on x (logistic fit on query inputs)
        lab = (pc.execute(teacher, xq) @ wv > 0)
        lin = np.linalg.lstsq(np.c_[xq, np.ones(len(xq))], 2.0 * lab - 1.0, rcond=None)[0]
        lin_acc = float(np.mean((np.c_[xq, np.ones(len(xq))] @ lin > 0) == lab))
        xq_t = torch.tensor(xq, dtype=torch.float32)
        row = {'task': i, 'C': C, 'A': A, 'B': B, 'linear_in_x_label_accuracy': lin_acc, 'sizes': {}}
        for n in SIZES:
            xs = torch.tensor(xs_all[:n], dtype=torch.float32)
            ys = torch.tensor(ys_all[:n], dtype=torch.float32)
            single = dr.exhaustive_route(lib, xs, ys, 3)
            lib.steps = 3
            with torch.no_grad():
                s_q = float(nmse(lib.hard(xq_t, single).numpy(), yq))
            res = {'SINGLE': s_q}
            for name, mid in (('INPUT', False), ('MID', True)):
                fits = [b1g.fit(lib, xs, ys, 3, mid, [5660, w, i, n, int(mid), r]) for r in range(b1g.RESTARTS)]
                _, c1, c2, gw, gb = min(fits, key=lambda f: f[0])
                with torch.no_grad():
                    pred = b1g.gated_predict(lib, xq_t, c1, c2, gw, gb, b1g.T_END, mid, hard=True)
                res[name] = float(nmse(pred.numpy(), yq))
            row['sizes'][str(n)] = res
        rows.append(row)
    return {'world': w, 'G0_reload_reproduces': g0, 'rows': rows}


def main():
    with ProcessPoolExecutor(max_workers=3) as ex:
        worlds = list(ex.map(one, WORLDS))
    rows = [r for wd in worlds for r in wd['rows']]
    summ = {'linear_in_x_label_accuracy_median': float(np.median([r['linear_in_x_label_accuracy'] for r in rows]))}
    for n in SIZES:
        k = str(n)
        med = {a: float(np.median([r['sizes'][k][a] for r in rows])) for a in ('SINGLE', 'INPUT', 'MID')}
        summ[f'n{n}'] = {'median_query_nmse': med,
                         'MID_below_INPUT': sum(r['sizes'][k]['MID'] < r['sizes'][k]['INPUT'] for r in rows),
                         'MID_below_0.05': sum(r['sizes'][k]['MID'] < 0.05 for r in rows),
                         'INPUT_below_0.05': sum(r['sizes'][k]['INPUT'] < 0.05 for r in rows),
                         'median_INPUT_over_MID': float(np.median([r['sizes'][k]['INPUT'] / r['sizes'][k]['MID']
                                                                   for r in rows])),
                         'tasks': len(rows)}
    out = {'tier': 0, 'descriptive': True, 'summary': summ, 'worlds': worlds, 'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps({'G0': [wd['G0_reload_reproduces'] for wd in worlds], **summ}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
