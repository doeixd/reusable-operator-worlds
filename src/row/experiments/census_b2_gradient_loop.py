"""B2 Tier 0 gradient check (descriptive; before any B2-online plan): can GRADIENT on `LoopLearner`'s soft stopping
mixture learn a loop task's body and halting predicate on a frozen formed vocabulary, from the zero initialization
wake uses? (The B1g analogue for loops.)

Library: B1-online's FINAL GATED library for worlds 44-46, stream 0, rebuilt exactly as in
`census_b2_iteration_necessity` (gates G0 and G1). Tasks: 16 loop tasks per world from `loop_stream.loop_tasks`
(the B2-online construction, 128 support examples, query = the task's evaluation set). Per task, a fresh
`LoopLearner` slot for the task (body code N(0, 0.1) restarts from seeds [5810, w, i, r], halting zero), library
frozen, temperature at its final value, Adam lr 0.05 on body code and halting, STEPS full-batch updates on the
soft-mixture MSE; RESTARTS restarts, the lowest final support loss kept. Query NMSE with hard semantics. Compared
with the support-only search of the census (slot search + count inference + logistic halting) on the same tasks.
Output `reports/b2_gradient_loop_census.json`.
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
from row.experiments import census_b2_iteration_necessity as c2
from row.experiments import deep_reroute as dr
from row.experiments import loop_stream as ls
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now
from row.models.branch_gated import BranchGatedLearner
from row.models.loop_gated import LoopLearner

OUTPUT = Path('reports/b2_gradient_loop_census.json')
WORLDS = (44, 45, 46)
TASKS = 16
STEPS = 600
RESTARTS = 4
LR = 0.05


def final_model(w):
    cfg, mixed, st, plan, canonical, btasks, meta = bs.stream(w, 0, b1o.N_BRANCH)
    model = b1o.build(BranchGatedLearner, cfg, plan, branch_plan={t.task_id: 'input' for t in btasks},
                      branch_seed=b1o.BRANCH_SEED)
    for t in st:
        model.begin_task(t.task_id)
    sd = torch.load(f'artifacts/b1_online/work/GATED_w{w}_s0/lifetime/model.pt', weights_only=True)['model_state_dict']
    model.load_state_dict(sd, strict=False)
    model.eval()
    ref = json.loads(Path(f'artifacts/b1_online/cells/GATED_w{w}_s0.json').read_text())['record']
    pool = o2d.reservoir(st, w, 0, b1o.MEMORY)
    bids = {t.task_id for t in btasks}
    dr.reroute(model, [t for t in st if t.task_id not in bids], plan, pool)
    b1o.refit_branches(model, btasks, pool, w)
    b1o.sleep(cfg, model, pool, [t.task_id for t in st], [1941, w, 0, b1o.MEMORY], model.branch_parameters(bids))
    g1 = score(model, SimpleNamespace(tasks=canonical))['per_task'] == ref['canonical_per_task']
    return cfg, mixed, model, g1


def one(w):
    torch.set_num_threads(1)
    cfg, mixed, model, g1 = final_model(w)
    lib = FrozenLibrary(model)
    ltasks, meta = ls.loop_tasks(mixed, TASKS, cfg.world.examples_per_task, cfg.world.evaluation_examples,
                                 cfg.world.state_dim)
    plan = {t.task_id: 1 for t in ltasks}
    learner = b1o.build(LoopLearner, cfg, plan, loop_plan={t.task_id: 'state' for t in ltasks})
    learner.load_state_dict({k: v for k, v in model.state_dict().items() if k.startswith('library.')}, strict=False)
    for p in learner.library.parameters():
        p.requires_grad_(False)
    learner.set_training_progress(1.0)
    rows = []
    for i, t in enumerate(ltasks):
        xs, ys = torch.tensor(t.train_x, dtype=torch.float32), torch.tensor(t.train_y, dtype=torch.float32)
        xq = torch.tensor(t.eval_x, dtype=torch.float32)
        learner.begin_task(t.task_id)
        best = None
        for r in range(RESTARTS):
            g = torch.Generator().manual_seed(int(np.random.SeedSequence([5810, w, i, r]).generate_state(1)[0]))
            with torch.no_grad():
                learner.task_codes[t.task_id].copy_(0.1 * torch.randn(learner.task_codes[t.task_id].shape, generator=g))
                learner.halt_w[t.task_id].zero_()
                learner.halt_b[t.task_id].zero_()
            opt = torch.optim.Adam([learner.task_codes[t.task_id], learner.halt_w[t.task_id], learner.halt_b[t.task_id]],
                                   lr=LR)
            learner.train()
            for _ in range(STEPS):
                opt.zero_grad(set_to_none=True)
                loss = torch.mean((learner(xs, t.task_id) - ys) ** 2)
                loss.backward()
                opt.step()
            learner.eval()
            with torch.no_grad():
                sup = float(torch.mean((learner(xs, t.task_id) - ys) ** 2))
                q = c2.nmse(learner(xq, t.task_id).numpy().astype(np.float64), t.eval_y)
                body = int(torch.argmax(learner.task_codes[t.task_id][0]))
            if best is None or sup < best[0]:
                best = (sup, q, body, r)
        # support-only search reference (the census's LOOP_LEARNED, slot searched)
        search = []
        for s in range(lib.slots):
            Us = c2.unrolled(lib, s, t.train_x)
            inferred = np.argmin(((Us - t.train_y[None]) ** 2).sum(-1), axis=0)
            Xh, yh = [], []
            for j in range(c2.K):
                at = inferred >= j
                Xh.append(Us[j][at])
                yh.append((inferred[at] > j).astype(int))
            Y = np.concatenate(yh)
            if len(set(Y.tolist())) < 2:
                continue
            Wh = c2.fit_linear(np.concatenate(Xh), Y, 2, seed=i)
            pred_s, _ = c2.loop_with(lambda Z: c2.predict_linear(Wh, Z) == 1, Us)
            Uq = c2.unrolled(lib, s, t.eval_x)
            pred_q, _ = c2.loop_with(lambda Z: c2.predict_linear(Wh, Z) == 1, Uq)
            search.append((float(np.mean((pred_s - t.train_y) ** 2)), c2.nmse(pred_q, t.eval_y), s))
        sb = min(search)
        rows.append({'task': t.task_id, 'P': meta[t.task_id]['P'], 'gradient_query': best[1], 'gradient_body': best[2],
                     'gradient_restart': best[3], 'search_query': sb[1], 'search_body': sb[2]})
    return {'world': w, 'G1_final_reproduces': g1, 'rows': rows}


def main():
    with ProcessPoolExecutor(max_workers=3) as ex:
        worlds = list(ex.map(one, WORLDS))
    rows = [r for wd in worlds for r in wd['rows']]
    g, s = np.array([r['gradient_query'] for r in rows]), np.array([r['search_query'] for r in rows])
    summ = {'tasks': len(rows), 'gradient_median': float(np.median(g)), 'search_median': float(np.median(s)),
            'gradient_p10_p90': np.percentile(g, [10, 90]).tolist(), 'search_p10_p90': np.percentile(s, [10, 90]).tolist(),
            'same_body': sum(r['gradient_body'] == r['search_body'] for r in rows),
            'gradient_within_1.5x_search': int(np.sum(g <= 1.5 * s)), 'gradient_below_0.05': int(np.sum(g < 0.05)),
            'search_below_0.05': int(np.sum(s < 0.05)), 'first_restart_kept': sum(r['gradient_restart'] == 0 for r in rows)}
    atomic_json(OUTPUT, {'tier': 0, 'descriptive': True, 'summary': summ, 'worlds': worlds, 'finished_utc': now()})
    print(json.dumps({'G1': [wd['G1_final_reproduces'] for wd in worlds], **summ}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
