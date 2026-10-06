"""B1-online post-hoc gate-accuracy check (descriptive; written after B1-online's result; changes no reading).

B1-online's descriptive gate accuracy scored the gate as if route 1 were the teacher's branch A; which learned route
plays A is arbitrary, so that field is uninterpretable. This re-runs one GATED cell (world 44, stream 0) with B1-online's
exact construction, requires its canonical and branch per-task scores to equal the committed cell record exactly
(same model), and then measures per branch task:
- ORIENTATION_FREE accuracy: max(acc, 1 - acc) against the true predicate on the query inputs;
- FUNCTION_MATCHED accuracy: route 1 is called A if its query outputs are closer (MSE) to the teacher's A than route
  2's are; the gate is scored in that orientation.
Output `reports/b1o_gate_check.json`.
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import b1_online as b1o
from row.experiments import branch_stream as bs
from row.experiments import deep_reroute as dr
from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.so1_storage import atomic_json, now
from row.models.branch_gated import BranchGatedLearner
from row.world import Program

OUTPUT = Path('reports/b1o_gate_check.json')
W, S = 44, 0


def main():
    torch.set_num_threads(1)
    cfg, mixed, st, plan, canonical, btasks, meta = bs.stream(W, S, b1o.N_BRANCH)
    model = b1o.build(BranchGatedLearner, cfg, plan, branch_plan={t.task_id: 'input' for t in btasks},
                      branch_seed=b1o.BRANCH_SEED)
    out = Path(tempfile.mkdtemp()) / 'lifetime'
    summary = ll.run(dataclasses.replace(o2.run_cfg(cfg, 1), output_directory=out), o2.KIND, world=mixed, model=model,
                     return_model=True, replay_seed=o2.replay_seed_for(W, S))
    model = summary.pop('terminal_model')
    pool = o2d.reservoir(st, W, S, b1o.MEMORY)
    branch_ids = {t.task_id for t in btasks}
    dr.reroute(model, [t for t in st if t.task_id not in branch_ids], plan, pool)
    b1o.refit_branches(model, btasks, pool, W)
    b1o.sleep(cfg, model, pool, [t.task_id for t in st], [1941, W, S, b1o.MEMORY], model.branch_parameters(branch_ids))
    ref = json.loads(Path(f'artifacts/b1_online/cells/GATED_w{W}_s{S}.json').read_text())['record']
    canon = score(model, SimpleNamespace(tasks=canonical))['per_task']
    branch = score(model, SimpleNamespace(tasks=btasks))['per_task']
    reproduces = canon == ref['canonical_per_task'] and branch == ref['branch_per_task']
    free, matched = [], []
    model.eval()
    with torch.no_grad():
        for t in btasks:
            x = torch.tensor(t.eval_x, dtype=torch.float32)
            truth = (t.eval_x @ np.array(meta[t.task_id]['w'])) > 0
            g = model.gate(x, t.task_id).numpy() > 0.5
            acc = float(np.mean(g == truth))
            free.append(max(acc, 1 - acc))
            d = model.depth_of(t.task_id)
            r1 = model._route(x, model._route_coefficients(model.task_codes[t.task_id]), d).numpy()
            r2 = model._route(x, model._route_coefficients(model.branch_codes[t.task_id]), d).numpy()
            yA = Program(tuple(meta[t.task_id]['A'])).execute(t.teacher_library, t.eval_x)
            route1_is_A = np.mean((r1 - yA) ** 2) < np.mean((r2 - yA) ** 2)
            matched.append(acc if route1_is_A else 1 - acc)
    result = {'tier': 0, 'post_hoc': True, 'cell': f'GATED_w{W}_s{S}', 'reproduces_committed_cell': reproduces,
              'orientation_free_accuracy_median': statistics.median(free),
              'function_matched_accuracy_median': statistics.median(matched),
              'function_matched_accuracy_p10': float(np.percentile(matched, 10)),
              'per_task': [{'orientation_free': f, 'function_matched': m} for f, m in zip(free, matched)],
              'finished_utc': now()}
    atomic_json(OUTPUT, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_task'}, indent=1))


if __name__ == '__main__':
    main()
