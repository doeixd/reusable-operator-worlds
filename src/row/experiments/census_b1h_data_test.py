"""B1-hard pre-launch Tier 0 check of the END-OF-STREAM DATA TEST alone (descriptive; changes no B1-hard rule).

B1-hard keeps a task's branch iff the best two-route gated fit (B1g, 4 restarts) on its 64 RETAINED examples has
hardened support MSE < BRANCH_RATIO x the best single route's. That comparison is in-sample on 64 examples, so a
two-route fit can lower a straight task's error by fitting noise and subsets: the specificity clause may be
unreachable by construction. This applies the data test, exactly as `b1_hard.end_of_stream` does, to B1-online's
saved GATED WAKE terminals (vocabulary formed, before re-route and sleep) on all 236 stream tasks, and also scores
both fits on each task's held-out query set. Reload gate G0: the reloaded model reproduces the committed
`wake_canonical_median` and `wake_branch_median` exactly.
Output `reports/b1h_data_test_census.json`.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import b1_hard as bh
from row.experiments import b1_online as b1o
from row.experiments import b1g_gradient_branch as b1g
from row.experiments import branch_stream as bs
from row.experiments import deep_reroute as dr
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now
from row.models.branch_gated import BranchGatedLearner

OUTPUT = Path('reports/b1h_data_test_census.json')
CELLS = [(44, 0), (45, 0), (46, 0)]


def one(ws):
    w, s = ws
    torch.set_num_threads(1)
    cfg, mixed, st, plan, canonical, btasks, meta = bs.stream(w, s, b1o.N_BRANCH)
    model = b1o.build(BranchGatedLearner, cfg, plan, branch_plan={t.task_id: 'input' for t in btasks},
                      branch_seed=b1o.BRANCH_SEED)
    for t in st:
        model.begin_task(t.task_id)
    state = torch.load(f'artifacts/b1_online/work/GATED_w{w}_s{s}/lifetime/model.pt', weights_only=True)['model_state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    # the only unexpected key is the terminal novel-composition probe code, unused here; G0 checks the reload
    model.eval()
    ref = json.loads(Path(f'artifacts/b1_online/cells/GATED_w{w}_s{s}.json').read_text())['record']
    g0 = (score(model, SimpleNamespace(tasks=canonical))['median'] == ref['wake_canonical_median']
          and score(model, SimpleNamespace(tasks=btasks))['median'] == ref['wake_branch_median'])
    pool = o2d.reservoir(st, w, s, b1o.MEMORY)
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    lib = FrozenLibrary(model)
    branch_ids = {t.task_id for t in btasks}
    rows = []
    for i, t in enumerate(st):
        xs = torch.tensor(np.stack([a for a, _ in by[t.task_id]]), dtype=torch.float32)
        ys = torch.tensor(np.stack([b for _, b in by[t.task_id]]), dtype=torch.float32)
        xq = torch.tensor(t.eval_x, dtype=torch.float32)
        yq = torch.tensor(t.eval_y, dtype=torch.float32)
        d = plan[t.task_id]
        single = dr.exhaustive_route(lib, xs, ys, d)
        lib.steps = d
        onehot = torch.nn.functional.one_hot(torch.tensor(single), lib.slots).float()
        with torch.no_grad():
            s_sup = float(torch.mean((lib.hard(xs, single) - ys) ** 2))
            s_q = float(torch.mean((lib.forward(xq, onehot) - yq) ** 2))
        fits = [b1g.fit(lib, xs, ys, d, False, [5900, w, i, r]) for r in range(b1g.RESTARTS)]
        loss, c1, c2, gw, gb = min(fits, key=lambda f: f[0])
        with torch.no_grad():
            b_q = float(torch.mean((b1g.gated_predict(lib, xq, c1, c2, gw, gb, b1g.T_END, False, hard=True) - yq) ** 2))
        rows.append({'task': t.task_id, 'branch': t.task_id in branch_ids, 'depth': d, 'single_support': s_sup,
                     'branch_support': loss, 'single_query': s_q, 'branch_query': b_q,
                     'kept': loss < bh.BRANCH_RATIO * s_sup})
    return {'cell': f'GATED_w{w}_s{s}', 'G0_reload_reproduces': g0, 'missing_keys': list(missing),
            'unexpected_keys': list(unexpected), 'rows': rows}


def main():
    with ProcessPoolExecutor(max_workers=3) as ex:
        cells = list(ex.map(one, CELLS))
    rows = [r for c in cells for r in c['rows']]
    br = [r for r in rows if r['branch']]
    stt = [r for r in rows if not r['branch']]
    ratio = lambda r, k: r[f'branch_{k}'] / r[f'single_{k}']   # noqa: E731
    summ = {'recall_at_0.5': sum(r['kept'] for r in br) / len(br),
            'specificity_at_0.5': sum(not r['kept'] for r in stt) / len(stt),
            'support_ratio_median': {'branch': float(np.median([ratio(r, 'support') for r in br])),
                                     'straight': float(np.median([ratio(r, 'support') for r in stt]))},
            'query_ratio_median': {'branch': float(np.median([ratio(r, 'query') for r in br])),
                                   'straight': float(np.median([ratio(r, 'query') for r in stt]))},
            'support_ratio_auc_branch_lower': bh.auc([-ratio(r, 'support') for r in br], [-ratio(r, 'support') for r in stt]),
            'query_ratio_auc_branch_lower': bh.auc([-ratio(r, 'query') for r in br], [-ratio(r, 'query') for r in stt])}
    for k in ('support', 'query'):
        for thr in (0.25, 0.5, 0.75, 0.9):
            summ[f'{k}_thr{thr}'] = {'recall': sum(ratio(r, k) < thr for r in br) / len(br),
                                     'specificity': sum(ratio(r, k) >= thr for r in stt) / len(stt)}
    out = {'tier': 0, 'descriptive': True, 'pre_launch_check_for': 'B1_HARD_PLAN.md', 'summary': summ,
           'cells': cells, 'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps({'G0': [c['G0_reload_reproduces'] for c in cells],
                      'keys': [(c['missing_keys'][:3], c['unexpected_keys'][:3]) for c in cells], **summ}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
