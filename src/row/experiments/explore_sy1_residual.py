"""EXPLORATORY, post hoc (no verdict): size SY1's residual. For each task still unsolved after SY1's single-position
polish, continue with a TWO-position polish (change any pair of positions jointly while support MSE falls), and record
whether it reaches the oracle route's support MSE and SY0's solved criterion. Sizes a possible SY2; decides nothing.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path

import torch

from row.experiments import census_sy0_search_budget as sy0
from row.experiments import census_sy1_route_diagnosis as sy1
from row.experiments.audit_e1_export import nmse
from row.experiments.audit_e5_synthesizer import execute
from row.support_split_world import _build_tasks

OUTPUT = Path('reports/sy1_residual_sizing.json')


def pair_polish(lib, x, y, route, slots):
    route, best = list(route), sy1.mse(lib, x, y, route)
    evals, improved = 0, True
    while improved:
        improved = False
        for p, q in itertools.combinations(range(len(route)), 2):
            for a in range(slots):
                for b in range(slots):
                    if a == route[p] and b == route[q]:
                        continue
                    cand = list(route)
                    cand[p], cand[q] = a, b
                    v = sy1.mse(lib, x, y, cand)
                    evals += 1
                    if v < best:
                        route, best, improved = cand, v, True
        if improved:   # re-run single moves cheaply after each pair sweep
            route, best, _, e = sy1.polish(lib, x, y, route, slots)
            evals += e
    return route, best, evals


def main():
    torch.set_num_threads(1)
    report = json.loads(Path('reports/sy1_route_diagnosis.json').read_text(encoding='utf-8'))
    residual = [t for t in report['tasks'] if not t['polish']['solved']]
    cells = {w: sy1.load(w) for w in sy0.WORLDS}
    out = []
    for t in residual:
        w, d, i = t['world'], t['depth'], int(t['key'].rsplit('_', 1)[1])
        cell = cells[w]
        program = sy1.e51_programs(cell, w, d)[i]
        task = _build_tasks(cell['config'].world, cell['world'].library, [program],
                            [f'task_e51_d{d}_{i}'], index_offset=51000 + d * 100 + i)[0]
        sx, sy = (torch.tensor(a, dtype=torch.float32) for a in (task.train_x, task.train_y))
        qx, qy = (torch.tensor(a, dtype=torch.float32) for a in (task.eval_x, task.eval_y))
        route, best, evals = pair_polish(cell['library'], sx, sy, t['polish']['route'], cell['slots'])
        q = nmse(execute(cell['library'], qx, route), qy)
        gap = math.log(q) - math.log(t['query_nmse']['O'])
        row = {'key': t['key'], 'depth': d, 'support_mse': best, 'oracle_support_mse': t['support_mse']['O'],
               'reaches_oracle_support': best <= t['support_mse']['O'] * (1 + 1e-6),
               'query_gap': gap, 'solved': gap <= sy0.GAP, 'evaluations': evals,
               'route': route, 'oracle_route': t['oracle_route'] if 'oracle_route' in t else None}
        out.append(row)
        print(row['key'], 'solved' if row['solved'] else 'hard', f"gap {gap:+.2f}", evals, flush=True)
    summary = {'residual_tasks': len(out), 'pair_polish_solved': sum(r['solved'] for r in out),
               'reaches_oracle_support': sum(r['reaches_oracle_support'] for r in out),
               'median_evaluations': sorted(r['evaluations'] for r in out)[len(out) // 2] if out else 0}
    payload = {'status': 'EXPLORATORY, post hoc, no verdict', 'summary': summary, 'tasks': out}
    tmp = OUTPUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    os.replace(tmp, OUTPUT)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
