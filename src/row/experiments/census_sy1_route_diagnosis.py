"""SY1 Tier 0 route diagnosis (`SY1_ROUTE_DIAGNOSIS_PLAN.md`): are SY0's HARD tasks search failures or selection
failures, and does a cheap local repair fix them?

Libraries: the three E1 discrete lifetimes, rebuilt from committed code (`artifacts/e1_disc_rebuild/world_*`) because the
originals were deleted. Gate R0 (fails closed): every cached E5.1 oracle route and every cached OPT route, executed
through the rebuilt library on its rebuilt task, reproduces the cached query NMSE (relative tolerance 1e-5), for all
192 tasks. Then, per task, support MSE (the search objective) of the oracle route and of each OPT_k route, and a
single-position coordinate-descent polish from the best-support OPT route.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from row.config import load_config
from row.experiments import census_sy0_search_budget as sy0
from row.experiments.audit_e0_export import assign, distance_matrix, load_model
from row.experiments.audit_e1_export import nmse
from row.experiments.audit_e5_synthesizer import execute
from row.support_split_world import _build_tasks
from row.world import World, WorldConfig

ROOT = Path('artifacts/e1_disc_rebuild')
OUTPUT = Path('reports/sy1_route_diagnosis.json')
REL_TOL = 1e-5
EQ_TOL = 1e-6        # relative tolerance for "support loss no worse than the oracle route"


def load(world, root=ROOT, config_path=Path('configs/v1.yaml')):
    path = root / f'world_{world}'
    raw = yaml.safe_load((path / 'config.yaml').read_text(encoding='utf-8'))
    config = load_config(config_path)
    config = replace(config, world=replace(config.world, **raw['world']))
    fields = set(config.discrete_model.__dataclass_fields__)
    config = replace(config, discrete_model=replace(
        config.discrete_model, **{k: v for k, v in raw['discrete_model'].items() if k in fields}))
    shipped, _, _ = load_model(config, path, 'discrete')
    world_obj = World.generate(WorldConfig(**raw['world']))
    probe = torch.tensor(np.random.default_rng(np.random.SeedSequence(
        [900, world, 1])).normal(size=(256, config.world.state_dim)), dtype=torch.float32)
    d = distance_matrix(shipped, world_obj, config.world.program_length, probe, 'library', config.world.seed)
    best, _, _ = assign(d['matrix'])
    return {'config': config, 'library': shipped.library, 'world': world_obj, 'assignment': best,
            'slots': shipped.operator_slots}


def e51_programs(cell, world, depth):
    """E5.1's program draw, verbatim."""
    config = cell['config']
    rng = np.random.default_rng(np.random.SeedSequence([951, world, depth]))
    trained = {tuple(t.program.primitive_ids) for t in cell['world'].tasks}
    programs = []
    while len(programs) < sy0.PER_DEPTH:
        c = tuple(int(v) for v in rng.integers(0, config.world.teacher_primitives, depth))
        if depth == config.world.program_length and c in trained:
            continue
        if c not in programs:
            programs.append(c)
    return programs


def mse(lib, x, y, route):
    return float(torch.mean((execute(lib, x, route) - y) ** 2))


def polish(lib, x, y, route, slots):
    route, best = list(route), mse(lib, x, y, route)
    sweeps = evals = 0
    improved = True
    while improved:
        improved, sweeps = False, sweeps + 1
        for p in range(len(route)):
            for s in range(slots):
                if s == route[p]:
                    continue
                cand = route[:p] + [s] + route[p + 1:]
                v = mse(lib, x, y, cand)
                evals += 1
                if v < best:
                    route, best, improved = cand, v, True
    return route, best, sweeps, evals


def run(root=ROOT):
    records = sy0.load_records()
    tasks_out, gate_fail = [], []
    for w in sy0.WORLDS:
        cell = load(w, root)
        lib = cell['library']
        for d in sy0.DEPTHS:
            for i, program in enumerate(e51_programs(cell, w, d)):
                rec = records[(d, w, i)]
                task = _build_tasks(cell['config'].world, cell['world'].library, [program],
                                    [f'task_e51_d{d}_{i}'], index_offset=51000 + d * 100 + i)[0]
                sx, sy = (torch.tensor(a, dtype=torch.float32) for a in (task.train_x, task.train_y))
                qx, qy = (torch.tensor(a, dtype=torch.float32) for a in (task.eval_x, task.eval_y))
                oracle = [cell['assignment'][int(p)] for p in program]
                if oracle != rec['O']['program']:
                    gate_fail.append(f'w{w}_d{d}_{i}: oracle route differs')
                arms = {'O': rec['O']['program']}
                arms.update({f'OPT{k}': rec[f'OPT{k}']['program'] for k in sy0.BUDGETS})
                q = {a: nmse(execute(lib, qx, r), qy) for a, r in arms.items()}
                for a in arms:
                    if abs(q[a] - rec[a]['nmse']) > REL_TOL * rec[a]['nmse']:
                        gate_fail.append(f'w{w}_d{d}_{i} {a}: {q[a]} vs cached {rec[a]["nmse"]}')
                sup = {a: mse(lib, sx, sy, r) for a, r in arms.items()}
                hard = not sy0.solved_budgets(rec)
                start = min((k for k in sy0.BUDGETS), key=lambda k: sup[f'OPT{k}'])
                pr, ps, sweeps, evals = polish(lib, sx, sy, arms[f'OPT{start}'], cell['slots'])
                pq = nmse(execute(lib, qx, pr), qy)
                best_opt_sup = sup[f'OPT{start}']
                tasks_out.append({
                    'key': f'w{w}_d{d}_{i}', 'world': w, 'depth': d, 'hard': hard,
                    'support_mse': sup, 'query_nmse': q,
                    'search_failure': best_opt_sup > sup['O'] * (1 + EQ_TOL),
                    'polish': {'start': f'OPT{start}', 'support_mse': ps, 'query_nmse': pq,
                               'query_gap': math.log(pq) - math.log(q['O']),
                               'solved': math.log(pq) - math.log(q['O']) <= sy0.GAP,
                               'reaches_oracle_support': ps <= sup['O'] * (1 + EQ_TOL),
                               'route': pr, 'sweeps': sweeps, 'evaluations': evals}})
    return tasks_out, gate_fail


def summarize(tasks):
    out = {}
    for d in sy0.DEPTHS:
        ts = [t for t in tasks if t['depth'] == d]
        hard = [t for t in ts if t['hard']]
        out[str(d)] = {
            'n': len(ts), 'hard': len(hard),
            'hard_search_failure': sum(t['search_failure'] for t in hard),
            'hard_selection_failure': sum(not t['search_failure'] for t in hard),
            'hard_polish_solved': sum(t['polish']['solved'] for t in hard),
            'all_polish_solved': sum(t['polish']['solved'] for t in ts),
            'median_polish_evaluations': float(np.median([t['polish']['evaluations'] for t in ts])),
        }
    deep = [t for t in tasks if t['depth'] >= 7 and t['hard']]
    n = len(deep)
    sf = sum(t['search_failure'] for t in deep)
    rep = sum(t['polish']['solved'] for t in deep)
    if n == 0:
        lab = 'NO_HARD_TASKS'
    elif sf >= 0.75 * n and rep >= 0.75 * n:
        lab = 'SEARCH_FAILURE_LOCALLY_REPAIRABLE'
    elif sf >= 0.75 * n:
        lab = 'SEARCH_FAILURE_NOT_LOCALLY_REPAIRABLE'
    elif sf <= 0.25 * n:
        lab = 'SELECTION_FAILURE'
    else:
        lab = 'MIXED'
    return out, {'label': lab, 'deep_hard': n, 'deep_search_failure': sf, 'deep_polish_solved': rep}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    tasks, gate_fail = run(args.root)
    import subprocess
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    payload = {'plan': 'SY1_ROUTE_DIAGNOSIS_PLAN.md', 'git_commit': commit, 'gate_R0': {
        'passes': not gate_fail, 'failures': gate_fail[:50], 'n_failures': len(gate_fail)}}
    if not gate_fail:
        per_depth, verdict = summarize(tasks)
        payload.update({'per_depth': per_depth, 'verdict': verdict, 'tasks': tasks})
    tmp = OUTPUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    os.replace(tmp, OUTPUT)
    if gate_fail:
        print(f'GATE R0 FAILS: {len(gate_fail)} mismatches; first: {gate_fail[:3]}')
        raise SystemExit(1)
    print(json.dumps(payload['verdict']))
    for d, v in payload['per_depth'].items():
        print(d, v)


if __name__ == '__main__':
    main()
