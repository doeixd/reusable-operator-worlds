"""A1 Tier 0 census (descriptive; SUCCESSOR_LADDER.md Phase A, issue #2 section 3): how good must routes be for
sleep to finish the repair?

On surviving D1 depth-4 SHUFFLED4 terminals (worlds 30, 31, 32, stream 0): compute the exact routes once
(`deep_reroute.exhaustive_route` on each stream task's 64 retained examples, the end-of-stream construction that
passed 21/21). Then, for each condition, install a route set on a fresh reload of the terminal with O5's minimal
logit swap and run O3's sleep verbatim (8,192 updates, sampling [1941, w, s, 64]):
- EXACT: the exact routes (the census_d1_end_reroute construction; reproduces it).
- STALE: the routes as committed at the end of the stream (no re-routing; D1's SLEEP4).
- P{f}_H{h}: the exact routes with a fraction f in {0.25, 0.5, 1.0} of ALL stream tasks corrupted in h positions
  (h = 1, or h = 'all' = every position of the task), each corrupted position moved to a uniformly chosen
  different slot; seed [5200, w, round(100 f), h or 9].
Recorded per condition: terminal median after sleep, terminal median before sleep (routes only), mean Hamming
distance to the exact routes, and the fraction of tasks whose installed route equals the exact one.

Reading, stated before running: the adequacy threshold is the largest corruption at which the median terminal
after sleep stays below 0.05 in all three worlds. Descriptive only. Output `reports/a1_route_adequacy_census.json`.
"""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import d1_depth4_formation as d1
from row.experiments import deep_reroute as dr
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now

OUTPUT = Path('reports/a1_route_adequacy_census.json')
WORLDS = (30, 31, 32)
CONDITIONS = ['EXACT', 'STALE'] + [f'P{f}_H{h}' for f in (0.25, 0.5, 1.0) for h in ('1', 'all')]
SLOTS = 12


def load(w, s=0):
    cfg, model, st, canonical = d1.load_terminal(d1.terminal_of(d1.ROOT, 'SHUFFLED4', w, s), w, s)
    plan = {t.task_id: len(t.program.primitive_ids) for t in st}
    return cfg, model, st, canonical, plan


def exact_routes(w, s=0):
    cfg, model, st, canonical, plan = load(w, s)
    pool = o2d.reservoir(st, w, s, 64)
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    lib = FrozenLibrary(model)
    out = {}
    for t in st:
        xs = torch.tensor(np.stack([a for a, _ in by[t.task_id]]), dtype=torch.float32)
        ys = torch.tensor(np.stack([b for _, b in by[t.task_id]]), dtype=torch.float32)
        out[t.task_id] = dr.exhaustive_route(lib, xs, ys, plan[t.task_id])
    return out, {t.task_id: [int(r) for r in model.hard_routes()[t.task_id][:plan[t.task_id]]] for t in st}


def corrupt(exact, st, w, f, h):
    rng = np.random.default_rng(np.random.SeedSequence([5200, w, int(round(100 * f)), 9 if h == 'all' else 1]))
    ids = [t.task_id for t in st]
    chosen = set(rng.choice(len(ids), size=int(round(f * len(ids))), replace=False).tolist())
    out = {}
    for i, tid in enumerate(ids):
        r = list(exact[tid])
        if i in chosen:
            positions = range(len(r)) if h == 'all' else [int(rng.integers(len(r)))]
            for p in positions:
                r[p] = int((r[p] + rng.integers(1, SLOTS)) % SLOTS)   # a uniformly chosen DIFFERENT slot
        out[tid] = r
    return out


def install(model, targets, plan):
    with torch.no_grad():
        for tid, new in targets.items():
            code = model.task_codes[tid]
            for step in range(plan[tid]):
                old = int(torch.argmax(code[step]))
                if old != new[step]:
                    a, b = code[step, old].clone(), code[step, new[step]].clone()
                    code[step, old], code[step, new[step]] = b, a
            if [int(r) for r in model.hard_routes()[tid][:plan[tid]]] != list(new):
                raise RuntimeError(f'{tid}: install failed')


def condition(args):
    torch.set_num_threads(1)
    w, cond, targets = args
    cfg, model, st, canonical, plan = load(w)
    if cond != 'STALE':
        install(model, targets, plan)
    before = score(model, SimpleNamespace(tasks=canonical))['median']
    pool = o2d.reservoir(st, w, 0, 64)
    o2c.consolidate(cfg, model, pool, [t.task_id for t in st], [1941, w, 0, 64], updates=8192)
    return w, cond, {'before_sleep_median': before, 'terminal_median': score(model, SimpleNamespace(tasks=canonical))['median']}


def main():
    torch.set_num_threads(1)
    started = time.time()
    jobs, meta = [], {}
    for w in WORLDS:
        _, _, st, _, plan = load(w)
        exact, stale = exact_routes(w)
        sets = {'EXACT': exact, 'STALE': stale}
        for f in (0.25, 0.5, 1.0):
            for h in ('1', 'all'):
                sets[f'P{f}_H{h}'] = corrupt(exact, st, w, f, h)
        for cond, routes in sets.items():
            ham = [sum(a != b for a, b in zip(routes[t], exact[t])) for t in routes]
            meta[(w, cond)] = {'mean_hamming_to_exact': float(np.mean(ham)),
                               'fraction_exact': float(np.mean([h == 0 for h in ham]))}
            jobs.append((w, cond, routes))
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(condition, jobs))
    cells = {}
    for w, cond, r in results:
        cells[f'w{w}_{cond}'] = r | meta[(w, cond)]
    summary = {}
    for cond in CONDITIONS:
        vals = [cells[f'w{w}_{cond}'] for w in WORLDS]
        summary[cond] = {'terminal_per_world': [v['terminal_median'] for v in vals],
                         'all_pass': all(v['terminal_median'] < 0.05 for v in vals),
                         'before_sleep_median': statistics.median(v['before_sleep_median'] for v in vals),
                         'mean_hamming_to_exact': statistics.median(v['mean_hamming_to_exact'] for v in vals),
                         'fraction_exact': statistics.median(v['fraction_exact'] for v in vals)}
    out = {'tier': 0, 'descriptive': True, 'summary': summary, 'cells': cells,
           'seconds': time.time() - started, 'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
