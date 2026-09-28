"""Tier 0 census (descriptive, read-only): does fresh-relaxation gradient re-routing match exhaustive re-routing?

Follows `census_o7_coordinate_reroute` (coordinate descent from the stale route and an
anchor-derived slot map both fail: stale routes need 2-3 coupled, position-specific changes).
Candidate here: SO1R's `optimize_route` verbatim (task code re-initialised to zero, i.e. a
uniform mixture; Adam 0.05; temperature 1.0 -> 0.1; library frozen; support = the task's 64
reservoir examples), then hardened by argmax. Its cost per step is d x 12 candidate
evaluations, linear in depth. Variants: `opt` alone, and `safe` = whichever of {current,
opt} has the lower support MSE (never worse than doing nothing on support).

Development terminals only (no sealed world). Canonical depth-3 tasks only (64 per cell), to
bound compute; exhaustive search is the reference. Output `reports/o7b_gradient_reroute_census.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o5_reroute_sleep as o5
from row.experiments.audit_j2a_staged_library import enum_route
from row.experiments.audit_so1r_route_only import FrozenLibrary, optimize_route
from row.experiments.so1_storage import atomic_json, now
from row.metrics import nmse

OUTPUT = Path('reports/o7b_gradient_reroute_census.json')
# both development collapse cells first, then one stream per remaining development world
CELLS = [('O2', 14, 0), ('O3', 22, 1)] + [('O2', w, 1) for w in (13, 15, 16, 17, 18, 19)] + \
        [('O3', w, 0) for w in (20, 21, 23, 24, 25, 26)]


def mse(library, x, y, route):
    with torch.no_grad():
        return float(torch.mean((library.hard(x, list(route)) - y) ** 2))


def cell(band, w, s, steps, limit=None):
    torch.manual_seed(0)
    _, model, stream_tasks, canonical = o3.load_shuffled_terminal(o5.terminal_path(band, w, s), w, s)
    pool = o2d.reservoir(stream_tasks, w, s, o3.MEMORY)
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    library = FrozenLibrary(model)
    library.steps = 3
    current = model.hard_routes()
    rows, query = [], {k: [] for k in ('current', 'exhaustive', 'opt', 'safe')}
    for task in canonical[:limit]:
        xs = torch.tensor(np.stack([a for a, _ in by[task.task_id]]), dtype=torch.float32)
        ys = torch.tensor(np.stack([b for _, b in by[task.task_id]]), dtype=torch.float32)
        cur = tuple(int(r) for r in current[task.task_id][:3])
        exh = tuple(int(r) for r in enum_route(library, xs, ys))
        opt = tuple(int(r) for r in optimize_route(library, xs, ys, steps)[0])
        m = {'current': mse(library, xs, ys, cur), 'exhaustive': mse(library, xs, ys, exh),
             'opt': mse(library, xs, ys, opt)}
        safe = opt if m['opt'] < m['current'] else cur
        rows.append({'stale': cur != exh, 'opt_agree': opt == exh, 'safe_agree': safe == exh,
                     'opt_ratio': m['opt'] / m['exhaustive'] if m['exhaustive'] > 0 else 1.0})
        ex = torch.tensor(task.eval_x, dtype=torch.float32)
        for name, route in (('current', cur), ('exhaustive', exh), ('opt', opt), ('safe', safe)):
            with torch.no_grad():
                query[name].append(float(nmse(library.hard(ex, list(route)).numpy(), task.eval_y)))
    stale = [r for r in rows if r['stale']]
    return {'tasks': len(rows), 'stale': len(stale),
            'opt_agrees': sum(r['opt_agree'] for r in rows), 'opt_agrees_on_stale': sum(r['opt_agree'] for r in stale),
            'safe_agrees': sum(r['safe_agree'] for r in rows),
            'safe_agrees_on_stale': sum(r['safe_agree'] for r in stale),
            'opt_worse_by_10pct': sum(r['opt_ratio'] > 1.1 for r in rows),
            'query_median': {k: float(np.median(v)) for k, v in query.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--timing', action='store_true', help='one cell, 4 tasks, prints seconds')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.timing:
        t = time.time()
        print(cell('O2', 14, 0, args.steps, limit=4), time.time() - t)
        return
    started, cells = time.time(), {}
    for band, w, s in CELLS:
        cells[f'{band}_w{w}_s{s}'] = cell(band, w, s, args.steps)
        print(band, w, s, cells[f'{band}_w{w}_s{s}'], flush=True)
        atomic_json(OUTPUT, {'tier': 0, 'descriptive': True, 'partial': True, 'steps': args.steps, 'cells': cells})
    atomic_json(OUTPUT, {'tier': 0, 'descriptive': True, 'partial': False, 'steps': args.steps,
                         'worlds': 'development only', 'cells': cells, 'seconds': time.time() - started,
                         'finished_utc': now()})


if __name__ == '__main__':
    main()
