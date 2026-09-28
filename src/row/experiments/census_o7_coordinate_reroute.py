"""Tier 0 census (descriptive, read-only): does a length-linear re-router match exhaustive re-routing?

O6 confirmed wake + EXHAUSTIVE re-route + sleep, but exhaustive search costs 12^d route
evaluations and is infeasible beyond depth ~4. Coordinate descent (CD) from the task's
current (stale) hard route re-optimises one step at a time with the others fixed, sweeping
until no step changes: 12*d evaluations per sweep, linear in program length.

On the saved DEVELOPMENT order-free terminals (O2 worlds 13-19, O3 20-26, streams 0-2; no
sealed world), for every stream task of depth >= 2, with the library frozen and only that
task's 64 reservoir examples (the O5/O6 construction): the exhaustive route, the CD route,
exact agreement, the support-MSE ratio CD/exhaustive, sweeps used, and whether CD lands in a
worse local minimum. Canonical-task query NMSE is also scored for each routing (no sleep).

Second method, added after the first pass showed CD rarely escapes a stale route (stale routes
need 2-3 coupled position changes; redundant slots drift): ANCHOR MAP. Re-route every depth-1
stream task exhaustively (12 evaluations each), take the majority map current slot -> new slot
over those anchors (identity for slots no anchor used), apply it position-wise to every longer
route, then polish with CD (`map_cd`). Cost is linear in program length. Output
`reports/o7_coordinate_reroute_census.json`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o5_reroute_sleep as o5
from row.experiments.audit_j2a_staged_library import enum_route
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now
from row.metrics import nmse

OUTPUT = Path('reports/o7_coordinate_reroute_census.json')
BANDS = {'O2': range(13, 20), 'O3': range(20, 27)}   # development only
MAX_SWEEPS = 20


def mse(library, x, y, route):
    with torch.no_grad():
        return float(torch.mean((library.hard(x, list(route)) - y) ** 2))


def coordinate_descent(library, x, y, start):
    route, best = list(start), mse(library, x, y, start)
    evaluations, sweeps = 1, 0
    for sweeps in range(1, MAX_SWEEPS + 1):
        changed = False
        for p in range(len(route)):
            for slot in range(library.slots):
                if slot == route[p]:
                    continue
                trial = route[:p] + [slot] + route[p + 1:]
                value = mse(library, x, y, trial)
                evaluations += 1
                if value < best:
                    route, best, changed = trial, value, True
        if not changed:
            break
    return tuple(route), best, evaluations, sweeps


def cell(band, w, s):
    torch.set_num_threads(1)
    _, model, stream_tasks, canonical = o3.load_shuffled_terminal(o5.terminal_path(band, w, s), w, s)
    _, _, _, plan, _ = o2.build_stream('SHUFFLED', w, s)
    pool = o2d.reservoir(stream_tasks, w, s, o3.MEMORY)
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    library = FrozenLibrary(model)
    current = model.hard_routes()
    canonical_ids = {t.task_id for t in canonical}

    def support(task):
        return (torch.tensor(np.stack([a for a, _ in by[task.task_id]]), dtype=torch.float32),
                torch.tensor(np.stack([b for _, b in by[task.task_id]]), dtype=torch.float32))

    votes = {}
    library.steps = 1
    for task in stream_tasks:   # anchor map: exhaustive depth-1 re-route, 12 evaluations per anchor
        if plan[task.task_id] == 1:
            xs, ys = support(task)
            old, new = int(current[task.task_id][0]), int(enum_route(library, xs, ys)[0])
            votes.setdefault(old, {}).setdefault(new, 0)
            votes[old][new] += 1
    slot_map = {a: max(v, key=v.get) for a, v in votes.items()}
    rows, query = [], {'current': [], 'exhaustive': [], 'cd': [], 'map': [], 'map_cd': []}
    for task in stream_tasks:
        d = plan[task.task_id]
        if d < 2:
            continue
        library.steps = d
        xs, ys = support(task)
        cur = tuple(int(r) for r in current[task.task_id][:d])
        exh = tuple(int(r) for r in enum_route(library, xs, ys))
        cd, cd_mse, evals, sweeps = coordinate_descent(library, xs, ys, cur)
        mapped = tuple(slot_map.get(r, r) for r in cur)
        map_cd, map_cd_mse, map_evals, _ = coordinate_descent(library, xs, ys, mapped)
        exh_mse = mse(library, xs, ys, exh)
        rows.append({'depth': d, 'stale': cur != exh, 'agree': cd == exh,
                     'ratio': cd_mse / exh_mse if exh_mse > 0 else 1.0, 'evaluations': evals, 'sweeps': sweeps,
                     'map_agree': mapped == exh, 'map_cd_agree': map_cd == exh,
                     'map_cd_ratio': map_cd_mse / exh_mse if exh_mse > 0 else 1.0, 'map_cd_evaluations': map_evals})
        if task.task_id in canonical_ids:
            ex = torch.tensor(task.eval_x, dtype=torch.float32)
            for name, route in (('current', cur), ('exhaustive', exh), ('cd', cd), ('map', mapped),
                                ('map_cd', map_cd)):
                with torch.no_grad():
                    query[name].append(float(nmse(library.hard(ex, list(route)).numpy(), task.eval_y)))
    return rows, {k: float(np.median(v)) for k, v in query.items()}


def summarize(rows):
    out = {}
    for d in sorted({r['depth'] for r in rows}):
        rs = [r for r in rows if r['depth'] == d]
        stale = [r for r in rs if r['stale']]
        out[str(d)] = {'tasks': len(rs), 'stale': len(stale), 'cd_agrees': sum(r['agree'] for r in rs),
                       'cd_agrees_on_stale': sum(r['agree'] for r in stale),
                       'map_agrees': sum(r['map_agree'] for r in rs),
                       'map_cd_agrees': sum(r['map_cd_agree'] for r in rs),
                       'map_cd_agrees_on_stale': sum(r['map_cd_agree'] for r in stale),
                       'map_cd_worse_by_10pct': sum(r['map_cd_ratio'] > 1.1 for r in rs),
                       'map_cd_evaluations_median': float(np.median([r['map_cd_evaluations'] for r in rs])),
                       'ratio_median': float(np.median([r['ratio'] for r in rs])),
                       'ratio_max': float(max(r['ratio'] for r in rs)),
                       'worse_by_10pct': sum(r['ratio'] > 1.1 for r in rs),
                       'evaluations_median': float(np.median([r['evaluations'] for r in rs])),
                       'exhaustive_evaluations': 12 ** d, 'sweeps_max': max(r['sweeps'] for r in rs)}
    return out


def main():
    started, all_rows, cells = time.time(), [], {}
    for band, worlds in BANDS.items():
        for w in worlds:
            for s in (0, 1, 2):
                rows, q = cell(band, w, s)
                all_rows += rows
                cells[f'{band}_w{w}_s{s}'] = {'query_median': q, 'depth_summary': summarize(rows)}
                print(band, w, s, q, flush=True)
    report = {'tier': 0, 'descriptive': True, 'worlds': 'development O2 13-19, O3 20-26; no sealed world',
              'max_sweeps': MAX_SWEEPS, 'overall': summarize(all_rows), 'cells': cells,
              'seconds': time.time() - started, 'finished_utc': now()}
    atomic_json(OUTPUT, report)
    print(json.dumps(report['overall'], indent=1))


if __name__ == '__main__':
    main()
