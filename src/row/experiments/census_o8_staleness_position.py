"""Tier 0 census (descriptive, read-only): WHEN do routes go stale in an online stream, and what does
exhaustive re-routing cost per depth?

Motivation (decision 14, 2026-10-03). O6 confirmed wake + exhaustive re-route + sleep at depth 3 and
O7 showed a gradient re-router matches it. The proposed next rung was a depth-4+ world "where
exhaustive search is infeasible". O6/O7 logs put exhaustive re-routing of all 188 tasks at ~1.7 s
per cell at depth 3, so the first question is where exhaustive search actually binds, and the
second is whether staleness is an END-OF-STREAM artefact (routes committed early, library moved
on) that an IN-STREAM re-router would address. Both are answerable from saved terminals.

Measures, on every saved SHUFFLED terminal of O2 (worlds 13-19) and O3 (worlds 20-26), and on
O3's INTERLEAVED terminals (consolidation during the stream):
- per stream task: arrival position, depth, whether its recorded hard route differs from the
  exhaustive-search route on its 64 reservoir examples (`stale`), and the support-MSE ratio
  recorded/exhaustive (how costly the staleness is on support);
- stale counts by arrival tercile and by depth;
- seconds for one exhaustive pass over the 188 tasks;
- on one library, all-route evaluation seconds at depth 3 and depth 4 on 64 examples, from which the
  12x-per-depth scaling is checked rather than assumed.

Development terminals only (no sealed world). Output `reports/o8_staleness_position_census.json`.
Nothing is trained or modified; the library is frozen and the models are not written back.
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
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now

OUTPUT = Path('reports/o8_staleness_position_census.json')
ARMS = {'SHUFFLED': ('O2', 'O3'), 'INTERLEAVED': ('O3',)}
TERCILES = ((0, 63), (63, 126), (126, 188))


def terminal_path(arm, band, w, s):
    return o5.BANDS[band]['work'] / f'{arm}_w{w}_s{s}' / 'lifetime' / 'model.pt'


def mse(library, x, y, route):
    with torch.no_grad():
        return float(torch.mean((library.hard(x, list(route)) - y) ** 2))


def cell(arm, band, w, s):
    _, model, stream_tasks, canonical = o3.load_shuffled_terminal(terminal_path(arm, band, w, s), w, s)
    from row.experiments import o2_online_reliability as o2
    _, _, _, plan, _ = o2.build_stream('SHUFFLED', w, s)
    pool = o2d.reservoir(stream_tasks, w, s, o3.MEMORY)
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    library = FrozenLibrary(model)
    current = model.hard_routes()
    rows, seconds = [], 0.0
    for position, task in enumerate(stream_tasks):
        d = plan[task.task_id]
        xs = torch.tensor(np.stack([a for a, _ in by[task.task_id]]), dtype=torch.float32)
        ys = torch.tensor(np.stack([b for _, b in by[task.task_id]]), dtype=torch.float32)
        cur = tuple(int(r) for r in current[task.task_id][:d])
        library.steps = d
        t0 = time.perf_counter()
        exh = tuple(int(r) for r in enum_route(library, xs, ys))
        seconds += time.perf_counter() - t0
        m_cur, m_exh = mse(library, xs, ys, cur), mse(library, xs, ys, exh)
        rows.append({'position': position, 'depth': d, 'stale': cur != exh,
                     'positions_changed': sum(a != b for a, b in zip(cur, exh)),
                     'support_ratio': m_cur / m_exh if m_exh > 0 else 1.0})
    by_tercile = {}
    for lo, hi in TERCILES:
        sub = [r for r in rows if lo <= r['position'] < hi]
        stale = [r for r in sub if r['stale']]
        by_tercile[f'{lo}-{hi - 1}'] = {
            'tasks': len(sub), 'stale': len(stale),
            'stale_depth3': sum(r['depth'] == 3 for r in stale), 'depth3_tasks': sum(r['depth'] == 3 for r in sub),
            'median_support_ratio_stale': float(np.median([r['support_ratio'] for r in stale])) if stale else None}
    by_depth = {str(d): {'tasks': sum(r['depth'] == d for r in rows), 'stale': sum(r['depth'] == d and r['stale'] for r in rows)}
                for d in (1, 2, 3)}
    stale_rows = [r for r in rows if r['stale']]
    return {'arm': arm, 'band': band, 'world': w, 'stream': s, 'tasks': len(rows), 'stale': len(stale_rows),
            'by_tercile': by_tercile, 'by_depth': by_depth,
            'positions_changed_hist': {str(k): sum(r['positions_changed'] == k for r in stale_rows) for k in (1, 2, 3)},
            'median_support_ratio_stale': float(np.median([r['support_ratio'] for r in stale_rows])) if stale_rows else None,
            'exhaustive_seconds': seconds, 'rows': rows}


def timing(band='O3', w=20, s=0, n_tasks=5):
    """All-route evaluation seconds at depth 3 and depth 4 on 64 examples; depth 4 is 12x the routes."""
    _, model, stream_tasks, canonical = o3.load_shuffled_terminal(terminal_path('SHUFFLED', band, w, s), w, s)
    pool = o2d.reservoir(stream_tasks, w, s, o3.MEMORY)
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    library = FrozenLibrary(model)
    out = {}
    for d in (3, 4):
        library.steps = d
        secs = []
        for task in canonical[:n_tasks]:
            xs = torch.tensor(np.stack([a for a, _ in by[task.task_id]]), dtype=torch.float32)
            ys = torch.tensor(np.stack([b for _, b in by[task.task_id]]), dtype=torch.float32)
            t0 = time.perf_counter()
            library.all_route_support_mse(xs, ys)
            secs.append(time.perf_counter() - t0)
        out[str(d)] = {'routes': library.slots ** d, 'median_seconds_per_task': float(np.median(secs)),
                       'examples': int(xs.shape[0])}
    out['ratio_4_over_3'] = out['4']['median_seconds_per_task'] / out['3']['median_seconds_per_task']
    out['projected_seconds_per_task'] = {str(d): out['3']['median_seconds_per_task'] * 12 ** (d - 3) for d in range(3, 9)}
    out['projected_seconds_per_188_task_pass'] = {d: v * 188 for d, v in out['projected_seconds_per_task'].items()}
    out['note'] = ('projection compounds the depth-3 time by 12 per depth; depth >= 5 needs chunking '
                   '(64 x 12^5 x 16 floats = 1.0 GiB per task unchunked), see l0d_depth5_memory_gate')
    return out


def summarize(cells):
    out = {}
    for arm in ARMS:
        sub = [c for c in cells.values() if c['arm'] == arm]
        if not sub:
            continue
        terc = {}
        for key in sub[0]['by_tercile']:
            terc[key] = {'stale': sum(c['by_tercile'][key]['stale'] for c in sub),
                         'tasks': sum(c['by_tercile'][key]['tasks'] for c in sub)}
            terc[key]['fraction'] = terc[key]['stale'] / terc[key]['tasks']
        depth = {}
        for d in ('1', '2', '3'):
            depth[d] = {'stale': sum(c['by_depth'][d]['stale'] for c in sub), 'tasks': sum(c['by_depth'][d]['tasks'] for c in sub)}
            depth[d]['fraction'] = depth[d]['stale'] / depth[d]['tasks']
        out[arm] = {'cells': len(sub), 'stale_total': sum(c['stale'] for c in sub),
                    'stale_per_cell_median': float(np.median([c['stale'] for c in sub])),
                    'by_tercile': terc, 'by_depth': depth,
                    'positions_changed_hist': {k: sum(c['positions_changed_hist'][k] for c in sub) for k in ('1', '2', '3')},
                    'exhaustive_seconds_median': float(np.median([c['exhaustive_seconds'] for c in sub]))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--one', action='store_true', help='one cell, prints it')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.one:
        c = cell('SHUFFLED', 'O3', 20, 0)
        c.pop('rows')
        print(json.dumps(c, indent=1))
        return
    started, cells = time.time(), {}
    for arm, bands in ARMS.items():
        for band in bands:
            for w in o5.BANDS[band]['worlds']:
                for s in o5.STREAMS:
                    key = f'{arm}_{band}_w{w}_s{s}'
                    cells[key] = cell(arm, band, w, s)
                    c = cells[key]
                    print(key, 'stale', c['stale'], 'terciles', [v['stale'] for v in c['by_tercile'].values()],
                          'depth', [v['stale'] for v in c['by_depth'].values()], f"{c['exhaustive_seconds']:.1f}s", flush=True)
                    atomic_json(OUTPUT, {'tier': 0, 'descriptive': True, 'partial': True, 'cells': cells})
    t = timing()
    print('timing', json.dumps(t, indent=1))
    atomic_json(OUTPUT, {'tier': 0, 'descriptive': True, 'partial': False, 'worlds': 'development only (O2 13-19, O3 20-26)',
                         'memory_per_task': o3.MEMORY, 'summary': summarize(cells), 'timing': t, 'cells': cells,
                         'seconds': time.time() - started, 'finished_utc': now()})


if __name__ == '__main__':
    main()
