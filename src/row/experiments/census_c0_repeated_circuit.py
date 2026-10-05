"""C0 Tier 0 census (descriptive; SUCCESSOR_LADDER.md, Track C, issue #3 section 2.1): with the correct route
GIVEN, does one learned operator keep its meaning when applied 1..32 times in a row?

Libraries: rebuilt deterministically from surviving D1 (depth 4) and D2 (depth 5) SHUFFLED terminals with the
end-of-stream construction that passed 21/21 and 21/21 (`deep_reroute.reroute` on all stream tasks, then O3's
sleep, 8,192 updates, sampling [1941, w, s, 64]). Cells: D1 worlds 30, 31, 32 and D2 worlds 37, 38, 39, stream 0.

Mapping teacher operation -> learned slot, support-only: the hard route of every length-1 anchor task (program
[k]) after re-route + sleep; the majority slot per k, with its purity recorded. Teacher primitive k is the world's
rotated primitive (reuse_rho = 1, so every task uses the world library).

Measured for n in 1, 2, 4, 8, 16, 32 on 256 fresh inputs per (cell, operation), seed [5100, w, k]:
- REPEAT: NMSE of the learned route [s_k] * n against the teacher's k applied n times;
- MIXED (control for generic depth drift): NMSE on 6 random teacher programs of length n per cell (seed
  [5101, w, n]) through their mapped slots;
- RANDOM_LIBRARY (instrument control): REPEAT through a freshly initialized library with the same slot map.
Also recorded: teacher output variance at each n (does the repeated teacher map contract or expand?).

Reading, stated before running: Track C is LIVE for execution if the median REPEAT NMSE is < 0.05 at n = 16 and
< 0.1 at n = 32; BINDS if it crosses 0.1 at n <= 8; otherwise read the shape. Descriptive only; no verdict.
Output `reports/c0_repeated_circuit_census.json`.
"""
from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from row.experiments import d1_depth4_formation as d1
from row.experiments import d2_depth5_formation as d2
from row.experiments import deep_reroute as dr
from row.experiments import dn_stream as dn
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now
from row.rotated_world import generate_rotated_world

OUTPUT = Path('reports/c0_repeated_circuit_census.json')
CELLS = [(4, 30), (4, 31), (4, 32), (5, 37), (5, 38), (5, 39)]
LENGTHS = (1, 2, 4, 8, 16, 32)
N_EVAL = 256
N_MIXED = 6


def rebuilt_library(depth, w, s=0):
    mod = d1 if depth == 4 else d2
    loaded = mod.load_terminal(mod.terminal_of(mod.ROOT, 'SHUFFLED4' if depth == 4 else 'SHUFFLED5', w, s), w, s)
    if depth == 4:
        cfg, model, st, canonical = loaded
        plan = {t.task_id: len(t.program.primitive_ids) for t in st}
    else:
        cfg, model, st, plan, canonical = loaded
    pool = o2d.reservoir(st, w, s, 64)
    dr.reroute(model, st, plan, pool)
    o2c.consolidate(cfg, model, pool, [t.task_id for t in st], [1941, w, s, 64], updates=8192)
    return cfg, model, st, plan


def slot_map(model, st, plan):
    routes = model.hard_routes()
    votes = {}
    for t in st:
        if plan[t.task_id] == 1:
            votes.setdefault(int(t.program.primitive_ids[0]), []).append(int(routes[t.task_id][0]))
    out = {}
    for k, v in votes.items():
        slot, count = Counter(v).most_common(1)[0]
        out[k] = {'slot': slot, 'purity': count / len(v), 'anchors': len(v)}
    return out


def teacher_apply(library, program, x):
    z = np.array(x, dtype=np.float64)
    for k in program:
        z = library[k](z)
    return z


def nmse(pred, target):
    return float(np.mean((pred - target) ** 2) / max(np.var(target), 1e-12))


def learned_apply(lib, route, x):
    lib.steps = len(route)
    with torch.no_grad():
        return lib.hard(torch.tensor(x, dtype=torch.float32), list(route)).numpy().astype(np.float64)


def cell(dw):
    torch.set_num_threads(1)
    depth, w = dw
    started = time.perf_counter()
    cfg, model, st, plan = rebuilt_library(depth, w)
    teacher = generate_rotated_world(dn.config(depth, w).world).library
    smap = slot_map(model, st, plan)
    lib = FrozenLibrary(model)
    fresh = FrozenLibrary(dn.planned_model(depth, cfg, plan))
    repeat, rand_lib, var = {}, {}, {}
    for k, m in sorted(smap.items()):
        x = np.random.default_rng(np.random.SeedSequence([5100, w, k])).normal(size=(N_EVAL, cfg.world.state_dim))
        for n in LENGTHS:
            y = teacher_apply(teacher, [k] * n, x)
            repeat[f'{k}_{n}'] = nmse(learned_apply(lib, [m['slot']] * n, x), y)
            rand_lib[f'{k}_{n}'] = nmse(learned_apply(fresh, [m['slot']] * n, x), y)
            var[f'{k}_{n}'] = float(np.var(y))
    mixed = {}
    ops = sorted(smap)
    for n in LENGTHS:
        rng = np.random.default_rng(np.random.SeedSequence([5101, w, n]))
        vals = []
        for i in range(N_MIXED):
            prog = [int(ops[j]) for j in rng.integers(0, len(ops), size=n)]
            x = rng.normal(size=(N_EVAL, cfg.world.state_dim))
            vals.append(nmse(learned_apply(lib, [smap[k]['slot'] for k in prog], x), teacher_apply(teacher, prog, x)))
        mixed[str(n)] = float(np.median(vals))
    return f'd{depth}_w{w}', {'depth': depth, 'world': w, 'slot_map': {str(k): v for k, v in smap.items()},
                              'repeat_nmse': repeat, 'random_library_nmse': rand_lib, 'teacher_var': var,
                              'mixed_median_nmse': mixed, 'seconds': time.perf_counter() - started}


def main():
    started = time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        cells = dict(pool.map(cell, CELLS))
    summary = {}
    for n in LENGTHS:
        rep = [v for c in cells.values() for key, v in c['repeat_nmse'].items() if key.endswith(f'_{n}')]
        rnd = [v for c in cells.values() for key, v in c['random_library_nmse'].items() if key.endswith(f'_{n}')]
        var = [v for c in cells.values() for key, v in c['teacher_var'].items() if key.endswith(f'_{n}')]
        mix = [c['mixed_median_nmse'][str(n)] for c in cells.values()]
        summary[str(n)] = {'repeat_median': statistics.median(rep), 'repeat_max': max(rep),
                           'repeat_q90': float(np.quantile(rep, 0.9)), 'mixed_median': statistics.median(mix),
                           'random_library_median': statistics.median(rnd), 'teacher_var_median': statistics.median(var)}
    r16, r32, early = summary['16']['repeat_median'], summary['32']['repeat_median'], \
        [n for n in LENGTHS if n <= 8 and summary[str(n)]['repeat_median'] >= 0.1]
    reading = 'LIVE' if (r16 < 0.05 and r32 < 0.1) else ('BINDS' if early else 'READ_SHAPE')
    purity = [m['purity'] for c in cells.values() for m in c['slot_map'].values()]
    out = {'tier': 0, 'descriptive': True, 'reading': reading, 'summary': summary,
           'slot_map_purity_min': min(purity), 'slot_map_purity_median': statistics.median(purity),
           'cells': cells, 'seconds': time.time() - started, 'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, indent=1))


if __name__ == '__main__':
    main()
