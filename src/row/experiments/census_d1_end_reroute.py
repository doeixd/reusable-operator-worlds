"""Tier 0 census (descriptive) on D1's saved depth-4 SHUFFLED4 terminals: does O6's END-OF-STREAM protocol
(one exhaustive re-route of every stream task on its 64 retained examples, then O3's sleep) form the depth-4
substrate, as it did at depth 3?

Why: in-stream re-routing repeats the route search after every arrival (~n/2 passes), which at depth 5 is roughly
10 hours of search per cell. One end-of-stream pass at depth 5 is ~64 x 3 s. If the end-of-stream protocol works at
depth 4, depth 5 is affordable without a new re-router. Construction: O6's `run_reroute_sleep` calls (o5.reroute,
then o2c.consolidate with 8,192 updates, sampling [1941, w, s, 64]) on the depth-4 stream and learner. Development
worlds 30-36 only (opened by D1). Output `reports/d1_end_reroute_census.json`.
"""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch

from row.experiments import d1_depth4_formation as d1
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o5_reroute_sleep as o5
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.so1_storage import atomic_json, now

OUTPUT = Path('reports/d1_end_reroute_census.json')
CELLS = [(w, s) for w in d1.WORLDS for s in d1.STREAMS]


def cell(ws):
    torch.set_num_threads(1)
    w, s = ws
    started = time.perf_counter()
    cfg, model, st, canonical = d1.load_terminal(d1.terminal_of(d1.ROOT, 'SHUFFLED4', w, s), w, s)
    plan = {t.task_id: len(t.program.primitive_ids) for t in st}
    pool = o2d.reservoir(st, w, s, d1.MEMORY)
    before = library_sha(model)
    t0 = time.perf_counter()
    changed = o5.reroute(model, st, plan, pool)
    reroute_seconds = time.perf_counter() - t0
    routes_only = score(model, SimpleNamespace(tasks=canonical))['median']
    o2c.consolidate(cfg, model, pool, [t.task_id for t in st], [1941, w, s, d1.MEMORY], updates=d1.EXTRA_UPDATES)
    terminal = score(model, SimpleNamespace(tasks=canonical))
    return f'w{w}_s{s}', {'routes_changed': changed, 'reroute_seconds': reroute_seconds,
                          'after_reroute_only': routes_only, 'terminal_median': terminal['median'],
                          'library_sha256_before': before, 'library_sha256': library_sha(model),
                          'seconds': time.perf_counter() - started}


def main():
    started = time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        cells = dict(pool.map(cell, CELLS))
    d1cells = json.loads(d1.OUTPUT.read_text())['cells']
    t = [c['terminal_median'] for c in cells.values()]
    insleep = {k: d1cells[f'RW_SLEEP4_{k}']['terminal_median'] for k in cells}
    out = {'tier': 0, 'descriptive': True, 'construction': 'O6 end-of-stream exhaustive re-route + O3 sleep on D1 SHUFFLED4 terminals',
           'passes': sum(x < 0.05 for x in t), 'cells_total': len(t), 'median': statistics.median(t), 'max': max(t),
           'after_reroute_only_median': statistics.median(c['after_reroute_only'] for c in cells.values()),
           'reroute_seconds_median': statistics.median(c['reroute_seconds'] for c in cells.values()),
           'below_instream_rw_sleep4': sum(cells[k]['terminal_median'] < insleep[k] for k in cells),
           'instream_rw_sleep4_median': statistics.median(insleep.values()),
           'cells': cells, 'seconds': time.time() - started, 'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, indent=1))


if __name__ == '__main__':
    main()
