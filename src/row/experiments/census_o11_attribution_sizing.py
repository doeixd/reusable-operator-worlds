"""Tier 0 census (descriptive) sizing O11's attribution clause before it is frozen.

O11 registers `n_better` = cells where wake + in-stream re-route + sleep (`RW_SLEEP`) ends strictly below wake +
sleep without re-routing (`SLEEP`). On sealed bands sleep alone passes 36-38 of 45, so the clause has room only if
in-stream re-routing also lowers terminals that sleep already passes. This measures the per-cell sign rate on O9's
opened worlds (930-944): `o3.run_sleep` verbatim on O9's saved SHUFFLED terminals, compared with O10's committed
`RW_SLEEP` cells on the same streams. Read-only on O9/O10; no sealed world beyond 930-944 is touched.
Output `reports/o11_attribution_sizing.json`.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from row.experiments import o3_online_sleep as o3
from row.experiments.so1_storage import atomic_json, now

OUTPUT = Path('reports/o11_attribution_sizing.json')
O9_WORK = Path('artifacts/o9_sealed_online/work')
O10_REPORT = Path('reports/o10_rw_sleep.json')
CELLS = [(w, s) for w in range(930, 945) for s in range(3)]


def cell(ws):
    torch.set_num_threads(1)
    w, s = ws
    rec = o3.run_sleep(w, s, O9_WORK / f'SHUFFLED_w{w}_s{s}' / 'lifetime' / 'model.pt')
    return w, s, rec['terminal_median']


def main():
    started = time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        sleep = {f'w{w}_s{s}': m for w, s, m in pool.map(cell, CELLS)}
    rws = {k: v['terminal_median'] for k, v in json.loads(O10_REPORT.read_text())['cells'].items()}
    better = [k for k in sleep if rws[k] < sleep[k]]
    fails = [k for k in sleep if not sleep[k] < 0.05]
    out = {'tier': 0, 'descriptive': True, 'cells': {k: {'SLEEP': sleep[k], 'RW_SLEEP': rws[k]} for k in sleep},
           'sleep_passes': 45 - len(fails), 'sleep_fails': fails,
           'rw_sleep_rescues_sleep_fails': sum(rws[k] < 0.05 for k in fails),
           'n_better': len(better), 'n_better_among_sleep_passes': sum(k in better for k in sleep if k not in fails),
           'median_ratio_rw_sleep_over_sleep': float(np.median([rws[k] / sleep[k] for k in sleep])),
           'seconds': time.time() - started, 'finished_utc': now()}
    atomic_json(OUTPUT, out)
    print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, indent=1))


if __name__ == '__main__':
    main()
