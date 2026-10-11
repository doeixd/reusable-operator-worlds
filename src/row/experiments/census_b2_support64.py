"""B2 Tier 0 census rerun at 64 support (descriptive): size the B2-online v2 loop bar against what the end-of-stream
re-fit can see (64 retained examples per task), not the 128 of the original census.

Runs `census_b2_iteration_necessity` unchanged (LIBRARY = 'final', gates G0/G1) with SIZES = (64, 128, 512). Support
sets are prefixes of one 512-draw per task, so the 128 and 512 rows must reproduce the committed
`reports/b2_iteration_necessity_census.json` exactly (anchor, checked in `main`). Output
`reports/b2_support64_census.json`.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from row.experiments import census_b2_iteration_necessity as c2
from row.experiments.so1_storage import atomic_json, now

c2.SIZES = (64, 128, 512)   # set at import so spawned workers see it too
OUTPUT = Path('reports/b2_support64_census.json')


def one(w):
    return c2.one(w)


def main():
    with ProcessPoolExecutor(max_workers=3) as ex:
        worlds = list(ex.map(one, c2.WORLDS))
    ref = json.loads(Path('reports/b2_iteration_necessity_census.json').read_text())
    anchor = all(r['sizes'][n] == q['sizes'][n] for wd, wr in zip(worlds, ref['worlds'])
                 for r, q in zip(wd['rows'], wr['rows']) for n in ('128', '512'))
    rows = [r for wd in worlds for r in wd['rows']]
    summ = {'G0': [wd['G0_reload_reproduces'] for wd in worlds], 'G1': [wd['G1_final_reproduces'] for wd in worlds],
            'anchor_128_512_reproduce': anchor,
            'LOOP_ORACLE_median': float(np.median([r['LOOP_ORACLE'] for r in rows]))}
    for n in c2.SIZES:
        k = str(n)
        summ[f'n{n}'] = {a: float(np.median([r['sizes'][k][a] for r in rows])) for a in ('REFUSAL', 'KWAY_INPUT', 'LOOP_LEARNED')}
        summ[f'n{n}']['LOOP_p10_p90'] = [float(np.percentile([r['sizes'][k]['LOOP_LEARNED'] for r in rows], q)) for q in (10, 90)]
        summ[f'n{n}']['inferred_count_accuracy_median'] = float(np.median([r['sizes'][k]['inferred_count_accuracy'] for r in rows]))
    atomic_json(OUTPUT, {'tier': 0, 'descriptive': True, 'summary': summ, 'worlds': worlds, 'finished_utc': now()})
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
