"""Independent scorer for M3 (`M3_FORK_CYCLE_PLAN.md`). Recomputes the registered decision rule from the raw
per-task query NMSE in `reports/m3_fork_cycle.json` with its own copy of the registered constants (it does not import
the runner's `summarize`) and checks agreement with the runner's summary. Refuses an incomplete, wrong-shape or
wrong-constant report. Writes `reports/m3_fork_cycle_score.json`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

REPORT = Path('reports/m3_fork_cycle.json')
OUT = Path('reports/m3_fork_cycle_score.json')
CELLS = {f'w{w}_s{s}' for w in (87, 88, 89) for s in (0, 1, 2)}
N_DEEP, N_SHALLOW = 48, 192
SOLVED, FORMED, BAR, GAIN = 0.20, 0.05, 0.8, 0.5
NEED_FORMED, NEED_INC, NEED_SOLVE, HARM_MAX = 7, 7, 8, 2


def score(report):
    if not report.get('complete') or set(report['cells']) != CELLS:
        raise ValueError('report incomplete or wrong cell set')
    p = report['protocol']
    if (p['solved'], p['formed'], p['need_streams'], p['scale']) != (
            SOLVED, FORMED, [NEED_FORMED, NEED_INC, NEED_SOLVE, HARM_MAX], 1):
        raise ValueError('report protocol constants differ from the registered rule')
    per = {}
    for key in sorted(CELLS):
        br = report['cells'][key]['branches']
        row = {}
        for b in ('WAKE', 'INC', 'CYC', 'FCYC'):
            deep, shallow = list(br[b]['deep'].values()), list(br[b]['shallow'].values())
            if len(deep) != N_DEEP or len(shallow) != N_SHALLOW or not all(map(math.isfinite, deep + shallow)):
                raise ValueError(f'{key} {b}: wrong count or non-finite score')
            row[b] = {'solved': sum(v < SOLVED for v in deep) / N_DEEP, 'shallow_median': statistics.median(shallow)}
        row['forks'] = len(br['FCYC']['fork']['forks'])
        row['formed'] = row['FCYC']['shallow_median'] < FORMED
        row['incumbent_solves'] = row['INC']['solved'] >= BAR
        row['fcyc_solves'] = row['FCYC']['solved'] >= BAR
        row['paired_gain'] = row['FCYC']['solved'] - row['INC']['solved'] >= GAIN
        row['harm'] = row['CYC']['solved'] >= BAR and row['FCYC']['solved'] < BAR
        row['rescue'] = row['CYC']['solved'] < BAR and row['FCYC']['solved'] >= BAR
        per[key] = row
    n = {k: sum(r[k] for r in per.values())
         for k in ('formed', 'incumbent_solves', 'fcyc_solves', 'paired_gain', 'harm', 'rescue')}
    if n['formed'] < NEED_FORMED:
        label = 'FORMATION_BROKEN'
    elif n['incumbent_solves'] >= NEED_INC:
        label = 'NO_NECESSITY'
    elif n['harm'] >= HARM_MAX:
        label = 'REPAIR_HARMS'
    elif n['fcyc_solves'] >= NEED_SOLVE and n['paired_gain'] >= NEED_SOLVE:
        label = 'REPAIRED_CYCLE_SOLVES'
    else:
        label = 'PARTIAL'
    runner = report.get('summary', {}).get('label')
    return {'label': label, 'denominator_streams': len(per), 'counts': n, 'per_stream': per, 'runner_label': runner,
            'agrees_with_runner': label == runner, 'report_protocol_sha256': report['protocol_sha256']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', default=str(REPORT))
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    result = score(json.loads(Path(args.report).read_text(encoding='utf-8')))
    out = Path(args.out)
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=1), encoding='utf-8')
    os.replace(tmp, out)
    print(result['label'], result['counts'], 'agrees' if result['agrees_with_runner'] else 'DISAGREES')
    if not result['agrees_with_runner']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
