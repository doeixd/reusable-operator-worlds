"""Independent scorer for M2 (`M2_MOTIF_CYCLE_PLAN.md`). Recomputes the registered decision rule from the raw
per-task query NMSE in `reports/m2_motif_cycle.json`, with its own copy of the registered constants (it does not
import the runner's `summarize`), and checks it agrees with the runner's summary. Refuses an incomplete, wrong-shape
or stale report. Writes `reports/m2_motif_cycle_score.json`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

REPORT = Path('reports/m2_motif_cycle.json')
OUT = Path('reports/m2_motif_cycle_score.json')
CELLS = {f'w{w}_s{s}' for w in (84, 85, 86) for s in (0, 1, 2)}
N_DEEP, N_SHALLOW = 48, 192
SOLVED, FORMED = 0.20, 0.05          # deep query NMSE below SOLVED is solved; CYC shallow median below FORMED
CYC_BAR, GAIN_BAR = 0.8, 0.5
NEED_FORMED, NEED_INC, NEED_CYC = 7, 7, 8


def frac(values):
    return sum(1 for v in values if v < SOLVED) / len(values)


def score(report):
    if not report.get('complete') or set(report['cells']) != CELLS:
        raise ValueError('report incomplete or wrong cell set')
    proto = report['protocol']
    if (proto['solved'], proto['formed'], proto['need_streams'], proto['scale']) != (
            SOLVED, FORMED, [NEED_FORMED, NEED_INC, NEED_CYC], 1):
        raise ValueError('report protocol constants differ from the registered rule')
    per = {}
    for key in sorted(CELLS):
        br = report['cells'][key]['branches']
        row = {}
        for b in ('WAKE', 'KEEP', 'INC', 'CYC'):
            deep, shallow = list(br[b]['deep'].values()), list(br[b]['shallow'].values())
            if len(deep) != N_DEEP or len(shallow) != N_SHALLOW or not all(map(math.isfinite, deep + shallow)):
                raise ValueError(f'{key} {b}: wrong count or non-finite score')
            row[b] = {'solved': frac(deep), 'shallow_median': statistics.median(shallow),
                      'deep_median': statistics.median(deep)}
        row['formed'] = row['CYC']['shallow_median'] < FORMED
        row['incumbent_solves'] = max(row['INC']['solved'], row['KEEP']['solved']) >= CYC_BAR
        row['cyc_solves'] = row['CYC']['solved'] >= CYC_BAR
        row['gain'] = row['CYC']['solved'] - max(row['INC']['solved'], row['KEEP']['solved'])
        row['paired_gain'] = row['gain'] >= GAIN_BAR
        per[key] = row
    n = {k: sum(r[k] for r in per.values()) for k in ('formed', 'incumbent_solves', 'cyc_solves', 'paired_gain')}
    if n['formed'] < NEED_FORMED:
        label = 'FORMATION_BROKEN'
    elif n['incumbent_solves'] >= NEED_INC:
        label = 'NO_NECESSITY'
    elif n['cyc_solves'] >= NEED_CYC and n['paired_gain'] >= NEED_CYC:
        label = 'CYCLE_SOLVES'
    else:
        label = 'PARTIAL'
    return {'label': label, 'denominator_streams': len(per), 'counts': n, 'per_stream': per,
            'runner_label': report.get('summary', {}).get('label'),
            'agrees_with_runner': label == report.get('summary', {}).get('label'),
            'report_protocol_sha256': report['protocol_sha256']}


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
