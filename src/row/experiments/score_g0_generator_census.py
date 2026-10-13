"""Independent scorer for G0 (`G0_GENERATOR_CENSUS_PLAN.md`). Recomputes every candidate's statistics from the
per-world records (not the runner's summary), applies the registered selection rule, and checks the base candidate
reproduces the committed baseline pre-measurement. Refuses a report whose plan or inputs changed since the run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PLAN = Path('G0_GENERATOR_CENSUS_PLAN.md')
REPORT = Path('reports/g0_generator_census.json')
BASELINE = Path('reports/g0_baseline_premeasure.json')
BASE_KEY = 'a0.35_g1.0_K6_m0.0'
WORLDS = (70, 71, 72)

S1_MOVED = 0.90                     # median held-out linear accuracy at k = 1 at or below this
S2_BAND = (0.01, 0.5)               # median teacher runner-up NMSE inside this (closed) band
S3_ARI_MIN = 0.30                   # depth-1 ARI in EVERY world, and above that world's depth-3 ARI
S4_VAR_RATIO_MAX, S4_NORM_MAX, S4_VAR_BAND = 4.0, 3.0, (0.25, 4.0)


def stats(rs):
    return {
        's1': float(np.median([v for r in rs for v in r['s1']['1']['accuracy']])),
        's2': float(np.median([t['runner_up_nmse'] for r in rs for t in r['s2']])),
        'ari1': [r['s3']['ari_depth1'] for r in rs], 'ari3': [r['s3']['ari_depth3_mean'] for r in rs],
        'var_med': float(np.median([r['s4']['output_variance_median'] for r in rs])),
        'var_ratio': float(max(r['s4']['output_variance_max_over_min'] for r in rs)),
        'norm_max': float(max(r['s4']['norm_ratio_max'] for r in rs)),
        'finite': all(r['s4']['finite'] for r in rs)}


def judge(st):
    s1_moved = st['s1'] <= S1_MOVED
    s2_in_band = S2_BAND[0] <= st['s2'] <= S2_BAND[1]
    s3 = all(a >= S3_ARI_MIN and a > b for a, b in zip(st['ari1'], st['ari3']))
    s4 = (st['finite'] and st['var_ratio'] <= S4_VAR_RATIO_MAX and st['norm_max'] <= S4_NORM_MAX
          and S4_VAR_BAND[0] <= st['var_med'] <= S4_VAR_BAND[1])
    moves = int(s1_moved) + int(s2_in_band)
    return {'s1_moved': s1_moved, 's2_in_band': s2_in_band, 's3_usable': s3, 's4_usable': s4, 'moves': moves,
            'passes': moves >= 1 and s3 and s4}


def changed(rec):
    return int(rec['alpha'] != 0.35) + int(rec['gain'] != 1.0) + int(rec['K'] != 6) + int(rec['shared'] != 0.0)


def _strip(r):
    return {k: v for k, v in r.items() if k != 'seconds'}


def score(report=REPORT, baseline=BASELINE, plan=PLAN):
    rep = json.loads(Path(report).read_text())
    hashes = rep['protocol']['input_sha256']
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{path} changed since the run')
    if Path(plan).as_posix() not in hashes:
        raise ValueError('report does not stamp the plan')
    by = {}
    for r in rep['records']:
        by.setdefault(r['candidate'], []).append(r)
    for key, rs in by.items():
        if sorted(r['world'] for r in rs) != list(WORLDS):
            raise ValueError(f'{key}: worlds')
    base_now = sorted((_strip(r) for r in by[BASE_KEY]), key=lambda r: r['world'])
    base_then = sorted((_strip(r) for r in json.loads(Path(baseline).read_text())['records']), key=lambda r: r['world'])
    anchor = base_now == base_then
    rows = {}
    for key, rs in by.items():
        st = stats(rs)
        rows[key] = {**st, **judge(st), 'changed_knobs': changed(rs[0]), 'K': rs[0]['K']}
    passing = sorted((k for k, v in rows.items() if v['passes']),
                     key=lambda k: (-rows[k]['moves'], rows[k]['K'], rows[k]['changed_knobs'], rows[k]['s1'], k))
    if not anchor:
        label, chosen = 'UNINTERPRETABLE', None
    elif passing:
        label, chosen = 'SELECTED', passing[0]
    else:
        label, chosen = 'NO_CANDIDATE', None
    best_s1 = min((k for k, v in rows.items() if v['s1_moved'] and v['s3_usable'] and v['s4_usable']),
                  key=lambda k: (rows[k]['K'], rows[k]['changed_knobs'], rows[k]['s1'], k), default=None)
    best_s2 = min((k for k, v in rows.items() if v['s2_in_band'] and v['s3_usable'] and v['s4_usable']),
                  key=lambda k: (rows[k]['K'], rows[k]['changed_knobs'], rows[k]['s1'], k), default=None)
    out = {'anchor_reproduces': anchor, 'label': label, 'chosen': chosen, 'passing': passing,
           'best_s1_mover': best_s1, 'best_s2_mover': best_s2, 'candidates': rows}
    s = rep.get('summary', {})
    out['summary_agrees'] = all(
        abs(s[k]['s1_k1_median'] - rows[k]['s1']) < 1e-12 and abs(s[k]['s2_runner_up_median'] - rows[k]['s2']) < 1e-12
        for k in rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', default=str(REPORT))
    args = ap.parse_args()
    out = score(args.report)
    print(json.dumps({k: v for k, v in out.items() if k != 'candidates'}, indent=1))
    return 0 if out['summary_agrees'] and out['label'] != 'UNINTERPRETABLE' else 1


if __name__ == '__main__':
    sys.exit(main())
