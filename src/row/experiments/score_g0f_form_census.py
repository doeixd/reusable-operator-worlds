"""Independent scorer for G0F (`G0F_FORM_CENSUS_PLAN.md`). Recomputes every candidate's statistics from the per-world
records (not the runner's summary), applies the registered rules and selection order, checks gate A0 against G0's and
G0F's baseline pre-measurements, and refuses a report whose plan or inputs changed since the run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PLAN = Path('G0F_FORM_CENSUS_PLAN.md')
REPORT = Path('reports/g0f_form_census.json')
G0_BASELINE = Path('reports/g0_baseline_premeasure.json')
BASELINE = Path('reports/g0f_baseline_premeasure.json')
ANCHOR = 'rot0.35'
WORLDS = (70, 71, 72)
CANDIDATES = 11

S1_MOVED = 0.90
S2_BAND = (0.01, 0.5)
S3_ARI_MIN = 0.30
S4_VAR_RATIO_MAX, S4_NORM_MAX, S4_VAR_BAND = 4.0, 3.0, (0.25, 4.0)
S6_MAX = 0.01


def stats(rs):
    return {
        's1': float(np.median([v for r in rs for v in r['s1']['1']['accuracy']])),
        's1_k3': float(np.median([v for r in rs for v in r['s1']['3']['accuracy']])),
        's2': float(np.median([t['runner_up_nmse'] for r in rs for t in r['s2']])),
        'ari1': [r['s3']['ari_depth1'] for r in rs], 'ari3': [r['s3']['ari_depth3_mean'] for r in rs],
        'var_med': float(np.median([r['s4']['output_variance_median'] for r in rs])),
        'var_ratio': float(max(r['s4']['output_variance_max_over_min'] for r in rs)),
        'norm_max': float(max(r['s4']['norm_ratio_max'] for r in rs)),
        'finite': all(r['s4']['finite'] for r in rs),
        's6': float(np.median([x['heldout_nmse'] for r in rs for x in r['s6']]))}


def judge(st):
    out = {'s1_moved': st['s1'] <= S1_MOVED,
           's3_usable': all(a >= S3_ARI_MIN and a > b for a, b in zip(st['ari1'], st['ari3'])),
           's4_usable': (st['finite'] and st['var_ratio'] <= S4_VAR_RATIO_MAX and st['norm_max'] <= S4_NORM_MAX
                         and S4_VAR_BAND[0] <= st['var_med'] <= S4_VAR_BAND[1]),
           's6_learnable': st['s6'] <= S6_MAX,
           's2_in_band': S2_BAND[0] <= st['s2'] <= S2_BAND[1]}
    out['passes'] = out['s1_moved'] and out['s3_usable'] and out['s4_usable'] and out['s6_learnable']
    return out


def _strip(r):
    return {k: v for k, v in r.items() if k != 'seconds'}


def _by_world(records):
    return {r['world']: r for r in records}


def gate(anchor_records, g0_baseline=G0_BASELINE, baseline=BASELINE):
    now = _by_world(anchor_records)
    g0b = _by_world(json.loads(Path(g0_baseline).read_text())['records'])
    own = _by_world(json.loads(Path(baseline).read_text())['records'])
    if sorted(now) != list(WORLDS) or sorted(g0b) != list(WORLDS) or sorted(own) != list(WORLDS):
        return False
    same_g0 = all(now[w][s] == g0b[w][s] for w in WORLDS for s in ('s1', 's2', 's4'))
    same_own = all(_strip(now[w]) == _strip(own[w]) for w in WORLDS)
    return same_g0 and same_own


def score(report=REPORT, plan=PLAN, g0_baseline=G0_BASELINE, baseline=BASELINE, check_inputs=True):
    rep = json.loads(Path(report).read_text())
    if check_inputs:
        hashes = rep['protocol']['input_sha256']
        for path, digest in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'{path} changed since the run')
        if Path(plan).as_posix() not in hashes:
            raise ValueError('report does not stamp the plan')
    by = {}
    for r in rep['records']:
        by.setdefault(r['candidate'], []).append(r)
    if len(by) != CANDIDATES:
        raise ValueError(f'{len(by)} candidates, expected {CANDIDATES}')
    for key, rs in by.items():
        if sorted(r['world'] for r in rs) != list(WORLDS):
            raise ValueError(f'{key}: worlds')
    a0 = gate(by[ANCHOR], g0_baseline, baseline)
    rows = {}
    for key, rs in by.items():
        st = stats(rs)
        rows[key] = {**st, **judge(st), 'form': rs[0]['form'], 'knob': rs[0]['knob']}
    passing = sorted((k for k, v in rows.items() if v['passes']),
                     key=lambda k: (-min(rows[k]['ari1']), rows[k]['s1'], k))
    if not a0:
        label, chosen = 'UNINTERPRETABLE', None
    elif passing:
        label, chosen = 'SELECTED', passing[0]
    else:
        label, chosen = 'NO_FORM', None
    per_form = {}
    for f in sorted({v['form'] for v in rows.values()}):
        ks = [k for k in rows if rows[k]['form'] == f]
        per_form[f] = {'passing': [k for k in ks if rows[k]['passes']],
                       'best_s1_mover': min(ks, key=lambda k: (rows[k]['s1'], k))}
    out = {'label': label, 'chosen': chosen, 'gate_a0': a0, 'passing': passing, 'per_form': per_form, 'rows': rows}
    ref = rep.get('reading')
    out['runner_agrees'] = None if ref is None else (ref.get('label') == label and ref.get('chosen') == chosen)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', default=str(REPORT))
    args = ap.parse_args()
    out = score(args.report)
    print(json.dumps(out, indent=1))
    return 0 if out['runner_agrees'] in (True, None) else 1


if __name__ == '__main__':
    sys.exit(main())
