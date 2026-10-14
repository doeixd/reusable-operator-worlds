"""Independent scorer for G1T (`G1T_TANH_FORMATION_PLAN.md`; Tier 1, exploratory).

Recomputes the reading from the durable per-cell records and G1's committed rotated reference (not the runner's
`reading`), re-checks every cell's record hash and stamp and the gates, and refuses a report whose plan, config, G0F
report or G1 report changed since the run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PLAN = Path('G1T_TANH_FORMATION_PLAN.md')
REPORT = Path('reports/g1t_tanh_formation.json')
CELLS = Path('artifacts/g1t_tanh_formation/cells')
G1_REPORT = Path('reports/g1_formation_probe.json')
WORLDS = (70, 71, 72)
ARMS = ('REROUTE_WAKE', 'SHUFFLED', 'STAGED', 'PLAIN', 'RW_SLEEP', 'SLEEP')
THRESHOLD = 0.05
IDENTITY_ARI = 0.8


def key(arm, w):
    return f'{arm}_w{w}'


def ok(v):
    return v is not None and math.isfinite(v) and v < THRESHOLD


def _fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def score(report=REPORT, cells=CELLS):
    rep = json.loads(Path(report).read_text())
    proto = rep['protocol']
    for path, digest in proto['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{path} changed since the run')
    for need in (PLAN, G1_REPORT):
        if need.as_posix() not in proto['input_sha256']:
            raise ValueError(f'report does not stamp {need}')
    if proto['threshold'] != THRESHOLD or proto['identity_ari'] != IDENTITY_ARI:
        raise ValueError('registered thresholds differ from the plan')
    g = rep['gates']
    if not (g['all_pass'] and all(g['E1'].values()) and all(g['E2'].values()) and g['F1'] and g['L1'] and g['L2']):
        raise ValueError('gates did not pass')
    records = {}
    for arm in ARMS:
        for w in WORLDS:
            k = key(arm, w)
            saved = json.loads((Path(cells) / f'{k}.json').read_text())
            if saved['stamp'] != {'protocol_sha256': rep['protocol_sha256']} or not saved['complete']:
                raise ValueError(f'{k}: stamp')
            if _fingerprint(saved['record']) != saved['record_sha256']:
                raise ValueError(f'{k}: record hash')
            if saved['record'] != rep['cells'][k]:
                raise ValueError(f'{k}: report differs from the durable cell')
            if saved['record']['scale'] != 1:
                raise ValueError(f'{k}: not a full-scale cell')
            records[k] = saved['record']
    g1 = json.loads(G1_REPORT.read_text())['cells']
    rotated = {w: g1[f'RW_SLEEP_m0_w{w}']['terminal_median'] for w in WORLDS}
    T = {k: v['terminal_median'] for k, v in records.items()}
    sleep = {w: records[key('RW_SLEEP', w)] for w in WORLDS}
    if any(s['learner_class'] != 'TanhFormLearner' for s in sleep.values()):
        raise ValueError('a slept cell is not the tanh learner')
    forms = {w: ok(sleep[w]['terminal_median']) and sleep[w]['formed']['d1_ari'] >= IDENTITY_ARI for w in WORLDS}
    anchored = [w for w in WORLDS if ok(rotated[w])]
    k = sum(forms[w] for w in anchored)
    if not anchored:
        label = 'ANCHOR_MISS'
    elif k == len(anchored):
        label = 'FORMS'
    elif k == 0:
        label = 'FAILS'
    else:
        label = 'MIXED'
    f = [sleep[w]['formed'] for w in WORLDS]
    out = {'label': label, 'anchored_worlds': anchored, 'forming_cells': k, 'denominator': len(anchored),
           'impostor_worlds': [w for w in anchored if ok(sleep[w]['terminal_median']) and not forms[w]],
           'passes': {a: sum(ok(T[key(a, w)]) for w in WORLDS) for a in ARMS},
           'terminal': {kk: T[kk] for kk in sorted(T)},
           'median_terminal': {a: float(np.median([T[key(a, w)] for w in WORLDS])) for a in ARMS},
           'rotated_reference': {str(w): rotated[w] for w in WORLDS},
           'd1_ari': [x['d1_ari'] for x in f],
           'd1_ari_wake': [records[key('REROUTE_WAKE', w)]['d1_ari_wake'] for w in WORLDS],
           'd1_ari_shuffled': [records[key('SHUFFLED', w)]['d1_ari_wake'] for w in WORLDS],
           'identifiability_median': [x['identifiability_median'] for x in f],
           'near_tie_tasks': [x['near_tie_tasks'] for x in f],
           'formed_runner_up_nmse_median': [x['formed_runner_up_nmse_median'] for x in f],
           'winner_query_below_0.05': [x['winner_query_below_0.05'] for x in f],
           'formed_s1_k1': [float(np.median(sleep[w]['formed_s1']['records']['1']['accuracy'])) for w in WORLDS],
           'formed_s1_k3': [float(np.median(sleep[w]['formed_s1']['records']['3']['accuracy'])) for w in WORLDS],
           'formed_s1_distinct_slots': [sleep[w]['formed_s1']['distinct_slots'] for w in WORLDS],
           'routes_changed_total': [records[key('REROUTE_WAKE', w)]['routes_changed_total'] for w in WORLDS]}
    out['runner_agrees'] = rep.get('reading', {}).get('label') == label
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', default=str(REPORT))
    ap.add_argument('--cells', default=str(CELLS))
    args = ap.parse_args()
    out = score(args.report, args.cells)
    print(json.dumps(out, indent=1))
    return 0 if out['runner_agrees'] else 1


if __name__ == '__main__':
    sys.exit(main())
