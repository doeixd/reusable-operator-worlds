"""Independent scorer for the G1 formation probe (`G1_FORMATION_PROBE_PLAN.md`; Tier 1, exploratory).

Recomputes the reading from the durable per-cell records (not the runner's `reading`), re-checks every cell's record
hash and stamp, the gates, and refuses a report whose plan, config or G0 report changed since the run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PLAN = Path('G1_FORMATION_PROBE_PLAN.md')
REPORT = Path('reports/g1_formation_probe.json')
CELLS = Path('artifacts/g1_formation_probe/cells')
WORLDS = (70, 71, 72)
SHARED = (0.0, 4.0)
ARMS = ('REROUTE_WAKE', 'PLAIN', 'RW_SLEEP')
THRESHOLD = 0.05


def key(arm, m, w):
    return f'{arm}_m{m:g}_w{w}'


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
    if PLAN.as_posix() not in proto['input_sha256']:
        raise ValueError('report does not stamp the plan')
    if not rep['gates']['all_pass'] or not rep['gates']['E1']['passes'] or not rep['gates']['E2']['passes']:
        raise ValueError('gates did not pass')
    records = {}
    for arm in ARMS:
        for m in SHARED:
            for w in WORLDS:
                k = key(arm, m, w)
                saved = json.loads((Path(cells) / f'{k}.json').read_text())
                if saved['stamp'] != {'protocol_sha256': rep['protocol_sha256']} or not saved['complete']:
                    raise ValueError(f'{k}: stamp')
                if _fingerprint(saved['record']) != saved['record_sha256']:
                    raise ValueError(f'{k}: record hash')
                if saved['record'] != rep['cells'][k]:
                    raise ValueError(f'{k}: report differs from the durable cell')
                records[k] = saved['record']
    T = {k: v['terminal_median'] for k, v in records.items()}
    anchored = [w for w in WORLDS if ok(T[key('RW_SLEEP', 0.0, w)])]
    k4 = sum(ok(T[key('RW_SLEEP', 4.0, w)]) for w in anchored)
    if not anchored:
        label = 'ANCHOR_MISS'
    elif k4 == len(anchored):
        label = 'FORMS'
    elif k4 == 0:
        label = 'FAILS'
    else:
        label = 'MIXED'
    formed = {m: [records[key('RW_SLEEP', m, w)]['formed'] for w in WORLDS] for m in SHARED}
    out = {'label': label, 'anchored_worlds': anchored, 'm4_passes': k4, 'denominator': len(anchored),
           'plain_passes': {f'm{m:g}': sum(ok(T[key('PLAIN', m, w)]) for w in WORLDS) for m in SHARED},
           'terminal': {k: T[k] for k in sorted(T)},
           'formed_by_m': {f'm{m:g}': {
               'd1_ari': [f['d1_ari'] for f in formed[m]],
               'identifiability_median': [f['identifiability_median'] for f in formed[m]],
               'near_tie_tasks': [f['near_tie_tasks'] for f in formed[m]],
               'formed_runner_up_nmse_median': [f['formed_runner_up_nmse_median'] for f in formed[m]],
               'winner_query_nmse_median': [f['winner_query_nmse_median'] for f in formed[m]],
               'winner_query_below_0.05': [f['winner_query_below_0.05'] for f in formed[m]]} for m in SHARED},
           'd1_ari_wake': {f'm{m:g}': [records[key('REROUTE_WAKE', m, w)]['d1_ari_wake'] for w in WORLDS]
                           for m in SHARED},
           'median_terminal': {f'{a}_m{m:g}': float(np.median([T[key(a, m, w)] for w in WORLDS]))
                               for a in ARMS for m in SHARED}}
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
