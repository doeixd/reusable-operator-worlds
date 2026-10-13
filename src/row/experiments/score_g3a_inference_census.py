"""Independent scorer for G3a (`G3A_INFERENCE_CENSUS_PLAN.md`).

Recomputes rule A (SG0's cell rule), rule B (program-level bind fraction) and the label from the per-program rows of
the durable cell records, re-checks stamps, record hashes, both anchors and the run's input hashes, and refuses a
report that differs from its cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PLAN = Path('G3A_INFERENCE_CENSUS_PLAN.md')
REPORT = Path('reports/g3a_inference_census.json')
CELLS = Path('artifacts/g3a_inference_census/cells')
WORLDS = (70, 71, 72)
SHARED = (0.0, 4.0)
SUPPORTS = ('128', '8', '2')
PROGRAMS = 64
RULE_B_MIN, RULE_B_MARGIN, HEADROOM_K = 0.10, 0.05, 2


def key(m, w):
    return f'm{m:g}_w{w}'


def _fp(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def label_of(k0, k4, b_supports):
    if k0 >= HEADROOM_K:
        return 'CONTROL_FIRES'
    if k4 >= HEADROOM_K:
        return 'HEADROOM'
    return 'PARTIAL' if b_supports else 'NO_HEADROOM'


def score(report=REPORT, cells=CELLS, check_inputs=True):
    rep = json.loads(Path(report).read_text())
    if check_inputs:
        for path, digest in rep['protocol']['input_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'{path} changed since the run')
        if PLAN.as_posix() not in rep['protocol']['input_sha256']:
            raise ValueError('report does not stamp the plan')
    recs = {}
    for m in SHARED:
        for w in WORLDS:
            saved = json.loads((Path(cells) / f'{key(m, w)}.json').read_text())
            if saved['stamp'] != {'protocol_sha256': rep['protocol_sha256']} or not saved['complete']:
                raise ValueError(f'{key(m, w)}: stamp')
            if _fp(saved['record']) != saved['record_sha256']:
                raise ValueError(f'{key(m, w)}: record hash')
            if saved['record'] != rep['cells'][key(m, w)]:
                raise ValueError(f'{key(m, w)}: report differs from the durable cell')
            r = saved['record']
            if not (r['anchor_terminal'] and r['anchor_formed']) or len(r['rows']) != PROGRAMS:
                raise ValueError(f'{key(m, w)}: anchors or program count')
            recs[(m, w)] = r
    k, frac, fails = {}, {}, {}
    for m in SHARED:
        k[m] = 0
        frac[m], fails[m] = {}, {}
        for s in SUPPORTS:
            binds = failures = 0
            for w in WORLDS:
                rows = [row['by_support'][s] for row in recs[(m, w)]['rows']]
                k[m] += int(np.median([r['regret'] for r in rows]) > np.median([r['null_floor'] for r in rows]))
                binds += sum(r['regret'] > r['null_floor'] for r in rows)
                failures += sum(r['nmse_hat_qb'] >= 0.05 and r['nmse_star_qb'] < 0.05 for r in rows)
            frac[m][s] = binds / (len(WORLDS) * PROGRAMS)
            fails[m][s] = failures
    b = [s for s in SUPPORTS if frac[4.0][s] >= RULE_B_MIN and frac[4.0][s] - frac[0.0][s] >= RULE_B_MARGIN]
    label = label_of(k[0.0], k[4.0], b)
    out = {'label': label, 'k': {f'm{m:g}': k[m] for m in SHARED},
           'bind_fraction': {f'm{m:g}': frac[m] for m in SHARED}, 'rule_b_supports': b,
           'inference_failures': {f'm{m:g}': fails[m] for m in SHARED}}
    out['runner_agrees'] = (rep.get('reading', {}).get('label') == label
                            and rep['reading']['k'] == out['k'] and rep['reading']['rule_b_supports'] == b)
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
