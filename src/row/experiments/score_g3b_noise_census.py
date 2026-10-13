"""Independent scorer for G3b (`G3B_NOISE_CENSUS_PLAN.md`).

Recomputes every (noise, support) bind fraction and the label from the per-program rows of the durable cell records,
re-checks stamps, record hashes, anchors A1 and A3 and the run's input hashes, and refuses a report that differs from
its cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PLAN = Path('G3B_NOISE_CENSUS_PLAN.md')
REPORT = Path('reports/g3b_noise_census.json')
CELLS = Path('artifacts/g3b_noise_census/cells')
WORLDS = (70, 71, 72)
SHARED = (0.0, 4.0)
SUPPORTS = ('128', '8', '2')
NOISE = ('0', '0.01', '0.03', '0.1', '0.3', '1')
PLAUSIBLE = ('0', '0.01', '0.03', '0.1')
PROGRAMS = 64
BIND_MIN, BIND_MARGIN = 0.10, 0.10


def key(m, w):
    return f'm{m:g}_w{w}'


def _fp(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def classify(frac):
    window, both, high = [], [], []
    for v in NOISE:
        for s in SUPPORTS:
            a, b = frac['m4'][v][s], frac['m0'][v][s]
            if a >= BIND_MIN and v in PLAUSIBLE:
                (window if a - b >= BIND_MARGIN else both).append([float(v), int(s)])
            elif a >= BIND_MIN:
                high.append([float(v), int(s)])
    label = ('NOISE_WINDOW' if window else 'NOISE_ONLY' if both else 'HIGH_NOISE_ONLY' if high else 'NO_BIND')
    return label, window, both, high


def score(report=REPORT, cells=CELLS, check_inputs=True):
    rep = json.loads(Path(report).read_text())
    if check_inputs:
        for path, digest in rep['protocol']['input_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'{path} changed since the run')
        if PLAN.as_posix() not in rep['protocol']['input_sha256']:
            raise ValueError('report does not stamp the plan')
    frac, fails = {}, {}
    for m in SHARED:
        arm = f'm{m:g}'
        counts = {v: {s: 0 for s in SUPPORTS} for v in NOISE}
        fcount = {v: {s: 0 for s in SUPPORTS} for v in NOISE}
        for w in WORLDS:
            saved = json.loads((Path(cells) / f'{key(m, w)}.json').read_text())
            if saved['stamp'] != {'protocol_sha256': rep['protocol_sha256']} or not saved['complete']:
                raise ValueError(f'{key(m, w)}: stamp')
            if _fp(saved['record']) != saved['record_sha256']:
                raise ValueError(f'{key(m, w)}: record hash')
            if saved['record'] != rep['cells'][key(m, w)]:
                raise ValueError(f'{key(m, w)}: report differs from the durable cell')
            r = saved['record']
            if not (r['anchor_terminal'] and r['anchor_g3a']) or len(r['rows']) != PROGRAMS:
                raise ValueError(f'{key(m, w)}: anchors or program count')
            for row in r['rows']:
                for v in NOISE:
                    for s in SUPPORTS:
                        x = row['by_noise'][v][s]
                        counts[v][s] += x['regret'] > x['null_floor']
                        fcount[v][s] += x['nmse_hat_qb'] >= 0.05 and x['nmse_star_qb'] < 0.05
        n = len(WORLDS) * PROGRAMS
        frac[arm] = {v: {s: counts[v][s] / n for s in SUPPORTS} for v in NOISE}
        fails[arm] = fcount
    label, window, both, high = classify(frac)
    out = {'label': label, 'window': window, 'both_bind': both, 'high_noise': high, 'bind_fraction': frac,
           'inference_failures': fails}
    rd = rep.get('reading', {})
    out['runner_agrees'] = (rd.get('label') == label and rd.get('window') == window and rd.get('both_bind') == both
                            and rd.get('high_noise') == high)
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
