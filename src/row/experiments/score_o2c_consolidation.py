"""Independent scorer for O2C (plan `O2C_CONSOLIDATION_OPPORTUNITY_PLAN.md`, frozen cee05a3).

Recomputes r (near-miss cells rescued, of 8) and b (passing cells broken, of 9)
from the durable cells and O2's committed report, the labels, and checks the
gates, non-vacuity, per-task medians and provenance. Refuses a stale report.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'o2c_consolidation.py'
THRESHOLD, NEAR_HIGH = 0.05, 0.2
WORLDS, STREAMS = range(13, 20), (0, 1, 2)


def label(r, b):
    if r >= 5 and b <= 1:
        return 'RESCUES'
    return 'HARMS' if b >= 2 else 'NO_RESCUE'


def score(root=Path('artifacts/o2c_consolidation'), output=Path('reports/o2c_consolidation.json')):
    root, output = Path(root), Path(output)
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads(output.read_text())
    protocol, sha = manifest['protocol'], manifest['protocol_sha256']
    if fingerprint(protocol) != sha or report['protocol_sha256'] != sha:
        raise ValueError('protocol mismatch')
    for path, expected in protocol['input_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'input changed since the run: {path}')
    here = Path(__file__).parent
    if digest(here / RUNNER) != protocol['implementation_sha256']:
        raise ValueError('runner changed since the run')
    if output.stat().st_mtime < (here / RUNNER).stat().st_mtime:
        raise ValueError('report is older than its runner')
    problems = []
    gates = report['gates']
    if not (gates['G0']['passes'] and gates['G1']['passes']):
        problems.append('gates did not pass')
    base = json.loads(Path('reports/o2_online_reliability.json').read_text())['cells']
    m0 = {(w, s): base[f'SHUFFLED_w{w}_s{s}']['terminal_median'] for w in WORLDS for s in STREAMS}
    near = [k for k, v in m0.items() if THRESHOLD <= v < NEAR_HIGH]
    passing = [k for k, v in m0.items() if v < THRESHOLD]
    if (len(near), len(passing)) != (8, 9):
        problems.append(f'near/passing sets are {len(near)}/{len(passing)}, plan says 8/9')
    out = {}
    for arm in protocol['arms']:
        m1 = {}
        for w in WORLDS:
            for s in STREAMS:
                key = f'{arm}_w{w}_s{s}'
                saved = json.loads((root / 'cells' / f'{key}.json').read_text())
                rec = saved['record']
                if (saved['stamp'] != {'protocol_sha256': sha} or not saved['complete']
                        or fingerprint(rec) != saved['record_sha256'] or report['cells'][key] != rec):
                    raise ValueError(f'cell integrity failure: {key}')
                vals = sorted(rec['terminal_per_task'].values())
                if len(vals) != 64 or not math.isclose((vals[31] + vals[32]) / 2, rec['terminal_median'],
                                                       rel_tol=1e-12, abs_tol=1e-15):
                    problems.append(f'{key}: terminal median not the median of 64 per-task values')
                if rec['library_sha256'] == rec['library_sha256_before']:
                    problems.append(f'{key}: G2 library unchanged')
                expected_pool = 24064 if arm == 'ORACLE_DATA' else 752
                if rec['pool_size'] != expected_pool:
                    problems.append(f'{key}: pool {rec["pool_size"]} != {expected_pool}')
                m1[(w, s)] = rec['terminal_median']
        r = sum(math.isfinite(m1[k]) and m1[k] < THRESHOLD for k in near)
        b = sum(not (math.isfinite(m1[k]) and m1[k] < THRESHOLD) for k in passing)
        out[arm] = {'r_rescued_of_8': r, 'b_broken_of_9': b, 'label': label(r, b),
                    'pass_of_21': sum(math.isfinite(v) and v < THRESHOLD for v in m1.values()),
                    'by_cell': {f'w{w}_s{s}': [m0[(w, s)], m1[(w, s)]] for (w, s) in sorted(m1)}}
        if report['summary']['arms'][arm]['label'] != out[arm]['label']:
            problems.append(f'{arm}: runner label {report["summary"]["arms"][arm]["label"]} != {out[arm]["label"]}')
    return {'valid': not problems, 'problems': problems, 'tier': 1, 'exploratory': True, 'arms': out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = score()
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
