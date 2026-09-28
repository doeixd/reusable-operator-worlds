"""Independent scorer for O7 (plan `O7_GRADIENT_REROUTE_PLAN.md`, frozen f3ec82e). Does not import the runner.

Recomputes r (target cells passing, denominator 10) and b (harm cells failing, denominator 12)
from the durable cells, the label, per-cell GRS/RS ratios against the committed O5 cells, and
checks provenance, gates and construction.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'o7_gradient_reroute.py'
TARGET = [['O2', 14, 0], ['O3', 22, 1], ['O4', 901, 0], ['O4', 907, 2], ['O4', 908, 1], ['O4', 909, 2],
          ['O4', 911, 0], ['O4', 911, 2], ['O4', 914, 1], ['O4', 914, 2]]
HARM = [['O2', w, 1] for w in (13, 15, 16, 17, 18, 19)] + [['O3', w, 0] for w in (20, 21, 23, 24, 25, 26)]


def label(r, b):
    if b >= 2:
        return 'HARMS'
    if r >= 8:
        return 'MATCHES_EXHAUSTIVE'
    return 'PARTIAL' if r >= 4 else 'NO_RESCUE'


def ok(m):
    return math.isfinite(m) and m < 0.05


def score(root=Path('artifacts/o7_gradient_reroute'), output=Path('reports/o7_gradient_reroute.json')):
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads(output.read_text())
    protocol, sha = manifest['protocol'], manifest['protocol_sha256']
    if fingerprint(protocol) != sha or report['protocol_sha256'] != sha:
        raise ValueError('protocol mismatch')
    for path, expected in protocol['input_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'input changed since the run: {path}')
    if digest(Path(__file__).parent / RUNNER) != protocol['implementation_sha256']:
        raise ValueError('runner changed since the run')
    problems = []
    if [list(c) for c in protocol['target']] != TARGET or [list(c) for c in protocol['harm']] != HARM:
        problems.append('not the registered cells')
    if not (report['gates']['G0']['passes'] and report['gates']['E1']['passes']):
        problems.append('gates failed')
    o5 = json.loads(Path('reports/o5_reroute_sleep.json').read_text())['cells']
    rows = {}
    for band, w, s in TARGET + HARM:
        key = f'{band}_w{w}_s{s}'
        saved = json.loads((root / 'cells' / f'{key}.json').read_text())
        rec = saved['record']
        if (saved['stamp'] != {'protocol_sha256': sha} or not saved['complete']
                or fingerprint(rec) != saved['record_sha256'] or report['cells'][key] != rec):
            raise ValueError(f'cell integrity failure: {key}')
        v = sorted(rec['terminal_per_task'].values())
        if len(v) != 64 or rec['pool_size'] != 188 * 64 or rec['chooser'] != 'gradient':
            problems.append(f'{key}: construction')
        elif math.isfinite(rec['terminal_median']) and not math.isclose((v[31] + v[32]) / 2, rec['terminal_median'],
                                                                         rel_tol=1e-12, abs_tol=1e-15):
            problems.append(f'{key}: median')
        if rec['library_sha256_before'] != o5[key]['library_sha256_before']:
            problems.append(f'{key}: different starting terminal from O5')
        rows[key] = {'grs': rec['terminal_median'], 'rs': o5[key]['terminal_median'],
                     'changed': rec['routes_changed'], 'rs_changed': o5[key]['routes_changed'],
                     'accepted': rec['gradient_accepted'], 'differs_from_exhaustive': rec['differs_from_exhaustive'],
                     'reroute_seconds': rec['reroute_seconds'], 'exhaustive_seconds': rec['exhaustive_seconds']}
    r = sum(ok(rows[f'{b}_w{w}_s{s}']['grs']) for b, w, s in TARGET)
    b = sum(not ok(rows[f'{b}_w{w}_s{s}']['grs']) for b, w, s in HARM)
    lab = label(r, b)
    if report['summary']['label'] != lab or report['summary']['r'] != r or report['summary']['b'] != b:
        problems.append('runner summary disagrees')
    ratio = [rows[k]['grs'] / rows[k]['rs'] for k in rows]
    return {'valid': not problems, 'problems': problems, 'tier': 1, 'exploratory': True, 'label': lab,
            'r': r, 'b': b, 'median_grs_over_rs': float(np.median(ratio)), 'max_grs_over_rs': float(max(ratio)),
            'rows': rows}


if __name__ == '__main__':
    result = score()
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result['valid'] else 1)
