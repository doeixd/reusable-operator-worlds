"""Independent scorer for O6 (plan `O6_SEALED_REROUTE_CONFIRMATION_PLAN.md`, frozen 666f961).

Recomputes, from the durable cells, k_RS over all 45 REROUTE_SLEEP cells (non-finite = not
passing) with CONFIRMED / NOT_CONFIRMED at k >= 41 and the PLAIN floor clause; the attribution
count n_better (REROUTE_SLEEP strictly below SLEEP; tie or non-finite RS = not better) with
ATTRIBUTED at >= 30 of 45; E4 on the real SHUFFLED cells; each arm's construction fields; the
gates; and provenance. Does not import the runner.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'o6_sealed_reroute.py'
WORLDS = list(range(915, 930))
ARM_STREAMS = {'SHUFFLED': [0, 1, 2], 'REROUTE_SLEEP': [0, 1, 2], 'SLEEP': [0, 1, 2], 'PLAIN': [0]}
THRESHOLD, CONFIRM_AT, ATTRIBUTE_AT = 0.05, 41, 30
GATES = ('E1', 'E2', 'E3', 'E3b', 'E4b')


def label(k_rs, floor):
    if floor:
        return 'FLOOR_FAILED'
    return 'CONFIRMED' if k_rs >= CONFIRM_AT else 'NOT_CONFIRMED'


def attribution(n_better):
    return 'ATTRIBUTED' if n_better >= ATTRIBUTE_AT else 'NOT_ATTRIBUTED'


def ok(m):
    return math.isfinite(m) and m < THRESHOLD


def better(rs, sleep):
    return math.isfinite(rs) and rs < sleep


def score(root=Path('artifacts/o6_sealed_reroute'), output=Path('reports/o6_sealed_reroute.json')):
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads(output.read_text())
    protocol, sha = manifest['protocol'], manifest['protocol_sha256']
    if fingerprint(protocol) != sha or report['protocol_sha256'] != sha or protocol.get('scale') != 1:
        raise ValueError('protocol mismatch or not full scale')
    for path, expected in protocol['input_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'input changed since the run: {path}')
    here = Path(__file__).parent
    if digest(here / RUNNER) != protocol['implementation_sha256']:
        raise ValueError('runner changed since the run')
    if output.stat().st_mtime < (here / RUNNER).stat().st_mtime:
        raise ValueError('report is older than its runner')
    problems = []
    gates = json.loads((root / 'gates' / 'gates.json').read_text())
    base = {k: v for k, v in protocol.items() if k != 'scale'}
    if (not gates.get('all_pass') or gates.get('protocol_sha256') != fingerprint(base)
            or not all(gates.get(g, {}).get('passes') for g in GATES)):
        problems.append('gates not passed at this protocol')
    if protocol['worlds'] != WORLDS or protocol['arm_streams'] != ARM_STREAMS:
        problems.append('not the registered sealed worlds and arms')
    M, recs = {}, {}
    for arm, streams in ARM_STREAMS.items():
        for w in WORLDS:
            for s in streams:
                key = f'{arm}_w{w}_s{s}'
                saved = json.loads((root / 'cells' / f'{key}.json').read_text())
                rec = saved['record']
                if (saved['stamp'] != {'protocol_sha256': sha} or not saved['complete']
                        or fingerprint(rec) != saved['record_sha256'] or report['cells'][key] != rec):
                    raise ValueError(f'cell integrity failure: {key}')
                v = sorted(rec['terminal_per_task'].values())
                if len(v) != 64:
                    problems.append(f'{key}: not 64 scored tasks')
                elif math.isfinite(rec['terminal_median']) and not math.isclose(
                        (v[31] + v[32]) / 2, rec['terminal_median'], rel_tol=1e-12, abs_tol=1e-15):
                    problems.append(f'{key}: median')
                if arm in ('SLEEP', 'REROUTE_SLEEP') and (
                        rec['extra_updates'] != 8192 or rec['pool_size'] != 188 * 64
                        or rec['library_sha256'] == rec['library_sha256_before']):
                    problems.append(f'{key}: sleep construction')
                if arm == 'REROUTE_SLEEP' and (rec.get('swap') is not True or rec.get('routes_changed') is None):
                    problems.append(f'{key}: re-route construction')
                if arm == 'SHUFFLED' and (rec['anchor_abs_error'] > 1e-6 or not rec['route_lengths_match_plan']):
                    problems.append(f'{key}: anchor or routes')
                M[key], recs[key] = rec['terminal_median'], rec
    for w in WORLDS:
        shas = [recs[f'SHUFFLED_w{w}_s{s}']['library_sha256'] for s in ARM_STREAMS['SHUFFLED']]
        if len(set(shas)) != len(shas):
            problems.append(f'E4: world {w} streams share a library')
        for s in ARM_STREAMS['SLEEP']:   # both dependent arms start from this cell's own terminal
            start = recs[f'SHUFFLED_w{w}_s{s}']['library_sha256']
            for arm in ('SLEEP', 'REROUTE_SLEEP'):
                if recs[f'{arm}_w{w}_s{s}']['library_sha256_before'] != start:
                    problems.append(f'{arm}_w{w}_s{s}: did not start from its SHUFFLED terminal')
    k = {a: sum(ok(M[f'{a}_w{w}_s{s}']) for w in WORLDS for s in ARM_STREAMS[a]) for a in ARM_STREAMS}
    floor = k['PLAIN'] > 0
    lab = label(k['REROUTE_SLEEP'], floor)
    pairs = [(w, s) for w in WORLDS for s in ARM_STREAMS['SLEEP']]
    n_better = sum(better(M[f'REROUTE_SLEEP_w{w}_s{s}'], M[f'SLEEP_w{w}_s{s}']) for w, s in pairs)
    att = attribution(n_better)
    summary = report.get('summary', {})
    if summary.get('label') != lab or summary.get('passes') != k:
        problems.append(f'runner summary {summary} disagrees with {lab} {k}')
    if summary.get('n_better') != n_better or summary.get('attribution') != att:
        problems.append(f'runner attribution disagrees with {att} {n_better}')
    if json.loads((root / 'status.json').read_text())['state'] != 'complete':
        problems.append('status not complete')
    if json.loads((root / 'exit.json').read_text())['exit_code'] != 0:
        problems.append('nonzero exit')
    collapsed = [(w, s) for w, s in pairs if not (math.isfinite(M[f'SHUFFLED_w{w}_s{s}'])
                                                  and M[f'SHUFFLED_w{w}_s{s}'] < 1.0)]
    ratio = [math.log10(max(M[f'REROUTE_SLEEP_w{w}_s{s}'], 1e-12) / max(M[f'SLEEP_w{w}_s{s}'], 1e-12))
             for w, s in pairs]
    changed = [recs[f'REROUTE_SLEEP_w{w}_s{s}']['routes_changed'] for w, s in pairs]
    return {'valid': not problems, 'problems': problems, 'sealed': True, 'label': lab, 'attribution': att,
            'passes': k, 'denominator': 45, 'confirm_at': CONFIRM_AT, 'n_better': n_better,
            'attribute_at': ATTRIBUTE_AT,
            'collapses': {a: sum(not (math.isfinite(M[f'{a}_w{w}_s{s}']) and M[f'{a}_w{w}_s{s}'] < 1.0)
                                 for w in WORLDS for s in ARM_STREAMS[a]) for a in ARM_STREAMS},
            'shuffled_collapses_rescued_by_rs': sum(ok(M[f'REROUTE_SLEEP_w{w}_s{s}']) for w, s in collapsed),
            'shuffled_collapses_rescued_by_sleep': sum(ok(M[f'SLEEP_w{w}_s{s}']) for w, s in collapsed),
            'median_log10_rs_over_sleep': float(np.median(ratio)),
            'routes_changed': {'median': float(np.median(changed)), 'min': min(changed), 'max': max(changed)},
            'per_world_passing': {str(w): {a: sum(ok(M[f'{a}_w{w}_s{s}']) for s in ARM_STREAMS[a])
                                           for a in ('SHUFFLED', 'SLEEP', 'REROUTE_SLEEP')} for w in WORLDS},
            'terminal_by_cell': dict(sorted(M.items()))}


if __name__ == '__main__':
    result = score()
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result['valid'] else 1)
