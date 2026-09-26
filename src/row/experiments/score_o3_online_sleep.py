"""Independent scorer for O3 (plan `O3_ONLINE_SLEEP_CONFIRMATION_PLAN.md`, frozen 5120482).

Recomputes the primary k-of-21 label, the paired phase contrast, the floor clause
and collapse counts from the durable cells; checks gates, E4 (streams distinct) on
the real cells, construction invariants and provenance; refuses a stale report.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'o3_online_sleep.py'
THRESHOLD = 0.05
EXTRA = 8192


def label(k):
    if k >= 19:
        return 'RELIABLE'
    return 'INTERMEDIATE' if k >= 17 else 'UNRELIABLE'


def r_value(ms, mi):
    fs, fi = math.isfinite(ms), math.isfinite(mi)
    if not (fs or fi):
        return 0.0
    if not fs:
        return math.inf
    if not fi:
        return -math.inf
    return math.log10(max(ms, 1e-12) / max(mi, 1e-12))


def contrast(rw):
    r = [x for ws in rw for x in ws]
    neg, pos = sum(x < 0 for x in r), sum(x > 0 for x in r)
    wneg = sum(sum(x < 0 for x in ws) >= 2 for ws in rw)
    wpos = sum(sum(x > 0 for x in ws) >= 2 for ws in rw)
    med = float(np.median(r))
    if neg >= 16 and wneg >= 6 and med <= -0.15:
        return 'PHASE_MATTERS'
    if pos >= 16 and wpos >= 6 and med >= 0.15:
        return 'INTERLEAVED_BETTER'
    if 7 <= neg <= 14 and abs(med) <= 0.10:
        return 'EQUIVALENT'
    return 'INDETERMINATE'


def score(root=Path('artifacts/o3_online_sleep_v2'), output=Path('reports/o3_online_sleep_v2.json')):
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
    base_protocol = {k: v for k, v in protocol.items() if k != 'scale'}
    if not gates.get('all_pass') or gates.get('protocol_sha256') != fingerprint(base_protocol):
        problems.append('gates not passed at this protocol')
    worlds, arm_streams = protocol['worlds'], protocol['arm_streams']
    cells = {}
    for arm, streams in arm_streams.items():
        for w in worlds:
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
                if arm in ('SLEEP', 'INTERLEAVED') and rec['extra_updates'] != EXTRA:
                    problems.append(f'{key}: extra updates {rec["extra_updates"]}')
                if arm == 'SLEEP' and (rec['pool_size'] != 188 * 64 or rec['library_sha256'] == rec['library_sha256_before']):
                    problems.append(f'{key}: sleep pool or non-vacuity')
                if arm in ('SHUFFLED', 'INTERLEAVED') and not rec['route_lengths_match_plan']:
                    problems.append(f'{key}: routes')
                if arm == 'SHUFFLED' and rec['anchor_abs_error'] > 1e-6:
                    problems.append(f'{key}: last-task anchor')
                cells[key] = rec
    for arm in ('SHUFFLED', 'INTERLEAVED', 'SLEEP'):
        for w in worlds:
            shas = [cells[f'{arm}_w{w}_s{s}']['library_sha256'] for s in arm_streams[arm]]
            if len(set(shas)) != len(shas):
                problems.append(f'E4: {arm} world {w} streams share a library')
    M = {k: v['terminal_median'] for k, v in cells.items()}
    ok = lambda m: math.isfinite(m) and m < THRESHOLD  # noqa: E731
    k = {a: sum(ok(M[f'{a}_w{w}_s{s}']) for w in worlds for s in arm_streams[a]) for a in arm_streams}
    floor = k['PLAIN'] > 0
    rw = [[r_value(M[f'SLEEP_w{w}_s{s}'], M[f'INTERLEAVED_w{w}_s{s}']) for s in arm_streams['SLEEP']] for w in worlds]
    primary = 'FLOOR_FAILED' if floor else 'WAKE_SLEEP_' + label(k['SLEEP'])
    con = 'FLOOR_FAILED' if floor else contrast(rw)
    summ = report.get('summary', {})
    if summ.get('primary') != primary or summ.get('contrast') != con or summ.get('passes') != k:
        problems.append(f"runner summary {summ.get('primary')}/{summ.get('contrast')} != {primary}/{con}")
    for f, want in (('status.json', 'complete'),):
        if json.loads((root / f).read_text())['state'] != want:
            problems.append('status not complete')
    if json.loads((root / 'exit.json').read_text())['exit_code'] != 0:
        problems.append('nonzero exit')
    sleep_effect = [math.log10(max(M[f'SLEEP_w{w}_s{s}'], 1e-12) / max(M[f'SHUFFLED_w{w}_s{s}'], 1e-12))
                    for w in worlds for s in arm_streams['SLEEP']]
    return {'valid': not problems, 'problems': problems, 'tier': 2, 'primary': primary, 'contrast': con,
            'interleaved_label': 'FLOOR_FAILED' if floor else label(k['INTERLEAVED']), 'passes': k,
            'r_by_world': {str(w): v for w, v in zip(worlds, rw)},
            'median_r': float(np.median([x for ws in rw for x in ws])),
            'median_sleep_effect_log10': float(np.median(sleep_effect)),
            'collapses': {a: sum(not (math.isfinite(M[f'{a}_w{w}_s{s}']) and M[f'{a}_w{w}_s{s}'] < 1.0)
                                 for w in worlds for s in arm_streams[a]) for a in arm_streams},
            'terminal_by_cell': {kk: M[kk] for kk in sorted(M)}}


if __name__ == '__main__':
    result = score()
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result['valid'] else 1)
