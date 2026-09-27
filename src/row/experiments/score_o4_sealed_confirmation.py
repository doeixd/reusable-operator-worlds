"""Independent scorer for O4 (plan `O4_SEALED_CONFIRMATION_PLAN.md`, frozen 397aa88).

Recomputes k_SLEEP over all 45 cells (non-finite = not passing), the CONFIRMED /
NOT_CONFIRMED label at k >= 41, the floor clause, E4 on the real cells, collapse
counts and the paired sleep effect, from the durable cells; checks gates and provenance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'o4_sealed_confirmation.py'
THRESHOLD, CONFIRM_AT = 0.05, 41


def label(k, floor):
    if floor:
        return 'FLOOR_FAILED'
    return 'CONFIRMED' if k >= CONFIRM_AT else 'NOT_CONFIRMED'


def score(root=Path('artifacts/o4_sealed_confirmation'), output=Path('reports/o4_sealed_confirmation.json')):
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
    if not gates.get('all_pass') or gates.get('protocol_sha256') != fingerprint(base):
        problems.append('gates not passed at this protocol')
    worlds, arm_streams = protocol['worlds'], protocol['arm_streams']
    if list(worlds) != list(range(900, 915)):
        problems.append('not the registered sealed worlds')
    M, recs = {}, {}
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
                if arm == 'SLEEP' and (rec['extra_updates'] != 8192 or rec['pool_size'] != 188 * 64
                                       or rec['library_sha256'] == rec['library_sha256_before']):
                    problems.append(f'{key}: sleep construction')
                if arm == 'SHUFFLED' and (rec['anchor_abs_error'] > 1e-6 or not rec['route_lengths_match_plan']):
                    problems.append(f'{key}: anchor or routes')
                M[key], recs[key] = rec['terminal_median'], rec
    for w in worlds:
        shas = [recs[f'SHUFFLED_w{w}_s{s}']['library_sha256'] for s in arm_streams['SHUFFLED']]
        if len(set(shas)) != len(shas):
            problems.append(f'E4: world {w} streams share a library')
    ok = lambda m: math.isfinite(m) and m < THRESHOLD  # noqa: E731
    k = {a: sum(ok(M[f'{a}_w{w}_s{s}']) for w in worlds for s in arm_streams[a]) for a in arm_streams}
    floor = k['PLAIN'] > 0
    lab = label(k['SLEEP'], floor)
    if report.get('summary', {}).get('label') != lab or report['summary'].get('passes') != k:
        problems.append(f"runner summary {report.get('summary')} disagrees with {lab} {k}")
    if json.loads((root / 'status.json').read_text())['state'] != 'complete':
        problems.append('status not complete')
    if json.loads((root / 'exit.json').read_text())['exit_code'] != 0:
        problems.append('nonzero exit')
    effect = [math.log10(max(M[f'SLEEP_w{w}_s{s}'], 1e-12) / max(M[f'SHUFFLED_w{w}_s{s}'], 1e-12))
              for w in worlds for s in arm_streams['SLEEP']]
    return {'valid': not problems, 'problems': problems, 'sealed': True, 'label': lab, 'passes': k,
            'denominator': 3 * len(worlds), 'confirm_at': CONFIRM_AT,
            'collapses': {a: sum(not (math.isfinite(M[f'{a}_w{w}_s{s}']) and M[f'{a}_w{w}_s{s}'] < 1.0)
                                 for w in worlds for s in arm_streams[a]) for a in arm_streams},
            'median_sleep_effect_log10': float(np.median(effect)),
            'per_world_passing': {str(w): {a: sum(ok(M[f'{a}_w{w}_s{s}']) for s in arm_streams[a])
                                           for a in ('SHUFFLED', 'SLEEP')} for w in worlds},
            'terminal_by_cell': dict(sorted(M.items()))}


if __name__ == '__main__':
    result = score()
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result['valid'] else 1)
