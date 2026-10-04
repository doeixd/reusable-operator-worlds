"""Independent scorer for O9 (plan `O9_SEALED_ONLINE_PLAN.md`, frozen 0b9df24). Sealed worlds 930-944.

Re-derives the primary (REROUTE_WAKE passes, floor) and the attribution (REROUTE_WAKE strictly below
SHUFFLED) from the per-cell records with its own label code, checks every cell's construction, the per-world
distinctness of SHUFFLED terminals, and refuses a report whose inputs moved since the run. Committed before the
sealed band is opened; it shares no rule code with the runner.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint

REPORT = Path('reports/o9_sealed_online.json')
WORLDS = list(range(930, 945))
STREAMS = [0, 1, 2]
THRESHOLD = 0.05


def ok(m):
    return isinstance(m, (int, float)) and math.isfinite(m) and m < THRESHOLD


def primary(k_rw, plain_passes):
    if plain_passes > 0:
        return 'FLOOR_FAILED'
    return 'CONFIRMED' if k_rw >= 41 else 'NOT_CONFIRMED'


def secondary(n_better):
    return 'ATTRIBUTED' if n_better >= 30 else 'NOT_ATTRIBUTED'


def score(report_path=REPORT):
    rep = json.loads(Path(report_path).read_text())
    problems = []
    proto = rep['protocol']
    if fingerprint(proto) != rep['protocol_sha256']:
        problems.append('protocol fingerprint mismatch')
    for p, sha in proto['input_sha256'].items():
        if not Path(p).exists() or digest(Path(p)) != sha:
            problems.append(f'input changed since the run: {p}')
    if not rep.get('complete') or proto.get('scale', 1) != 1 or not proto.get('sealed'):
        problems.append('not a complete full-scale sealed report')
    if proto.get('worlds') != WORLDS:
        problems.append(f'worlds {proto.get("worlds")} are not the registered 930-944')
    c = rep['cells']

    def get(arm, w, s):
        rec = c.get(f'{arm}_w{w}_s{s}')
        if rec is None:
            problems.append(f'missing cell {arm}_w{w}_s{s}')
            return None
        if rec['arm'] != arm or rec['world'] != w or rec['stream'] != s or len(rec['terminal_per_task']) != 64:
            problems.append(f'{arm}_w{w}_s{s}: identity or scoring')
        return rec

    k_rw = n_better = 0
    rows = []
    for w in WORLDS:
        libs = set()
        for s in STREAMS:
            rw, sh, rs = get('REROUTE_WAKE', w, s), get('SHUFFLED', w, s), get('REROUTE_SLEEP', w, s)
            if None in (rw, sh, rs):
                continue
            if rw['reroute_passes'] != 187 or rw['extra_updates'] != 0 or rw['anchor_abs_error'] > 1e-6 \
                    or rw['trained_tasks'] != 188 or not rw['reroute_enabled']:
                problems.append(f'REROUTE_WAKE_w{w}_s{s}: construction')
            if rs['extra_updates'] != 8192 or rs['pool_size'] != 188 * 64 or not rs['swap']:
                problems.append(f'REROUTE_SLEEP_w{w}_s{s}: construction')
            if sh['trained_tasks'] != 188 or sh['stream_tasks'] != 188:
                problems.append(f'SHUFFLED_w{w}_s{s}: construction')
            libs.add(sh['library_sha256'])
            a, b = rw['terminal_median'], sh['terminal_median']
            k_rw += ok(a)
            n_better += isinstance(a, (int, float)) and math.isfinite(a) and a < b
            rows.append({'cell': f'w{w}_s{s}', 'REROUTE_WAKE': a, 'SHUFFLED': b,
                         'REROUTE_SLEEP': rs['terminal_median'], 'stale': rw['terminal_stale'],
                         'routes_changed_total': rw['routes_changed_total']})
        if len(libs) != 3:
            problems.append(f'w{w}: SHUFFLED terminal libraries are not distinct across streams (E4)')
    plain = [get('PLAIN', w, 0) for w in WORLDS]
    plain_passes = sum(ok(p['terminal_median']) for p in plain if p is not None)
    out = {'valid': None, 'problems': problems, 'denominator': len(WORLDS) * len(STREAMS),
           'k_REROUTE_WAKE': k_rw, 'k_SHUFFLED': sum(ok(r['SHUFFLED']) for r in rows),
           'k_REROUTE_SLEEP': sum(ok(r['REROUTE_SLEEP']) for r in rows), 'plain_passes': plain_passes,
           'label': primary(k_rw, plain_passes), 'n_better': n_better, 'attribution': secondary(n_better),
           'cells': rows}
    runner = rep.get('summary', {})
    if runner and (runner.get('label') != out['label'] or runner.get('attribution') != out['attribution']):
        problems.append(f"runner labels {runner.get('label')}/{runner.get('attribution')} != scorer "
                        f"{out['label']}/{out['attribution']}")
    out['valid'] = not problems
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(REPORT))
    args = parser.parse_args()
    result = score(args.report)
    print(json.dumps({k: v for k, v in result.items() if k != 'cells'}, indent=1))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
