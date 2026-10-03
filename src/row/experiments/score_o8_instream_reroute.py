"""Independent scorer for O8 (plan `O8_INSTREAM_REROUTE_PLAN.md`, frozen b693ea2).

Re-derives rule A (REROUTE_WAKE against O3's committed SHUFFLED) and rule B (REROUTE_INTERLEAVED against
O3's committed INTERLEAVED) from the per-cell records and the O3 report, with its own label functions,
and refuses a report whose inputs have moved since the run (stale-report rule). Written and committed
before launch; it shares no rule code with the runner.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint

PLAN = 'O8_INSTREAM_REROUTE_PLAN.md'
O3_REPORT = 'reports/o3_online_sleep_v2.json'
REPORT = Path('reports/o8_instream_reroute.json')
WORLDS = (20, 21, 22, 23, 24, 25, 26)
STREAMS = (0, 1, 2)
THRESHOLD = 0.05
RULES = {'REROUTE_WAKE': 'SHUFFLED', 'REROUTE_INTERLEAVED': 'INTERLEAVED'}


def ok(m):
    return isinstance(m, (int, float)) and math.isfinite(m) and m < THRESHOLD


def label_a(k, n_better, h):
    return 'HARMS' if h >= 3 else 'PREVENTS' if (k >= 14 and n_better >= 16) else \
        'PARTIAL' if (k >= 14 or n_better >= 16) else 'NO_EFFECT'


def label_b(f, n_better, h):
    return 'HARMS_B' if h >= 3 else 'REACHES_CEILING' if (f <= 1 and n_better >= 16) else \
        'IMPROVES' if n_better >= 16 else 'NO_EFFECT_B'


def score(report_path=REPORT):
    rep = json.loads(Path(report_path).read_text())
    problems = []
    proto = rep['protocol']
    if fingerprint(proto) != rep['protocol_sha256']:
        problems.append('protocol fingerprint mismatch')
    for p, sha in proto['input_sha256'].items():
        if not Path(p).exists() or digest(Path(p)) != sha:
            problems.append(f'input changed since the run: {p}')
    if not rep.get('complete'):
        problems.append('report not complete')
    if proto.get('scale', 1) != 1 or proto.get('tier') != 1 or not proto.get('exploratory'):
        problems.append('not a full-scale exploratory Tier 1 report')
    cells = rep['cells']
    o3 = json.loads(Path(O3_REPORT).read_text())['cells']
    out = {'valid': None, 'problems': problems, 'denominator': len(WORLDS) * len(STREAMS)}
    for arm, ref in RULES.items():
        k = better = h = 0
        rows = []
        for w in WORLDS:
            for s in STREAMS:
                key = f'{arm}_w{w}_s{s}'
                rec = cells.get(key)
                if rec is None:
                    problems.append(f'missing cell {key}')
                    continue
                if rec['arm'] != arm or rec['world'] != w or rec['stream'] != s or not rec['reroute_enabled']:
                    problems.append(f'{key}: identity')
                if len(rec['terminal_per_task']) != 64 or rec['trained_tasks'] != 188 or rec['reroute_passes'] != 187:
                    problems.append(f'{key}: construction')
                if arm == 'REROUTE_WAKE' and (rec['extra_updates'] != 0 or rec['anchor_abs_error'] > 1e-6):
                    problems.append(f'{key}: REROUTE_WAKE anchor/updates')
                if arm == 'REROUTE_INTERLEAVED' and rec['extra_updates'] != 8192:
                    problems.append(f'{key}: REROUTE_INTERLEAVED updates')
                new, old = rec['terminal_median'], o3[f'{ref}_w{w}_s{s}']['terminal_median']
                k += ok(new)
                better += ok(new) and new < old
                h += ok(old) and not ok(new)
                rows.append({'cell': f'w{w}_s{s}', 'new': new, 'reference': old, 'passes': ok(new),
                             'better': bool(ok(new) and new < old), 'stale': rec['terminal_stale'],
                             'routes_changed_total': rec['routes_changed_total']})
        n = len(WORLDS) * len(STREAMS)
        out[arm] = {'reference': ref, 'k': k, 'f': n - k, 'n_better': better, 'h': h,
                    'label': label_a(k, better, h) if arm == 'REROUTE_WAKE' else label_b(n - k, better, h),
                    'cells': rows}
        runner = rep.get('summary', {}).get(arm, {})
        if runner and runner.get('label') != out[arm]['label']:
            problems.append(f'{arm}: runner label {runner.get("label")} != scorer {out[arm]["label"]}')
    out['valid'] = not problems
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(REPORT))
    args = parser.parse_args()
    result = score(args.report)
    print(json.dumps({k: v for k, v in result.items() if k not in ('REROUTE_WAKE', 'REROUTE_INTERLEAVED')}
                     | {a: {k: v for k, v in result[a].items() if k != 'cells'} for a in RULES if a in result},
                     indent=1))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
