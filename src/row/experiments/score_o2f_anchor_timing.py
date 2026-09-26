"""O2F Tier 0 census scorer: does early anchor supply predict which order-free streams form?

Plan `O2F_ANCHOR_TIMING_CENSUS_PLAN.md` (registered 2026-09-25 before any O2 cell
finished; sha256 2a85c6f4...d347). Observational: reads O2's validated report and
the pre-computed predictor file, and trains nothing. The predictor is re-derived
here from the registered order seeds and must equal the pre-registered file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PLAN = Path('O2F_ANCHOR_TIMING_CENSUS_PLAN.md')
PLAN_SHA256 = '2a85c6f4a038da1e1303eb0c3a84de207890badbec17eac59dc128595460d347'
PREDICTORS = Path('reports/o2_design/o2f_predictors.json')
O2_REPORT = Path('reports/o2_online_reliability.json')
OUTPUT = Path('reports/o2f_anchor_timing.json')
THRESHOLD = 0.05
ARMS = ('SHUFFLED', 'MIXED_L1')


def sha(path):
    """sha256 of the file with CRLF normalised to LF, so a Windows checkout verifies."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def recompute_predictors():
    from row.experiments import o2_online_reliability as o2
    out = {}
    for arm in ARMS:
        for w in o2.WORLDS:
            for s in o2.STREAMS:
                _, _, stream, plan, _ = o2.build_stream(arm, w, s)
                out[(arm, w, s)] = sum(1 for t in stream[:32] if plan[t.task_id] == 1)
    return out


def label(n, a):
    if n < 5:
        return 'UNMEASURABLE_WORLD_DRIVEN'
    if a / n >= 0.85:
        return 'EARLY_DOSE_PREDICTS'
    if a / n <= 0.60:
        return 'NO_TIMING_EFFECT'
    return 'INCONCLUSIVE'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-recompute', action='store_true', help='tests only')
    args = parser.parse_args()
    if sha(PLAN) != PLAN_SHA256:
        raise SystemExit('O2F plan changed since registration')
    registered = json.loads(PREDICTORS.read_text())['cells']
    E = {(c['arm'], c['world'], c['stream']): c['E_len1_in_first32'] for c in registered}
    if not args.skip_recompute and recompute_predictors() != E:
        raise SystemExit('predictors do not re-derive from the registered order seeds')
    o2 = json.loads(O2_REPORT.read_text())
    if o2['summary']['primary'] == 'ORDER_FREE_RELIABLE':
        result = {'label': 'MOOT', 'reason': 'O2 primary is ORDER_FREE_RELIABLE'}
    else:
        groups, n, a = [], 0, 0
        for arm in ARMS:
            for w in o2['protocol']['worlds']:
                rows = []
                for s in o2['protocol']['arm_streams'][arm]:
                    m = o2['cells'][f'{arm}_w{w}_s{s}']['terminal_median']
                    rows.append({'stream': s, 'E': E[(arm, w, s)],
                                 'pass': bool(math.isfinite(m) and m < THRESHOLD), 'terminal': m})
                ps = [r['E'] for r in rows if r['pass']]
                fl = [r['E'] for r in rows if not r['pass']]
                informative = bool(ps and fl and sum(ps) / len(ps) != sum(fl) / len(fl))
                direction = None
                if informative:
                    direction = 'passing_higher' if sum(ps) / len(ps) > sum(fl) / len(fl) else 'passing_lower'
                    n += 1
                    a += direction == 'passing_higher'
                groups.append({'arm': arm, 'world': w, 'streams': rows, 'informative': informative,
                               'direction': direction})
        result = {'label': label(n, a), 'informative_groups': n, 'agreeing': a,
                  'fraction': None if n == 0 else a / n, 'groups': groups}
    result |= {'plan_sha256': PLAN_SHA256, 'predictors_sha256': sha(PREDICTORS),
               'o2_report_sha256': sha(O2_REPORT), 'o2_primary': o2['summary']['primary']}
    OUTPUT.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'groups'}, indent=1))


if __name__ == '__main__':
    main()
