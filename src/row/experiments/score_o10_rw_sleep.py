"""Independent scorer for O10 (plan `O10_REROUTE_WAKE_SLEEP_PLAN.md`, frozen 264abd1).

Re-derives r (target rescues) and b (harm-set breaks) from the cell records with its own rule code, takes the
target set from O9's report (failing REROUTE_WAKE cells), checks construction and gates, and refuses a report whose
inputs moved since the run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint

REPORT = Path('reports/o10_rw_sleep.json')
O9_REPORT = Path('reports/o9_sealed_online.json')


def ok(m):
    return isinstance(m, (int, float)) and math.isfinite(m) and m < 0.05


def verdict(r, b):
    return 'HARMS' if b >= 3 else 'REPAIRS' if r >= 7 else 'PARTIAL' if r >= 4 else 'NO_REPAIR'


def score(report_path=REPORT):
    rep = json.loads(Path(report_path).read_text())
    problems = []
    if fingerprint(rep['protocol']) != rep['protocol_sha256']:
        problems.append('protocol fingerprint mismatch')
    for p, sha in rep['protocol']['input_sha256'].items():
        if not Path(p).exists() or digest(Path(p)) != sha:
            problems.append(f'input changed since the run: {p}')
    if not rep.get('complete') or not rep['gates'].get('all_pass'):
        problems.append('incomplete report or gates not passed')
    o9 = json.loads(O9_REPORT.read_text())['cells']
    target = {(w, s) for w in range(930, 945) for s in range(3)
              if not ok(o9[f'REROUTE_WAKE_w{w}_s{s}']['terminal_median'])}
    if target != {tuple(c) for c in rep['protocol']['target']}:
        problems.append('registered target set is not O9\'s failing REROUTE_WAKE cells')
    r = b = 0
    for w in range(930, 945):
        for s in range(3):
            rec = rep['cells'].get(f'w{w}_s{s}')
            if rec is None:
                problems.append(f'missing w{w}_s{s}')
                continue
            if rec['pool_size'] != 188 * 64 or rec['extra_updates'] != 8192 or len(rec['terminal_per_task']) != 64 \
                    or not rec['rebuild_matches']:
                problems.append(f'w{w}_s{s}: construction')
            if (w, s) in target:
                r += ok(rec['terminal_median'])
            else:
                b += not ok(rec['terminal_median'])
    out = {'valid': None, 'problems': problems, 'r': r, 'target_cells': len(target), 'b': b,
           'harm_cells': 45 - len(target), 'label': verdict(r, b)}
    if rep.get('summary', {}).get('label') != out['label']:
        problems.append(f"runner label {rep.get('summary', {}).get('label')} != scorer {out['label']}")
    out['valid'] = not problems
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(REPORT))
    result = score(parser.parse_args().report)
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
