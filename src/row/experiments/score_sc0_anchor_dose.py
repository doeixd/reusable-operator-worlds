"""Independent scorer for SC0 (`SC0_ANCHOR_DOSE_PLAN.md`). Recomputes the label from the durable cell records alone
(per-task terminal NMSE on the canonical 64), without importing the runner, and requires agreement with the runner's
report. Refuses a report whose cells do not match their stamp, whose populations are wrong, or whose plan changed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

WORLD, STREAMS, DOSES = 64, (0, 1, 2, 3, 4), (4, 8, 16)
CONTROLS = ((60, 0), (0, 0))
THRESHOLD = 0.05


def score(root=Path('artifacts/sc0_anchor_dose'), report=Path('reports/sc0_anchor_dose.json'),
          plan=Path('SC0_ANCHOR_DOSE_PLAN.md'), world=WORLD):
    root, report = Path(root), Path(report)
    rep = json.loads(report.read_text())
    sha = rep['protocol_sha256']
    if hashlib.sha256(Path(plan).read_bytes()).hexdigest() != rep['protocol']['input_sha256'][Path(plan).as_posix()]:
        raise ValueError('plan changed since the run')
    keys = [(n, s) for n in DOSES for s in STREAMS] + list(CONTROLS)
    med = {}
    for n, s in keys:
        key = f'D{n}_w{world}_s{s}'
        saved = json.loads((root / 'cells' / f'{key}.json').read_text())
        if saved['stamp'] != {'protocol_sha256': sha} or not saved['complete']:
            raise ValueError(f'{key}: stamp')
        if rep['cells'][key] != saved['record']:
            raise ValueError(f'{key}: report and cell record differ')
        r = saved['record']
        if len(r['canonical_per_task']) != 64 or len(r['anchor_ids']) != n or r['stream_tasks'] != 128 + n:
            raise ValueError(f'{key}: population')
        med[(n, s)] = float(np.median(list(r['canonical_per_task'].values())))

    def ok(m):
        return math.isfinite(m) and m < THRESHOLD

    k = {n: sum(ok(med[(n, s)]) for s in STREAMS) for n in DOSES}
    top, floor = ok(med[(60, 0)]), not ok(med[(0, 0)])
    if not (top and floor):
        lab = 'UNINTERPRETABLE'
    elif k[4] == 5:
        lab = 'NO_OPPORTUNITY'
    elif k[4] <= 1 and k[16] >= 4:
        lab = 'OPPORTUNITY'
    else:
        lab = 'GRADED'
    out = {'k': {str(n): k[n] for n in DOSES}, 'top_passes': top, 'floor_fails': floor, 'label': lab}
    out['agrees_with_runner'] = all(out[x] == rep['summary'][x] for x in ('k', 'top_passes', 'floor_fails', 'label'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='artifacts/sc0_anchor_dose')
    ap.add_argument('--report', default='reports/sc0_anchor_dose.json')
    args = ap.parse_args()
    out = score(args.root, args.report)
    print(json.dumps(out, indent=1))
    return 0 if out['agrees_with_runner'] else 1


if __name__ == '__main__':
    sys.exit(main())
