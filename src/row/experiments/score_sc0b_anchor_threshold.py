"""Independent scorer for SC0b (`SC0B_ANCHOR_THRESHOLD_PLAN.md`). Recomputes the label from the durable cell records
alone (per-task terminal NMSE on the canonical 64) and the committed SC0 report, without importing the runner, and
requires agreement with the runner's report. Refuses a report whose cells do not match their stamp, whose populations
are wrong, or whose plan or SC0 reference changed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

WORLD, STREAMS, DOSES = 64, (0, 1, 2, 3, 4), (24, 32, 40)
ANCHOR_KEY = 'D16_w64_s0'
THRESHOLD = 0.05


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(root=Path('artifacts/sc0b_anchor_threshold'), report=Path('reports/sc0b_anchor_threshold.json'),
          plan=Path('SC0B_ANCHOR_THRESHOLD_PLAN.md'), sc0_report=Path('reports/sc0_anchor_dose.json')):
    root, report = Path(root), Path(report)
    rep = json.loads(report.read_text())
    sha = rep['protocol_sha256']
    hashes = rep['protocol']['input_sha256']
    if _sha(plan) != hashes[Path(plan).as_posix()]:
        raise ValueError('plan changed since the run')
    if _sha(sc0_report) != hashes[Path(sc0_report).as_posix()]:
        raise ValueError('SC0 report changed since the run')
    sc0_cells = json.loads(Path(sc0_report).read_text())['cells']

    def load(key, n):
        saved = json.loads((root / 'cells' / f'{key}.json').read_text())
        if saved['stamp'] != {'protocol_sha256': sha} or not saved['complete']:
            raise ValueError(f'{key}: stamp')
        if rep['cells'][key] != saved['record']:
            raise ValueError(f'{key}: report and cell record differ')
        r = saved['record']
        if len(r['canonical_per_task']) != 64 or len(r['anchor_ids']) != n or r['stream_tasks'] != 128 + n:
            raise ValueError(f'{key}: population')
        return r

    def ok(per_task):
        m = float(np.median(list(per_task.values())))
        return math.isfinite(m) and m < THRESHOLD

    if len(rep['cells']) != 1 + len(DOSES) * len(STREAMS):
        raise ValueError('cell count')
    anchor = load(ANCHOR_KEY, 16)
    old = sc0_cells[ANCHOR_KEY]
    anchor_ok = (anchor['library_sha256'] == old['library_sha256']
                 and anchor['canonical_per_task'] == old['canonical_per_task'])
    k = {n: sum(ok(load(f'D{n}_w{WORLD}_s{s}', n)['canonical_per_task']) for s in STREAMS) for n in DOSES}
    if not anchor_ok:
        lab = 'UNINTERPRETABLE'
    elif k[32] == 5 and k[40] >= 4:
        lab = 'LENGTHS_SUFFICE'
    elif k[40] <= 1:
        lab = 'LENGTHS_INSUFFICIENT'
    else:
        lab = 'GRADED'
    out = {'k': {str(n): k[n] for n in DOSES}, 'anchor_reproduces': anchor_ok, 'label': lab}
    out['agrees_with_runner'] = all(out[x] == rep['summary'][x] for x in ('k', 'anchor_reproduces', 'label'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='artifacts/sc0b_anchor_threshold')
    ap.add_argument('--report', default='reports/sc0b_anchor_threshold.json')
    args = ap.parse_args()
    out = score(args.root, args.report)
    print(json.dumps(out, indent=1))
    return 0 if out['agrees_with_runner'] else 1


if __name__ == '__main__':
    sys.exit(main())
