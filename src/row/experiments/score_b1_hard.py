"""Independent scorer for B1-hard (`B1_HARD_PLAN.md`). Recomputes every registered clause from the durable cell
records alone (per-task wake gate norms, per-task end-of-stream decisions, terminal medians), without importing the
runner's summary code, and requires agreement with the runner's report. Refuses a report whose protocol stamp does
not match its cells or whose plan has changed since the run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ARMS = ('ALLGATE', 'CONSTMIX')
WORLDS, STREAMS = (44, 45, 46), (0, 1, 2)
N_BRANCH, N_TASKS = 48, 236


def _auc(pos, neg):
    hits = 0.0
    for p in pos:
        for n in neg:
            hits += 1.0 if p > n else 0.5 if p == n else 0.0
    return hits / (len(pos) * len(neg))


def score(root=Path('artifacts/b1_hard'), report=Path('reports/b1_hard.json'), plan=Path('B1_HARD_PLAN.md')):
    root, report = Path(root), Path(report)
    rep = json.loads(report.read_text())
    sha = rep['protocol_sha256']
    if hashlib.sha256(Path(plan).read_bytes()).hexdigest() != rep['protocol']['input_sha256'][Path(plan).as_posix()]:
        raise ValueError('plan changed since the run')
    recs = {}
    for a in ARMS:
        for w in WORLDS:
            for s in STREAMS:
                key = f'{a}_w{w}_s{s}'
                saved = json.loads((root / 'cells' / f'{key}.json').read_text())
                if saved['stamp'] != {'protocol_sha256': sha} or not saved['complete']:
                    raise ValueError(f'{key}: stamp')
                if rep['cells'][key] != saved['record']:
                    raise ValueError(f'{key}: report and cell record differ')
                recs[key] = saved['record']
    aucs, kept_b, single_s, k_form, n_ctrl = [], 0, 0, 0, 0
    for w in WORLDS:
        for s in STREAMS:
            A, C = recs[f'ALLGATE_w{w}_s{s}'], recs[f'CONSTMIX_w{w}_s{s}']
            b = set(A['branch_task_ids'])
            if len(b) != N_BRANCH or len(A['decisions']) != N_TASKS or len(A['wake_gate_norms']) != N_TASKS:
                raise ValueError(f'w{w}_s{s}: population')
            if any(C['decisions'].values()):
                raise ValueError(f'w{w}_s{s}: control kept a branch')
            aucs.append(_auc([v for t, v in A['wake_gate_norms'].items() if t in b],
                             [v for t, v in A['wake_gate_norms'].items() if t not in b]))
            kept_b += sum(A['decisions'][t] for t in b)
            single_s += sum(not v for t, v in A['decisions'].items() if t not in b)
            m = float(np.median(list(A['canonical_per_task'].values())))
            k_form += int(math.isfinite(m) and m < 0.05)
            n_ctrl += int(float(np.median(list(A['branch_per_task'].values())))
                          < float(np.median(list(C['branch_per_task'].values()))))
    cells = len(WORLDS) * len(STREAMS)
    out = {'k_formation': k_form, 'wake_gate_auc_median': float(np.median(aucs)),
           'branch_recall': kept_b / (N_BRANCH * cells), 'straight_specificity': single_s / ((N_TASKS - N_BRANCH) * cells),
           'n_better_than_control': n_ctrl}
    if k_form <= 6:
        out['label'] = 'FORMATION_BROKEN'
    elif (out['wake_gate_auc_median'] >= 0.9 and out['branch_recall'] >= 0.8 and out['straight_specificity'] >= 0.95
          and n_ctrl >= 8):
        out['label'] = 'DISCOVERS'
    else:
        out['label'] = 'NOT_DISCOVERED'
    runner = rep['summary']
    agree = all((abs(out[k] - runner[k]) < 1e-12) if isinstance(out[k], float) else out[k] == runner[k]
                for k in out)
    out['agrees_with_runner'] = agree
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='artifacts/b1_hard')
    ap.add_argument('--report', default='reports/b1_hard.json')
    args = ap.parse_args()
    out = score(args.root, args.report)
    print(json.dumps(out, indent=1))
    return 0 if out['agrees_with_runner'] else 1


if __name__ == '__main__':
    sys.exit(main())
