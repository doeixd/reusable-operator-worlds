"""Independent scorer for B2-hard v2 (`B2_HARD_V2_PLAN.md`). Recomputes every registered clause from the durable cell
records alone (per-task wake state-weight norms, per-task end-of-stream decisions, per-task terminal errors, and the
wake library's per-operation best fixed counts), without importing the runner's summary code, and requires agreement
with the runner's report. Refuses a report whose cells do not match their stamp, whose populations are wrong, or
whose plan has changed since the run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ARMS = ('ONESTEP', 'ZERO', 'GIVEN')
WORLDS, STREAMS = (57, 58, 59), (0, 1, 2)
N_LOOP, N_CAND = 48, 108
THRESHOLD, LOOP_BAR = 0.05, 0.38


def _auc(pos, neg):
    hits = 0.0
    for p in pos:
        for n in neg:
            hits += 1.0 if p > n else 0.5 if p == n else 0.0
    return hits / (len(pos) * len(neg))


def _rooted(by_op):
    return any(len(Ls) > 0 and min(Ls) >= 2 for Ls in by_op.values())


def score(root=Path('artifacts/b2_hard_v2'), report=Path('reports/b2_hard_v2.json'), plan=Path('B2_HARD_V2_PLAN.md'),
          worlds=WORLDS):
    root, report = Path(root), Path(report)
    rep = json.loads(report.read_text())
    sha = rep['protocol_sha256']
    if hashlib.sha256(Path(plan).read_bytes()).hexdigest() != rep['protocol']['input_sha256'][Path(plan).as_posix()]:
        raise ValueError('plan changed since the run')
    recs = {}
    for a in ARMS:
        for w in worlds:
            for s in STREAMS:
                key = f'{a}_w{w}_s{s}'
                saved = json.loads((root / 'cells' / f'{key}.json').read_text())
                if saved['stamp'] != {'protocol_sha256': sha} or not saved['complete']:
                    raise ValueError(f'{key}: stamp')
                if rep['cells'][key] != saved['record']:
                    raise ValueError(f'{key}: report and cell record differ')
                r = saved['record']
                if (len(r['loop_per_task']) != N_LOOP or len(r['canonical_per_task']) != 64
                        or sum(len(v) for v in r['wake_anchor_best_L_by_op'].values()) != N_CAND - N_LOOP):
                    raise ValueError(f'{key}: population')
                recs[key] = r

    def med(r, k):
        return float(np.median(list(r[k].values())))

    aucs, kept, single, k_formation, n_terminal = [], 0, 0, 0, 0
    rooted = {a: 0 for a in ARMS}
    for w in worlds:
        for s in STREAMS:
            O = recs[f'ONESTEP_w{w}_s{s}']
            loops = set(O['loop_task_ids'])
            norms, dec = O['wake_state_norms'], O['decisions']
            if len(loops) != N_LOOP or len(norms) != N_CAND or len(dec) != N_CAND or not loops <= set(norms):
                raise ValueError(f'w{w}_s{s}: candidates')
            aucs.append(_auc([v for t, v in norms.items() if t in loops], [v for t, v in norms.items() if t not in loops]))
            kept += sum(dec[t]['loop'] for t in loops)
            single += sum(not v['loop'] for t, v in dec.items() if t not in loops)
            c = med(O, 'canonical_per_task')
            k_formation += int(math.isfinite(c) and c < THRESHOLD)
            n_terminal += int(med(O, 'loop_per_task') < LOOP_BAR)
            for a in ARMS:
                rooted[a] += int(_rooted(recs[f'{a}_w{w}_s{s}']['wake_anchor_best_L_by_op']))
    n = len(worlds) * len(STREAMS)
    auc_median = float(np.median(aucs))
    recall, specificity = kept / (N_LOOP * n), single / ((N_CAND - N_LOOP) * n)
    if k_formation <= 6:
        lab = 'FORMATION_BROKEN'
    elif auc_median >= 0.9 and recall >= 0.8 and specificity >= 0.95 and n_terminal >= 8:
        lab = 'DISCOVERS'
    else:
        lab = 'NOT_DISCOVERED'
    if rooted['ZERO'] <= 1:
        roots = 'NO_ROOTS_IN_CONTROL'
    elif rooted['ONESTEP'] == 0:
        roots = 'ROOTS_PREVENTED'
    else:
        roots = 'ROOTS_NOT_PREVENTED'
    out = {'k_formation': k_formation, 'wake_state_auc_median': auc_median, 'loop_recall': recall,
           'straight_specificity': specificity, 'n_terminal': n_terminal, 'label': lab,
           'rooted_cells': rooted, 'roots_label': roots}
    out['agrees_with_runner'] = all(
        (abs(out[k] - rep['summary'][k]) < 1e-12 if isinstance(out[k], float) else out[k] == rep['summary'][k])
        for k in out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='artifacts/b2_hard_v2')
    ap.add_argument('--report', default='reports/b2_hard_v2.json')
    args = ap.parse_args()
    out = score(args.root, args.report)
    print(json.dumps(out, indent=1))
    return 0 if out['agrees_with_runner'] else 1


if __name__ == '__main__':
    sys.exit(main())
