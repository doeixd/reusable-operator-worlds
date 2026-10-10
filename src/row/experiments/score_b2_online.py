"""Independent scorer for B2-online (`B2_ONLINE_PLAN.md`). Recomputes every registered clause from the durable cell
records alone (per-task terminal canonical and loop errors, wake medians), without importing the runner's summary
code, and requires agreement with the runner's report. Refuses a report whose cells do not match their stamp or
whose plan has changed since the run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ARMS = ('LOOP', 'CONSTCOUNT', 'REFUSAL')
WORLDS, STREAMS = (44, 45, 46), (0, 1, 2)
N_LOOP = 48


def score(root=Path('artifacts/b2_online'), report=Path('reports/b2_online.json'), plan=Path('B2_ONLINE_PLAN.md')):
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
                if len(saved['record']['loop_per_task']) != N_LOOP or len(saved['record']['canonical_per_task']) != 64:
                    raise ValueError(f'{key}: population')
                recs[key] = saved['record']
    k_formation = n_wake = n_terminal = n_control = 0
    for w in WORLDS:
        for s in STREAMS:
            L, C = recs[f'LOOP_w{w}_s{s}'], recs[f'CONSTCOUNT_w{w}_s{s}']
            canon = float(np.median(list(L['canonical_per_task'].values())))
            loop = float(np.median(list(L['loop_per_task'].values())))
            k_formation += int(math.isfinite(canon) and canon < 0.05)
            n_terminal += int(math.isfinite(loop) and loop < 0.25)
            n_wake += int(L['wake_loop_median'] < 0.5 * C['wake_loop_median'])
            n_control += int(loop < float(np.median(list(C['loop_per_task'].values()))))
    if k_formation <= 6:
        lab = 'FORMATION_BROKEN'
    elif n_terminal >= 8 and n_wake >= 8:
        lab = 'LOOPS_ONLINE'
    elif n_terminal >= 8:
        lab = 'LOOPS_AFTER_SLEEP'
    else:
        lab = 'NO_LOOPS'
    out = {'k_formation': k_formation, 'n_wake': n_wake, 'n_terminal': n_terminal, 'n_control': n_control,
           'label': lab}
    out['agrees_with_runner'] = all(out[k] == rep['summary'][k] for k in out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='artifacts/b2_online')
    ap.add_argument('--report', default='reports/b2_online.json')
    args = ap.parse_args()
    out = score(args.root, args.report)
    print(json.dumps(out, indent=1))
    return 0 if out['agrees_with_runner'] else 1


if __name__ == '__main__':
    sys.exit(main())
