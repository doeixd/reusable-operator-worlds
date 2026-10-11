"""Independent scorer for B2-online v2 (`B2_ONLINE_V2_PLAN.md`). Recomputes every registered clause from the durable
cell records alone (per-task terminal canonical and loop errors, wake medians), without importing the runner's
summary code, and requires agreement with the runner's report. Refuses a report whose cells do not match their stamp
or whose plan has changed since the run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ARMS = ('LOOP', 'LOOP_ATTACHED', 'CONSTCOUNT')
WORLDS, STREAMS = (50, 51, 52), (0, 1, 2)
N_LOOP = 48


def score(root=Path('artifacts/b2_online_v2'), report=Path('reports/b2_online_v2.json'), plan=Path('B2_ONLINE_V2_PLAN.md'),
          worlds=WORLDS):
    root, report = Path(root), Path(report)
    rep = json.loads(report.read_text())
    sha = rep['protocol_sha256']
    if hashlib.sha256(Path(plan).read_bytes()).hexdigest() != rep['protocol']['input_sha256'][Path(plan).as_posix()]:
        raise ValueError('plan changed since the run')
    bar = rep['protocol']['loop_bar']
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
                if len(saved['record']['loop_per_task']) != N_LOOP or len(saved['record']['canonical_per_task']) != 64:
                    raise ValueError(f'{key}: population')
                recs[key] = saved['record']
    med = lambda r, k: float(np.median(list(r[k].values())))   # noqa: E731
    k_formation = n_wake = n_terminal = n_mechanism = n_control = 0
    for w in worlds:
        for s in STREAMS:
            L, A, C = (recs[f'{a}_w{w}_s{s}'] for a in ARMS)
            canon, loop = med(L, 'canonical_per_task'), med(L, 'loop_per_task')
            k_formation += int(math.isfinite(canon) and canon < 0.05)
            n_terminal += int(math.isfinite(loop) and loop < bar)
            n_wake += int(L['wake_loop_median'] < 0.5 * C['wake_loop_median'])
            n_mechanism += int(canon < med(A, 'canonical_per_task'))
            n_control += int(loop < med(C, 'loop_per_task'))
    if k_formation <= 6:
        lab = 'FORMATION_BROKEN'
    elif n_terminal >= 8 and n_wake >= 8:
        lab = 'LOOPS_ONLINE'
    elif n_terminal >= 8:
        lab = 'LOOPS_AFTER_SLEEP'
    else:
        lab = 'NO_LOOPS'
    out = {'k_formation': k_formation, 'n_wake': n_wake, 'n_terminal': n_terminal, 'n_mechanism': n_mechanism,
           'n_control': n_control, 'label': lab,
           'mechanism_label': 'DETACH_PROTECTS' if n_mechanism >= 8 else 'DETACH_NOT_SHOWN'}
    out['agrees_with_runner'] = all(out[k] == rep['summary'][k] for k in out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='artifacts/b2_online_v2')
    ap.add_argument('--report', default='reports/b2_online_v2.json')
    args = ap.parse_args()
    out = score(args.root, args.report)
    print(json.dumps(out, indent=1))
    return 0 if out['agrees_with_runner'] else 1


if __name__ == '__main__':
    sys.exit(main())
