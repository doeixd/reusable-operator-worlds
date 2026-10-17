"""Independent scorer for G1S2 (`G1S2_TANH_BUDGET_PLAN.md`; Tier 1, exploratory sizing).

Recomputes the reading from durable per-cell records (the run directory's `cells/`, or the copy the runner archives
into `reports/`) and G1S's committed cells, not from the runner's `reading`: anchor A0, the identity filter, the chosen
(wake updates, sleep length) point, the label and the compounding ratios. Re-checks stamps, record hashes, that each
cell ran at its budget and learning rate, the gates, and refuses a report whose plan, config or G1S report changed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PLAN = Path('G1S2_TANH_BUDGET_PLAN.md')
REPORT = Path('reports/g1s2_tanh_budget.json')
CELLS = Path('artifacts/g1s2_tanh_budget/cells')
G1S_REPORT = Path('reports/g1s_tanh_sizing.json')
WORLD = 70
LR = 0.003
WAKE_UPDATES = (1, 2)
CHECKPOINTS = (8192, 16384, 32768, 65536)
FITS, IMPROVES, IDENTITY_ARI = 0.05, 0.1, 0.8


def _fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def score(report=REPORT, cells=CELLS):
    rep = json.loads(Path(report).read_text())
    proto = rep['protocol']
    for path, digest in proto['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{path} changed since the run')
    for need in (PLAN, G1S_REPORT):
        if need.as_posix() not in proto['input_sha256']:
            raise ValueError(f'report does not stamp {need}')
    if (proto['fits'], proto['improves'], proto['identity_ari']) != (FITS, IMPROVES, IDENTITY_ARI):
        raise ValueError('registered thresholds differ from the plan')
    if (tuple(proto['wake_updates']), tuple(proto['checkpoints']), proto['world'], proto['setting'][0]) != (
            WAKE_UPDATES, CHECKPOINTS, WORLD, LR):
        raise ValueError('budgets, world or setting differ from the plan')
    if not rep['gates']['all_pass']:
        raise ValueError('gates did not pass')
    records = {}
    for arm in ('WAKE', 'SLEEP'):
        for u in WAKE_UPDATES:
            k = f'{arm}_u{u}'
            saved = json.loads((Path(cells) / f'{k}.json').read_text())
            if saved['stamp'] != {'protocol_sha256': rep['protocol_sha256']} or not saved['complete']:
                raise ValueError(f'{k}: stamp')
            if _fingerprint(saved['record']) != saved['record_sha256']:
                raise ValueError(f'{k}: record hash')
            r = saved['record']
            if r != rep['cells'][k] or r['scale'] != 1 or r['world'] != WORLD or r['wake_updates'] != u:
                raise ValueError(f'{k}: report, scale, world or budget')
            if arm == 'WAKE' and (r['lifetime_lr'] != LR or r['lifetime_updates_per_example'] != u):
                raise ValueError(f'{k}: lifetime ran at the wrong lr or updates')
            if arm == 'SLEEP' and (r['sleep_lr'] != LR or [int(n) for n in r['curve']] != list(CHECKPOINTS)):
                raise ValueError(f'{k}: sleep lr or checkpoints')
            records[k] = r
    g1s = json.loads(G1S_REPORT.read_text())['cells']
    a0_wake = records['WAKE_u1']['terminal_per_task'] == g1s['WAKE_lr0.003_g1']['terminal_per_task']
    a0_sleep = (records['SLEEP_u1']['curve'][str(CHECKPOINTS[0])]['terminal_per_task']
                == g1s['SLEEP_lr0.003_g1']['terminal_per_task'])
    points = {}
    for u in WAKE_UPDATES:
        for n in CHECKPOINTS:
            c = records[f'SLEEP_u{u}']['curve'][str(n)]
            per = c['terminal_per_task']
            if len(per) != 64 or not all(math.isfinite(v) for v in per.values()):
                raise ValueError(f'u{u} s{n}: terminal')
            points[f'u{u}_s{n}'] = {'terminal': c['terminal_median'], 'd1_ari': c['d1_ari'],
                                    'depth': {d: c['depth'][d]['median'] for d in '123'},
                                    'compounding': c['depth']['3']['median'] / c['depth']['1']['median']}
    eligible = {t: p['terminal'] for t, p in points.items() if p['d1_ari'] >= IDENTITY_ARI}
    chosen = min(eligible, key=eligible.get) if eligible else None
    best = eligible[chosen] if chosen else None
    if not (a0_wake and a0_sleep):
        label = 'ANCHOR_MISS'
    elif chosen is None:
        label = 'IDENTITY_LOST'
    elif best < FITS:
        label = 'FITS'
    elif best < IMPROVES:
        label = 'IMPROVES'
    else:
        label = 'NO_GAIN'
    out = {'label': label, 'anchor_A0': {'wake': a0_wake, 'sleep': a0_sleep}, 'chosen': chosen,
           'chosen_terminal': best, 'points': points,
           'last_doubling_ratio': {f'u{u}': points[f'u{u}_s{CHECKPOINTS[-1]}']['terminal']
                                   / points[f'u{u}_s{CHECKPOINTS[-2]}']['terminal'] for u in WAKE_UPDATES},
           'wake_terminal': {f'u{u}': records[f'WAKE_u{u}']['terminal_median'] for u in WAKE_UPDATES},
           'seconds': {k: r['seconds'] for k, r in records.items()}}
    mine = rep.get('reading', {})
    out['runner_agrees'] = (mine.get('label') == label and mine.get('chosen') == chosen
                            and mine.get('chosen_terminal') == best
                            and mine.get('last_doubling_ratio') == out['last_doubling_ratio'])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(REPORT))
    parser.add_argument('--cells', default=str(CELLS))
    args = parser.parse_args()
    print(json.dumps(score(args.report, args.cells), indent=1))


if __name__ == '__main__':
    main()
