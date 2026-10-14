"""Independent scorer for G1S (`G1S_TANH_SIZING_PLAN.md`; Tier 1, exploratory sizing).

Recomputes the reading from the durable per-cell records and G1T's committed world-70 cells (not the runner's
`reading`): anchor A0, the identity filter, the chosen setting and the label. Re-checks every cell's stamp and record
hash, that each cell ran at its setting (wake lifetime and sleep learning rates), the gates, and refuses a report
whose plan, config or G1T report changed since the run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PLAN = Path('G1S_TANH_SIZING_PLAN.md')
REPORT = Path('reports/g1s_tanh_sizing.json')
CELLS = Path('artifacts/g1s_tanh_sizing/cells')
G1T_REPORT = Path('reports/g1t_tanh_formation.json')
WORLD = 70
SETTINGS = ((0.001, 1.0), (0.003, 1.0), (0.01, 1.0), (0.001, 2.0), (0.003, 2.0), (0.01, 2.0))
BASELINE = (0.001, 1.0)
FITS, IMPROVES, IDENTITY_ARI = 0.05, 0.2, 0.8


def tag(s):
    return f'lr{s[0]:g}_g{s[1]:g}'


def _fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def score(report=REPORT, cells=CELLS):
    rep = json.loads(Path(report).read_text())
    proto = rep['protocol']
    for path, digest in proto['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{path} changed since the run')
    for need in (PLAN, G1T_REPORT):
        if need.as_posix() not in proto['input_sha256']:
            raise ValueError(f'report does not stamp {need}')
    if (proto['fits'], proto['improves'], proto['identity_ari']) != (FITS, IMPROVES, IDENTITY_ARI):
        raise ValueError('registered thresholds differ from the plan')
    if [tuple(s) for s in proto['settings']] != list(SETTINGS) or proto['world'] != WORLD:
        raise ValueError('settings or world differ from the plan')
    if not rep['gates']['all_pass']:
        raise ValueError('gates did not pass')
    records = {}
    for arm in ('WAKE', 'SLEEP'):
        for s in SETTINGS:
            k = f'{arm}_{tag(s)}'
            saved = json.loads((Path(cells) / f'{k}.json').read_text())
            if saved['stamp'] != {'protocol_sha256': rep['protocol_sha256']} or not saved['complete']:
                raise ValueError(f'{k}: stamp')
            if _fingerprint(saved['record']) != saved['record_sha256']:
                raise ValueError(f'{k}: record hash')
            r = saved['record']
            if r != rep['cells'][k] or r['scale'] != 1 or r['world'] != WORLD:
                raise ValueError(f'{k}: report, scale or world')
            if (r['lr'], r['init_gain']) != s:
                raise ValueError(f'{k}: cell is not its setting')
            ran_at = r['lifetime_lr'] if arm == 'WAKE' else r['sleep_lr']
            if ran_at != s[0]:
                raise ValueError(f'{k}: ran at lr {ran_at}')
            if len(r['terminal_per_task']) != 64 or not all(math.isfinite(v) for v in r['terminal_per_task'].values()):
                raise ValueError(f'{k}: terminal')
            records[k] = r
    g1t = json.loads(G1T_REPORT.read_text())['cells']
    a0_wake = records[f'WAKE_{tag(BASELINE)}']['terminal_per_task'] == g1t[f'SHUFFLED_w{WORLD}']['terminal_per_task']
    a0_sleep = records[f'SLEEP_{tag(BASELINE)}']['terminal_per_task'] == g1t[f'SLEEP_w{WORLD}']['terminal_per_task']
    slept = {tag(s): records[f'SLEEP_{tag(s)}'] for s in SETTINGS}
    eligible = {t: r['terminal_median'] for t, r in slept.items() if r['d1_ari'] >= IDENTITY_ARI}
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
           'chosen_terminal': best,
           'slept_terminal': {t: r['terminal_median'] for t, r in slept.items()},
           'wake_terminal': {tag(s): records[f'WAKE_{tag(s)}']['terminal_median'] for s in SETTINGS},
           'slept_depth_median': {t: {d: r['depth'][d]['median'] for d in '123'} for t, r in slept.items()},
           'slept_d1_ari': {t: r['d1_ari'] for t, r in slept.items()},
           'wake_d1_ari': {tag(s): records[f'WAKE_{tag(s)}']['d1_ari'] for s in SETTINGS},
           'seconds': {k: r['seconds'] for k, r in records.items()}}
    mine = rep.get('reading', {})
    out['runner_agrees'] = (mine.get('label') == label and mine.get('chosen') == chosen
                            and mine.get('chosen_terminal') == best)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(REPORT))
    parser.add_argument('--cells', default=str(CELLS))
    args = parser.parse_args()
    print(json.dumps(score(args.report, args.cells), indent=1))


if __name__ == '__main__':
    main()
