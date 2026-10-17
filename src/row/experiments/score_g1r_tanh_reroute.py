"""Independent scorer for G1R (`G1R_TANH_REROUTE_PLAN.md`; Tier 0 re-route census).

Recomputes the reading from durable per-cell records (the run's `cells/` or the copy archived into `reports/`) and
G1S2's committed cells, not from the runner's `reading`: anchor A0, the collapsed sets, the stale fraction, the primary
and the label. Re-checks stamps, record hashes, lr, gain and pool per cell, the gates, and refuses a report whose plan,
config, G1S2 report or saved terminals changed since the run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PLAN = Path('G1R_TANH_REROUTE_PLAN.md')
REPORT = Path('reports/g1r_tanh_reroute.json')
CELLS = Path('artifacts/g1r_tanh_reroute/cells')
G1S2_REPORT = Path('reports/g1s2_tanh_budget.json')
WAKES = (1, 2)
CHECKPOINTS = (8192, 16384, 32768, 65536)
COLLAPSED, RESCUED, FITS, IDENTITY_ARI = 0.5, 0.2, 0.05, 0.8


def _fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def score(report=REPORT, cells=CELLS, check_terminals=True):
    rep = json.loads(Path(report).read_text())
    proto = rep['protocol']
    for path, digest in proto['input_sha256'].items():
        if path.endswith('model.pt') and not check_terminals:
            continue
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{path} changed since the run')
    for need in (PLAN, G1S2_REPORT):
        if need.as_posix() not in proto['input_sha256']:
            raise ValueError(f'report does not stamp {need}')
    if (proto['collapsed'], proto['rescued'], proto['fits'], proto['identity_ari']) != (
            COLLAPSED, RESCUED, FITS, IDENTITY_ARI) or tuple(proto['checkpoints']) != CHECKPOINTS:
        raise ValueError('registered thresholds or checkpoints differ from the plan')
    if not rep['gates']['all_pass']:
        raise ValueError('gates did not pass')
    records = {}
    for arm in ('RR_SLEEP', 'SLEEP_RR'):
        for u in WAKES:
            k = f'{arm}_u{u}'
            saved = json.loads((Path(cells) / f'{k}.json').read_text())
            if saved['stamp'] != {'protocol_sha256': rep['protocol_sha256']} or not saved['complete']:
                raise ValueError(f'{k}: stamp')
            if _fingerprint(saved['record']) != saved['record_sha256']:
                raise ValueError(f'{k}: record hash')
            r = saved['record']
            if r != rep['cells'][k] or r['wake_updates'] != u or r['world'] != 70:
                raise ValueError(f'{k}: report, wake or world')
            if (r['lr'], r['init_gain'], r['sleep_lr'], r['pool_size']) != (0.003, 1.0, 0.003, 188 * 64):
                raise ValueError(f'{k}: construction')
            records[k] = r
    g = json.loads(G1S2_REPORT.read_text())['cells']
    a0 = all(records[f'{a}_u{u}']['wake']['terminal_per_task'] == g[f'WAKE_u{u}']['terminal_per_task']
             for a in ('RR_SLEEP', 'SLEEP_RR') for u in WAKES)
    a0 = a0 and all(records[f'SLEEP_RR_u{u}']['curve'][str(n)]['terminal_per_task']
                    == g[f'SLEEP_u{u}']['curve'][str(n)]['terminal_per_task'] for u in WAKES for n in CHECKPOINTS)
    collapsed = {u: [t for t, v in g[f'SLEEP_u{u}']['curve'][str(CHECKPOINTS[-1])]['terminal_per_task'].items()
                     if v > COLLAPSED] for u in WAKES}
    rescued = {u: [t for t in collapsed[u] if records[f'SLEEP_RR_u{u}']['rr']['terminal_per_task'][t] < RESCUED]
               for u in WAKES}
    n_c = sum(len(v) for v in collapsed.values())
    stale = sum(len(v) for v in rescued.values()) / n_c if n_c else None
    curve = records['RR_SLEEP_u1']['curve']
    for n, c in curve.items():
        if not all(math.isfinite(v) for v in c['terminal_per_task'].values()):
            raise ValueError(f'RR_SLEEP_u1 {n}: terminal')
    eligible = {n: c['terminal_median'] for n, c in curve.items() if c['d1_ari'] >= IDENTITY_ARI}
    best_n = min(eligible, key=eligible.get) if eligible else None
    best = eligible[best_n] if best_n else None
    if not a0:
        label = 'ANCHOR_MISS'
    elif best is not None and best < FITS:
        label = 'FITS'
    elif stale is not None and stale >= 0.5:
        label = 'STALE_ONLY'
    else:
        label = 'NOT_STALE'
    out = {'label': label, 'anchor_A0': a0, 'collapsed_tasks': {f'u{u}': len(v) for u, v in collapsed.items()},
           'rescued_by_reroute': {f'u{u}': len(v) for u, v in rescued.items()}, 'stale_fraction': stale,
           'rr_sleep_u1_best_checkpoint': best_n, 'rr_sleep_u1_best': best,
           'medians': {k: {'wake': r['wake']['terminal_median'],
                           **{f's{n}': r['curve'][str(n)]['terminal_median'] for n in CHECKPOINTS},
                           **({'rerouted': r['rerouted']['terminal_median']} if 'rerouted' in r else
                              {'rr': r['rr']['terminal_median'], 'rr_sleep': r['rr_sleep']['terminal_median']}),
                           'routes_changed': r['routes_changed']} for k, r in records.items()}}
    mine = rep.get('reading', {})
    out['runner_agrees'] = all(mine.get(k) == out[k] for k in
                               ('label', 'stale_fraction', 'rr_sleep_u1_best', 'rr_sleep_u1_best_checkpoint'))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(REPORT))
    parser.add_argument('--cells', default=str(CELLS))
    parser.add_argument('--skip-terminal-hashes', action='store_true',
                        help='only if the saved terminals in artifacts/ were deleted after the run (disclose it)')
    args = parser.parse_args()
    print(json.dumps(score(args.report, args.cells, not args.skip_terminal_hashes), indent=1))


if __name__ == '__main__':
    main()
