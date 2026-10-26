"""SY0 Tier 0 census (`SY0_SEARCH_BUDGET_CENSUS_PLAN.md`): does a fixed route-optimization budget FAIL on deep
straight-line programs, or only cost more? A re-analysis of the committed E5.1 per-task cache; no model is run.

A task is SOLVED at budget k iff log(nmse_OPTk) - log(nmse_O) <= GAP; HARD iff solved at no budget. U(D) is the HARD
fraction over the 24 tasks of depth D. The label compares U(D) with U(D-2), the optimistic stand-in for a search made
two steps shorter by one length-3 macro. `--verify` recomputes the label on a separate code path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from row.necessity_gate import incumbent_degrades

CACHE = Path('reports/e5_1_cache')
PROTOCOL = 'f945fe71aec9f1bf'
OUTPUT = Path('reports/sy0_search_budget_census.json')
WORLDS = (0, 1, 2)
DEPTHS = tuple(range(3, 11))
PER_DEPTH = 8
BUDGETS = (250, 500, 1000, 2000)
GAP = 0.15
U_MIN, U_DROP, Z_MIN = 0.25, 0.10, 1.645
DEEP = (8, 9, 10)


def load_records(cache=CACHE):
    out = {}
    for d in DEPTHS:
        for w in WORLDS:
            for i in range(PER_DEPTH):
                path = cache / f'w{w}_d{d}_{i}.json'
                if not path.exists():
                    raise SystemExit(f'missing record {path}; refuse to shrink a denominator')
                raw = json.loads(path.read_text(encoding='utf-8'))
                if raw.get('protocol') != PROTOCOL:
                    raise SystemExit(f'{path} under protocol {raw.get("protocol")}, expected {PROTOCOL}')
                out[(d, w, i)] = raw['value']
    return out


def solved_budgets(rec):
    o = math.log(rec['O']['nmse'])
    return [k for k in BUDGETS if math.log(rec[f'OPT{k}']['nmse']) - o <= GAP]


def label(u, n_per_depth=len(WORLDS) * PER_DEPTH):
    """u: {depth: hard fraction}. Pooled trend over all depths (first rule, per-depth differences, failed
    discrimination: see the plan). Fires iff the deep level U(8..10) >= U_MIN AND the fitted rise over two depths
    (the macro proxy) is >= U_DROP AND that slope is significant one-sided (z >= 1.645, pooled binomial variance)."""
    ds = np.array(DEPTHS, dtype=float)
    us = np.array([u[d] for d in DEPTHS])
    w = np.full(len(ds), n_per_depth, dtype=float)
    dbar = np.sum(w * ds) / np.sum(w)
    sxx = np.sum(w * (ds - dbar) ** 2)
    slope = float(np.sum(w * (ds - dbar) * us) / sxx)
    pbar = float(np.sum(w * us) / np.sum(w))
    se = math.sqrt(max(pbar * (1 - pbar), 1e-12) / sxx)
    deep = float(np.mean([u[d] for d in DEEP]))
    stats = {'slope_per_depth': slope, 'rise_over_two_depths': 2 * slope, 'z': slope / se, 'U_deep': deep}
    fire = deep >= U_MIN and 2 * slope >= U_DROP and slope / se >= Z_MIN
    return ('OPPORTUNITY' if fire else 'NO_OPPORTUNITY'), stats


def discrimination(draws=20000, seed=2610):
    rng = np.random.default_rng(seed)
    n = len(WORLDS) * PER_DEPTH

    def rate(profile):
        p = np.array([profile(d) for d in DEPTHS])
        u = rng.binomial(n, p, size=(draws, len(DEPTHS))) / n
        fires = 0
        for row in u:
            if label(dict(zip(DEPTHS, row)))[0] == 'OPPORTUNITY':
                fires += 1
        return fires / draws

    nulls = {f'null_{p}': rate(lambda d, p=p: p) for p in (0.05, 0.15, 0.25, 0.35)}
    effects = {
        'linear_0.05_to_0.45': rate(lambda d: 0.05 + 0.4 * max(0, d - 4) / 6),
        'step_at_8': rate(lambda d: 0.05 if d < 8 else 0.45),
    }
    ok = max(nulls.values()) <= 0.05 and min(effects.values()) >= 0.80
    return {'false_fire': nulls, 'detection': effects, 'adequate': ok, 'draws': draws}


def census(records):
    per_depth = {}
    for d in DEPTHS:
        tasks = [(w, i, solved_budgets(records[(d, w, i)])) for w in WORLDS for i in range(PER_DEPTH)]
        n = len(tasks)
        hard = [t for t in tasks if not t[2]]
        bstar = [min(s) if s else None for _, _, s in tasks]
        nonmono = sum(1 for _, _, s in tasks
                      if any(k in s and 2 * k in BUDGETS and 2 * k not in s for k in BUDGETS))
        sortable = sorted(b if b is not None else 10 ** 9 for b in bstar)
        med = sortable[(n - 1) // 2]
        per_depth[d] = {
            'n': n, 'U': len(hard) / n,
            'F': {str(k): sum(1 for _, _, s in tasks if k in s) / n for k in BUDGETS},
            'median_b_star': med if med < 10 ** 9 else '>2000',
            'non_monotone_tasks': nonmono,
            'U_by_world': {str(w): sum(1 for t in hard if t[0] == w) / PER_DEPTH for w in WORLDS},
            'hard_tasks': [f'w{w}_d{d}_{i}' for w, i, _ in hard],
        }
    u = {d: per_depth[d]['U'] for d in DEPTHS}
    lab, stats = label(u)

    def need(d):   # smallest budget reaching F >= 0.75, 4000 if none
        return next((k for k in BUDGETS if per_depth[d]['F'][str(k)] >= 0.75), 4000)
    scaling = incumbent_degrades([need(3), need(10)], [3, 10])
    return {'per_depth': {str(d): v for d, v in per_depth.items()}, 'U': {str(d): v for d, v in u.items()},
            'label': lab, 'trend': stats,
            'budget_for_F75': {str(d): need(d) for d in DEPTHS}, 'budget_scaling_vs_depth': scaling}


def verify(records):
    """Independent path: per-task gaps as arrays, HARD = all gaps above GAP."""
    u = {}
    for d in DEPTHS:
        gaps = np.array([[np.log(records[(d, w, i)][f'OPT{k}']['nmse']) - np.log(records[(d, w, i)]['O']['nmse'])
                          for k in BUDGETS] for w in WORLDS for i in range(PER_DEPTH)])
        u[d] = float(np.mean(np.all(gaps > GAP, axis=1)))
    x = np.repeat(np.arange(3, 11, dtype=float), len(WORLDS) * PER_DEPTH)
    y = np.repeat([u[d] for d in range(3, 11)], len(WORLDS) * PER_DEPTH)  # same mean per depth
    slope = np.polyfit(x, y, 1)[0]
    se = math.sqrt(y.mean() * (1 - y.mean()) / np.sum((x - x.mean()) ** 2))
    deep = (u[8] + u[9] + u[10]) / 3
    fire = deep >= 0.25 and 2 * slope >= 0.10 and slope / se >= 1.645
    return ('OPPORTUNITY' if fire else 'NO_OPPORTUNITY'), u


def write(payload, path):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--discrimination-only', action='store_true')
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    disc = discrimination()
    if args.discrimination_only:
        print(json.dumps(disc, indent=2))
        return
    records = load_records()
    if args.verify:
        lab, u = verify(records)
        stored = json.loads(OUTPUT.read_text(encoding='utf-8'))
        agree = lab == stored['census']['label'] and all(
            abs(u[d] - stored['census']['U'][str(d)]) < 1e-12 for d in DEPTHS)
        print(f'verify: {lab}; agrees with report: {agree}')
        raise SystemExit(0 if agree else 1)
    import subprocess
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    result = census(records)
    if not disc['adequate']:
        result['label'] = 'UNDERPOWERED:' + result['label']
    write({'plan': 'SY0_SEARCH_BUDGET_CENSUS_PLAN.md', 'git_commit': commit, 'source_protocol': PROTOCOL,
           'gap': GAP, 'rule': {'U_deep_min': U_MIN, 'rise_over_two_min': U_DROP, 'z_min': Z_MIN, 'deep': list(DEEP)},
           'discrimination': disc, 'census': result}, OUTPUT)
    print(f"label {result['label']} trend {result['trend']}")
    for d in DEPTHS:
        p = result['per_depth'][str(d)]
        print(f"d{d} U {p['U']:.3f} F " + ' '.join(f"{k}:{p['F'][str(k)]:.2f}" for k in BUDGETS)
              + f" med_b* {p['median_b_star']} nonmono {p['non_monotone_tasks']}")


if __name__ == '__main__':
    main()
