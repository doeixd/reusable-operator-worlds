"""CF1 Tier 0 census (`CF1_RELAXED_STRUCTURE_CENSUS_PLAN.md`): with the FAIR incumbent (SY1's: a relaxed gradient
descent, then a discrete polish), is finding the structure of a composite loop program hard, and harder than finding
a straight program of the same length?

Same tasks as CF0 (`census_cf0_loop_structure.make_task`, worlds 76-78, teacher operators). Two arms per task:
- LOOP: target `B(loop_{w,K}(P, A(x)))`; relaxation = softmax operator choice per position, soft ABSORBING halting
  (alive_{k+1} = alive_k * sigmoid(BETA * w.z_k), z_{k+1} = z_k + alive_{k+1} (P(z_k) - z_k)); after descent, the
  argmax skeleton is polished with CF0's count-inference score (one-position moves, then two-position moves).
- STRAIGHT: target `B(P(A(x)))` from the same draws; relaxation without the loop; polished with plain MSE.
RESTARTS random initializations of STEPS // RESTARTS steps each (matched total steps); best hard skeleton kept.
Structure success: support score <= EXACT_SCORE after polish.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from row.experiments import census_cf0_loop_structure as cf0

STEPS, RESTARTS, LR, BETA = 2100, 3, 0.05, 10.0
POLISH_CAP = 30000
EXACT_SCORE = cf0.EXACT_SCORE
SIZES = cf0.SIZES
SUPPORT = 128
PASS_RATE, FAIL_RATE, LOOP_GAP = 0.9, 0.6, 0.3
SECONDS_OK, SECONDS_BIND = 60.0, 600.0
OUTPUT = Path('reports/cf1_relaxed_structure_census.json')


def torch_prims(lib):
    out = []
    for p in lib:
        U, V, b, Q = (torch.tensor(np.asarray(t), dtype=torch.float64) for t in (p.U, p.V, p.b, p.Q))
        out.append(lambda z, U=U, V=V, b=b, Q=Q, a=p.alpha: (z + a * torch.tanh(z @ V.T + b) @ U.T) @ Q.T)
    return out


def soft_apply(prims, coeffs, z):
    return sum(c * f(z) for c, f in zip(coeffs, prims))


def relaxed_forward(prims, logits, x, a, b, loop, w):
    z = x
    for i in range(a):
        z = soft_apply(prims, torch.softmax(logits[i], 0), z)
    cP = torch.softmax(logits[a], 0)
    if loop:
        alive = torch.ones(len(z), 1, dtype=z.dtype)
        for _ in range(cf0.K):
            alive = alive * torch.sigmoid(BETA * (z @ w)).unsqueeze(1)
            z = z + alive * (soft_apply(prims, cP, z) - z)
    else:
        z = soft_apply(prims, cP, z)
    for i in range(a + 1, a + 1 + b):
        z = soft_apply(prims, torch.softmax(logits[i], 0), z)
    return z


def hard_score(lib, seq, a, b, x, y, loop, cache):
    if loop:
        return cf0.score(lib, cf0.split(seq, a, b), x, y, cache)[0]
    return float(np.mean((cf0.apply_seq(lib, seq, x.copy()) - y) ** 2))


def descend(lib, prims, x, y, a, b, loop, seed):
    n = a + 1 + b
    xt, yt = torch.tensor(x), torch.tensor(y)
    best = (math.inf, None)
    for r in range(RESTARTS):
        g = torch.Generator().manual_seed(int(seed * 10 + r))
        logits = (0.1 * torch.randn(n, len(lib), generator=g, dtype=torch.float64)).requires_grad_(True)
        w = (torch.randn(x.shape[1], generator=g, dtype=torch.float64) / math.sqrt(x.shape[1])).requires_grad_(True)
        params = [logits, w] if loop else [logits]
        opt = torch.optim.Adam(params, lr=LR)
        for _ in range(STEPS // RESTARTS):
            opt.zero_grad()
            loss = torch.mean((relaxed_forward(prims, logits, xt, a, b, loop, w) - yt) ** 2)
            loss.backward()
            opt.step()
        seq = [int(v) for v in torch.argmax(logits.detach(), dim=1)]
        s = hard_score(lib, seq, a, b, x, y, loop, {})
        if s < best[0]:
            best = (s, seq)
    return best


def polish(lib, seq, a, b, x, y, loop):
    cache = {}
    cur, evals = hard_score(lib, seq, a, b, x, y, loop, cache), 0
    seq = list(seq)
    for width in (1, 2):
        improved = True
        while improved and cur > EXACT_SCORE and evals < POLISH_CAP:
            improved = False
            for pos in itertools.combinations(range(len(seq)), width):
                for vals in itertools.product(range(len(lib)), repeat=width):
                    if all(seq[p] == v for p, v in zip(pos, vals)) or evals >= POLISH_CAP:
                        continue
                    cand = list(seq)
                    for p, v in zip(pos, vals):
                        cand[p] = v
                    s = hard_score(lib, cand, a, b, x, y, loop, cache)
                    evals += 1
                    if s < cur:
                        seq, cur, improved = cand, s, True
            if improved and width == 2:
                break   # return to one-position moves after a two-position improvement
        if cur <= EXACT_SCORE:
            break
    return seq, cur, evals


def run_task(w_obj, prims, seed, a, b, i):
    lib = w_obj.library
    t = cf0.make_task(w_obj, seed, a, b, i, SUPPORT)
    truth = list(t['A']) + [t['P']] + list(t['B'])
    out = {'world': seed, 'a': a, 'b': b, 'i': i, 'truth': truth}
    for arm, loop in (('LOOP', True), ('STRAIGHT', False)):
        y = t['ys'] if loop else cf0.apply_seq(lib, truth, t['xs'].copy())
        t0 = time.process_time()
        d_score, d_seq = descend(lib, prims, t['xs'], y, a, b, loop, seed * 1000 + a * 100 + b * 10 + i)
        p_seq, p_score, evals = polish(lib, d_seq, a, b, t['xs'], y, loop)
        out[arm] = {'descent_score': d_score, 'descent_exact': d_seq == truth,
                    'descent_success': d_score <= EXACT_SCORE, 'polish_score': p_score,
                    'success': p_score <= EXACT_SCORE, 'exact': p_seq == truth, 'found': p_seq,
                    'polish_evaluations': evals, 'seconds': time.process_time() - t0}
    return out


def label(rows):
    per = {}
    for a, b in SIZES:
        rs = [r for r in rows if (r['a'], r['b']) == (a, b)]
        per[f'{a}_{b}'] = {arm: {'n': len(rs), 'success_rate': sum(r[arm]['success'] for r in rs) / len(rs),
                                 'descent_success_rate': sum(r[arm]['descent_success'] for r in rs) / len(rs),
                                 'median_seconds': float(np.median([r[arm]['seconds'] for r in rs])),
                                 'median_polish_evaluations': float(np.median([r[arm]['polish_evaluations']
                                                                               for r in rs]))}
                           for arm in ('LOOP', 'STRAIGHT')}
    loop_rates = [v['LOOP']['success_rate'] for v in per.values()]
    loop_secs = [v['LOOP']['median_seconds'] for v in per.values()]
    last = per[f'{SIZES[-1][0]}_{SIZES[-1][1]}']
    if all(r >= PASS_RATE for r in loop_rates) and all(s <= SECONDS_OK for s in loop_secs):
        lab = 'NO_SEARCH_NECESSITY'
    elif last['LOOP']['success_rate'] < FAIL_RATE or max(loop_secs) > SECONDS_BIND:
        lab = 'SEARCH_BINDS'
    else:
        lab = 'GRADED'
    specific = last['STRAIGHT']['success_rate'] - last['LOOP']['success_rate'] >= LOOP_GAP
    return lab, ('LOOP_SPECIFIC' if specific else 'NOT_LOOP_SPECIFIC'), per


def discrimination(draws=100000, seed=2612):
    rng = np.random.default_rng(seed)
    n = len(cf0.WORLDS) * cf0.PER_CELL
    def gap(p_loop, p_straight):
        lo = rng.binomial(n, p_loop, draws) / n
        st = rng.binomial(n, p_straight, draws) / n
        return float(np.mean(st - lo >= LOOP_GAP))
    return {'loop_specific_fire': {'equal_0.9': gap(0.9, 0.9), 'equal_0.5': gap(0.5, 0.5),
                                   'gap_0.1': gap(0.5, 0.6), 'gap_0.5': gap(0.4, 0.9), 'gap_0.6': gap(0.3, 0.9)},
            'cf0_size_clauses': cf0.discrimination()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--discrimination-only', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(1)
    disc = discrimination()
    if args.discrimination_only:
        print(json.dumps(disc, indent=1))
        return
    worlds = {s: cf0.world(s) for s in cf0.WORLDS}
    prims = {s: torch_prims(worlds[s].library) for s in cf0.WORLDS}
    x = np.random.default_rng(0).normal(size=(4, 16))
    for s in cf0.WORLDS:   # gate: torch primitives reproduce the teacher
        for k, p in enumerate(worlds[s].library):
            err = np.max(np.abs(prims[s][k](torch.tensor(x)).numpy() - p(x)))
            if err > 1e-10:
                raise SystemExit(f'torch primitive {k} of world {s} differs from the teacher by {err}')
    rows = []
    sizes = ((1, 1), (3, 3)) if args.smoke else SIZES
    for (a, b) in sizes:
        for s in cf0.WORLDS:
            for i in range(1 if args.smoke else cf0.PER_CELL):
                r = run_task(worlds[s], prims[s], s, a, b, i)
                rows.append(r)
                print(f"a{a}b{b} w{s} i{i} LOOP {'OK ' if r['LOOP']['success'] else 'BAD'} "
                      f"(descent {'ok' if r['LOOP']['descent_success'] else '--'}, ev {r['LOOP']['polish_evaluations']}, "
                      f"{r['LOOP']['seconds']:.1f}s) STRAIGHT {'OK ' if r['STRAIGHT']['success'] else 'BAD'} "
                      f"(descent {'ok' if r['STRAIGHT']['descent_success'] else '--'}, "
                      f"ev {r['STRAIGHT']['polish_evaluations']}, {r['STRAIGHT']['seconds']:.1f}s)", flush=True)
    if args.smoke:
        return
    lab, spec, per = label(rows)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    payload = {'plan': 'CF1_RELAXED_STRUCTURE_CENSUS_PLAN.md', 'git_commit': commit, 'label': lab,
               'loop_specificity': spec, 'per_size': per, 'discrimination': disc, 'rows': rows}
    tmp = OUTPUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    os.replace(tmp, OUTPUT)
    print(lab, spec, json.dumps(per, indent=1))


if __name__ == '__main__':
    main()
