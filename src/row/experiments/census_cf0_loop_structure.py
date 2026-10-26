"""CF0 Tier 0 census (`CF0_LOOP_STRUCTURE_CENSUS_PLAN.md`): is finding the STRUCTURE of a composite loop program
hard for a cheap incumbent search, with the teacher's exact operators?

Tasks `z = A(x); repeat at most K times: stop if w.z <= 0, else z = P(z); y = B(z)` on rotated worlds 76-78, A and
B random teacher-primitive sequences of length a and b, P one primitive, w a random unit hyperplane. The searcher
never sees A, P, B, w or the per-example iteration count.

Skeleton score (A, P, B): run P up to K times on A(x) for every example, apply B after each count, take each
example's best count; score = mean over examples of that minimum error. Search = coordinate descent over the
a + 1 + b positions from random starts (restarts until the score is ~0 or EVAL_BUDGET skeleton scores are spent);
exhaustive enumeration is also run where the skeleton space is <= ENUM_MAX, as a reference. Then w is fitted by
logistic regression on the states the inferred counts label (continue before the count, stop at it), and the query is
executed with the fitted w. STRUCTURE success: the found skeleton explains every support example exactly (support
score <= EXACT_SCORE). Query NMSE with the fitted predicate is recorded separately (predicate estimation is a
sample-size question B0b already measured, not a search question).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.rotated_world import generate_rotated_world

WORLDS = (76, 77, 78)
SIZES = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4))
PER_CELL = 8
K = 6
SUPPORTS = (128, 16)
N_QUERY = 256
EVAL_BUDGET = 3000
ENUM_MAX = 7776
EXACT_SCORE = 1e-12
PASS_RATE, FAIL_RATE = 0.9, 0.6
SECONDS_OK, SECONDS_BIND = 60.0, 600.0
OUTPUT = Path('reports/cf0_loop_structure_census.json')


def world(seed):
    cfg = load_config('configs/v1.yaml')
    return generate_rotated_world(replace(cfg.world, seed=seed))


def apply_seq(lib, seq, z):
    for p in seq:
        z = lib[p](z)
    return z


def teacher(lib, A, P, B, w, x):
    z = apply_seq(lib, A, np.array(x, dtype=np.float64))
    for _ in range(K):
        go = z @ w > 0
        if not go.any():
            break
        z[go] = lib[P](z[go])
    return apply_seq(lib, B, z)


def make_task(w_obj, seed, a, b, i, n_support):
    lib = w_obj.library
    d = w_obj.config.state_dim
    rng = np.random.default_rng(np.random.SeedSequence([6100, seed, a, b, i]))
    A = tuple(int(v) for v in rng.integers(0, len(lib), a))
    P = int(rng.integers(0, len(lib)))
    B = tuple(int(v) for v in rng.integers(0, len(lib), b))
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    xs, xq = rng.normal(size=(max(SUPPORTS), d)), rng.normal(size=(N_QUERY, d))
    xs = xs[:n_support]
    return {'A': A, 'P': P, 'B': B, 'w': w, 'xs': xs, 'ys': teacher(lib, A, P, B, w, xs),
            'xq': xq, 'yq': teacher(lib, A, P, B, w, xq)}


def iterates(lib, P, z0):
    zs = [z0]
    for _ in range(K):
        zs.append(lib[P](zs[-1]))
    return zs


def score(lib, skel, x, y, cache):
    A, P, B = skel
    key = (A, P)
    if key not in cache:
        cache[key] = iterates(lib, P, apply_seq(lib, A, np.array(x, dtype=np.float64)))
    errs = np.stack([np.mean((apply_seq(lib, B, z) - y) ** 2, axis=1) for z in cache[key]], axis=1)
    return float(np.mean(errs.min(axis=1))), errs.argmin(axis=1)


def split(seq, a, b):
    return tuple(seq[:a]), int(seq[a]), tuple(seq[a + 1:])


def local_search(lib, x, y, a, b, rng, n_prims):
    n = a + 1 + b
    cache, evals, best = {}, 0, (math.inf, None)
    while evals < EVAL_BUDGET and best[0] > 1e-12:
        seq = [int(v) for v in rng.integers(0, n_prims, n)]
        cur, _ = score(lib, split(seq, a, b), x, y, cache)
        evals += 1
        improved = True
        while improved and evals < EVAL_BUDGET and cur > 1e-12:
            improved = False
            for pos in range(n):
                for s in range(n_prims):
                    if s == seq[pos] or evals >= EVAL_BUDGET:
                        continue
                    cand = list(seq)
                    cand[pos] = s
                    v, _ = score(lib, split(cand, a, b), x, y, cache)
                    evals += 1
                    if v < cur:
                        seq, cur, improved = cand, v, True
        if cur < best[0]:
            best = (cur, list(seq))
    return split(best[1], a, b), best[0], evals


def enumerate_all(lib, x, y, a, b, n_prims):
    cache, best = {}, (math.inf, None)
    for seq in itertools.product(range(n_prims), repeat=a + 1 + b):
        v, _ = score(lib, split(seq, a, b), x, y, cache)
        if v < best[0]:
            best = (v, seq)
    return split(best[1], a, b), best[0], n_prims ** (a + 1 + b)


def fit_predicate(lib, skel, x, y, seed):
    A, P, B = skel
    zs = iterates(lib, P, apply_seq(lib, A, np.array(x, dtype=np.float64)))
    _, counts = score(lib, skel, x, y, {})
    states, labels = [], []
    for n, k in enumerate(counts):
        for i in range(k):
            states.append(zs[i][n]); labels.append(1.0)
        if k < K:
            states.append(zs[k][n]); labels.append(0.0)
    S = torch.tensor(np.array(states), dtype=torch.float32)
    L = torch.tensor(labels, dtype=torch.float32)
    torch.manual_seed(seed)
    w = torch.zeros(S.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(500):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(S @ w * 10.0, L)
        loss.backward()
        opt.step()
    v = w.detach().numpy().astype(np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def nmse(pred, y):
    return float(np.mean((pred - y) ** 2) / (np.var(y) + 1e-12))


def run_task(w_obj, seed, a, b, i, n_support):
    lib = w_obj.library
    t = make_task(w_obj, seed, a, b, i, n_support)
    rng = np.random.default_rng(np.random.SeedSequence([6101, seed, a, b, i, n_support]))
    t0 = time.process_time()
    skel, sc, evals = local_search(lib, t['xs'], t['ys'], a, b, rng, len(lib))
    w_hat = fit_predicate(lib, skel, t['xs'], t['ys'], seed)
    secs = time.process_time() - t0
    q = nmse(teacher(lib, *skel, w_hat, t['xq']), t['yq'])
    truth = (t['A'], t['P'], t['B'])
    q_true_skel = nmse(teacher(lib, *truth, fit_predicate(lib, truth, t['xs'], t['ys'], seed), t['xq']), t['yq'])
    row = {'world': seed, 'a': a, 'b': b, 'i': i, 'support': n_support, 'truth': [list(t['A']), t['P'], list(t['B'])],
           'found': [list(skel[0]), skel[1], list(skel[2])], 'exact': skel == truth, 'support_score': sc,
           'evaluations': evals, 'seconds': secs, 'query_nmse': q, 'success': sc <= EXACT_SCORE,
           'query_nmse_true_skeleton': q_true_skel,
           'stop_at_zero_fraction': float(np.mean(apply_seq(lib, t['A'], t['xs'].copy()) @ t['w'] <= 0))}
    if len(lib) ** (a + 1 + b) <= ENUM_MAX:
        t1 = time.process_time()
        eskel, esc, _ = enumerate_all(lib, t['xs'], t['ys'], a, b, len(lib))
        row['enum'] = {'exact': eskel == truth, 'support_score': esc, 'seconds': time.process_time() - t1,
                       'query_nmse': nmse(teacher(lib, *eskel, fit_predicate(lib, eskel, t['xs'], t['ys'], seed),
                                                  t['xq']), t['yq'])}
    return row


def label(rows):
    per = {}
    for a, b in SIZES:
        rs = [r for r in rows if (r['a'], r['b']) == (a, b) and r['support'] == max(SUPPORTS)]
        rate = sum(r['success'] for r in rs) / len(rs)
        med = float(np.median([r['seconds'] for r in rs]))
        q16 = [r['query_nmse'] for r in rows if (r['a'], r['b']) == (a, b) and r['support'] == min(SUPPORTS)]
        per[f'{a}_{b}'] = {'n': len(rs), 'success_rate': rate, 'median_seconds': med,
                           'median_query_nmse_128': float(np.median([r['query_nmse'] for r in rs])),
                           'median_query_nmse_16': float(np.median(q16)),
                           'success_rate_16': sum(r['success'] for r in rows if (r['a'], r['b']) == (a, b)
                                                  and r['support'] == min(SUPPORTS)) / len(q16),
                           'exact_rate': sum(r['exact'] for r in rs) / len(rs),
                           'median_evaluations': float(np.median([r['evaluations'] for r in rs]))}
    rates = [v['success_rate'] for v in per.values()]
    secs = [v['median_seconds'] for v in per.values()]
    if all(r >= PASS_RATE for r in rates) and all(s <= SECONDS_OK for s in secs):
        lab = 'NO_SEARCH_NECESSITY'
    elif per[f'{SIZES[-1][0]}_{SIZES[-1][1]}']['success_rate'] < FAIL_RATE or max(secs) > SECONDS_BIND:
        lab = 'SEARCH_BINDS'
    else:
        lab = 'GRADED'
    return lab, per


def discrimination(draws=100000, seed=2611):
    """Per-size pass clause (>= 0.9 of 24) and fail clause (< 0.6 of 24) under binomial sampling."""
    rng = np.random.default_rng(seed)
    n = len(WORLDS) * PER_CELL
    def p_pass(p):
        return float(np.mean(np.all(rng.binomial(n, p, size=(draws, len(SIZES))) / n >= PASS_RATE, axis=1)))
    def p_fail_last(p):
        return float(np.mean(rng.binomial(n, p, size=draws) / n < FAIL_RATE))
    return {'pass_all_sizes_at_true_rate': {str(p): p_pass(p) for p in (0.99, 0.97, 0.8, 0.7)},
            'fail_clause_at_true_rate': {str(p): p_fail_last(p) for p in (0.9, 0.8, 0.5, 0.4)}}


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
    rows = []
    worlds = {s: world(s) for s in WORLDS}
    sizes = SIZES[:2] if args.smoke else SIZES
    for (a, b) in sizes:
        for n_support in SUPPORTS:
            for s in WORLDS:
                for i in range(1 if args.smoke else PER_CELL):
                    r = run_task(worlds[s], s, a, b, i, n_support)
                    rows.append(r)
                    print(f"a{a}b{b} n{n_support} w{s} i{i} {'OK ' if r['success'] else 'BAD'} exact={r['exact']} "
                          f"q={r['query_nmse']:.4f} sc={r['support_score']:.2e} ev={r['evaluations']} "
                          f"{r['seconds']:.1f}s", flush=True)
    if args.smoke:
        return
    lab, per = label(rows)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    payload = {'plan': 'CF0_LOOP_STRUCTURE_CENSUS_PLAN.md', 'git_commit': commit, 'label': lab, 'per_size': per,
               'discrimination': disc, 'rows': rows}
    tmp = OUTPUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    os.replace(tmp, OUTPUT)
    print(lab, json.dumps(per, indent=1))


if __name__ == '__main__':
    main()
