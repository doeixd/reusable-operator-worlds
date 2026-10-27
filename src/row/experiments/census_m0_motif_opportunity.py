"""M0 Tier 0 census (`M0_MOTIF_OPPORTUNITY_CENSUS_PLAN.md`): on a HIERARCHICAL (motif) task distribution over the
rotated teacher operators, does deep straight-line search bind for the incumbent while macros mined from solved
shallow tasks make it cheap, and only when the distribution really recurs?

Teacher operators throughout (no learner, no execution drift). Per world:
- MOTIF generator: K_MOTIFS distinct non-constant length-3 primitive sequences. SHALLOW tasks = one motif with 0-2
  random filler primitives split around it (depth 3-5); DEEP tasks = three motifs drawn with replacement (depth 9).
- UNIFORM generator (null): the same counts and depths, every primitive drawn uniformly.
Shallow tasks are solved by exhaustive prefix-shared enumeration at their known length. MINED alphabet: 3-grams at all
offsets of the found shallow routes with count >= MIN_COUNT, most frequent first, at most MAX_ALPHABET. Deep arms
(support MSE <= EXACT_SCORE is success):
- MINED / RANDOM / ORACLE: enumerate every triple over the alphabet (mined; random grams of the same size; the true
  motifs), then CF1's one/two-position primitive polish if not exact. ORACLE is an implementation check.
- INCUMBENT: CF1's relaxed descent + polish (`census_cf1_relaxed_structure`), straight program of length 9.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from row.experiments import census_cf0_loop_structure as cf0

WORLDS = (80, 81, 82)
SMOKE_WORLD = 79
K_MOTIFS, MOTIF_LEN, DEEP_MOTIFS = 6, 3, 3
N_SHALLOW, N_DEEP = 48, 12
MAX_FILLER = 2
SUPPORT, N_QUERY = 128, 256
MIN_COUNT, MAX_ALPHABET = 3, 12
EXACT_SCORE = 1e-12
MINED_PASS, RANDOM_MAX, NULL_MAX = 0.9, 0.2, 0.2
INCUMBENT_BINDS, INCUMBENT_FREE = 0.5, 0.8
GENERATORS = ('MOTIF', 'UNIFORM')
OUTPUT = Path('reports/m0_motif_opportunity_census.json')
RUN_DIR = Path('reports/m0_motif_opportunity_20261027')


def draw_motifs(rng, n_prims):
    motifs = []
    while len(motifs) < K_MOTIFS:
        m = tuple(int(v) for v in rng.integers(0, n_prims, MOTIF_LEN))
        if len(set(m)) > 1 and m not in motifs:
            motifs.append(m)
    return motifs


def make_programs(seed, generator, n_prims):
    """Shallow and deep programs of one (world, generator); the UNIFORM null matches every depth of the MOTIF draw."""
    rng = np.random.default_rng(np.random.SeedSequence([6300, seed]))
    motifs = draw_motifs(rng, n_prims)
    shallow, deep = [], []
    for _ in range(N_SHALLOW):
        m = motifs[int(rng.integers(0, K_MOTIFS))]
        f = int(rng.integers(0, MAX_FILLER + 1))
        left = int(rng.integers(0, f + 1))
        fill = [int(v) for v in rng.integers(0, n_prims, f)]
        shallow.append(tuple(fill[:left]) + m + tuple(fill[left:]))
    for _ in range(N_DEEP):
        deep.append(sum((motifs[int(k)] for k in rng.integers(0, K_MOTIFS, DEEP_MOTIFS)), ()))
    if generator == 'UNIFORM':
        u = np.random.default_rng(np.random.SeedSequence([6301, seed]))
        shallow = [tuple(int(v) for v in u.integers(0, n_prims, len(p))) for p in shallow]
        deep = [tuple(int(v) for v in u.integers(0, n_prims, len(p))) for p in deep]
    return motifs, shallow, deep


def task_data(w_obj, seed, generator, kind, i, prog):
    d = w_obj.config.state_dim
    rng = np.random.default_rng(np.random.SeedSequence([6302, seed, GENERATORS.index(generator), kind, i]))
    xs, xq = rng.normal(size=(SUPPORT, d)), rng.normal(size=(N_QUERY, d))
    lib = w_obj.library
    return xs, cf0.apply_seq(lib, prog, xs.copy()), xq, cf0.apply_seq(lib, prog, xq.copy())


def mse(lib, seq, x, y):
    return float(np.mean((cf0.apply_seq(lib, seq, x.copy()) - y) ** 2))


def enum_prefix_shared(lib, x, y, depth):
    best = [np.inf, None]

    def rec(z, seq):
        if len(seq) == depth:
            v = float(np.mean((z - y) ** 2))
            if v < best[0]:
                best[0], best[1] = v, tuple(seq)
            return
        for p in range(len(lib)):
            rec(lib[p](z), seq + [p])
    rec(np.array(x, dtype=np.float64), [])
    return best[1], best[0]


def mine(routes):
    counts = Counter(tuple(r[j:j + MOTIF_LEN]) for r in routes for j in range(len(r) - MOTIF_LEN + 1))
    ranked = sorted((g for g, c in counts.items() if c >= MIN_COUNT), key=lambda g: (-counts[g], g))
    return ranked[:MAX_ALPHABET], {''.join(map(str, g)): c for g, c in counts.most_common(20)}


def random_alphabet(seed, generator, size, n_prims):
    rng = np.random.default_rng(np.random.SeedSequence([6303, seed, GENERATORS.index(generator)]))
    out = []
    while len(out) < size:
        g = tuple(int(v) for v in rng.integers(0, n_prims, MOTIF_LEN))
        if len(set(g)) > 1 and g not in out:
            out.append(g)
    return out


def macro_search(lib, alphabet, x, y):
    """Enumerate every DEEP_MOTIFS-tuple over the alphabet with prefix sharing, then CF1's polish if not exact."""
    from row.experiments import census_cf1_relaxed_structure as cf1
    t0 = time.process_time()
    best, evals = (np.inf, None), 0
    if alphabet:
        def rec(z, seq, k):
            nonlocal best, evals
            if k == DEEP_MOTIFS:
                evals += 1
                v = float(np.mean((z - y) ** 2))
                if v < best[0]:
                    best = (v, seq)
                return
            for g in alphabet:
                rec(cf0.apply_seq(lib, g, z.copy()), seq + g, k + 1)
        rec(np.array(x, dtype=np.float64), (), 0)
        seq, score = list(best[1]), best[0]
    else:
        seq, score = [0] * (MOTIF_LEN * DEEP_MOTIFS), None
    p_evals = 0
    if score is None or score > EXACT_SCORE:
        n = len(seq)
        seq, score, p_evals = cf1.polish(lib, seq, (n - 1) // 2, n - 1 - (n - 1) // 2, x, y, False)
    return {'found': [int(v) for v in seq], 'score': float(score), 'success': score <= EXACT_SCORE,
            'enum_evaluations': evals, 'polish_evaluations': p_evals, 'seconds': time.process_time() - t0}


def incumbent(lib, prims, x, y, seed):
    from row.experiments import census_cf1_relaxed_structure as cf1
    n = MOTIF_LEN * DEEP_MOTIFS
    a, b = (n - 1) // 2, n - 1 - (n - 1) // 2
    t0 = time.process_time()
    d_score, d_seq = cf1.descend(lib, prims, x, y, a, b, False, seed)
    p_seq, p_score, evals = cf1.polish(lib, d_seq, a, b, x, y, False)
    return {'found': [int(v) for v in p_seq], 'descent_score': float(d_score), 'score': float(p_score),
            'success': p_score <= EXACT_SCORE, 'polish_evaluations': evals, 'seconds': time.process_time() - t0}


def aliases(lib, truth, x, y):
    """One-position neighbours of the true program that also explain the support exactly."""
    out = 0
    for pos in range(len(truth)):
        for v in range(len(lib)):
            if v != truth[pos]:
                cand = list(truth)
                cand[pos] = v
                out += mse(lib, cand, x, y) <= EXACT_SCORE
    return out


def run_cell(seed, generator, smoke=False):
    import torch
    from row.experiments import census_cf1_relaxed_structure as cf1
    torch.set_num_threads(1)
    w_obj = cf0.world(seed)
    lib = w_obj.library
    prims = cf1.torch_prims(lib)
    probe = np.random.default_rng(0).normal(size=(4, w_obj.config.state_dim))
    for k, p in enumerate(lib):   # gate: torch primitives reproduce the teacher
        err = float(np.max(np.abs(prims[k](torch.tensor(probe)).numpy() - p(probe))))
        if err > 1e-10:
            raise RuntimeError(f'torch primitive {k} of world {seed} differs from the teacher by {err}')
    motifs, shallow, deep = make_programs(seed, generator, len(lib))
    n_shallow = 6 if smoke else N_SHALLOW
    n_deep = 2 if smoke else N_DEEP
    t0 = time.process_time()
    shallow_rows = []
    for i, prog in enumerate(shallow[:n_shallow]):
        xs, ys, _, _ = task_data(w_obj, seed, generator, 0, i, prog)
        found, score = enum_prefix_shared(lib, xs, ys, len(prog))
        shallow_rows.append({'truth': list(prog), 'found': list(found), 'score': score,
                             'exact': tuple(found) == prog, 'solved': score <= EXACT_SCORE})
    shallow_seconds = time.process_time() - t0
    t0 = time.process_time()
    alphabet, top = mine([tuple(r['found']) for r in shallow_rows if r['solved']])
    mine_seconds = time.process_time() - t0
    rand = random_alphabet(seed, generator, len(alphabet), len(lib))
    deep_rows = []
    for i, prog in enumerate(deep[:n_deep]):
        xs, ys, xq, yq = task_data(w_obj, seed, generator, 1, i, prog)
        row = {'truth': list(prog), 'aliases_1pos': aliases(lib, prog, xs, ys)}
        arms = {'MINED': alphabet, 'RANDOM': rand}
        if generator == 'MOTIF':
            arms['ORACLE'] = motifs
        for arm, alpha in arms.items():
            row[arm] = macro_search(lib, alpha, xs, ys)
        if generator == 'MOTIF':
            row['INCUMBENT'] = incumbent(lib, prims, xs, ys, seed * 1000 + i)
        var = float(np.mean((yq - yq.mean(0)) ** 2))
        for arm in [a for a in ('MINED', 'RANDOM', 'ORACLE', 'INCUMBENT') if a in row]:
            row[arm]['exact'] = tuple(row[arm]['found']) == prog
            row[arm]['query_nmse'] = mse(lib, row[arm]['found'], xq, yq) / var
        deep_rows.append(row)
    return {'world': seed, 'generator': generator, 'motifs': [list(m) for m in motifs],
            'alphabet': [list(g) for g in alphabet], 'random_alphabet': [list(g) for g in rand],
            'motifs_recovered': sum(m in alphabet for m in motifs) if generator == 'MOTIF' else None,
            'top_grams': top, 'shallow': shallow_rows, 'shallow_seconds': shallow_seconds,
            'mine_seconds': mine_seconds, 'deep': deep_rows, 'complete': True}


def rate(cells, generator, arm):
    rows = [r for c in cells if c['generator'] == generator for r in c['deep']]
    return sum(r[arm]['success'] for r in rows) / len(rows), len(rows)


def label(cells):
    n_expected = len(WORLDS) * N_DEEP
    rates = {f'{g}_{a}': rate(cells, g, a) for g, a in (('MOTIF', 'MINED'), ('MOTIF', 'RANDOM'), ('MOTIF', 'ORACLE'),
                                                       ('MOTIF', 'INCUMBENT'), ('UNIFORM', 'MINED'),
                                                       ('UNIFORM', 'RANDOM'))}
    if any(n != n_expected for _, n in rates.values()):
        raise SystemExit(f'incomplete census: {rates}')
    r = {k: v for k, (v, _) in rates.items()}
    if r['MOTIF_ORACLE'] < 1.0:
        lab = 'INVALID_ORACLE'
    elif r['MOTIF_MINED'] < MINED_PASS or r['MOTIF_RANDOM'] > RANDOM_MAX or r['UNIFORM_MINED'] > NULL_MAX:
        lab = 'CONSTRUCT_FAILS'
    elif r['MOTIF_INCUMBENT'] > INCUMBENT_FREE:
        lab = 'NO_SEARCH_NECESSITY'
    elif r['MOTIF_INCUMBENT'] <= INCUMBENT_BINDS:
        lab = 'MOTIF_OPPORTUNITY'
    else:
        lab = 'GRADED'
    return lab, r


def discrimination(draws=100000, seed=2710):
    rng = np.random.default_rng(seed)
    n = len(WORLDS) * N_DEEP

    def p(q, op, bar):
        k = rng.binomial(n, q, draws) / n
        return float(np.mean(k >= bar if op == 'ge' else (k > bar if op == 'gt' else k <= bar)))
    return {'n_per_rate': n,
            'mined_pass_ge_0.9': {str(q): p(q, 'ge', MINED_PASS) for q in (0.99, 0.97, 0.95, 0.85, 0.8, 0.7)},
            'control_fail_gt_0.2': {str(q): p(q, 'gt', RANDOM_MAX) for q in (0.02, 0.05, 0.1, 0.3, 0.4)},
            'incumbent_binds_le_0.5': {str(q): p(q, 'le', INCUMBENT_BINDS) for q in (0.2, 0.3, 0.4, 0.6, 0.7)},
            'incumbent_free_gt_0.8': {str(q): p(q, 'gt', INCUMBENT_FREE) for q in (0.7, 0.75, 0.9, 0.95)}}


def write_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1), encoding='utf-8')
    os.replace(tmp, path)


def log(msg):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    with open(RUN_DIR / 'run.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--discrimination-only', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--jobs', type=int, default=3)
    args = ap.parse_args()
    disc = discrimination()
    if args.discrimination_only:
        print(json.dumps(disc, indent=1))
        return
    if args.smoke:
        for g in GENERATORS:
            c = run_cell(SMOKE_WORLD, g, smoke=True)
            print(json.dumps({k: v for k, v in c.items() if k not in ('shallow',)}, indent=1))
        return
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise SystemExit('refusing to run from a dirty tree')
    keys = [(s, g) for g in GENERATORS for s in WORLDS]
    cells, todo = {}, []
    for s, g in keys:
        path = RUN_DIR / 'cells' / f'{g}_w{s}.json'
        if path.exists():
            c = json.loads(path.read_text(encoding='utf-8'))
            if c.get('complete') and c.get('git_commit') == commit:
                cells[(s, g)] = c
                continue
            if c.get('git_commit') != commit:
                raise SystemExit(f'{path} was written at {c.get("git_commit")}, not {commit}: refusing to mix')
        todo.append((s, g))
    log(f'launch commit {commit} cells {len(keys)} done {len(cells)} todo {len(todo)} pid {os.getpid()}')
    status = {'git_commit': commit, 'pid': os.getpid(), 'start': time.time(), 'total': len(keys)}

    def update(running):
        write_atomic(RUN_DIR / 'status.json', {**status, 'done': len(cells), 'running': running,
                                               'last_update': time.time()})
    update([f'{g}_w{s}' for s, g in todo])
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_cell, s, g): (s, g) for s, g in todo}
        for fut in as_completed(futs):
            s, g = futs[fut]
            c = fut.result()
            c['git_commit'] = commit
            write_atomic(RUN_DIR / 'cells' / f'{g}_w{s}.json', c)
            cells[(s, g)] = c
            arms = [a for a in ('MINED', 'RANDOM', 'ORACLE', 'INCUMBENT') if a in c['deep'][0]]
            log(f'cell {g}_w{s} done alphabet {len(c["alphabet"])} recovered {c["motifs_recovered"]} ' +
                ' '.join(f'{a} {sum(r[a]["success"] for r in c["deep"])}/{len(c["deep"])}' for a in arms))
            update([f'{gg}_w{ss}' for ss, gg in todo if (ss, gg) not in cells])
    lab, rates = label(list(cells.values()))
    payload = {'plan': 'M0_MOTIF_OPPORTUNITY_CENSUS_PLAN.md', 'git_commit': commit, 'label': lab, 'rates': rates,
               'discrimination': disc, 'cells': [cells[k] for k in keys]}
    write_atomic(OUTPUT, payload)
    log(f'exit 0 label {lab} rates {rates}')
    print(lab, json.dumps(rates, indent=1))


if __name__ == '__main__':
    main()
