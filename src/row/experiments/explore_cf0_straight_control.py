"""EXPLORATORY, post hoc (no verdict): CF0's straight-line control. The same random-start coordinate-descent searcher,
budget and tasks' (A, P, B) draws as CF0, but the target is the straight program A, P, B (P applied exactly once, no
loop) and the score has no count inference. If this also fails at 7-9 positions, CF0's SEARCH_BINDS is a property of
the searcher, not of loop structure.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from row.experiments import census_cf0_loop_structure as cf0

OUTPUT = Path('reports/cf0_straight_control.json')
SIZES = ((2, 2), (3, 3), (4, 4))


def run(seed, w_obj, a, b, i):
    lib = w_obj.library
    t = cf0.make_task(w_obj, seed, a, b, i, 128)
    truth = list(t['A']) + [t['P']] + list(t['B'])
    ys = cf0.apply_seq(lib, truth, t['xs'].copy())
    rng = np.random.default_rng(np.random.SeedSequence([6102, seed, a, b, i]))
    n, evals, best = len(truth), 0, (math.inf, None)

    def sc(seq):
        return float(np.mean((cf0.apply_seq(lib, seq, t['xs'].copy()) - ys) ** 2))
    t0 = time.process_time()
    while evals < cf0.EVAL_BUDGET and best[0] > 1e-12:
        seq = [int(v) for v in rng.integers(0, len(lib), n)]
        cur = sc(seq); evals += 1
        improved = True
        while improved and evals < cf0.EVAL_BUDGET and cur > 1e-12:
            improved = False
            for pos in range(n):
                for s in range(len(lib)):
                    if s == seq[pos] or evals >= cf0.EVAL_BUDGET:
                        continue
                    cand = list(seq); cand[pos] = s
                    v = sc(cand); evals += 1
                    if v < cur:
                        seq, cur, improved = cand, v, True
        if cur < best[0]:
            best = (cur, seq)
    return {'world': seed, 'a': a, 'b': b, 'i': i, 'success': best[0] <= cf0.EXACT_SCORE,
            'exact': best[1] == truth, 'evaluations': evals, 'seconds': time.process_time() - t0}


def main():
    worlds = {s: cf0.world(s) for s in cf0.WORLDS}
    rows = [run(s, worlds[s], a, b, i) for (a, b) in SIZES for s in cf0.WORLDS for i in range(cf0.PER_CELL)]
    per = {f'{a}_{b}': {'success_rate': float(np.mean([r['success'] for r in rows if (r['a'], r['b']) == (a, b)])),
                        'median_evaluations': float(np.median([r['evaluations'] for r in rows
                                                               if (r['a'], r['b']) == (a, b)]))}
           for (a, b) in SIZES}
    payload = {'status': 'EXPLORATORY, post hoc, no verdict', 'per_size': per, 'rows': rows}
    tmp = OUTPUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    os.replace(tmp, OUTPUT)
    print(json.dumps(per, indent=1))


if __name__ == '__main__':
    main()
