"""EXPLORATORY timing (no verdict): exhaustive straight-line route search on the rotated TEACHER library with prefix
sharing (depth-first over the program tree, states reused across siblings), world 79, depths 5-7, one task each."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from row.experiments import census_cf0_loop_structure as cf0


def enum_prefix_shared(lib, x, y, depth):
    best = [np.inf, None]
    n = len(lib)

    def rec(z, seq):
        if len(seq) == depth:
            v = float(np.mean((z - y) ** 2))
            if v < best[0]:
                best[0], best[1] = v, list(seq)
            return
        for p in range(n):
            rec(lib[p](z), seq + [p])
    rec(np.array(x, dtype=np.float64), [])
    return best


def main():
    w = cf0.world(79)
    out = []
    for depth in (5, 6, 7):
        a = (depth - 1) // 2
        t = cf0.make_task(w, 79, a, depth - 1 - a, 0, 128)
        truth = list(t['A']) + [t['P']] + list(t['B'])
        y = cf0.apply_seq(w.library, truth, t['xs'].copy())
        t0 = time.perf_counter()
        score, seq = enum_prefix_shared(w.library, t['xs'], y, depth)
        secs = time.perf_counter() - t0
        out.append({'depth': depth, 'routes': 6 ** depth, 'seconds': secs, 'exact': seq == truth, 'score': score})
        print(out[-1], flush=True)
    Path('reports/cf1_tuning/enum_cost.json').write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
