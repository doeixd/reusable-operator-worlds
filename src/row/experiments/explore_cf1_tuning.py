"""EXPLORATORY (Tier 0 sizing, no verdict): tune CF1's relaxed descent on world 79 (not a census world), straight
arm, before CF1 is frozen. Settings: learning rate, restarts at matched total steps, logit init scale."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from row.experiments import census_cf0_loop_structure as cf0
from row.experiments import census_cf1_relaxed_structure as cf1

WORLD = 79
SETTINGS = [(0.05, 3, 0.1), (0.01, 1, 0.0), (0.01, 3, 0.1), (0.2, 3, 0.1), (0.05, 7, 0.1), (0.05, 3, 1.0)]


def main():
    torch.set_num_threads(1)
    w = cf0.world(WORLD)
    prims = cf1.torch_prims(w.library)
    rows = []
    for lr, restarts, scale in SETTINGS:
        cf1.LR, cf1.RESTARTS = lr, restarts
        orig = torch.randn
        for (a, b) in ((2, 2), (3, 3)):
            for i in range(6):
                t = cf0.make_task(w, WORLD, a, b, i, 128)
                truth = list(t['A']) + [t['P']] + list(t['B'])
                y = cf0.apply_seq(w.library, truth, t['xs'].copy())
                torch.randn = lambda *s, orig=orig, scale=scale, **k: orig(*s, **k) * (scale / 0.1 if scale else 0.0)
                t0 = time.process_time()
                ds, dseq = cf1.descend(w.library, prims, t['xs'], y, a, b, False, i)
                torch.randn = orig
                ps, psc, ev = cf1.polish(w.library, dseq, a, b, t['xs'], y, False)
                rows.append({'lr': lr, 'restarts': restarts, 'init': scale, 'a': a, 'b': b, 'i': i,
                             'descent_ok': ds <= cf1.EXACT_SCORE, 'ok': psc <= cf1.EXACT_SCORE,
                             'seconds': time.process_time() - t0})
        for (a, b) in ((2, 2), (3, 3)):
            g = [r for r in rows if (r['lr'], r['restarts'], r['init'], r['a']) == (lr, restarts, scale, a)]
            print(f"lr {lr} R {restarts} init {scale} size {a}_{b}: descent {sum(r['descent_ok'] for r in g)}/6 "
                  f"polished {sum(r['ok'] for r in g)}/6", flush=True)
    Path('reports/cf1_tuning/tuning.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
