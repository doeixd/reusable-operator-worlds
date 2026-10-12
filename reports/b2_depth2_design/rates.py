"""B2-depth2 rule rates (seed 41). DISCOVERS clauses: AUC_med >= 0.9 (48 loop vs 64 straight candidates per cell, 9
cells), recall >= 0.8 of 432, specificity >= 0.95 of 576, n_terminal >= 8 of 9, formation guard >= 7 of 9.
Per-task null/effect rates as in reports/b2_hard_v2_design/rates.py; sizing anchor (world 60): AUC 0.979, recall
43/48 = 0.896, specificity 64/64, terminal loop 0.368 against the bar 0.5."""
import numpy as np
from scipy.stats import binom

rng = np.random.default_rng(41)


def auc(p, n):
    return (p[:, None] > n[None, :]).mean()


for dprime in (0.0, 1.5, 2.0):
    meds = np.array([np.median([auc(rng.normal(dprime, 1, 48), rng.normal(0, 1, 64)) for _ in range(9)])
                     for _ in range(2000)])
    print(f"AUC clause d'={dprime}: P(median >= 0.9)={np.mean(meds >= 0.9):.3f}")
print(f'recall per-task 0.75 / 0.85 / 0.896: {binom.sf(345, 432, 0.75):.4f} / {binom.sf(345, 432, 0.85):.4f} / '
      f'{binom.sf(345, 432, 0.896):.4f}')
print(f'specificity per-task 0.93 / 0.97: {binom.sf(547, 576, 0.93):.4f} / {binom.sf(547, 576, 0.97):.4f}')
for p in (0.5, 0.7, 0.9, 0.98):
    print(f'n_terminal / formation guard, per-cell pass {p}: P(>= 8 of 9)={binom.sf(7, 9, p):.4f}  '
          f'P(>= 7 of 9)={binom.sf(6, 9, p):.4f}')
