"""B1-hard rule rates. DISCOVERS needs: median over 9 cells of the wake gate-norm AUC (48 branch vs 188 straight
tasks per cell) >= 0.9; pooled branch recall >= 0.8 (of 432); pooled straight specificity >= 0.95 (of 1,692); the
control clause (>= 8/9) is a construction check. Simulations: 20,000 draws, seed 41."""
import numpy as np
from scipy.stats import binom
rng = np.random.default_rng(41)


def auc(p, n):
    return (p[:, None] > n[None, :]).mean()


for dprime in (0.0, 1.0, 1.5, 2.0, 3.0):
    meds = []
    for _ in range(2000):
        aucs = [auc(rng.normal(dprime, 1, 48), rng.normal(0, 1, 188)) for _ in range(9)]
        meds.append(np.median(aucs))
    meds = np.array(meds)
    print(f"gate separation d'={dprime}: median AUC ~{np.median(meds):.3f}; P(median >= 0.9)={np.mean(meds >= 0.9):.3f}")
for p in (0.6, 0.75, 0.8, 0.85, 0.9):
    print(f'recall: per-task {p}: P(pooled >= 0.8 of 432)={binom.sf(int(np.ceil(0.8 * 432)) - 1, 432, p):.4f}')
for q in (0.9, 0.93, 0.95, 0.97, 0.99):
    print(f'specificity: per-task {q}: P(pooled >= 0.95 of 1692)={binom.sf(int(np.ceil(0.95 * 1692)) - 1, 1692, q):.4f}')
