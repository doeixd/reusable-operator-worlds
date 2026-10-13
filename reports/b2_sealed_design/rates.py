"""B2 sealed rule rates (seed 43). ONESTEP: 30 cells (15 worlds x streams 0-1); ZERO and GIVEN: 15 cells (stream 0).
CONFIRMED needs: k_formation >= 27 of 30, AUC_med >= 0.9 (median of 30 cells), recall >= 0.8 of 1440, specificity
>= 0.95 of 1800, n_terminal >= 27 of 30. Gate: GIVEN forms in >= 12 of 15. ROOTS_PREVENTED: r_zero >= 2 of 15 and
r_one <= 1 of 30. Zero-start root base rate per cell 7/19 (B2-hard 6/18, sizing 1/1; B2-hard v2 ZERO 7/9 not pooled
in, reported separately)."""
import numpy as np
from scipy.stats import binom

rng = np.random.default_rng(43)


def auc(p, n):
    return (p[:, None] > n[None, :]).mean()


for dprime in (0.0, 1.5, 2.0):
    meds = np.array([np.median([auc(rng.normal(dprime, 1, 48), rng.normal(0, 1, 60)) for _ in range(30)])
                     for _ in range(1000)])
    print(f"AUC clause, 30 cells, d'={dprime}: P(median >= 0.9)={np.mean(meds >= 0.9):.3f}")
print(f'recall of 1440, per-task 0.75 / 0.85: {binom.sf(1151, 1440, 0.75):.4f} / {binom.sf(1151, 1440, 0.85):.4f}')
print(f'specificity of 1800, per-task 0.93 / 0.97: {binom.sf(1709, 1800, 0.93):.4f} / {binom.sf(1709, 1800, 0.97):.4f}')
for p in (0.67, 0.75, 0.8, 0.85, 0.95, 0.98):
    print(f'formation / n_terminal, per-cell {p}: P(>= 27 of 30)={binom.sf(26, 30, p):.4f}')
for p in (0.5, 0.67, 0.9, 0.98):
    print(f'GIVEN gate, per-cell {p}: P(>= 12 of 15)={binom.sf(11, 15, p):.4f}')
p0 = 7 / 19
z_inf = binom.sf(1, 15, p0)
print(f'zero-start base rate {p0:.3f}: P(r_zero >= 2 of 15)={z_inf:.4f}')
for p1 in (p0, 0.2, 0.1, 0.05, 0.02, 0.0):
    print(f'ROOTS_PREVENTED, one-step root rate {p1:.3f}: P(r_one <= 1 of 30 and r_zero >= 2)={binom.cdf(1, 30, p1) * z_inf:.4f}')
