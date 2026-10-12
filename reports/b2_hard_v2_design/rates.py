"""B2-hard v2 rule rates (seed 41). Primary label (DISCOVERS) clauses AUC / recall / specificity as in
reports/b2_hard_design/rates.py, plus n_terminal >= 8 of 9 and the formation guard >= 7 of 9. Secondary label
ROOTS_PREVENTED: r_one == 0 and r_zero >= 2 (cells with a rooted operation at end of wake, of 9 per arm). Zero-start
base rate per cell: 7 of 19 candidate cells so far (B2-hard ALLLOOP 3/9, ALLCOUNT 3/9; sizing 1/1)."""
import numpy as np
from scipy.stats import binom

rng = np.random.default_rng(41)


def auc(p, n):
    return (p[:, None] > n[None, :]).mean()


for dprime in (0.0, 1.5, 2.0):
    meds = np.array([np.median([auc(rng.normal(dprime, 1, 48), rng.normal(0, 1, 60)) for _ in range(9)])
                     for _ in range(2000)])
    print(f"AUC clause d'={dprime}: P(median >= 0.9)={np.mean(meds >= 0.9):.3f}")
print(f'recall per-task 0.75 / 0.85: {binom.sf(345, 432, 0.75):.4f} / {binom.sf(345, 432, 0.85):.4f}')
print(f'specificity per-task 0.93 / 0.97: {binom.sf(512, 540, 0.93):.4f} / {binom.sf(512, 540, 0.97):.4f}')
for p in (0.5, 0.7, 0.9, 0.98):
    print(f'n_terminal / formation guard, per-cell pass {p}: P(>= 8 of 9)={binom.sf(7, 9, p):.4f}  P(>= 7 of 9)={binom.sf(6, 9, p):.4f}')
p0 = 7 / 19
z_inf = binom.sf(1, 9, p0)
print(f'zero-start base rate {p0:.3f}: P(r_zero >= 2)={z_inf:.3f} (informative control)')
for p1 in (p0, 0.2, 0.1, 0.05, 0.02, 0.0):
    print(f'ROOTS_PREVENTED, one-step per-cell root rate {p1:.3f}: P(r_one == 0 and r_zero >= 2)={binom.pmf(0, 9, p1) * z_inf:.4f}')
