"""Discrimination rates for one G3b (v, s) cell (G3B_NOISE_CENSUS_PLAN.md). 192 programs per arm; a cell is in the
WINDOW iff the m=4 bind fraction a >= 0.10 and a - b >= 0.10 (b = m=0 fraction); BOTH iff a >= 0.10 and a - b < 0.10.
Arms are independent binomials at true per-program bind rates (qa, qb); 100,000 draws, seed 12."""
import numpy as np

rng = np.random.default_rng(12)
N, D = 192, 100_000
print('qa   qb   P(window)  P(both)  P(neither)')
for qa, qb in ((0.0, 0.0), (0.05, 0.0), (0.10, 0.0), (0.15, 0.0), (0.20, 0.0), (0.15, 0.15), (0.30, 0.30),
               (0.30, 0.20), (0.30, 0.10), (0.50, 0.30)):
    a = rng.binomial(N, qa, D) / N
    b = rng.binomial(N, qb, D) / N
    win = (a >= 0.10) & (a - b >= 0.10)
    both = (a >= 0.10) & (a - b < 0.10)
    print(f'{qa:.2f} {qb:.2f}  {win.mean():.4f}     {both.mean():.4f}   {1 - win.mean() - both.mean():.4f}')
