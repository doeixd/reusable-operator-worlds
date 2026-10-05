"""D2 Tier 1 rule rates (D1's rule unchanged; RS5 replaces RW_SLEEP4, SLEEP5 replaces SLEEP4) (exact binomials). k = RW_SLEEP4 cells < 0.05 of 21; h = cells SLEEP4 passes and RW_SLEEP4 fails.
TRANSFERS k >= 18; PARTIAL 12-17; DOES_NOT_TRANSFER k <= 11; HARMS h >= 3 (checked first)."""
from scipy.stats import binom
for p in (0.43, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
    print(f'p_pass={p}: P(k>=18)={binom.sf(17, 21, p):.4f} P(12<=k<=17)={binom.cdf(17, 21, p) - binom.cdf(11, 21, p):.4f} P(k<=11)={binom.cdf(11, 21, p):.4f}')
for q in (0.02, 0.05, 0.1, 0.2):
    for n in (10, 21):
        print(f'p_break={q} over n={n} SLEEP4 passes: P(h>=3)={binom.sf(2, n, q):.4f}')
