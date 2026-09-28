"""O7 Tier 1 rule rates: r = rescues among the 10 target cells (exact binomial); b = breaks among 12 harm-check cells."""
from scipy.stats import binom
for p in (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
    print(f'p_rescue={p}: P(r>=8)={binom.sf(7, 10, p):.4f} P(4<=r<=7)={binom.cdf(7, 10, p) - binom.cdf(3, 10, p):.4f} P(r<=3)={binom.cdf(3, 10, p):.4f}')
for q in (0.02, 0.05, 0.1, 0.2, 0.3):
    print(f'p_break={q}: P(b>=2)={binom.sf(1, 12, q):.4f}')
