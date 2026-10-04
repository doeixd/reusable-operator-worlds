"""O10 Tier 1 rule rates, exact binomials. Target set F = O9's 8 failing REROUTE_WAKE cells (r = rescued by sleep);
harm set H = O9's 37 passing REROUTE_WAKE cells (b = broken by sleep). Rule: HARMS if b >= 3; REPAIRS if r >= 7 and
b <= 2; PARTIAL if 4 <= r <= 6; NO_REPAIR if r <= 3."""
from scipy.stats import binom
for p in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
    print(f'p_rescue={p}: P(r>=7)={binom.sf(6, 8, p):.4f} P(4<=r<=6)={binom.cdf(6, 8, p) - binom.cdf(3, 8, p):.4f} P(r<=3)={binom.cdf(3, 8, p):.4f}')
for q in (0.01, 0.02, 0.05, 0.1, 0.2):
    print(f'p_break={q}: P(b>=3 of 37)={binom.sf(2, 37, q):.4f}')
