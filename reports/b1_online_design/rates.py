"""B1-online rule rates, exact binomials over the 9 paired cells (3 worlds x 3 streams).
ONLINE_BRANCHES needs n_better >= 8 of 9 (GATED branch median strictly below REFUSAL's) AND k_formation >= 7 of 9
(GATED canonical median < 0.05). FORMATION_BROKEN if k_formation <= 6 (checked first)."""
from scipy.stats import binom
for p in (0.5, 0.7, 0.8, 0.9, 0.95):
    print(f'n_better: p={p}: P(>=8/9)={binom.sf(7, 9, p):.4f}')
for q in (0.5, 0.7, 0.85, 0.95, 0.98):
    print(f'formation: per-cell pass {q}: P(k>=7/9)={binom.sf(6, 9, q):.4f}  P(FORMATION_BROKEN)={binom.cdf(6, 9, q):.4f}')
