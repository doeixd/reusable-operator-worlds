"""O8 Tier 1 rule rates, exact binomials, n = 21 paired development cells (O3 worlds 20-26 x 3 streams).

Rule A (REROUTE_WAKE against the committed SHUFFLED): k = cells with terminal median < 0.05;
n_better = cells whose terminal median is strictly below SHUFFLED's. Null per-cell pass rate =
SHUFFLED's observed 8/21 = 0.381; null for the paired sign = 0.5.
Rule B (REROUTE_INTERLEAVED against the committed INTERLEAVED, 17/21): paired sign n_better
and the fail count f. INTERLEAVED leaves only 4 failing cells, so k alone has no room; the sign
test over all 21 is the clause with power.
"""
from scipy.stats import binom
n = 21
print('Rule A pass-count clause k >= K')
for K in (13, 14, 15):
    print(f'  K={K}: null p=0.381 -> {binom.sf(K - 1, n, 0.381):.4f}; p=0.6 -> {binom.sf(K - 1, n, 0.6):.4f}; '
          f'p=0.7 -> {binom.sf(K - 1, n, 0.7):.4f}; p=0.86 (sleep level 18/21) -> {binom.sf(K - 1, n, 0.86):.4f}')
print('Paired sign clause n_better >= U (both rules)')
for U in (15, 16, 17):
    print(f'  U={U}: null p=0.5 -> {binom.sf(U - 1, n, 0.5):.4f}; p=0.8 -> {binom.sf(U - 1, n, 0.8):.4f}; '
          f'p=0.9 -> {binom.sf(U - 1, n, 0.9):.4f}')
print('Harm clause: h = cells SHUFFLED passes (8) that the re-routed arm fails; h >= H fires')
for H in (2, 3):
    for q in (0.05, 0.1, 0.25, 0.5):
        print(f'  H={H} q={q}: {binom.sf(H - 1, 8, q):.4f}')
print('Rule B fail clause f <= 1 of 21 at per-cell fail rate')
for q in (0.02, 0.05, 0.1, 0.19):
    print(f'  q={q}: P(f<=1)={binom.cdf(1, n, q):.4f}')
print('Rule B harm clause: h = cells INTERLEAVED passes (17) that REROUTE_INTERLEAVED fails; h >= 3 fires')
for q in (0.05, 0.1, 0.25, 0.5):
    print(f'  q={q}: {binom.sf(2, 17, q):.4f}')
