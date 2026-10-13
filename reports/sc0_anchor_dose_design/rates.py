"""SC0 rule rates (exact binomials). Per dose, k = passing streams of 5 (independent streams, common per-cell pass
probability p at that dose).
NO_OPPORTUNITY: k_4 == 5. OPPORTUNITY: k_4 <= 1 and k_16 >= 4. Controls (one cell each): dose 60 passes, dose 0 fails.
Baselines: confirmed O6 construction (60 anchors) 45/45 sealed; no anchors 0/15 (O9 PLAIN), offline N1b/N1c 6 anchors
fail, 18 pass 2/3."""
from scipy.stats import binom

print('NO_OPPORTUNITY (k_4 == 5):')
for p in (0.5, 0.7, 0.8, 0.9, 0.98):
    print(f'  p_4={p}: P(fire)={binom.pmf(5, 5, p):.4f}')
print('OPPORTUNITY (k_4 <= 1 and k_16 >= 4), null = no dose effect (p_4 = p_16 = p):')
worst = max((binom.cdf(1, 5, p / 100) * binom.sf(3, 5, p / 100), p / 100) for p in range(1, 100))
print(f'  worst-case false-fire over p: {worst[0]:.4f} at p={worst[1]:.2f}')
for p4, p16 in ((0.1, 0.95), (0.2, 0.9), (0.3, 0.8)):
    print(f'  effect p_4={p4}, p_16={p16}: detection={binom.cdf(1, 5, p4) * binom.sf(3, 5, p16):.4f}')
for p in (0.9, 0.98):
    print(f'controls both as expected at per-cell {p}: {p * p:.4f}')
