"""SC0b rule rates. Per dose, k = passing streams of 5. Doses are NESTED within a stream, so k_32 and k_40 may be
positively correlated; every rate below is given at independence and at the worst case over that correlation.
LENGTHS_SUFFICE: k_32 == 5 and k_40 >= 4. LENGTHS_INSUFFICIENT: k_40 <= 1.
Baselines: SC0 (world 64): k_4 = 0, k_8 = 0, k_16 = 2 of 5; dose 60 passed (O6 sealed 45/45)."""
from scipy.stats import binom

print('LENGTHS_SUFFICE (k_32 == 5 and k_40 >= 4)')
print('  null: dose 32 unreliable (p_32 <= 0.5); false-fire <= P(k_32 == 5) under ANY correlation:')
for p in (0.3, 0.5):
    print(f'    p_32={p}: bound={binom.pmf(5, 5, p):.4f}  independent (p_40=0.98)={binom.pmf(5, 5, p) * binom.sf(3, 5, 0.98):.4f}')
print('  effect: both doses reliable; detection independent / fully correlated (= P(k_32 == 5)):')
for p32, p40 in ((0.98, 0.98), (0.96, 0.98), (0.95, 0.95)):
    print(f'    p_32={p32}, p_40={p40}: {binom.pmf(5, 5, p32) * binom.sf(3, 5, p40):.4f} / {binom.pmf(5, 5, p32):.4f}')
print('LENGTHS_INSUFFICIENT (k_40 <= 1), one dose so correlation does not enter')
for p in (0.7, 0.8, 0.9):
    print(f'  null p_40={p}: false-fire={binom.cdf(1, 5, p):.4f}')
for p in (0.1, 0.2):
    print(f'  effect p_40={p}: detection={binom.cdf(1, 5, p):.4f}')
