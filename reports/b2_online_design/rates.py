"""B2-online design: exact binomial rates for the registered '>= 8 of 9 cells' clauses and the formation guard."""
from math import comb


def at_least(k, n, p):
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


for p in (0.5, 0.7, 0.9, 0.95):
    print(f'per-cell clause probability {p}: P(>= 8 of 9) = {at_least(8, 9, p):.4f}')
for p in (0.98, 0.85, 0.5):
    print(f'per-cell formation probability {p}: P(>= 7 of 9) = {at_least(7, 9, p):.4f}')
