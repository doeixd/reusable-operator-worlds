"""O2G design: SYSTEMATIC / STOCHASTIC rule for STAGED stage-3 collapse, 2 fresh reruns per cell.

C = the 3 O2 cells that collapsed (6 reruns); H = the 18 that did not (36 reruns).
SYSTEMATIC: >= 5 of 6 C-reruns collapse AND <= 3 of 36 H-reruns collapse.
STOCHASTIC: <= 2 of 6 C-reruns collapse. Otherwise MIXED.
Stochastic world: every rerun collapses independently with hazard h (O2: 3/21 = 0.143).
Systematic world: C-reruns collapse with q (0.9 / 0.8), H-reruns with a small h0 (0.03 / 0.06). Exact binomial.
"""
from math import comb


def b_ge(n, k, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def b_le(n, k, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


for h in (0.1, 0.143, 0.25):
    sysfire = b_ge(6, 5, h) * b_le(36, 3, h)
    print(f'stochastic h={h}: SYSTEMATIC fires {sysfire:.5f}; STOCHASTIC fires {b_le(6, 2, h):.4f}')
for q, h0 in ((0.9, 0.03), (0.8, 0.03), (0.9, 0.06), (0.8, 0.06)):
    print(f'systematic q={q} h0={h0}: SYSTEMATIC fires {b_ge(6, 5, q) * b_le(36, 3, h0):.4f}; STOCHASTIC fires {b_le(6, 2, q):.5f}')
