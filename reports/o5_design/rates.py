"""O5 design: collapse-repair rule on the 8 collapsed order-free cells (O2 1, O3 1, O4 6). Exact binomials.
Null: re-routing adds nothing, so a collapse cell passes after sleep with p0 (sleep alone rescued 1 of 8: 0.125; 0.25 harder).
Effect: re-routing restores the library's usable routes and sleep repairs the rest: p1 = 0.85 / 0.75."""
from math import comb


def ge(n, k, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


for p in (0.125, 0.25, 0.75, 0.85):
    print(f'p={p}: P(r>=6 of 8)={ge(8, 6, p):.4f}  P(r>=3)={ge(8, 3, p):.4f}  P(r<=2)={1 - ge(8, 3, p):.4f}')
