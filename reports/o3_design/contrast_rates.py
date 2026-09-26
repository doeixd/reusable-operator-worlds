"""O3 design: paired phase contrast on per-cell r = log10(M_SLEEP / M_INTERLEAVED).

r = u_w + e_c + delta, u_w ~ N(0, s_w) per world, e_c ~ N(0, s_e) per cell. 20,000 draws, seed 11.
PHASE_MATTERS: >= 16 of 21 cells r < 0 AND >= 6 of 7 worlds with >= 2 of 3 streams r < 0 AND median(r) <= -0.15.
INTERLEAVED_BETTER: the mirror image. EQUIVALENT: 7 <= cells r<0 <= 14 AND |median r| <= 0.10. Else INDETERMINATE.
"""
import numpy as np

rng = np.random.default_rng(11)


def rule(r):
    neg = (r < 0).sum(); pos = 21 - neg
    wneg = ((r.reshape(7, 3) < 0).sum(1) >= 2).sum(); wpos = ((r.reshape(7, 3) > 0).sum(1) >= 2).sum()
    med = np.median(r)
    if neg >= 16 and wneg >= 6 and med <= -0.15:
        return 'PHASE'
    if pos >= 16 and wpos >= 6 and med >= 0.15:
        return 'INTER_BETTER'
    if 7 <= neg <= 14 and abs(med) <= 0.10:
        return 'EQUIV'
    return 'INDET'


for sw, se in ((0.0, 0.3), (0.2, 0.3), (0.4, 0.3), (0.2, 0.6)):
    for delta in (0.0, -0.15, -0.3, -0.5):
        c = {}
        for _ in range(20000):
            r = np.repeat(rng.normal(0, sw, 7), 3) + rng.normal(0, se, 21) + delta
            k = rule(r); c[k] = c.get(k, 0) + 1
        print(f'sw{sw} se{se} delta{delta:+.2f} ' + ' '.join(f'{k}={c.get(k, 0) / 20000:.3f}' for k in ('PHASE', 'INTER_BETTER', 'EQUIV', 'INDET')))
