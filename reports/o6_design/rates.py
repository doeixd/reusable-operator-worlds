"""O6 sealed design, 15 worlds x 3 streams = 45 cells: rates of 'k >= 41' by true per-cell rate and world heterogeneity.
Same rule and sampler as O4 (reports/o4_design/rates_w15.py); adds m=0.844 (SLEEP's sealed O4 rate, the
're-routing adds nothing' null) and m=0.98 (near O5's 87/87). Each world draws a logit offset N(0, sd); streams
pass independently given it. 20,000 draws, seed 12."""
import numpy as np

rng = np.random.default_rng(12)
logit = lambda p: np.log(p / (1 - p))  # noqa: E731
for sd in (0.0, 0.5, 1.0):
    ks = {}
    for m in (0.43, 0.80, 0.844, 0.90, 0.95, 0.98):
        a = rng.normal(0, sd, size=(20000, 15, 1)) * np.ones((1, 1, 3))
        p = 1 / (1 + np.exp(-(logit(m) + a)))
        ks[m] = (rng.random(p.shape) < p).sum(axis=(1, 2))
    print(f'sd{sd} T=41: ' + ' '.join(f'm{m}={np.mean(ks[m] >= 41):.3f}' for m in ks))
