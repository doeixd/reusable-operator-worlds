"""O4 sealed design, 15 worlds x 3 streams = 45 cells: rates of 'k >= T' by true per-cell rate and world heterogeneity.
Each world draws a logit offset N(0, sd); streams pass independently given it. 20,000 draws, seed 11."""
import numpy as np

rng = np.random.default_rng(11)
logit = lambda p: np.log(p / (1 - p))  # noqa: E731
for sd in (0.0, 0.5, 1.0):
    ks = {}
    for m in (0.43, 0.70, 0.80, 0.85, 0.90, 0.95):
        a = rng.normal(0, sd, size=(20000, 15, 1)) * np.ones((1, 1, 3))
        p = 1 / (1 + np.exp(-(logit(m) + a)))
        ks[m] = (rng.random(p.shape) < p).sum(axis=(1, 2))
    for T in (40, 41):
        print(f'sd{sd} T={T}: ' + ' '.join(f'm{m}={np.mean(ks[m] >= T):.3f}' for m in ks))
