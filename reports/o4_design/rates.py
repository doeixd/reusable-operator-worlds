"""O4 (sealed) design: k-of-30 rule (10 worlds x 3 streams), per-world logit offset N(0, sd). 20,000 draws, seed 11."""
import numpy as np

rng = np.random.default_rng(11)
logit = lambda p: np.log(p / (1 - p))  # noqa: E731
for sd in (0.0, 1.0):
    for m in (0.43, 0.70, 0.80, 0.90, 0.95):
        a = rng.normal(0, sd, size=(20000, 10, 1)) * np.ones((1, 1, 3))
        p = 1 / (1 + np.exp(-(logit(m) + a)))
        k = (rng.random(p.shape) < p).sum(axis=(1, 2))
        print(f'sd{sd} m={m}: ' + ' '.join(f'P(k>={t})={np.mean(k >= t):.3f}' for t in (25, 26, 27, 28)) + f'  P(k<=24)={np.mean(k <= 24):.3f}')
