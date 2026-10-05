"""B1 rule rates. k = LS_LP tasks with query NMSE < 0.05 of n = 192 (3 worlds x 64). LEARNABLE k >= 0.8 n (154),
PARTIAL 0.5 n <= k < 0.8 n, NOT_LEARNABLE k < 0.5 n. Tasks share a library within a world, so a world-level logit
offset N(0, sd) is drawn (3 worlds x 64 tasks); 20,000 draws, seed 31."""
import numpy as np
rng = np.random.default_rng(31)
lg = lambda p: np.log(p / (1 - p))  # noqa: E731
for sd in (0.0, 0.5, 1.0):
    row = []
    for m in (0.3, 0.6, 0.7, 0.75, 0.85, 0.9, 0.95):
        a = rng.normal(0, sd, size=(20000, 3, 1)) * np.ones((1, 1, 64))
        k = (rng.random(a.shape) < 1 / (1 + np.exp(-(lg(m) + a)))).sum(axis=(1, 2))
        row.append(f'm{m}: L={np.mean(k >= 154):.3f}')
    print(f'sd{sd}', ' '.join(row))
