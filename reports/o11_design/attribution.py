"""O11 attribution (same sampler and seed as O9): n_better = cells where RWS is strictly below SLEEP (wake + sleep, no re-routing); rule n_better >= 30
of 45. O6's sampler (reports/o6_design/attribution.py): per-world logit offset N(0, sd), streams independent given
it. Null 0.5 (re-routing adds nothing); O8's development rate was 20/21 = 0.952. 20,000 draws, seed 22."""
import numpy as np

rng = np.random.default_rng(22)
logit = lambda p: np.log(p / (1 - p))  # noqa: E731
for sd in (0.0, 0.5, 1.0):
    out = []
    for q in (0.5, 0.6, 0.7, 0.8, 0.952):
        a = rng.normal(0, sd, size=(20000, 15, 1)) * np.ones((1, 1, 3))
        p = 1 / (1 + np.exp(-(logit(q) + a)))
        n = (rng.random(p.shape) < p).sum(axis=(1, 2))
        out.append(f'q{q}={np.mean(n >= 30):.3f}')
    print(f'sd{sd} U=30: ' + ' '.join(out))
