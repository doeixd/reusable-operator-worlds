"""O6 secondary (attribution) rule: paired sign test over all 45 (world, stream) cells.
ATTRIBUTED iff REROUTE_SLEEP's terminal median is strictly below SLEEP's in >= K = 30 cells. The independent-cell
exact threshold would be 29, but under world clustering (sd 1) 29 false-fires 6.2%; 30 keeps it <= 5% at every sd. Non-finite RS or a tie counts as not improved.
Null: per-cell improvement probability 0.5 (re-routing adds nothing), also with world clustering.
Effect: improvement probability m (O5 measured 87/87 cells improved, 78/78 among SLEEP-passing).
Each world draws a logit offset N(0, sd); streams independent given it. 20,000 draws, seed 13."""
import numpy as np
from scipy.stats import binom

K = 30
print(f'K = {K}; exact null tail P(Bin(45,0.5) >= K) = {binom.sf(K - 1, 45, 0.5):.4f}')
rng = np.random.default_rng(13)
logit = lambda p: np.log(p / (1 - p))  # noqa: E731
for sd in (0.0, 0.5, 1.0):
    out = []
    for m in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        a = rng.normal(0, sd, size=(20000, 15, 1)) * np.ones((1, 1, 3))
        p = 1 / (1 + np.exp(-(logit(m) + a)))
        out.append(f'm{m}={np.mean((rng.random(p.shape) < p).sum(axis=(1, 2)) >= K):.3f}')
    print(f'sd{sd}: ' + ' '.join(out))
