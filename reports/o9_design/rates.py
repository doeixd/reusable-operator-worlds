"""O9 sealed design, 15 worlds x 3 streams = 45 cells. Primary rule: k_RW >= 41 (REROUTE_WAKE cells < 0.05).
O6's rule and sampler (reports/o6_design/rates.py) with nulls/effects re-centred for O9: 0.43 (wake alone,
O2-O3 order-free rate), 0.80 (partially reliable: the registered null), 0.844 (sealed O4 SLEEP rate),
0.90, 0.952 (O8's development 20/21), 0.98. Each world draws a logit offset N(0, sd); streams independent
given it. 20,000 draws, seed 21."""
import numpy as np

rng = np.random.default_rng(21)
logit = lambda p: np.log(p / (1 - p))  # noqa: E731
for sd in (0.0, 0.5, 1.0):
    ks = {}
    for m in (0.43, 0.80, 0.844, 0.90, 0.952, 0.98):
        a = rng.normal(0, sd, size=(20000, 15, 1)) * np.ones((1, 1, 3))
        p = 1 / (1 + np.exp(-(logit(m) + a)))
        ks[m] = (rng.random(p.shape) < p).sum(axis=(1, 2))
    print(f'sd{sd} T=41: ' + ' '.join(f'm{m}={np.mean(ks[m] >= 41):.3f}' for m in ks))
