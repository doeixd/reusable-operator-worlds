"""O2 design: false-fire and detection rates of the registered k-of-21 rule (DRAFT plan).

Null m=0.667 (incumbent online rate), effect m=0.95; per-world Beta(m c, (1-m) c)
heterogeneity at c = inf (0), 4, 2; three Bernoulli streams per world; seed 11.
"""
import numpy as np
W, S, N = 7, 3, 20000
def draw(rng, mean, conc):
    a, b = mean*conc, (1-mean)*conc
    p = rng.beta(a, b, size=W) if conc else np.full(W, mean)
    return (rng.random((W, S)) < p[:, None])
for conc in (0, 4, 2):
    rng = np.random.default_rng(11)
    for mean in (0.667, 0.8, 0.9, 0.95, 0.98):
        ks = np.array([draw(rng, mean, conc).sum() for _ in range(N)])
        print(f"conc={conc} mean={mean}: " + " ".join(f"P(k>={t})={np.mean(ks>=t):.3f}" for t in (16,17,18,19,20)) + f"  P(k<=16)={np.mean(ks<=16):.3f} P(k<=14)={np.mean(ks<=14):.3f}")
