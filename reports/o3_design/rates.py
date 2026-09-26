"""O3 design: false-fire / detection for the primary k-of-21 rule and the paired phase contrast.

Cells: 7 worlds x 3 streams; each world draws a latent logit a_w ~ N(0, sd_w); arm pass prob = sigmoid(logit(m_arm) + a_w).
Arms in a cell are conditionally independent given the world (paired only through the world). 20,000 draws, seed 11.
Primary: SLEEP RELIABLE if k >= 19. Contrast: d = k_SLEEP - k_INTER; PHASE_MATTERS if d >= 3; INTER_BETTER if d <= -3;
EQUIVALENT if |d| <= 1; else INDETERMINATE.
"""
import numpy as np

rng = np.random.default_rng(11)
logit = lambda p: np.log(p / (1 - p))  # noqa: E731


def draw(m, a):
    p = 1 / (1 + np.exp(-(logit(m) + a)))
    return (rng.random(p.shape) < p).sum(axis=(1, 2))


def run(ms, mi, sd, n=20000):
    a = rng.normal(0, sd, size=(n, 7, 1)) * np.ones((1, 1, 3))
    ks, ki = draw(ms, a), draw(mi, a)
    d = ks - ki
    return {'RELIABLE': np.mean(ks >= 19), 'PHASE': np.mean(d >= 3), 'INTER_BETTER': np.mean(d <= -3),
            'EQUIV': np.mean(np.abs(d) <= 1)}


for sd in (0.0, 1.0):
    for ms, mi, tag in ((0.43, 0.43, 'null: sleep does not transfer'), (0.80, 0.80, 'partial transfer, equal'),
                        (0.95, 0.95, 'both reliable, equal'), (0.95, 0.80, 'phase effect (0.95 vs 0.80)'),
                        (0.95, 0.70, 'phase effect (0.95 vs 0.70)'), (0.95, 0.60, 'phase effect (0.95 vs 0.60)')):
        r = run(ms, mi, sd)
        print(f'sd{sd} {tag:32s} ' + ' '.join(f'{k}={v:.3f}' for k, v in r.items()))
