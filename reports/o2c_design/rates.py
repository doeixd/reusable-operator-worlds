"""O2C design: false-fire / detection of the registered near-miss rescue rule, on O2's actual SHUFFLED terminals.

Rule RESCUES: >= 5 of the 8 near-miss cells (0.05 <= M < 0.2) pass (< 0.05) AND <= 1 of the 9 passing cells breaks.
Null: consolidation multiplies every terminal by exp(N(0, sd)) (perturbation, no systematic gain).
Effect: exp(log(f) + N(0, sd)) with f < 1 a systematic reduction. 20,000 draws, seed 11.
"""
import json
import numpy as np

d = json.load(open('reports/o2_online_reliability.json'))['cells']
v = np.array([d[f'SHUFFLED_w{w}_s{s}']['terminal_median'] for w in range(13, 20) for s in (0, 1, 2)])
near, ok = (v >= 0.05) & (v < 0.2), v < 0.05
rng = np.random.default_rng(11)


def rate(shift, sd):
    hits = 0
    for _ in range(20000):
        new = v * np.exp(shift + rng.normal(0, sd, size=v.shape))
        hits += (((new < 0.05) & near).sum() >= 5) and (((new >= 0.05) & ok).sum() <= 1)
    return hits / 20000


print('near-miss cells', int(near.sum()), 'passing cells', int(ok.sum()))
for label, shift in (('null', 0.0), ('x0.5', np.log(0.5)), ('x0.33', np.log(1 / 3)), ('x0.25', np.log(0.25))):
    print(label, ' '.join(f'sd{sd}:{rate(shift, sd):.4f}' for sd in (0.3, 0.6, 1.0)))
