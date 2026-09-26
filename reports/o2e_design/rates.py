"""O2E design: per-arm near-miss rescue rule on O2's STAGED and MIXED_L1 terminals.

Rule RESCUES: >= 4 of the arm's 6 near-miss cells (0.05 <= M < 0.2) pass after sleep AND <= 1 of its passing cells breaks.
Null: every terminal x exp(N(0, sd)). Effect: x f with the same noise. 20,000 draws, seed 11.
"""
import json
import numpy as np

d = json.load(open('reports/o2_online_reliability.json'))['cells']
rng = np.random.default_rng(11)
for arm in ('STAGED', 'MIXED_L1'):
    v = np.array([d[f'{arm}_w{w}_s{s}']['terminal_median'] for w in range(13, 20) for s in (0, 1, 2)])
    near, ok, collapsed = (v >= 0.05) & (v < 0.2), v < 0.05, v >= 1.0
    print(arm, 'near', int(near.sum()), 'passing', int(ok.sum()), 'collapsed', int(collapsed.sum()))

    def rate(shift, sd):
        hits = 0
        for _ in range(20000):
            new = v * np.exp(shift + rng.normal(0, sd, size=v.shape))
            hits += (((new < 0.05) & near).sum() >= 4) and (((new >= 0.05) & ok).sum() <= 1)
        return hits / 20000

    for label, shift in (('null', 0.0), ('x0.5', np.log(0.5)), ('x0.33', np.log(1 / 3)), ('x0.25', np.log(0.25))):
        print('  ', label, ' '.join(f'sd{sd}:{rate(shift, sd):.4f}' for sd in (0.3, 0.6, 1.0)))
