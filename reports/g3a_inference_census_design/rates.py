"""Discrimination rates for the G3a decision rule (G3A_INFERENCE_CENSUS_PLAN.md).

A cell = 64 programs; it COUNTS iff median regret > median null floor (SG0's cell rule). The rule fires
(`HEADROOM`) iff k >= 2 of the 9 m=4 cells (3 worlds x supports 128, 8, 2). Units are SG0's depth-3 per-program
(regret, floor) pairs, resampled within support: null = SG0's formed (staged) libraries, effect = SG0's control
libraries, and mixtures in which a fraction f of programs come from the control pool.
"""
import json
from collections import defaultdict

import numpy as np

r = json.load(open('reports/sg0_full_v2.json'))
pool = defaultdict(list)
for c in r['cells'].values():
    if c['depth'] == 3:
        for row in c['rows']:
            pool[(c['staged'], c['support'])].append((row['regret'], row['null_floor']))
pool = {k: np.array(v) for k, v in pool.items()}
SUPPORTS, WORLDS, PROGRAMS, DRAWS = (128, 8, 2), 3, 64, 4000
rng = np.random.default_rng(11)


def cell(f, s):
    n_bad = rng.binomial(PROGRAMS, f)
    rows = np.concatenate([pool[(False, s)][rng.integers(0, 96, n_bad)],
                           pool[(True, s)][rng.integers(0, 96, PROGRAMS - n_bad)]])
    return np.median(rows[:, 0]) > np.median(rows[:, 1])


print('f (fraction of programs drawn from SG0 control libraries) -> P(HEADROOM: k >= 2 of 9); mean k; P(k>=1 at s=2)')
for f in (0.0, 0.25, 0.5, 0.6, 0.75, 1.0):
    ks, k2 = [], []
    for _ in range(DRAWS):
        counts = {s: sum(cell(f, s) for _ in range(WORLDS)) for s in SUPPORTS}
        ks.append(sum(counts.values()))
        k2.append(counts[2] >= 1)
    ks = np.array(ks)
    print(f'f={f:.2f}  P(fire)={np.mean(ks >= 2):.4f}  mean k={ks.mean():.2f}  P(any s=2 cell)={np.mean(k2):.3f}')


# Rule B (program level): at some support, the pooled fraction of the 192 m=4 programs (3 worlds x 64) with
# regret > its own floor is >= 0.10 AND exceeds the m=0 control's pooled fraction at that support by >= 0.05.
# Null: both arms drawn from SG0's formed pool (bind rate 0 at every support). Effect: m=4 programs bind at a true
# rate q at one support, the control at SG0's formed rate (0); the other supports at 0.
from scipy.stats import binom

print()
print('Rule B: true per-program bind rate q at one support -> P(fire), 192 programs, control 0')
print('null (q=0, both arms from the formed pool): P(fire) = 0.0000 (no formed SG0 program binds at any support)')
for q in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
    print(f'q={q:.2f}  P(fire)={binom.sf(int(np.ceil(0.10 * 192)) - 1, 192, q):.4f}')
