"""O2D design: false-fire / detection of 'pass >= T of 21 and <= 1 of the 9 O2-passing cells broken'.

Null: the arm behaves like O2C REPLAY_ONLY (4 per task), each terminal x exp(N(0, sd)).
Effect: the arm behaves like O2C ORACLE_DATA (all data), each terminal x exp(N(0, sd)). 20,000 draws, seed 11.
"""
import json
import numpy as np

c = json.load(open('reports/o2c_consolidation.json'))['cells']
o2 = json.load(open('reports/o2_online_reliability.json'))['cells']
keys = [(w, s) for w in range(13, 20) for s in (0, 1, 2)]
rep = np.array([c[f'REPLAY_ONLY_w{w}_s{s}']['terminal_median'] for w, s in keys])
ora = np.array([c[f'ORACLE_DATA_w{w}_s{s}']['terminal_median'] for w, s in keys])
ok = np.array([o2[f'SHUFFLED_w{w}_s{s}']['terminal_median'] < 0.05 for w, s in keys])
rng = np.random.default_rng(11)


def rate(base, sd, T):
    hits = 0
    for _ in range(20000):
        new = base * np.exp(rng.normal(0, sd, size=base.shape))
        hits += ((new < 0.05).sum() >= T) and (((new >= 0.05) & ok).sum() <= 1)
    return hits / 20000


for T in (18, 19):
    print(f'T={T}', 'null(REPLAY)', ' '.join(f'sd{sd}:{rate(rep, sd, T):.4f}' for sd in (0.2, 0.4)),
          '| effect(ORACLE)', ' '.join(f'sd{sd}:{rate(ora, sd, T):.4f}' for sd in (0.2, 0.4)))
