"""B3 Tier 0 opportunity census (2026-10-12). Teacher programs only; no learner, no training.
Question: does the B2-depth2 stream generator contain recurring repeated structure that a loop abstraction
(REPEAT(A, n), or a shared loop over a recurring body) could amortize? Priced with E6's registered alphabet-tax
threshold H* = [L log2 K + N D log2((K+1)/K)] / [(L-1) log2(K+1)] (PREDICTIONS.md, review 82)."""
import json
import math
from collections import Counter

import numpy as np

from row.experiments import loop_stream as ls

K_SLOTS = 12


def h_star(L, N, D, K=K_SLOTS):
    return (L * math.log2(K) + N * D * math.log2((K + 1) / K)) / ((L - 1) * math.log2(K + 1))


out = {'cells': {}}
for w in (61, 62, 63):
    for s in (0, 1, 2):
        cfg, world, st, plan, canonical, ltasks, meta = ls.stream(w, s, 48, 2)
        loop_ids = {t.task_id for t in ltasks}
        straight = [tuple(t.program.primitive_ids)
                    for t in st if t.task_id not in loop_ids]
        by_len = Counter(len(p) for p in straight)
        runs = Counter()
        for p in straight:
            for i in range(len(p) - 1):
                if p[i] == p[i + 1]:
                    runs[len(p)] += 1
                    break
        full = Counter(len(p) for p in straight if len(p) > 1 and len(set(p)) == 1)
        # most frequent repeated pair AA over all straight programs (uses of a REPEAT(A,2) macro)
        aa = Counter()
        for p in straight:
            for i in range(len(p) - 1):
                if p[i] == p[i + 1]:
                    aa[p[i]] += 1
        bodies = Counter(tuple(meta[t.task_id]['P']) for t in ltasks)
        N = len(st)
        D = float(np.mean([len(p) for p in straight]))
        out['cells'][f'w{w}_s{s}'] = {
            'n_tasks': N, 'straight_by_length': dict(by_len), 'programs_with_adjacent_repeat': dict(runs),
            'all_same_programs': dict(full), 'max_uses_of_one_AA': max(aa.values()) if aa else 0,
            'loop_body_max_recurrence': max(bodies.values()), 'distinct_loop_bodies': len(bodies),
            'mean_length': D, 'H_star_L2': h_star(2, N, D), 'H_star_L3': h_star(3, N, D)}
        print(w, s, out['cells'][f'w{w}_s{s}'], flush=True)
c = list(out['cells'].values())
out['summary'] = {k: [min(x[k] for x in c), max(x[k] for x in c)]
                  for k in ('max_uses_of_one_AA', 'loop_body_max_recurrence', 'H_star_L2', 'H_star_L3')}
with open('reports/b3_opportunity/census.json', 'w') as f:
    json.dump(out, f, indent=1)
print(out['summary'])
