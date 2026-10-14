"""G0F design: false-fire and detection of the S1 rule (median of 72 held-out accuracies <= 0.90).

Null: the rotated anchor's own 72 accuracies (reports/g0f_baseline_premeasure.json), resampled with replacement.
Effect and near-bar samplers: 72 i.i.d. units with a chosen true median and a WIDER spread than the anchor's
(new forms may give hyperplanes very different accuracies): normal with SD 0.03, 0.08 or 0.15, clipped to [0.5, 1].
"""
import json
import numpy as np

rng = np.random.default_rng(13)
DRAWS, N, BAR = 20000, 72, 0.90
base = json.load(open('reports/g0f_baseline_premeasure.json'))
units = np.array([v for r in base['records'] for v in r['s1']['1']['accuracy']])
assert len(units) == N
out = {'null_anchor_median': float(np.median(units)),
       'null_false_fire': float(np.mean([np.median(rng.choice(units, N)) <= BAR for _ in range(DRAWS)]))}
for sd in (0.03, 0.08, 0.15):
    for true in (0.94, 0.92, 0.91, 0.89, 0.88, 0.86):
        x = np.clip(true + sd * rng.standard_normal((DRAWS, N)), 0.5, 1.0)
        out[f'sd{sd}_true{true}'] = float(np.mean(np.median(x, axis=1) <= BAR))
print(json.dumps(out, indent=1))
