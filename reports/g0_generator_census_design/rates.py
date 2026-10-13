"""Discrimination rates for G0's per-candidate rules (G0_GENERATOR_CENSUS_PLAN.md), from the baseline pre-measurement.

Units are resampled with replacement WITHIN each world (72 hyperplanes = 3 x 24 for S1; 96 tasks = 3 x 32 for S2),
which is the sampling the census itself does. Null = the current generator's own units; effects = the same units
shifted to a target median. 2,000 draws, seed 11."""
import json
import numpy as np

rep = json.load(open('reports/g0_baseline_premeasure.json'))
rng = np.random.default_rng(11)
D = 2000
s1 = [np.array(r['s1']['1']['accuracy']) for r in rep['records']]
s2 = [np.array([t['runner_up_nmse'] for t in r['s2']]) for r in rep['records']]


def draw(units):
    return np.median(np.concatenate([rng.choice(u, size=len(u)) for u in units]))


def rate(units, rule):
    return float(np.mean([rule(draw(units)) for _ in range(D)]))


base1, base2 = np.median(np.concatenate(s1)), np.median(np.concatenate(s2))
s1_rule = lambda m: m <= 0.90
s2_rule = lambda m: 0.01 <= m <= 0.5
out = {'baseline_s1_median': float(base1), 'baseline_s2_median': float(base2)}
out['s1_null_false_fire'] = rate(s1, s1_rule)
for target in (0.92, 0.91, 0.89, 0.88, 0.85):
    out[f's1_rate_at_true_median_{target}'] = rate([u - (base1 - target) for u in s1], s1_rule)
out['s2_null_false_fire'] = rate(s2, s2_rule)
for target in (0.6, 0.55, 0.45, 0.4, 0.2, 0.02):
    out[f's2_rate_at_true_median_{target}'] = rate([u * (target / base2) for u in s2], s2_rule)
out['s2_rate_at_true_median_0.008'] = rate([u * (0.008 / base2) for u in s2], s2_rule)
out['s1_bootstrap_sd'] = float(np.std([draw(s1) for _ in range(D)]))
out['s2_bootstrap_sd'] = float(np.std([draw(s2) for _ in range(D)]))
print(json.dumps(out, indent=1))
