"""Re-derive N1's discriminating power for the CORRECTED arms.

The original rates assumed SHAM was drawn at the published floor U(0.92, 0.97).
Under the corrected construction SHAM is 188 length-3 slots at the same update
budget, which is NOT the published 64-slot floor, so its level is unmeasured.
The false-fire and detection rates must therefore be shown robust across a range
of plausible SHAM locations rather than computed at one borrowed value.
"""
import json
import numpy as np
from row.design_adequacy import discriminating_power, partitions_outcomes

RNG = np.random.default_rng(11)
DRAWS = 2000
WORLDS = 3
SUFFICE_T = 0.05
RATIO_T = 5.0


def triage(sample):
    m, s = sample['m'], sample['s']
    if sum(1 for v in m if v < SUFFICE_T) >= 2:
        return 'SUFFICE'
    if float(np.median(s)) / max(float(np.median(m)), 1e-12) >= RATIO_T:
        return 'PARTIAL'
    return 'INSUFFICIENT'


def draw(m_lo, m_hi, s_lo, s_hi, n=DRAWS):
    return [{'m': RNG.uniform(m_lo, m_hi, WORLDS), 's': RNG.uniform(s_lo, s_hi, WORLDS)}
            for _ in range(n)]


def rates(samples):
    out = {'SUFFICE': 0, 'PARTIAL': 0, 'INSUFFICIENT': 0}
    for s in samples:
        out[triage(s)] += 1
    return {k: v / len(samples) for k, v in out.items()}


# SHAM location is unmeasured; sweep it across everything plausible, from
# "barely worse than usable" to the published 64-slot floor.
SHAM_BANDS = [(0.20, 0.25), (0.30, 0.35), (0.45, 0.50), (0.60, 0.65),
              (0.75, 0.80), (0.92, 0.97)]

report = {'rule': {'suffice_threshold': SUFFICE_T, 'ratio_threshold': RATIO_T,
                   'worlds': WORLDS, 'draws': DRAWS, 'seed': 11},
          'partition': None, 'sweep': []}

part = partitions_outcomes([('SUFFICE', (2, 3)), ('NOT_SUFFICE', (0, 1))], (0, 3))
report['partition'] = {'passes': part['passes'], 'detail': part['detail']}

for lo, hi in SHAM_BANDS:
    # NULL: anchors do nothing, so INTERLEAVED behaves like SHAM.
    null = draw(lo, hi, lo, hi)
    # FULL: anchors reproduce staged formation (published J1c).
    full = draw(0.0047, 0.0072, lo, hi)
    # HALF: anchors help but miss usability.
    half = draw(0.05, 0.09, lo, hi)
    row = {
        'sham_band': [lo, hi],
        'null': rates(null),
        'full': rates(full),
        'half': rates(half),
    }
    fires = lambda s: triage(s) in ('SUFFICE', 'PARTIAL')
    dp_full = discriminating_power(fires, null, full)
    dp_half = discriminating_power(fires, null, half)
    row['false_fire'] = dp_full['detail']['false_fire_rate']
    row['detection_full'] = dp_full['detail']['detection_rate']
    row['detection_half'] = dp_half['detail']['detection_rate']
    row['passes_full'] = dp_full['passes']
    row['passes_half'] = dp_half['passes']
    report['sweep'].append(row)

print(f"{'SHAM band':14} {'false-fire':>11} {'det(full)':>10} {'det(half)':>10}  half-ok")
for r in report['sweep']:
    b = f"{r['sham_band'][0]:.2f}-{r['sham_band'][1]:.2f}"
    print(f"{b:14} {r['false_fire']:11.4f} {r['detection_full']:10.4f} "
          f"{r['detection_half']:10.4f}  {r['passes_half']}")
print()
print('partition over 0..3 worlds:', report['partition']['passes'])
Path = __import__('pathlib').Path
out = Path(__file__).resolve().parent / 'corrected_rates.json'
out.write_text(json.dumps(report, indent=1), encoding='utf-8')
print('wrote', out)
