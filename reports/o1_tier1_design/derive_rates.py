"""O1 Tier 1 discrimination, from committed values: SO2 PLAIN (null), SO3's
seven passing BASE streams (effect), SO2's failing STAGED range (intermediate)."""
import json
from pathlib import Path
import numpy as np
from row.design_adequacy import discriminating_power, partitions_outcomes

rng = np.random.default_rng(11)
rule = lambda m: sum(v < 0.05 for v in m) >= 2
draw = lambda lo, hi: [rng.uniform(lo, hi, 3) for _ in range(2000)]
null, eff, mid = draw(1.910, 1.967), draw(0.0088, 0.0400), draw(0.05, 0.13)
out = {'false_fire': float(sum(map(rule, null)) / 2000), 'detection': float(sum(map(rule, eff)) / 2000),
       'intermediate': float(sum(map(rule, mid)) / 2000),
       'passes': discriminating_power(rule, null, eff)['passes'],
       'partitions': partitions_outcomes([('live', (2, 3)), ('not', (0, 1))], (0, 3))['passes'],
       'samplers': {'null': [1.910, 1.967], 'effect': [0.0088, 0.0400], 'intermediate': [0.05, 0.13]}}
Path(__file__).with_name('rates.json').write_text(json.dumps(out, indent=1))
print(out)
