"""N1c discriminating power for the registered 2-of-3 rule, from committed values."""
import json
from pathlib import Path
import numpy as np
from row.design_adequacy import discriminating_power, partitions_outcomes

rng = np.random.default_rng(11)
rule = lambda m: sum(v < 0.05 for v in m) >= 2
draw = lambda lo, hi: [rng.uniform(lo, hi, 3) for _ in range(2000)]
null, full, mid = draw(1.037, 1.087), draw(0.0067, 0.0175), draw(0.05, 0.09)
out = {'false_fire': sum(map(rule, null)) / 2000, 'detection': sum(map(rule, full)) / 2000,
       'intermediate': sum(map(rule, mid)) / 2000,
       'passes': discriminating_power(rule, null, full)['passes'],
       'partitions': partitions_outcomes([('suffices', (2, 3)), ('fails', (0, 1))], (0, 3))['passes']}
Path(__file__).with_name('rates.json').write_text(json.dumps(out, indent=1))
print(out)
