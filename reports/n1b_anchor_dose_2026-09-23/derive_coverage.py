"""POST-HOC, descriptive only: length-1 anchor count and operation coverage per
N1b cell. Not a registered estimand. Regenerates coverage.json."""
import json
from pathlib import Path
from row.experiments import n1_anchor_supply as n1, n1b_anchor_dose as nb

here = Path(__file__).resolve().parent
rows = []
for arm in nb.ARMS:
    for w in nb.WORLDS:
        r = json.loads((here / 'cells' / f'{arm}_w{w}.json').read_text())['record']
        chosen = nb.select_anchors(arm, n1.anchor_tasks(w), w)
        ops = sorted({t.program[0] for t in chosen if t.depth == 1})
        rows.append({'cell': f'{arm}_w{w}', 'length1_anchors': r['anchor_depths']['1'],
                     'length2_anchors': r['anchor_depths']['2'], 'ops_covered': ops,
                     'coverage': len(ops), 'terminal_median': r['terminal_median'],
                     'passes': r['terminal_median'] < 0.05})
(here / 'coverage.json').write_text(json.dumps({'status': 'POST-HOC DESCRIPTIVE', 'rows': rows}, indent=1))
full = [x['passes'] for x in rows if x['coverage'] == 6]
part = [x['passes'] for x in rows if x['coverage'] < 6]
print(f'full coverage: {sum(full)}/{len(full)} pass | partial coverage: {sum(part)}/{len(part)} pass')
