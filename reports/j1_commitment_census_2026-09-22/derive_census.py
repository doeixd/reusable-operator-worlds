"""Read-only census of J1's commitment dynamics, for the C1 candidate rung.

Regenerates `census.json` from the committed `reports/j1_search_loop.json` and
records that file's digest. No experiment, no artifact, no learner: this is a
recomputation over a report already in version control.

Question it answers: J1's recorded lesson says the assignment "never changed
again (0% after round 3-5)". How sticky is it exactly, what does the sham do,
and is the CONFIDENCE signal that candidate rung C1 proposes to gate on present
anywhere in the record?

Run from the repository root:
    python reports/j1_commitment_census_2026-09-22/derive_census.py
"""
from __future__ import annotations

import json
from pathlib import Path

from row.experiments.so1_storage import digest

SOURCE = Path('reports/j1_search_loop.json')
OUTPUT = Path(__file__).resolve().parent / 'census.json'

# Fields C1 would need in order to gate a commitment on its confidence.
CONFIDENCE_FIELDS = {'margin', 'route_margin', 'confidence', 'gap', 'posterior',
                     'second_best', 'relative_gap', 'entropy'}


def main() -> int:
    report = json.loads(SOURCE.read_text(encoding='utf-8'))
    cells = report['cells']

    rows = {}
    round_fields = set()
    for key, cell in sorted(cells.items()):
        tasks = len(cell['routes'])
        rounds = cell['rounds']
        for record in rounds:
            round_fields.update(record)
        # `changed` is a FRACTION of tasks; convert to a count so the number is
        # legible without knowing the denominator.
        changed = [(r['round'], round(r['changed'] * tasks))
                   for r in rounds if 'changed' in r]
        frozen = next((r['round'] for r in rounds if r.get('changed') == 0.0), None)
        rows[key] = {
            'mode': cell.get('mode'),
            'world': cell.get('world'),
            'tasks': tasks,
            'rounds': len(rounds),
            'changed_counts_first_five': [c for _, c in changed[:5]],
            'total_changes_after_round_one': sum(c for r, c in changed if r > 1),
            'first_frozen_round': frozen,
            'terminal_median': cell['terminal_median'],
            'pinned_one_hot': cell.get('pinned_one_hot'),
        }

    j1 = [r for r in rows.values() if r['mode'] == 'j1']
    sham = [r for r in rows.values() if r['mode'] == 'sham']
    census = {
        'source': SOURCE.as_posix(),
        'source_sha256': digest(SOURCE),
        'status': 'READ-ONLY CENSUS over a committed report; no experiment, no verdict',
        'cells': rows,
        'round_record_fields': sorted(round_fields),
        'confidence_fields_present': sorted(round_fields & CONFIDENCE_FIELDS),
        'confidence_signal_recorded': bool(round_fields & CONFIDENCE_FIELDS),
        'summary': {
            'j1_changed_in_round_one': sorted(r['changed_counts_first_five'][0] for r in j1),
            'j1_total_changes_after_round_one': sorted(r['total_changes_after_round_one'] for r in j1),
            'j1_first_frozen_round': sorted(r['first_frozen_round'] for r in j1),
            'j1_terminal_median': sorted(round(r['terminal_median'], 4) for r in j1),
            'sham_changed_in_round_one': sorted(r['changed_counts_first_five'][0] for r in sham),
            'sham_first_frozen_round': sorted(
                (-1 if r['first_frozen_round'] is None else r['first_frozen_round']) for r in sham),
            'sham_terminal_median': sorted(round(r['terminal_median'], 4) for r in sham),
            'tasks_per_cell': sorted({r['tasks'] for r in rows.values()}),
        },
    }
    OUTPUT.write_text(json.dumps(census, indent=1), encoding='utf-8')
    print(json.dumps(census['summary'], indent=1))
    print('confidence signal recorded anywhere in J1 rounds:',
          census['confidence_signal_recorded'])
    print('wrote', OUTPUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
