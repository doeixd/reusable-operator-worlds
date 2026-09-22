"""SG0 NO-HEADROOM sub-triage, scored read-only over a committed SG0 report.

Implements `SG0_SUBTRIAGE_AMENDMENT.md`, which amends the sub-triage registered
in `SYNTHESIS_OPPORTUNITY_GATE_PLAN.md` Revision 3. The amendment lives in its
own file because the plan is hashed into the SG0 run's protocol: editing it
would make `score_sg0_headroom_gate` refuse the report as `input changed since
the run`.

Nothing here re-runs or regenerates an artifact. Every quantity is a reduction
over records the registered runner already wrote.

The sub-triage is a DIAGNOSTIC that decides a research-program question (ladder
decision 6). It is not preregistered for this run and supports no scientific
claim; see the amendment's "Registration order" section.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from row.experiments.so1_storage import atomic_json

#: Tolerances at which the near-tie set's EXISTENCE is probed. A program has a
#: rival within `eps` exactly when its recorded identifiability is <= eps.
EPS_CURVE = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)

#: The tolerance the runner used, and the only one at which the query-spread
#: clause is evaluable (`near_tie_disagreement` is recorded only there).
REGISTERED_EPS = 0.01

#: `j <= J_BAND` -> SATURATED, `j >= J_BAND + 1` -> EVIDENTIAL.
J_BAND = 2

#: Widest tolerance at which the bound is claimed; see the amendment, "Scope".
EPS_SCOPE = 0.10

STAGED_CELLS_IN_FULL_GRID = 36


def _cell_counts(cell):
    """Per-program rule (Amendment A): a cell counts if ANY program has a rival
    in its near-tie set whose query-NMSE spread exceeds that program's floor."""
    return any(row['near_tie_size'] >= 2 and row['near_tie_disagreement'] > row['null_floor']
               for row in cell['rows'])


def _arm(cells):
    programs = [row for cell in cells for row in cell['rows']]
    j = sum(1 for cell in cells if _cell_counts(cell))
    tie_programs = [row for row in programs if row['near_tie_size'] >= 2]
    disagreements = sorted(row['near_tie_disagreement'] for row in tie_programs)
    identifiability = sorted(row['identifiability'] for row in programs)
    return {
        'cells': len(cells),
        'programs': len(programs),
        'j': j,
        'reading': 'EVIDENTIAL' if j > J_BAND else 'SATURATED',
        'programs_with_nontrivial_near_tie': len(tie_programs),
        'cells_with_any_nontrivial_near_tie': sum(1 for c in cells
                                                  if any(r['near_tie_size'] >= 2 for r in c['rows'])),
        'max_near_tie_disagreement': disagreements[-1] if disagreements else None,
        'median_identifiability': identifiability[len(identifiability) // 2] if identifiability else None,
        'min_identifiability': identifiability[0] if identifiability else None,
        # Upper bound on j(eps): j requires the tie to exist (Amendment B).
        'j_upper_bound_by_eps': {
            f'{eps:g}': sum(1 for c in cells if any(r['identifiability'] <= eps for r in c['rows']))
            for eps in EPS_CURVE},
        'programs_with_rival_by_eps': {
            f'{eps:g}': sum(1 for r in programs if r['identifiability'] <= eps)
            for eps in EPS_CURVE},
    }


def subtriage(report=Path('reports/sg0_full_v2.json')):
    report = Path(report)
    data = json.loads(report.read_text())
    protocol = data['protocol']

    problems = []
    if not data.get('complete'):
        problems.append('report is not complete')
    if protocol.get('label') != 'full':
        problems.append(f"sub-triage is defined over the full grid; label is {protocol.get('label')!r}")
    if abs(protocol.get('near_tie_eps', 0.0) - REGISTERED_EPS) > 1e-15:
        problems.append(f"runner near_tie_eps {protocol.get('near_tie_eps')} is not {REGISTERED_EPS}")

    cells = list(data['cells'].values())
    staged = [c for c in cells if c['staged']]
    control = [c for c in cells if not c['staged']]
    if len(staged) != STAGED_CELLS_IN_FULL_GRID:
        problems.append(f'expected {STAGED_CELLS_IN_FULL_GRID} staged cells, found {len(staged)}')

    staged_arm, control_arm = _arm(staged), _arm(control)

    # Amendment C: the borrowed floor is only harmless while staged near-tie
    # sets are empty. Refuse rather than read if that ever stops being true.
    if staged_arm['programs_with_nontrivial_near_tie'] and staged_arm['j'] == 0:
        problems.append('staged near-tie sets are non-empty: the query-spread floor must be '
                        'derived before this sub-triage may be read (amendment C)')

    # Amendment D: non-vacuity. Controls must detect the ambiguity they contain.
    non_vacuity = control_arm['reading'] == 'EVIDENTIAL'
    if not non_vacuity:
        problems.append('non-vacuity failed: the control arm does not read EVIDENTIAL, '
                        'so the statistic is withdrawn rather than read')

    verdict = f"NO-HEADROOM-{staged_arm['reading']}" if not problems else 'UNREADABLE'
    return {
        'source_report': report.as_posix(),
        'source_protocol_sha256': data['protocol_sha256'],
        'amendment': 'SG0_SUBTRIAGE_AMENDMENT.md',
        'status': 'DIAGNOSTIC; not preregistered for this run; decides ladder decision 6 only',
        'registered_eps': REGISTERED_EPS,
        'eps_scope': EPS_SCOPE,
        'j_band': J_BAND,
        'valid': not problems,
        'problems': problems,
        'non_vacuity_passes': non_vacuity,
        'verdict': verdict,
        'staged': staged_arm,
        'control': control_arm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('reports/sg0_full_v2.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/sg0_subtriage.json'))
    args = parser.parse_args()
    result = subtriage(args.report)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
