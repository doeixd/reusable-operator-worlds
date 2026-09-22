"""Tests for the SG0 NO-HEADROOM sub-triage (`SG0_SUBTRIAGE_AMENDMENT.md`).

Exercises the EVALUATOR, not its arithmetic: every branch of the amended rule
gets a fixture that can only pass if the rule is implemented as written. The
depth-five lesson is that a gate's tests must run its evaluator.
"""
import json
import tempfile
import unittest
from pathlib import Path

from row.experiments import sg0_subtriage as st


def row(task, near_tie_size=1, disagreement=0.0, floor=0.0, identifiability=10.0):
    return {'task': task, 'near_tie_size': near_tie_size,
            'near_tie_disagreement': disagreement, 'null_floor': floor,
            'identifiability': identifiability}


def cell(name, staged, rows):
    return {'name': name, 'staged': staged, 'rows': rows}


def report(staged_cells, control_cells, label='full', complete=True, eps=st.REGISTERED_EPS):
    cells = {}
    for i, c in enumerate(staged_cells):
        cells[f'S{i}'] = c
    for i, c in enumerate(control_cells):
        cells[f'C{i}'] = c
    return {'complete': complete, 'protocol_sha256': 'deadbeef',
            'protocol': {'label': label, 'near_tie_eps': eps}, 'cells': cells}


def scored(data):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'r.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        return st.subtriage(path)


def singleton_cells(n, staged, identifiability=10.0):
    return [cell(f'{staged}{i}', staged, [row(t, identifiability=identifiability) for t in range(4)])
            for i in range(n)]


def ambiguous_cells(n, staged):
    """Cells whose first program has a rival whose query spread beats its floor."""
    out = []
    for i in range(n):
        rows = [row(0, near_tie_size=2, disagreement=0.5, floor=0.1, identifiability=0.005)]
        rows += [row(t) for t in range(1, 4)]
        out.append(cell(f'{staged}{i}', staged, rows))
    return out


class TestSubtriage(unittest.TestCase):
    def test_saturated_when_staged_has_no_near_ties(self):
        result = scored(report(singleton_cells(36, True), ambiguous_cells(36, False)))
        self.assertTrue(result['valid'], result['problems'])
        self.assertEqual(result['verdict'], 'NO-HEADROOM-SATURATED')
        self.assertEqual(result['staged']['j'], 0)
        self.assertTrue(result['non_vacuity_passes'])

    def test_evidential_when_staged_cells_clear_the_band(self):
        staged = ambiguous_cells(3, True) + singleton_cells(33, True)
        result = scored(report(staged, ambiguous_cells(36, False)))
        self.assertEqual(result['staged']['j'], 3)
        self.assertEqual(result['verdict'], 'NO-HEADROOM-EVIDENTIAL')

    def test_band_boundary_two_cells_is_still_saturated(self):
        staged = ambiguous_cells(2, True) + singleton_cells(34, True)
        result = scored(report(staged, ambiguous_cells(36, False)))
        self.assertEqual(result['staged']['j'], st.J_BAND)
        self.assertEqual(result['verdict'], 'NO-HEADROOM-SATURATED')

    def test_non_vacuity_fails_when_controls_are_silent(self):
        result = scored(report(singleton_cells(36, True), singleton_cells(36, False)))
        self.assertFalse(result['valid'])
        self.assertFalse(result['non_vacuity_passes'])
        self.assertEqual(result['verdict'], 'UNREADABLE')

    def test_spread_clause_requires_the_floor_to_be_exceeded(self):
        """A rival that does not beat its floor must not make a cell count."""
        rows = [row(0, near_tie_size=2, disagreement=0.05, floor=0.10, identifiability=0.005)]
        staged = [cell('S', True, rows)] + singleton_cells(35, True)
        result = scored(report(staged, ambiguous_cells(36, False)))
        self.assertEqual(result['staged']['j'], 0)
        # ...and amendment C must refuse, because the staged near-tie set is non-empty.
        self.assertFalse(result['valid'])
        self.assertTrue(any('amendment c' in p.lower() for p in result['problems']), result['problems'])

    def test_eps_curve_is_an_upper_bound_on_j(self):
        staged = singleton_cells(36, True, identifiability=0.07)
        result = scored(report(staged, ambiguous_cells(36, False)))
        bound = result['staged']['j_upper_bound_by_eps']
        self.assertEqual(bound['0.01'], 0)
        self.assertEqual(bound['0.05'], 0)
        self.assertEqual(bound['0.1'], 36)
        self.assertGreaterEqual(bound['0.1'], result['staged']['j'])

    def test_partial_or_incomplete_reports_are_refused(self):
        for kwargs in ({'label': 'dry-run'}, {'complete': False}, {'eps': 0.05}):
            data = report(singleton_cells(36, True), ambiguous_cells(36, False), **kwargs)
            self.assertFalse(scored(data)['valid'], kwargs)

    def test_wrong_staged_cell_count_is_refused(self):
        result = scored(report(singleton_cells(35, True), ambiguous_cells(36, False)))
        self.assertFalse(result['valid'])


class TestAgainstCommittedReport(unittest.TestCase):
    REPORT = Path('reports/sg0_full_v2.json')

    @unittest.skipUnless(REPORT.exists(), 'committed SG0 grid not present')
    def test_committed_grid_reads_saturated(self):
        result = st.subtriage(self.REPORT)
        self.assertTrue(result['valid'], result['problems'])
        self.assertEqual(result['verdict'], 'NO-HEADROOM-SATURATED')
        self.assertEqual(result['staged']['j'], 0)
        self.assertEqual(result['staged']['programs_with_nontrivial_near_tie'], 0)
        self.assertEqual(result['control']['reading'], 'EVIDENTIAL')
        # The admissibility claim: inside the band across a tenfold widening.
        bound = result['staged']['j_upper_bound_by_eps']
        for eps in ('0.01', '0.05', '0.1'):
            self.assertLessEqual(bound[eps], st.J_BAND, eps)


if __name__ == '__main__':
    unittest.main()
