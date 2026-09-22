"""Tests for the SG6 opportunity gate.

Exercises the evaluator on fixtures that can only pass if the bimodality and
within-cluster clauses are implemented as written, plus the rank statistic on
the constant and tied inputs that caught the old `spearman` helper.
"""
import json
import tempfile
import unittest
from math import isnan
from pathlib import Path

from row.experiments import sg6_quality_gate as g


def cell(name, world, staged, quality, identifiability, depth=3, support=128):
    return {'name': name, 'world': world, 'staged': staged, 'depth': depth, 'support': support,
            'rows': [{'nmse_hat_qb': quality, 'identifiability': identifiability}]}


def report(cells):
    return {'complete': True, 'protocol_sha256': 'abc', 'protocol': {'label': 'full'},
            'cells': {f'{c["name"]}_w{c["world"]}_d{c["depth"]}_s{c["support"]}': c for c in cells}}


def gated(cells):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'r.json'
        path.write_text(json.dumps(report(cells)), encoding='utf-8')
        return g.gate(path)


class TestSpearman(unittest.TestCase):
    def test_constant_input_is_nan_not_one(self):
        self.assertTrue(isnan(g.spearman([1, 1, 1], [1, 2, 3])))
        self.assertTrue(isnan(g.spearman([1, 2, 3], [5, 5, 5])))

    def test_ties_use_average_ranks(self):
        self.assertAlmostEqual(g.spearman([1, 2, 2, 3], [1, 2, 2, 3]), 1.0)

    def test_perfect_inversion(self):
        self.assertAlmostEqual(g.spearman([1, 2, 3], [3, 2, 1]), -1.0)


class TestGate(unittest.TestCase):
    def test_bimodal_without_within_cluster_relation_is_not_measurable(self):
        cells = [cell('S', w, True, 0.005 + 0.001 * w, 100 - 10 * w) for w in range(3)]
        cells += [cell('C', w, False, 1.26 + 0.01 * w, 0.01 + 0.001 * w) for w in range(3)]
        result = gated(cells)
        self.assertTrue(result['bimodal'])
        self.assertEqual(result['verdict'], 'NOT-MEASURABLE-HERE')
        self.assertGreater(result['gap_over_spread'], g.BIMODALITY_RATIO)

    def test_graded_axis_with_within_cluster_relation_passes(self):
        """Worse quality (higher nmse) with lower identifiability, inside both clusters."""
        cells = [cell('S', w, True, 0.10 + 0.30 * w, 100 - 30 * w) for w in range(3)]
        cells += [cell('C', w, False, 0.90 + 0.30 * w, 20 - 5 * w) for w in range(3)]
        result = gated(cells)
        self.assertFalse(result['bimodal'])
        self.assertEqual(result['verdict'], 'AXIS-PRESENT')

    def test_wrong_sign_within_cluster_does_not_pass(self):
        cells = [cell('S', w, True, 0.10 + 0.30 * w, 10 + 30 * w) for w in range(3)]
        cells += [cell('C', w, False, 0.90 + 0.30 * w, 20 + 5 * w) for w in range(3)]
        self.assertNotEqual(gated(cells)['verdict'], 'AXIS-PRESENT')

    def test_only_the_anchor_cells_are_read(self):
        cells = [cell('S', w, True, 0.005, 100) for w in range(3)]
        cells += [cell('C', w, False, 1.26, 0.01) for w in range(3)]
        cells += [cell('S', w, True, 99.0, 99.0, depth=4, support=2) for w in range(3)]
        result = gated(cells)
        self.assertEqual(result['n_libraries'], 6)

    def test_partial_report_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'r.json'
            data = report([cell('S', 0, True, 0.005, 100)])
            data['protocol']['label'] = 'dry-run'
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                g.gate(path)


class TestAgainstCommittedReport(unittest.TestCase):
    REPORT = Path('reports/sg0_full_v2.json')

    @unittest.skipUnless(REPORT.exists(), 'committed SG0 grid not present')
    def test_held_artifacts_have_no_graded_quality_axis(self):
        result = g.gate(self.REPORT)
        self.assertEqual(result['n_libraries'], 12)
        self.assertEqual(result['verdict'], 'NOT-MEASURABLE-HERE')
        self.assertTrue(result['bimodal'])
        # Pooled looks supportive; within-cluster does not survive. That gap is
        # the whole finding, so both halves are asserted.
        self.assertLess(result['pooled_rho'], -0.5)
        for arm in ('staged', 'control'):
            self.assertGreater(result['within_rho'][arm], -g.MIN_WITHIN_RHO, arm)


if __name__ == '__main__':
    unittest.main()
