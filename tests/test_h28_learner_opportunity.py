"""Smoke checks for the provisional H28 learner/core opportunity harness."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('h28_learner_opportunity',
    Path(__file__).resolve().parents[1] / 'tools/h28_learner_opportunity.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class LearnerOpportunityTests(unittest.TestCase):
    def test_smoke_has_registered_controls_but_is_not_a_verdict(self):
        result = pilot.fit(smoke=True)
        self.assertEqual(len(result['arms_present']), 5)
        self.assertTrue(result['oracle_anchor_present'])
        self.assertTrue(result['independent_control_present'])
        self.assertTrue(result['random_core_control_present'])
        self.assertTrue(result['non_vacuity_pass'])
        self.assertTrue(result['reconstruction_pass'])
        self.assertTrue(result['paired_cost_pass'])
        self.assertFalse(result['economic_value_measured'])
        self.assertFalse(result['query_used_for_fit'])
        self.assertFalse(result['core_changed_after_context2'])

    def test_controls_are_non_vacuous_and_finite(self):
        result = pilot.fit(smoke=True)
        oracle = result['control_rows']['ORACLE_CORE_ADAPTER']
        random = result['control_rows']['RANDOM_CORE_ADAPTER']
        independent = result['control_rows']['INDEPENDENT']
        oracle_mean = sum(r['query_canonical_nmse'] for r in oracle) / len(oracle)
        random_mean = sum(r['query_canonical_nmse'] for r in random) / len(random)
        self.assertLess(oracle_mean, 1e-8)
        self.assertGreater(random_mean, oracle_mean * 100)
        self.assertTrue(all(r['query_canonical_nmse'] >= 0 for r in independent))

    def test_shared_adapter_beats_no_adapter_in_smoke(self):
        result = pilot.fit(smoke=True)
        for context in ('context_1', 'context_2'):
            rows = [r for r in result['rows'] if r['context'] == context]
            fitted = sum(r['query_canonical_nmse'] for r in rows) / len(rows)
            baseline = sum(r['no_adapter_canonical_nmse'] for r in rows) / len(rows)
            self.assertLess(fitted, baseline / 10)


if __name__ == '__main__':
    unittest.main()
