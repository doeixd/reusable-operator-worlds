"""Smoke checks for the provisional H28 learner/core opportunity harness."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('h28_learner_opportunity',
    Path(__file__).resolve().parents[1] / 'tools/h28_learner_opportunity.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class LearnerOpportunityTests(unittest.TestCase):
    def test_smoke_is_explicitly_incomplete(self):
        result = pilot.fit(smoke=True)
        self.assertEqual(result['arms_present'], ['SHARED_CORE_ADAPTER', 'SHARED_NO_ADAPTER'])
        self.assertFalse(result['oracle_anchor_present'])
        self.assertFalse(result['independent_control_present'])
        self.assertFalse(result['random_core_control_present'])
        self.assertFalse(result['query_used_for_fit'])
        self.assertFalse(result['core_changed_after_context2'])

    def test_shared_adapter_beats_no_adapter_in_smoke(self):
        result = pilot.fit(smoke=True)
        for context in ('context_1', 'context_2'):
            rows = [r for r in result['rows'] if r['context'] == context]
            fitted = sum(r['query_canonical_nmse'] for r in rows) / len(rows)
            baseline = sum(r['no_adapter_canonical_nmse'] for r in rows) / len(rows)
            self.assertLess(fitted, baseline / 10)


if __name__ == '__main__':
    unittest.main()
