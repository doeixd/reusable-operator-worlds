"""Tests for the necessity gate.

As with the adequacy checks, the load-bearing half is the historical
reconstructions: each check must FIRE on the ROW failure it claims to catch,
using that failure's own committed numbers where they exist.
"""
import unittest

from row import necessity_gate as ng


class TestRefusalCost(unittest.TestCase):
    def test_a_costly_refusal_passes(self):
        result = ng.refusal_cost(oracle_loss=100.0, denied_loss=160.0)
        self.assertTrue(result['passes'])
        self.assertAlmostEqual(result['detail']['cost'], 60.0)

    def test_a_free_refusal_fails(self):
        result = ng.refusal_cost(oracle_loss=100.0, denied_loss=100.5)
        self.assertFalse(result['passes'])
        self.assertIn('free at this scale', result['detail']['reason'])

    def test_scale_is_the_contribution_not_the_total(self):
        """V4.1 divided by total output variance while the object contributed ~0.2%."""
        contribution = 2.0
        result = ng.refusal_cost(oracle_loss=100.0, denied_loss=101.0, scale=contribution)
        self.assertTrue(result['passes'])
        # Against the total instead, the same refusal reads as free.
        self.assertFalse(ng.refusal_cost(100.0, 101.0, scale=1000.0)['passes'])

    def test_h48b_the_denied_learner_beat_the_oracle(self):
        """Review 68: label-free learner ~500 nats BETTER than the told oracle."""
        result = ng.refusal_cost(oracle_loss=0.0, denied_loss=-500.0, scale=500.0)
        self.assertFalse(result['passes'])
        self.assertIn('harmful', result['detail']['reason'])
        self.assertLess(result['detail']['cost'], 0)


class TestImpostor(unittest.TestCase):
    def test_target_must_beat_the_cheapest_simpler_construct(self):
        self.assertTrue(ng.no_cheaper_impostor(50.0, {'COMPRESS': 100.0})['passes'])

    def test_e6_macro_matches_the_loop(self):
        """A trace-compressing macro is the expected impostor for a loop."""
        result = ng.no_cheaper_impostor(98.0, {'MACRO': 100.0, 'KEEP': 140.0})
        self.assertFalse(result['passes'])
        self.assertEqual(result['detail']['best_impostor'], 'MACRO')
        self.assertIn('elicits it instead', result['detail']['reason'])

    def test_v4_2_factorization_lost_to_compress_at_equal_bits(self):
        result = ng.no_cheaper_impostor(target_score=104.0, impostors={'COMPRESS': 100.0})
        self.assertFalse(result['passes'])

    def test_no_impostor_evaluated_is_a_failure_not_a_pass(self):
        self.assertFalse(ng.no_cheaper_impostor(1.0, {})['passes'])


class TestIncumbent(unittest.TestCase):
    def test_e5_1_search_does_not_degrade(self):
        """Space grew 3.58e7-fold; route-optimization seconds grew 3.30-fold.

        A 3.3x slowdown reads as degradation until it is put against the axis:
        the exponent is 0.068, effectively flat.
        """
        result = ng.incumbent_degrades(costs=[1.0, 3.30], axis=[1.0, 3.58e7])
        self.assertFalse(result['passes'])
        self.assertAlmostEqual(result['detail']['scaling_exponent'], 0.0686, places=3)
        self.assertIn('buys no difficulty', result['detail']['reason'])

    def test_a_ratio_that_looks_like_degradation_is_not_one(self):
        """The same 3.3x cost growth passes when the axis grows only 10x."""
        self.assertTrue(ng.incumbent_degrades(costs=[1.0, 3.30], axis=[1.0, 10.0])['passes'])

    def test_exhaustive_enumeration_scales_with_the_space(self):
        """ENUM/OPT seconds went 0.04x, 0.42x, 4.66x: enumeration does degrade."""
        result = ng.incumbent_degrades(costs=[0.04, 4.66], axis=[1.0, 100.0])
        self.assertTrue(result['passes'])
        self.assertGreater(result['detail']['scaling_exponent'], 1.0)

    def test_mismatched_series_are_refused(self):
        with self.assertRaises(ValueError):
            ng.incumbent_degrades([1.0], [1.0])
        with self.assertRaises(ValueError):
            ng.incumbent_degrades([1.0, 2.0], [5.0, 5.0])


class TestDifficultyBand(unittest.TestCase):
    def test_in_band_passes(self):
        self.assertTrue(ng.difficulty_band([0.4, 0.5, 0.6], (0.1, 0.9))['passes'])

    def test_sg0_route_identification_is_too_easy(self):
        """Staged median identifiability 53.9: the runner-up is ~54x worse.

        Identifiability is HIGHER-IS-EASIER, so the flag must be set or the
        reading inverts - which is exactly what this check got wrong first.
        """
        result = ng.difficulty_band([53.9], band=(0.001, 5.0), higher_is_harder=False)
        self.assertFalse(result['passes'])
        self.assertEqual(result['detail']['state'], 'TOO_EASY')
        self.assertIn('cannot elicit', result['detail']['reason'])

    def test_the_control_arm_is_in_band_on_the_same_scale(self):
        """Control median identifiability 0.025: ambiguous, and readable."""
        result = ng.difficulty_band([0.025], band=(0.001, 5.0), higher_is_harder=False)
        self.assertTrue(result['passes'])

    def test_direction_flag_is_not_cosmetic(self):
        """The same score and band read oppositely under the two directions."""
        high = ng.difficulty_band([53.9], band=(0.001, 5.0), higher_is_harder=True)
        low = ng.difficulty_band([53.9], band=(0.001, 5.0), higher_is_harder=False)
        self.assertEqual(high['detail']['state'], 'UNINTERPRETABLE')
        self.assertEqual(low['detail']['state'], 'TOO_EASY')

    def test_too_easy_end(self):
        result = ng.difficulty_band([0.01, 0.02], band=(0.1, 0.9))
        self.assertEqual(result['detail']['state'], 'TOO_EASY')
        self.assertIn('cannot elicit', result['detail']['reason'])

    def test_unreachable_end_is_uninterpretable_not_negative(self):
        result = ng.difficulty_band([5.0], band=(0.1, 0.9))
        self.assertEqual(result['detail']['state'], 'UNINTERPRETABLE')
        self.assertIn('uninterpretable rather than evidence', result['detail']['reason'])

    def test_empty_band_is_refused(self):
        with self.assertRaises(ValueError):
            ng.difficulty_band([1.0], band=(0.9, 0.1))


class TestCapacity(unittest.TestCase):
    def test_review_68_one_channel_absorbs_two_subspaces(self):
        """1x64 matched or beat 2x32; the learner ignored true orthogonal groups."""
        result = ng.capacity_forces_structure(monolithic_score=100.0, structured_score=100.5)
        self.assertFalse(result['passes'])
        self.assertIn('not necessary at this capacity', result['detail']['reason'])

    def test_a_capacity_where_splitting_wins_passes(self):
        self.assertTrue(ng.capacity_forces_structure(100.0, 60.0)['passes'])


class TestAudit(unittest.TestCase):
    def test_audit_reports_every_failure(self):
        result = ng.audit([
            ng.refusal_cost(100.0, 100.1),
            ng.no_cheaper_impostor(98.0, {'MACRO': 100.0}),
            ng.incumbent_degrades([1.0, 40.0], [1.0, 100.0]),
        ])
        self.assertFalse(result['elicits'])
        self.assertEqual(result['failed'], ['refusal_cost', 'no_cheaper_impostor'])

    def test_a_task_that_elicits_passes_every_check(self):
        result = ng.audit([
            ng.refusal_cost(100.0, 160.0),
            ng.no_cheaper_impostor(50.0, {'COMPRESS': 100.0}),
            ng.incumbent_degrades([1.0, 40.0], [1.0, 100.0]),
            ng.difficulty_band([0.5], (0.1, 0.9)),
            ng.capacity_forces_structure(100.0, 60.0),
        ])
        self.assertTrue(result['elicits'])


if __name__ == '__main__':
    unittest.main()
