"""Tests for the design-adequacy checks.

The important half is `TestHistoricalFailures`: each check is run against a
reconstruction of the ROW defect it claims to catch, and must FIRE. A checker
that never fires is not being run - the same rule the checks themselves apply
to experiments.
"""
import random
import unittest

from row import design_adequacy as da
from row.experiments.sg6_quality_gate import spearman


class TestDiscriminatingPower(unittest.TestCase):
    def test_a_separating_rule_passes(self):
        rule = lambda s: s > 1.0
        result = da.discriminating_power(rule, [0.0] * 100, [2.0] * 100)
        self.assertTrue(result['passes'])
        self.assertEqual(result['detail']['false_fire_rate'], 0.0)
        self.assertEqual(result['detail']['detection_rate'], 1.0)

    def test_a_rule_that_fires_under_the_null_fails(self):
        result = da.discriminating_power(lambda s: s > -100, [0.0] * 100, [2.0] * 100)
        self.assertFalse(result['passes'])
        self.assertIn('null worlds', result['detail']['reason'])

    def test_a_rule_that_cannot_detect_the_effect_fails(self):
        result = da.discriminating_power(lambda s: s > 1e6, [0.0] * 100, [2.0] * 100)
        self.assertFalse(result['passes'])
        self.assertIn('effect worlds', result['detail']['reason'])

    def test_both_sample_sets_are_required(self):
        with self.assertRaises(ValueError):
            da.discriminating_power(lambda s: True, [], [1.0])


class TestOtherChecks(unittest.TestCase):
    def test_graded_axis_accepts_a_spread_sample(self):
        result = da.graded_axis([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(result['passes'])
        self.assertEqual(result['detail']['effective_n'], 5)

    def test_partition_detects_a_hole_and_an_overlap(self):
        self.assertFalse(da.partitions_outcomes([('a', (0, 2)), ('b', (4, 10))], (0, 10))['passes'])
        self.assertFalse(da.partitions_outcomes([('a', (0, 5)), ('b', (3, 10))], (0, 10))['passes'])
        self.assertTrue(da.partitions_outcomes([('a', (0, 2)), ('b', (3, 10))], (0, 10))['passes'])

    def test_floor_must_be_built_for_its_statistic(self):
        self.assertTrue(da.floor_matches_statistic('regret', 'regret')['passes'])
        self.assertFalse(da.floor_matches_statistic('regret', 'near_tie_spread')['passes'])

    def test_audit_aggregates(self):
        result = da.audit([da.floor_matches_statistic('a', 'a'), da.floor_matches_statistic('a', 'b')])
        self.assertFalse(result['adequate'])
        self.assertEqual(result['failed'], ['floor_matches_statistic'])


class TestHistoricalFailures(unittest.TestCase):
    """Each check must fire on the defect it claims to catch."""

    def test_e5_1_first_crossing_always_names_a_horizon(self):
        """`min{D : X_D > tau}` on pure noise always returns a value.

        E5.1 registered it, depth 7 failed, 8-10 passed, and the rule printed
        SEARCH BINDS FIRST off a single cell.
        """
        rng = random.Random(51)
        def first_crossing_fires(series):
            return any(x > 0.5 for x in series)
        null = [[rng.gauss(0.0, 0.35) for _ in range(8)] for _ in range(400)]
        effect = [[rng.gauss(1.0, 0.35) for _ in range(8)] for _ in range(400)]
        result = da.discriminating_power(first_crossing_fires, null, effect)
        self.assertFalse(result['passes'])
        # Not tuned to a round number: what matters is that a pure-noise series
        # trips the rule far above the 5%% bound, which is E5.1's defect.
        self.assertGreater(result['detail']['false_fire_rate'], da.MAX_FALSE_FIRE * 4)

    def test_e6_macro_threshold_is_cleared_by_almost_everything(self):
        """`H* = L/(L-1)` is 1.5 uses at L=3, which nearly every pattern clears."""
        rng = random.Random(6)
        uses_null = [rng.choice([1, 2, 2, 3, 4]) for _ in range(400)]
        uses_effect = [rng.choice([8, 9, 10, 14]) for _ in range(400)]
        result = da.discriminating_power(lambda u: u >= 1.5, uses_null, uses_effect)
        self.assertFalse(result['passes'])
        self.assertGreater(result['detail']['false_fire_rate'], 0.5)
        # The alphabet-tax threshold (5-14) discriminates, which is why it was adopted.
        self.assertTrue(da.discriminating_power(lambda u: u >= 7, uses_null, uses_effect)['passes'])

    def test_sg6_bimodal_quality_axis(self):
        """The real committed numbers: staged ~0.005, control ~1.26."""
        staged = [0.00465, 0.00516, 0.00531, 0.00639, 0.00648, 0.00725]
        control = [1.26012, 1.26589, 1.26681, 1.27661, 1.28023, 1.30157]
        result = da.graded_axis(staged + control)
        self.assertFalse(result['passes'])
        self.assertEqual(result['detail']['effective_n'], 2)
        self.assertGreater(result['detail']['gap_over_spread'], da.MAX_GAP_OVER_SPREAD)

    def test_sg6_pooled_relation_does_not_survive_within_cluster(self):
        quality = [0.00465, 0.00516, 0.00531, 0.00639, 0.00648, 0.00725,
                   1.26012, 1.26589, 1.26681, 1.27661, 1.28023, 1.30157]
        identifiability = [121.7, 117.7, 1.786, 2.653, 22.15, 141.1,
                           0.01109, 0.008884, 0.02794, 0.01653, 0.02432, 0.01254]
        clusters = ['staged'] * 6 + ['control'] * 6
        self.assertLess(spearman(quality, identifiability), -0.5)  # pooled looks supportive
        result = da.survives_within_cluster(quality, identifiability, clusters, spearman)
        self.assertFalse(result['passes'])

    def test_sealed_c2_denominator_resolves_to_neither(self):
        """"Within 15% in >= 4 of 6 cells" once a cell can be unobservable."""
        intervals = [('pass', (4, 6)), ('fail', (0, 2))]
        self.assertFalse(da.partitions_outcomes(intervals, (0, 6))['passes'])

    def test_s0_threshold_registered_without_a_baseline(self):
        self.assertFalse(da.baseline_registered(0.5, None)['passes'])
        # And a threshold the control already meets is a vacuous pass.
        self.assertFalse(da.baseline_registered(0.5, 0.6)['passes'])
        # S0's own numbers: the control does not already satisfy it, so THIS
        # check does not catch S0's real defect. Recorded honestly in the doc.
        self.assertTrue(da.baseline_registered(0.5, 0.25)['passes'])

    def test_sg0_borrowed_floor(self):
        result = da.floor_matches_statistic('regret', 'near_tie_query_spread')
        self.assertFalse(result['passes'])


class TestSG0WouldHavePassed(unittest.TestCase):
    """The check must not condemn a design that was in fact adequate."""

    def test_sg0_triage_partitions_and_discriminates(self):
        self.assertTrue(da.partitions_outcomes(
            [('NO-HEADROOM', (0, 2)), ('HEADROOM', (3, 36))], (0, 36))['passes'])
        rng = random.Random(0)
        # Null: cells whose median regret sits at the floor. Effect: cells above it.
        null = [[rng.uniform(-0.01, 0.01) for _ in range(36)] for _ in range(200)]
        effect = [[rng.uniform(0.05, 0.6) for _ in range(36)] for _ in range(200)]
        rule = lambda cells: sum(1 for c in cells if c > 0.02) >= 3
        self.assertTrue(da.discriminating_power(rule, null, effect)['passes'])


if __name__ == '__main__':
    unittest.main()
