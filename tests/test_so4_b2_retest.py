import math
import unittest

from row.experiments import audit_so4_b2_retest as so4

GATES_OK = {"G0_replay_seed_neutral": True, "G1_prefix_sharing": True, "G2_streams_distinct": True,
            "G3_anchor": True, "G4_transfer": True}


def medians(rows):
    """rows: {world: [s0, s1, s2]}"""
    return {w: dict(enumerate(v)) for w, v in rows.items()}


def flat(value):
    return {w: [value] * 3 for w in so4.WORLDS}


def margins(value):
    return {w: value for w in so4.WORLDS}


class LadderTests(unittest.TestCase):
    def test_passes(self):
        self.assertEqual(so4.classify(medians(flat(0.02)), margins(2.0), GATES_OK)["program"], "SO4_PASSES")

    def test_three_of_four_worlds_suffice(self):
        rows = flat(0.02) | {9: [0.2, 0.2, 0.2]}
        self.assertEqual(so4.classify(medians(rows), margins(2.0), GATES_OK)["program"], "SO4_PASSES")
        rows = flat(0.02) | {8: [0.2] * 3, 9: [0.2] * 3}
        self.assertEqual(so4.classify(medians(rows), margins(2.0), GATES_OK)["program"], "SO4_FAILS")

    def test_original_sub_clause_was_redundant(self):
        # The defect Amendment 1 fixes: with three streams, median <= t <=> at least 2 of 3 <= t.
        for rows in ([0.01, 0.02, 0.9], [0.9, 0.04, 0.05], [0.049, 0.051, 0.001], [0.06, 0.07, 0.01]):
            self.assertEqual(sorted(rows)[1] <= so4.THRESHOLD, sum(v <= so4.THRESHOLD for v in rows) >= 2)

    def test_stream_fragile_can_fire(self):
        # Median passes everywhere, but world 7 has one collapsed stream (0.12 > 0.10).
        rows = flat(0.02) | {7: [0.01, 0.02, 0.12]}
        result = so4.classify(medians(rows), margins(2.0), GATES_OK)
        self.assertEqual(result["program"], "SO4_STREAM_FRAGILE")
        self.assertFalse(result["stream_robust"][7])
        # SO3-BASE-like spread (worst stream 0.066) stays robust.
        rows = flat(0.02) | {7: [0.009, 0.015, 0.066]}
        self.assertEqual(so4.classify(medians(rows), margins(2.0), GATES_OK)["program"], "SO4_PASSES")

    def test_collapse_in_a_non_passing_world_does_not_make_it_fragile(self):
        rows = flat(0.02) | {9: [0.2, 0.3, 0.9]}
        self.assertEqual(so4.classify(medians(rows), margins(2.0), GATES_OK)["program"], "SO4_PASSES")

    def test_acquires_only_when_margin_misses(self):
        m = margins(2.0) | {7: 0.1, 8: 0.2}
        self.assertEqual(so4.classify(medians(flat(0.02)), m, GATES_OK)["program"], "SO4_ACQUIRES_ONLY")

    def test_harness_failed(self):
        self.assertEqual(so4.classify(medians(flat(0.02)), margins(2.0), GATES_OK | {"G3_anchor": False})["program"],
                         "HARNESS_FAILED")
        rows = medians(flat(0.02))
        del rows[9][2]
        self.assertEqual(so4.classify(rows, margins(2.0), GATES_OK)["program"], "HARNESS_FAILED")


class ConstructionTests(unittest.TestCase):
    def test_stream_seeds_and_constants(self):
        self.assertIsNone(so4.replay_seed_for(6, 0))
        self.assertEqual(len({so4.replay_seed_for(w, s) for w in so4.WORLDS for s in (1, 2)}), 8)
        self.assertEqual((so4.MODEL_SEED, so4.STREAM_ROOT, so4.HELD_OUT, so4.MARGIN_STREAM), (7000, 7400, 12, 0))
        cfg, _, _, _ = so4.stage_setup(6, 3, so4.MODEL_SEED)
        self.assertEqual(cfg.discrete_model.seed, 7000)

    def test_margin_is_natural_log_of_geometric_means(self):
        pairs = [{"trained": 0.01, "scratch": 0.1}, {"trained": 0.04, "scratch": 0.4}]
        self.assertAlmostEqual(so4.margin_from_pairs(pairs), math.log(10), places=12)


if __name__ == "__main__":
    unittest.main()
