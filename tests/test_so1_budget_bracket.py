import unittest

from row.experiments.audit_so1_budget_bracket import (
    ANCHORS,
    BATCHES,
    GRADIENT_LEVELS,
    ORACLE_CELL_INDEX,
    STAGE2_CELL_INDEX,
    anchor_check,
    cell_key,
    checkpoints_for,
    classify,
    envelope,
    monotone,
    paired_differences,
    persistence,
)


def oracle_cell(median, passes, persistence_label="not crossed"):
    return {"terminal_median": median, "passes": passes, "persistence": persistence_label}


def curve(batch, medians, passes, labels=None):
    cells = {}
    for g, m, p in zip(GRADIENT_LEVELS, medians, passes):
        label = "not crossed"
        if labels:
            label = labels[GRADIENT_LEVELS.index(g)]
        cells[cell_key(batch, g, True)] = {str(w): oracle_cell(m, p, label) for w in (0, 1, 2)}
    return cells


class SO1BracketTests(unittest.TestCase):
    def test_sampling_streams_are_unique_and_disjoint_from_stage_d(self):
        indices = list(ORACLE_CELL_INDEX.values()) + list(STAGE2_CELL_INDEX.values())
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(len(ORACLE_CELL_INDEX), len(BATCHES) * len(GRADIENT_LEVELS))
        # Stage D used cell indices 0-2.
        self.assertTrue(all(i >= 100 for i in indices))

    def test_grid_holds_one_axis_fixed_in_each_direction(self):
        for batch in BATCHES:
            updates = {g: g // batch for g in GRADIENT_LEVELS}
            self.assertEqual(len(set(updates.values())), len(GRADIENT_LEVELS))
        for g in GRADIENT_LEVELS:
            # Equal gradients, 32x difference in updates and in diversity.
            self.assertEqual(g // 2 // (g // 64), 32)

    def test_checkpoints_are_five_fractions_of_the_cell_budget(self):
        self.assertEqual(checkpoints_for(8192), (1024, 2048, 4096, 6144, 8192))
        # Degenerate small budgets collapse duplicates rather than repeating.
        self.assertEqual(checkpoints_for(16), (2, 4, 8, 12, 16))

    def test_persistence_needs_two_later_sub_threshold_checkpoints(self):
        self.assertEqual(persistence({"1": 2.0, "2": 1.0}), (None, "not crossed"))
        self.assertEqual(
            persistence({"1": 2.0, "2": 0.04, "3": 0.03, "4": 0.02}),
            (2, "persistent"),
        )
        self.assertEqual(
            persistence({"1": 2.0, "2": 0.04, "3": 0.9, "4": 0.8}),
            (2, "crossed, not persistent"),
        )
        self.assertEqual(
            persistence({"1": 2.0, "2": 1.0, "3": 0.04}),
            (3, "crossed, persistence unobservable"),
        )
        self.assertTrue(monotone({"1": 2.0, "2": 1.0, "3": 0.5}))
        self.assertFalse(monotone({"1": 2.0, "2": 0.5, "3": 1.0}))

    def test_envelope_reports_both_persistence_readings(self):
        cells = curve(
            64,
            [1.0, 0.9, 0.8, 0.3, 0.004],
            [False, False, False, False, True],
            ["not crossed"] * 4 + ["crossed, persistence unobservable"],
        )
        result = envelope(cells, 64)
        self.assertEqual(result["lowest_passing"], 262144)
        self.assertIsNone(result["lowest_passing_excluding_unobservable"])
        cells = curve(
            2,
            [1.0, 0.9, 0.02, 0.01, 0.005],
            [False, False, True, True, True],
            ["not crossed", "not crossed", "persistent", "persistent", "persistent"],
        )
        result = envelope(cells, 2)
        self.assertEqual(result["lowest_passing"], 65536)
        self.assertEqual(result["lowest_passing_excluding_unobservable"], 65536)

    def test_envelope_ignores_incomplete_cells(self):
        cells = curve(2, [0.01] * 5, [True] * 5)
        del cells[cell_key(2, 16384, True)]["2"]
        self.assertEqual(envelope(cells, 2)["lowest_passing"], 32768)

    def test_paired_difference_is_b2_minus_b64_per_world(self):
        cells = {}
        cells.update(curve(2, [0.5] * 5, [False] * 5))
        cells.update(curve(64, [0.8] * 5, [False] * 5))
        differences = paired_differences(cells)
        self.assertAlmostEqual(differences["16384"]["0"], -0.3)
        self.assertEqual(set(differences), {str(g) for g in GRADIENT_LEVELS})

    def test_anchor_requires_tolerance_and_matching_verdict(self):
        stage_d = {"cells": {
            "C_lo": {str(w): {"terminal_median": 0.9, "passes": False} for w in (0, 1, 2)},
            "C_hi": {"0": {"terminal_median": 0.72, "passes": False},
                     "1": {"terminal_median": 0.0063, "passes": True},
                     "2": {"terminal_median": 0.0054, "passes": True}},
        }}
        cells = {
            cell_key(2, 16384, True): {str(w): oracle_cell(0.905, False) for w in (0, 1, 2)},
            cell_key(64, 262144, True): {"0": oracle_cell(0.73, False),
                                         "1": oracle_cell(0.0065, True),
                                         "2": oracle_cell(0.0051, True)},
        }
        self.assertTrue(all(a["passes"] for a in anchor_check(cells, stage_d).values()))
        # A verdict flip fails even inside the numeric tolerance.
        cells[cell_key(64, 262144, True)]["1"] = oracle_cell(0.0063, False)
        self.assertFalse(anchor_check(cells, stage_d)[cell_key(64, 262144, True)]["passes"])
        # A large numeric drift fails even with matching verdicts.
        cells[cell_key(64, 262144, True)]["1"] = oracle_cell(0.3, True)
        self.assertFalse(anchor_check(cells, stage_d)[cell_key(64, 262144, True)]["passes"])
        # Both anchors are Stage D corners.
        self.assertEqual(set(ANCHORS.values()), {"C_lo", "C_hi"})

    def test_decision_ladder(self):
        self.assertEqual(classify(False, None), "NO_ORACLE_CELL_PASSES")
        self.assertEqual(classify(True, None), "ORACLE_PASSES_STAGE2_PENDING")
        self.assertEqual(classify(True, False), "ORACLE_PASSES_LEARNED_FAILS")
        self.assertEqual(classify(True, True), "ORACLE_AND_LEARNED_PASS")


if __name__ == "__main__":
    unittest.main()
