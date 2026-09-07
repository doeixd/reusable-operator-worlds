import math
import unittest

from row.experiments.audit_so0_census import (
    axis_isolation,
    budget_row,
    expected_distinct_tasks,
    first_crossing,
)


class SO0CensusTests(unittest.TestCase):
    def test_expected_distinct_tasks_matches_plan_constants(self):
        self.assertAlmostEqual(expected_distinct_tasks(1), 1.0)
        self.assertAlmostEqual(expected_distinct_tasks(2), 1.984375)
        self.assertAlmostEqual(expected_distinct_tasks(64), 40.5, places=1)
        self.assertLess(expected_distinct_tasks(10_000), 64.0)

    def test_persistence_rule_needs_two_later_checkpoints(self):
        # Terminal-only crossing: unobservable.
        crossing, label, monotone = first_crossing(
            {"0": 2.0, "256": 1.2, "1024": 0.9, "4096": 0.006}
        )
        self.assertEqual(crossing, 4096)
        self.assertEqual(label, "crossed, persistence unobservable")
        self.assertTrue(monotone)
        # Crossing with two later sub-threshold checkpoints: persistent.
        crossing, label, _ = first_crossing(
            {"0": 2.0, "256": 0.04, "1024": 0.03, "4096": 0.02, "8192": 0.01}
        )
        self.assertEqual((crossing, label), (256, "persistent"))
        # A single excursion below threshold is a crossing, not persistence.
        crossing, label, monotone = first_crossing(
            {"0": 2.0, "256": 0.04, "1024": 0.3, "4096": 0.2, "8192": 0.1}
        )
        self.assertEqual((crossing, label), (256, "crossed, not persistent"))
        self.assertFalse(monotone)
        crossing, label, _ = first_crossing({"0": 2.0, "256": 1.0, "1024": 0.9})
        self.assertIsNone(crossing)
        self.assertEqual(label, "not crossed")

    def test_axis_isolation_flags_pairs_holding_two_axes_fixed(self):
        cells = {
            "C_hi": budget_row("C_hi", 4096, 64),
            "C_lo": budget_row("C_lo", 8192, 2),
        }
        result = axis_isolation(cells)
        self.assertEqual(len(result["pairs"]), 1)
        self.assertEqual(result["pairs"][0]["axes_held_fixed"], [])
        self.assertEqual(
            result["axes_not_yet_isolated_by_any_pair"],
            ["updates", "diversity", "example_gradients"],
        )
        # The axes have two degrees of freedom, so a pair controls an axis by
        # holding it fixed: C_mid shares batch (hence diversity) with C_hi and
        # updates with C_lo, giving two controlled axes; example-gradients is
        # still uncontrolled until an equal-gradient pair exists.
        cells["C_mid"] = budget_row("C_mid", 8192, 64)
        result = axis_isolation(cells)
        by_pair = {tuple(p["pair"]): p for p in result["pairs"]}
        self.assertEqual(by_pair[("C_hi", "C_mid")]["axes_held_fixed"], ["diversity"])
        self.assertEqual(by_pair[("C_lo", "C_mid")]["axes_held_fixed"], ["updates"])
        self.assertEqual(result["axes_controlled_by_some_pair"], ["diversity", "updates"])
        self.assertEqual(result["axes_not_yet_isolated_by_any_pair"], ["example_gradients"])
        cells["C_eq"] = budget_row("C_eq", 2048, 64)   # 131,072 gradients
        cells["C_eq2"] = budget_row("C_eq2", 65536, 2)  # 131,072 gradients
        result = axis_isolation(cells)
        self.assertEqual(result["axes_not_yet_isolated_by_any_pair"], [])

    def test_budget_row_example_gradients(self):
        row = budget_row("x", 8192, 2)
        self.assertEqual(row["example_gradients"], 16384)
        self.assertTrue(math.isclose(row["diversity"], expected_distinct_tasks(2)))


if __name__ == "__main__":
    unittest.main()
