from __future__ import annotations

import unittest

from row.experiments.summarize_robustness import _effect_summary


class SummarizeRobustnessTests(unittest.TestCase):
    def test_effect_summary_reports_world_wins_and_interval(self) -> None:
        result = _effect_summary([1.0, 2.0, -1.0], seed=1)
        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["worlds"], 3)
        self.assertAlmostEqual(result["median"], 1.0)
        self.assertEqual(len(result["bootstrap_95_percent_ci_of_mean"]), 2)


if __name__ == "__main__":
    unittest.main()
