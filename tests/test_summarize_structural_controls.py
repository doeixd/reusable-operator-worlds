from __future__ import annotations

import unittest

from row.experiments.summarize_structural_controls import _effect_summary


class SummarizeStructuralControlsTests(unittest.TestCase):
    def test_effect_summary_uses_positive_as_named_model_win(self) -> None:
        rows = [
            {"loss": 2.0, "novel": -0.1},
            {"loss": -1.0, "novel": 0.2},
            {"loss": 3.0, "novel": 0.3},
        ]
        summary = _effect_summary(rows, "loss", "novel")
        self.assertEqual(summary["loss_wins"], 2)
        self.assertEqual(summary["novel_32_shot_wins"], 2)
        self.assertAlmostEqual(summary["mean_gaussian_log_loss_advantage"], 4 / 3)


if __name__ == "__main__":
    unittest.main()
