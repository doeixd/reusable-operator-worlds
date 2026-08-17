from __future__ import annotations

import unittest

from row.experiments.sweep_checkpoints import CHECKPOINTS, _aggregate


class SweepCheckpointsTests(unittest.TestCase):
    def test_checkpoint_sequence_matches_plan(self) -> None:
        self.assertEqual(CHECKPOINTS, (8, 16, 32, 64))

    def test_aggregate_reports_learning_gain_and_pairing(self) -> None:
        records = [
            {
                "world_seed": 0,
                "model": "continuous",
                "checkpoint_32_shot_nmse": {"8": 0.04, "16": 0.03, "32": 0.02, "64": 0.01},
            },
            {
                "world_seed": 0,
                "model": "dense",
                "checkpoint_32_shot_nmse": {"8": 0.05, "16": 0.04, "32": 0.03, "64": 0.02},
            },
        ]
        report = _aggregate(records)
        self.assertEqual(report["checkpoint_summaries"][-1]["continuous_wins"], 1)
        self.assertAlmostEqual(report["learning_gain"]["continuous"]["mean_8_to_64_ratio"], 4.0)
        self.assertTrue(report["learning_gain"]["dense"]["improves_in_all_worlds"])


if __name__ == "__main__":
    unittest.main()
