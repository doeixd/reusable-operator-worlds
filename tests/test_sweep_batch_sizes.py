from __future__ import annotations

import unittest

from row.experiments.sweep_batch_sizes import _aggregate


class SweepBatchSizesTests(unittest.TestCase):
    def test_aggregate_pairs_models_and_sizes_within_world(self) -> None:
        records = []
        for batch_size in (2, 8):
            records.extend(
                [
                    {
                        "world_seed": 0,
                        "model": "continuous",
                        "batch_size": batch_size,
                        "gaussian_log_loss": -10.0 - batch_size,
                        "novel_32_shot_nmse": 0.01,
                    },
                    {
                        "world_seed": 0,
                        "model": "dense",
                        "batch_size": batch_size,
                        "gaussian_log_loss": -8.0 - batch_size,
                        "novel_32_shot_nmse": 0.02,
                    },
                ]
            )
        report = _aggregate(records)
        self.assertEqual(report["architecture_effects"][1]["continuous_lifetime_wins"], 1)
        self.assertEqual(
            report["batch_size_effects"][0]["batch8_lifetime_improves"], 1
        )


if __name__ == "__main__":
    unittest.main()
