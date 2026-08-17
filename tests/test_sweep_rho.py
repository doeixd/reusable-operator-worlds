from __future__ import annotations

import unittest

from row.experiments.sweep_rho import STAGE2_EXACT_REUSE_ROOTS, _aggregate, _rho_label


class SweepRhoTests(unittest.TestCase):
    def test_rho_label_is_path_safe(self) -> None:
        self.assertEqual(_rho_label(0.25), "0p25")
        self.assertEqual(_rho_label(1.0), "1")

    def test_exact_reuse_roots_match_selected_learning_rates(self) -> None:
        self.assertIn("global_3em03", str(STAGE2_EXACT_REUSE_ROOTS["continuous"]))
        self.assertIn("global_1em03", str(STAGE2_EXACT_REUSE_ROOTS["dense"]))

    def test_aggregate_pairs_models_within_world(self) -> None:
        records = [
            {
                "world_seed": 1,
                "configured_rho": 0.5,
                "model": "continuous",
                "measured_residual_correlation": 0.2,
                "gaussian_log_loss": -12.0,
                "novel_32_shot_nmse": 0.02,
            },
            {
                "world_seed": 1,
                "configured_rho": 0.5,
                "model": "dense",
                "measured_residual_correlation": 0.2,
                "gaussian_log_loss": -10.0,
                "novel_32_shot_nmse": 0.03,
            },
        ]
        summary = _aggregate(records)["rho_summaries"][0]
        self.assertEqual(summary["paired_worlds"], 1)
        self.assertEqual(summary["continuous_wins"], 1)
        self.assertAlmostEqual(
            summary["mean_dense_minus_continuous_gaussian_log_loss"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
