from __future__ import annotations

import unittest

from pathlib import Path

from row.experiments.analyze_rho_bridge import (
    _artifact_path,
    _linear_fit_quality,
    _summarize,
)


class AnalyzeRhoBridgeTests(unittest.TestCase):
    def test_exact_reuse_stage_two_uses_imported_selected_artifacts(self) -> None:
        path = _artifact_path(Path("artifacts"), 3, 1.0, "continuous")
        self.assertIn("continuous_3em03", str(path))

    def test_linear_fit_identifies_exact_line(self) -> None:
        result = _linear_fit_quality([0.0, 0.5, 1.0], [-1.0, 0.0, 1.0])
        self.assertAlmostEqual(result["r_squared"], 1.0)
        self.assertAlmostEqual(result["root_mean_square_residual"], 0.0)

    def test_summary_reports_declining_crossover(self) -> None:
        rows = []
        for truncation, crossing in ((2048, 0.8), (4096, 0.7), (8192, 0.6)):
            for world in (0, 1):
                for rho in (0.0, 0.5, 1.0):
                    rows.append(
                        {
                            "world_seed": world,
                            "configured_rho": rho,
                            "measured_residual_correlation": rho,
                            "online_examples": truncation,
                            "dense_minus_continuous_gaussian_log_loss": rho - crossing,
                        }
                    )
        report = _summarize(rows)
        self.assertEqual(report["h5a_amortization_prediction"]["status"], "supported")
        self.assertAlmostEqual(
            report["crossings_by_lifetime"][-1]["mean_curve_crossing"]["configured_rho"],
            0.6,
        )


if __name__ == "__main__":
    unittest.main()
