from __future__ import annotations

import unittest

from row.experiments.tune_shared_residual import _select


class TuneSharedResidualTests(unittest.TestCase):
    def test_selection_rejects_escape_hatch_even_with_better_loss(self) -> None:
        records = [
            {
                "residual_learning_rate": 0.01,
                "residual_penalty": 0.001,
                "gaussian_log_loss": -20.0,
                "maximum_task_functional_ratio": 1.2,
            },
            {
                "residual_learning_rate": 0.005,
                "residual_penalty": 0.01,
                "gaussian_log_loss": -10.0,
                "maximum_task_functional_ratio": 0.4,
            },
        ]
        report = _select(records)
        self.assertEqual(report["selected"]["residual_penalty"], 0.01)


if __name__ == "__main__":
    unittest.main()
