import unittest

import numpy as np

from row.experiments.scratch_difficulty import summarize
from row.metrics import examples_to_criterion, gaussian_nll, nmse


class MetricTests(unittest.TestCase):
    def test_perfect_nmse_is_zero(self) -> None:
        target = np.array([[1.0, -1.0], [0.0, 2.0]])
        self.assertEqual(nmse(target, target), 0.0)

    def test_examples_to_criterion_and_censoring(self) -> None:
        curve = {0: 1.0, 1: 0.4, 2: 0.09, 4: 0.03}
        self.assertEqual(examples_to_criterion(curve, 0.1, 4), 2)
        self.assertEqual(examples_to_criterion(curve, 0.01, 4), 5)

    def test_gaussian_nll_prefers_exact_prediction(self) -> None:
        target = np.zeros((2, 3))
        exact = gaussian_nll(target, target)
        wrong = gaussian_nll(np.ones_like(target), target)
        self.assertLess(exact, wrong)

    def test_scratch_summary_reports_censoring_and_reverse_trend(self) -> None:
        rows = [
            {
                "record_type": "task_summary",
                "task_index": i,
                "final_nmse": 0.1 + i,
                "examples_to_0.1": value,
            }
            for i, value in enumerate((1, 2, 5))
        ]
        result = summarize(rows, max_examples=4)
        self.assertAlmostEqual(result["criteria"]["0.1"]["censored_fraction"], 1.0 / 3.0)
        self.assertGreater(result["final_nmse"]["forward_order"]["slope_per_task"], 0.0)
        self.assertLess(result["final_nmse"]["reverse_order"]["slope_per_task"], 0.0)


if __name__ == "__main__":
    unittest.main()
