import unittest

from row.experiments.tune_mdl import _select


class TuneMDLTests(unittest.TestCase):
    def test_selects_shortest_sufficient_library_then_loss(self) -> None:
        records = [
            {
                "active_operators_at_threshold": 8,
                "final_median_nmse": 0.01,
                "novel_32_shot_nmse": 0.01,
                "gaussian_log_loss": -100.0,
                "presence_learning_rate": 1e-4,
                "library_presence_penalty": 1e-5,
            },
            {
                "active_operators_at_threshold": 7,
                "final_median_nmse": 0.019,
                "novel_32_shot_nmse": 0.019,
                "gaussian_log_loss": -90.0,
                "presence_learning_rate": 1e-3,
                "library_presence_penalty": 1e-5,
            },
            {
                "active_operators_at_threshold": 6,
                "final_median_nmse": 0.03,
                "novel_32_shot_nmse": 0.01,
                "gaussian_log_loss": -110.0,
                "presence_learning_rate": 1e-3,
                "library_presence_penalty": 1e-4,
            },
        ]
        report = _select(records, 12)
        self.assertEqual(report["selected"]["active_operators_at_threshold"], 7)

    def test_requires_pruning_and_sufficiency(self) -> None:
        record = {
            "active_operators_at_threshold": 12,
            "final_median_nmse": 0.01,
            "novel_32_shot_nmse": 0.01,
            "gaussian_log_loss": -100.0,
            "presence_learning_rate": 1e-4,
            "library_presence_penalty": 1e-5,
        }
        with self.assertRaisesRegex(ValueError, "shorter sufficient"):
            _select([record], 12)


if __name__ == "__main__":
    unittest.main()
