import unittest

from row.experiments.summarize_shared_residual import summarize


def _shared_record(world: int, rho: float, loss: float, ratio: float) -> dict:
    return {
        "world_seed": world,
        "configured_rho": rho,
        "measured_residual_correlation": rho,
        "gaussian_log_loss": loss,
        "novel_32_shot_nmse": 0.1,
        "mean_functional_ratio": ratio,
        "maximum_task_functional_ratio": ratio + 0.1,
        "shared_parameter_count": 10,
        "task_state_scalar_count": 20,
        "training_forward_multiply_adds_per_sample": 30,
        "inference_multiply_adds_per_sample": 30,
    }


class SummarizeSharedResidualTests(unittest.TestCase):
    def test_fixed_envelope_and_residual_prediction(self) -> None:
        shared = {
            "scope": "test",
            "escape_hatch_max_ratio": 1.0,
            "selected_configuration": {},
            "records": [
                _shared_record(0, 0.5, -12.0, 0.4),
                _shared_record(0, 1.0, -15.0, 0.1),
            ],
        }
        baselines = []
        for rho, continuous, dense in ((0.5, -10.0, -11.0), (1.0, -14.0, -9.0)):
            baselines.extend(
                [
                    {
                        "world_seed": 0,
                        "configured_rho": rho,
                        "model": "continuous",
                        "gaussian_log_loss": continuous,
                        "novel_32_shot_nmse": 0.2,
                    },
                    {
                        "world_seed": 0,
                        "configured_rho": rho,
                        "model": "dense",
                        "gaussian_log_loss": dense,
                        "novel_32_shot_nmse": 0.3,
                    },
                ]
            )

        report = summarize(shared, baselines)

        self.assertEqual(
            [row["best_fixed_loss_model"] for row in report["comparisons"]],
            ["dense", "continuous"],
        )
        self.assertEqual(
            report["predictions"]["shared_loss_wins_over_fixed_envelope_total"],
            2,
        )
        self.assertEqual(
            report["predictions"]["residual_ratio_lower_at_high_reuse_worlds"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
