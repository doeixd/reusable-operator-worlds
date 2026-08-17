import unittest

import torch

from row.experiments.quantize_artifact import (
    _build_from_artifact,
    _inference_multiply_adds,
    symmetric_int8_dequantize,
)
from row.models import HypernetworkLearner


class QuantizationTests(unittest.TestCase):
    def test_symmetric_quantization_is_bounded_and_preserves_zero(self) -> None:
        values = torch.tensor([-2.0, -0.25, 0.0, 0.75, 2.0])
        restored, scale = symmetric_int8_dequantize(values)
        self.assertGreater(scale, 0.0)
        self.assertEqual(float(restored[2]), 0.0)
        self.assertLessEqual(float(torch.max(torch.abs(restored - values))), scale / 2 + 1e-7)

    def test_builds_and_accounts_for_hypernetwork_artifact(self) -> None:
        raw = {
            "world": {"state_dim": 16, "alpha": 0.35},
            "hypernetwork_model": {
                "step_code_dim": 8,
                "hypernetwork_hidden_dim": 8,
                "operator_rank": 8,
                "task_steps": 3,
                "operator_alpha_init": 0.2,
                "learnable_alpha": True,
                "operator_activation": "tanh",
                "seed": 6000,
            },
        }
        model = _build_from_artifact(raw, "hypernetwork")
        self.assertIsInstance(model, HypernetworkLearner)
        self.assertEqual(_inference_multiply_adds(raw, "hypernetwork"), 7296)


if __name__ == "__main__":
    unittest.main()
