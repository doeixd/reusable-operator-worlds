import unittest

import torch

from row.models import OracleCompositor


class OracleModelTests(unittest.TestCase):
    def test_forward_shapes_and_parameter_count(self) -> None:
        model = OracleCompositor(d=4, rank=2, operators=3, alpha=0.35, seed=7)
        x = torch.randn(5, 4)
        prediction = model(x, (0, 1, 2))
        self.assertEqual(prediction.shape, x.shape)
        self.assertEqual(model.parameter_count, 3 * (2 * 4 + 4 * 2 + 2))

    def test_variable_route_batch_backpropagates(self) -> None:
        model = OracleCompositor(d=4, rank=2, operators=3, alpha=0.35, seed=7)
        x = torch.randn(2, 4)
        target = torch.zeros_like(x)
        prediction = model.forward_routes(x, ((0, 1), (2, 0)))
        torch.nn.functional.mse_loss(prediction, target).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
