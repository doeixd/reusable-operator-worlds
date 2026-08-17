import unittest

import torch

from row.models import ContinuousBasisLearner, DenseLearner, DiscreteLibraryLearner


class LearnedModelTests(unittest.TestCase):
    def test_dense_uses_identically_initialized_task_codes(self) -> None:
        model = DenseLearner(4, 3, 8, 2, seed=1)
        first = model.begin_task("task_a")
        second = model.begin_task("task_b")
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(model(torch.randn(5, 4), "task_a").shape, (5, 4))
        self.assertEqual(model.task_state_scalar_count, 6)

    def test_continuous_basis_codes_receive_gradients(self) -> None:
        model = ContinuousBasisLearner(4, 3, 2, 2, 0.35, seed=1)
        code = model.begin_task("task_a")
        prediction = model(torch.randn(5, 4), "task_a")
        prediction.square().mean().backward()
        self.assertIsNotNone(code.grad)
        self.assertEqual(model.task_state_scalar_count, 6)
        diagnostics = model.routing_diagnostics()
        self.assertAlmostEqual(diagnostics["mean_max_coefficient"], 1.0 / 3.0, places=6)

    def test_variable_task_batches_preserve_shape(self) -> None:
        model = DenseLearner(4, 3, 8, 2, seed=1)
        model.begin_task("task_a")
        model.begin_task("task_b")
        output = model.forward_tasks(torch.randn(2, 4), ("task_a", "task_b"))
        self.assertEqual(output.shape, (2, 4))

    def test_discrete_library_trains_soft_and_evaluates_hard(self) -> None:
        model = DiscreteLibraryLearner(4, 3, 2, 2, 0.35, 1.0, 0.1, seed=2)
        code = model.begin_task("task_a")
        with torch.no_grad():
            code[0, 1] = 2.0
            code[1, 2] = 2.0
        model.train()
        soft = model._coefficients("task_a")
        self.assertTrue(torch.all((soft > 0.0) & (soft < 1.0)))
        model.eval()
        hard = model._coefficients("task_a")
        self.assertTrue(torch.all((hard == 0.0) | (hard == 1.0)))
        self.assertEqual(model.hard_routes()["task_a"], [1, 2])


if __name__ == "__main__":
    unittest.main()
