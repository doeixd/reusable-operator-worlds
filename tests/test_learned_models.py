from dataclasses import replace
import unittest

import torch

from row.config import load_config
from row.experiments.learned_lifetime import (
    TaskReplayBuffer,
    _true_route_operator_quality,
    _update_batch_counts,
)
from row.models import (
    ContinuousBasisLearner,
    DenseLearner,
    DiscreteLibraryLearner,
    HypernetworkLearner,
)
from row.world import World


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

    def test_continuous_identity_slot_adds_code_capacity_not_shared_weights(self) -> None:
        plain = ContinuousBasisLearner(4, 3, 2, 2, 0.2, seed=1)
        identity = ContinuousBasisLearner(
            4, 3, 2, 2, 0.2, seed=1, include_identity=True
        )
        plain.begin_task("task_a")
        code = identity.begin_task("task_a")
        self.assertEqual(plain.shared_parameter_count, identity.shared_parameter_count)
        self.assertEqual(code.shape, (2, 4))

    def test_variable_task_batches_preserve_shape(self) -> None:
        model = DenseLearner(4, 3, 8, 2, seed=1)
        model.begin_task("task_a")
        model.begin_task("task_b")
        output = model.forward_tasks(torch.randn(2, 4), ("task_a", "task_b"))
        self.assertEqual(output.shape, (2, 4))

    def test_hypernetwork_matches_task_state_budget_and_receives_code_gradients(self) -> None:
        model = HypernetworkLearner(4, 3, 5, 2, 2, 0.2, seed=1)
        code = model.begin_task("task_a")
        prediction = model(torch.randn(5, 4), "task_a")
        prediction.square().mean().backward()
        self.assertEqual(code.shape, (2, 3))
        self.assertEqual(model.task_state_scalar_count, 6)
        self.assertIsNotNone(code.grad)
        self.assertGreater(float(torch.linalg.vector_norm(code.grad)), 0.0)

    def test_hypernetwork_zero_code_generates_shared_base_operator(self) -> None:
        model = HypernetworkLearner(4, 3, 5, 2, 2, 0.2, seed=1)
        code = model.begin_task("task_a")
        with torch.no_grad():
            U, V, b = model._generated_parameters(code[0])
        self.assertTrue(torch.equal(U, model.U))
        self.assertTrue(torch.equal(V, model.V))
        self.assertTrue(torch.equal(b, model.b))

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

    def test_true_route_operator_diagnostic_does_not_require_future_task_codes(self) -> None:
        config = load_config("configs/v1.yaml")
        config = replace(
            config,
            world=replace(config.world, tasks=3, evaluation_examples=8),
        )
        world = World.generate(config.world)
        model = ContinuousBasisLearner(16, 8, 8, 3, 0.2, seed=1)
        model.begin_task(world.tasks[0].task_id)
        result = _true_route_operator_quality(model, world, config, 1)
        self.assertIn("true_route_future_programs_nmse_mean", result)
        self.assertGreaterEqual(result["one_to_one_mean_primitive_distance"], 0.0)
        self.assertTrue(model.training)

    def test_batch_eight_splits_evenly_at_one_to_one_replay(self) -> None:
        self.assertEqual(_update_batch_counts(8, 1.0), (4, 4))
        self.assertEqual(_update_batch_counts(2, 1.0), (1, 1))

    def test_split_replay_rng_keeps_buffer_construction_independent_of_sampling(self) -> None:
        config = replace(
            load_config("configs/v1.yaml").world,
            tasks=2,
            examples_per_task=8,
            evaluation_examples=4,
        )
        world = World.generate(config)
        small = TaskReplayBuffer(11, sampling_seed=12)
        large = TaskReplayBuffer(11, sampling_seed=12)
        small.add_task(world.tasks[0], 4)
        large.add_task(world.tasks[0], 4)
        small.sample(1)
        large.sample(4)
        small.add_task(world.tasks[1], 4)
        large.add_task(world.tasks[1], 4)
        for left, right in zip(small.items, large.items, strict=True):
            self.assertEqual(left[2], right[2])
            self.assertTrue(torch.equal(torch.from_numpy(left[0]), torch.from_numpy(right[0])))


if __name__ == "__main__":
    unittest.main()
