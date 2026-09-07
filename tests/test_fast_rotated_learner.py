import io
import unittest

import torch

from row.config import load_config
from row.experiments.learned_lifetime import _build_model
from row.models import FastRotatedDiscreteLibraryLearner, RotatedDiscreteLibraryLearner
from row.models.learned_models import batched_householder
from row.models.torch_oracle import HouseholderOrthogonal


def make(cls, seed=5000, slots=12):
    model = cls(
        d=16, operator_slots=slots, operator_rank=8, task_steps=3, alpha=0.2,
        initial_temperature=1.0, final_temperature=0.1, seed=seed,
        learnable_alpha=True, activation="tanh",
    )
    generator = torch.Generator().manual_seed(11)
    for index in range(4):
        code = model.begin_task(f"t{index}")
        with torch.no_grad():
            code.normal_(generator=generator)
    return model


class FastRotatedLearnerTests(unittest.TestCase):
    def test_batched_householder_matches_sequential_reflections(self):
        rotations = [HouseholderOrthogonal(16, 16, 1), HouseholderOrthogonal(16, 15, 2)]
        Q = batched_householder([r.vectors for r in rotations])
        z = torch.randn(5, 16)
        for index, rotation in enumerate(rotations):
            expected = rotation(z)
            actual = z @ Q[index]
            self.assertLess((expected - actual).abs().max().item(), 1e-5)
        # Orthogonal by construction, in both determinant components.
        for index in range(2):
            self.assertLess((Q[index] @ Q[index].T - torch.eye(16)).abs().max().item(), 1e-5)

    def test_same_parameters_same_state_dict_same_function(self):
        slow = make(RotatedDiscreteLibraryLearner)
        fast = make(FastRotatedDiscreteLibraryLearner)
        self.assertEqual(list(slow.state_dict()), list(fast.state_dict()))
        fast.load_state_dict(slow.state_dict())
        for batch in (2, 64):
            x = torch.randn(batch, 16, generator=torch.Generator().manual_seed(batch))
            for training in (True, False):
                slow.train(training)
                fast.train(training)
                for progress in (0.0, 0.5, 1.0):
                    slow.set_training_progress(progress)
                    fast.set_training_progress(progress)
                    a = slow(x, "t1")
                    b = fast(x, "t1")
                    self.assertLess((a - b).abs().max().item(), 1e-5,
                                    f"batch {batch} training {training} progress {progress}")

    def test_gradients_agree(self):
        slow = make(RotatedDiscreteLibraryLearner)
        fast = make(FastRotatedDiscreteLibraryLearner)
        fast.load_state_dict(slow.state_dict())
        x = torch.randn(8, 16, generator=torch.Generator().manual_seed(3))
        y = torch.randn(8, 16, generator=torch.Generator().manual_seed(4))
        ids = ["t0", "t1", "t2", "t3"] * 2
        for model in (slow, fast):
            model.train()
            model.set_training_progress(0.3)
            torch.nn.functional.mse_loss(model.forward_tasks(x, ids), y).backward()
        for (name, ps), (_, pf) in zip(slow.named_parameters(), fast.named_parameters()):
            self.assertIsNotNone(pf.grad, name)
            self.assertLess((ps.grad - pf.grad).abs().max().item(), 1e-5, name)

    def test_kind_switch_recovers_the_sequential_class(self):
        config = load_config("configs/smoke.yaml")
        self.assertIs(type(_build_model(config, "rotated_discrete")), RotatedDiscreteLibraryLearner)
        self.assertIs(type(_build_model(config, "rotated_discrete_fast")), FastRotatedDiscreteLibraryLearner)
        self.assertEqual(FastRotatedDiscreteLibraryLearner.implementation, "batched_rotation_v1")

    def test_checkpoint_round_trips_between_kinds(self):
        slow = make(RotatedDiscreteLibraryLearner)
        buffer = io.BytesIO()
        torch.save(slow.state_dict(), buffer)
        buffer.seek(0)
        fast = make(FastRotatedDiscreteLibraryLearner)
        fast.load_state_dict(torch.load(buffer, weights_only=True))
        x = torch.randn(3, 16)
        slow.eval()
        fast.eval()
        self.assertLess((slow(x, "t2") - fast(x, "t2")).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
