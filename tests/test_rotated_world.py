"""The rotated substrate's generator, and the reduction it must satisfy."""
import unittest

import numpy as np

from row.config import WorldConfig
from row.rotated_world import RotatedPrimitive, generate_rotated_world, rotated_library
from row.world import Primitive, World


def _config(**over):
    base = dict(seed=3, state_dim=8, teacher_rank=4, teacher_primitives=4,
                program_length=3, tasks=6, examples_per_task=16,
                evaluation_examples=16)
    base.update(over)
    return WorldConfig(**base)


class TestRotatedWorld(unittest.TestCase):
    def test_reduces_to_the_standard_primitive_when_Q_is_identity(self):
        """The rotation replaces the outer tanh; with Q = I only that differs."""
        cfg = _config()
        rot = RotatedPrimitive.random(cfg.seed, 0, cfg.state_dim, cfg.teacher_rank, cfg.alpha)
        std = Primitive.random(cfg.seed, 0, cfg.state_dim, cfg.teacher_rank, cfg.alpha)
        # same residual parameters: Q is the only difference between substrates
        np.testing.assert_allclose(rot.U, std.U)
        np.testing.assert_allclose(rot.V, std.V)
        np.testing.assert_allclose(rot.b, std.b)
        identity = RotatedPrimitive(U=rot.U, V=rot.V, b=rot.b,
                                    Q=np.eye(cfg.state_dim), alpha=rot.alpha)
        z = np.random.default_rng(0).normal(size=(32, cfg.state_dim))
        hidden = np.tanh(z @ std.V.T + std.b)
        np.testing.assert_allclose(identity(z), z + std.alpha * (hidden @ std.U.T))

    def test_Q_is_orthogonal_and_deterministic(self):
        cfg = _config()
        a = RotatedPrimitive.random(cfg.seed, 1, cfg.state_dim, cfg.teacher_rank, cfg.alpha)
        b = RotatedPrimitive.random(cfg.seed, 1, cfg.state_dim, cfg.teacher_rank, cfg.alpha)
        np.testing.assert_allclose(a.Q, b.Q)
        np.testing.assert_allclose(a.Q @ a.Q.T, np.eye(cfg.state_dim), atol=1e-10)

    def test_iterates_do_not_converge(self):
        """The property the substrate exists for: no fixed point."""
        cfg = _config()
        p = rotated_library(cfg)[0]
        z = np.random.default_rng(1).normal(size=(256, cfg.state_dim))
        states = []
        for _ in range(6):
            z = p(z)
            states.append(z.copy())
        late = np.mean((states[5] - states[4]) ** 2) / np.mean(states[4] ** 2)
        self.assertGreater(late, 0.05, "rotated iterates converged like the standard family")

    def test_world_generates_and_targets_are_finite(self):
        world = generate_rotated_world(_config())
        self.assertEqual(len(world.tasks), 6)
        for t in world.tasks:
            self.assertTrue(np.all(np.isfinite(t.train_y)))
            self.assertTrue(np.all(np.isfinite(t.eval_y)))

    def test_refuses_rho_below_one_rather_than_dropping_the_rotation(self):
        with self.assertRaises(ValueError):
            generate_rotated_world(_config(reuse_rho=0.9))


if __name__ == "__main__":
    unittest.main()
