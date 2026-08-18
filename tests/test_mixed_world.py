import unittest

import numpy as np

from row.mixed_world import (
    generate_mixed_world,
    per_primitive_recurrence,
    usage_by_task_index,
)
from row.world import World, WorldConfig


class MixedWorldTest(unittest.TestCase):
    def test_uniform_profile_reproduces_homogeneous_world(self) -> None:
        config = WorldConfig(seed=0, tasks=6, examples_per_task=8,
                             evaluation_examples=8, reuse_rho=0.5)
        homogeneous = World.generate(config)
        mixed = generate_mixed_world(config, (0.5,) * 6)
        for a, b in zip(homogeneous.tasks, mixed.tasks, strict=True):
            self.assertEqual(a.task_id, b.task_id)
            np.testing.assert_array_equal(a.train_y, b.train_y)
            np.testing.assert_array_equal(a.eval_y, b.eval_y)
            for pa, pb in zip(a.teacher_library, b.teacher_library, strict=True):
                np.testing.assert_array_equal(pa.U, pb.U)

    def test_exact_reuse_primitive_shared_others_perturbed(self) -> None:
        config = WorldConfig(seed=1, tasks=4, examples_per_task=8,
                             evaluation_examples=8)
        mixed = generate_mixed_world(config, (1.0, 0.0, 0.5, 0.5, 0.5, 0.5))
        base = mixed.library
        for task in mixed.tasks:
            np.testing.assert_array_equal(task.teacher_library[0].U, base[0].U)
            self.assertFalse(np.array_equal(task.teacher_library[1].U, base[1].U))

    def test_per_primitive_recurrence_orders_with_profile(self) -> None:
        config = WorldConfig(seed=2, tasks=8, examples_per_task=8,
                             evaluation_examples=8)
        profile = (1.0, 0.95, 0.8, 0.5, 0.2, 0.0)
        mixed = generate_mixed_world(config, profile)
        rows = per_primitive_recurrence(mixed, probe_examples=128, max_tasks=8)
        measured = [row["measured_recurrence"] for row in rows]
        self.assertEqual(measured, sorted(measured, reverse=True))
        self.assertAlmostEqual(measured[0], 1.0, places=6)
        self.assertLess(abs(measured[5]), 0.15)

    def test_novel_task_library_uses_profile(self) -> None:
        config = WorldConfig(seed=3, tasks=4, examples_per_task=8,
                             evaluation_examples=8)
        mixed = generate_mixed_world(config, (1.0, 0.0, 0.5, 0.5, 0.5, 0.5))
        novel = mixed.library_for_task(config.tasks + 1)
        np.testing.assert_array_equal(novel[0].U, mixed.library[0].U)
        self.assertFalse(np.array_equal(novel[1].U, mixed.library[1].U))

    def test_usage_gate_returns_bounded_correlations(self) -> None:
        config = WorldConfig(seed=4, tasks=16, examples_per_task=8,
                             evaluation_examples=8)
        mixed = generate_mixed_world(config, (1.0, 0.95, 0.8, 0.5, 0.2, 0.0))
        gate = usage_by_task_index(mixed)
        self.assertLessEqual(gate["max_abs_correlation"], 1.0)


if __name__ == "__main__":
    unittest.main()
