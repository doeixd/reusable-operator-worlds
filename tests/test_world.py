import unittest

import numpy as np

from row.world import World, WorldConfig


class WorldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WorldConfig(tasks=8, examples_per_task=12, evaluation_examples=16)

    def test_world_is_deterministic(self) -> None:
        first = World.generate(self.config)
        second = World.generate(self.config)
        self.assertEqual(first.programs_json(), second.programs_json())
        np.testing.assert_array_equal(first.tasks[0].train_x, second.tasks[0].train_x)
        np.testing.assert_array_equal(first.tasks[0].train_y, second.tasks[0].train_y)

    def test_different_seed_changes_world(self) -> None:
        first = World.generate(self.config)
        second = World.generate(WorldConfig(seed=1, tasks=8, examples_per_task=12, evaluation_examples=16))
        self.assertNotEqual(first.programs_json(), second.programs_json())

    def test_programs_and_task_ids_are_unique(self) -> None:
        world = World.generate(self.config)
        programs = [task.program.primitive_ids for task in world.tasks]
        ids = [task.task_id for task in world.tasks]
        self.assertEqual(len(programs), len(set(programs)))
        self.assertEqual(len(ids), len(set(ids)))

    def test_shapes_and_bounds(self) -> None:
        world = World.generate(self.config)
        task = world.tasks[0]
        self.assertEqual(task.train_x.shape, (12, 16))
        self.assertEqual(task.train_y.shape, (12, 16))
        self.assertEqual(task.eval_x.shape, (16, 16))
        self.assertTrue(np.all(np.abs(task.train_y) <= 1.0))

    def test_public_task_does_not_expose_program(self) -> None:
        world = World.generate(self.config)
        task_id, train_x, train_y = world.public_task(0)
        self.assertIsInstance(task_id, str)
        self.assertEqual(train_x.shape, train_y.shape)

    def test_scrambled_ids_preserve_all_task_contents_and_order(self) -> None:
        world = World.generate(self.config)
        scrambled = world.with_scrambled_task_ids(7)
        self.assertTrue(
            {task.task_id for task in world.tasks}.isdisjoint(
                task.task_id for task in scrambled.tasks
            )
        )
        self.assertEqual(
            [task.program for task in world.tasks],
            [task.program for task in scrambled.tasks],
        )
        for original, relabeled in zip(world.tasks, scrambled.tasks, strict=True):
            self.assertIs(original.train_x, relabeled.train_x)
            self.assertIs(original.train_y, relabeled.train_y)
            self.assertIs(original.eval_x, relabeled.eval_x)
            self.assertIs(original.eval_y, relabeled.eval_y)
        self.assertEqual(
            [task.task_id for task in scrambled.tasks],
            [task.task_id for task in world.with_scrambled_task_ids(7).tasks],
        )

    def test_too_many_tasks_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            WorldConfig(teacher_primitives=2, program_length=2, tasks=5)

    def test_rho_endpoints_control_task_specific_operator_recurrence(self) -> None:
        exact = World.generate(
            WorldConfig(reuse_rho=1.0, tasks=4, examples_per_task=4, evaluation_examples=4)
        )
        independent = World.generate(
            WorldConfig(reuse_rho=0.0, tasks=4, examples_per_task=4, evaluation_examples=4)
        )
        self.assertIs(exact.tasks[0].teacher_library, exact.library)
        self.assertFalse(
            np.array_equal(
                independent.tasks[0].teacher_library[0].U,
                independent.tasks[1].teacher_library[0].U,
            )
        )
        exact_diagnostic = exact.functional_reuse_diagnostics(probe_examples=32, max_tasks=4)
        independent_diagnostic = independent.functional_reuse_diagnostics(
            probe_examples=32, max_tasks=4
        )
        self.assertAlmostEqual(exact_diagnostic["mean_pairwise_residual_correlation"], 1.0)
        self.assertLess(
            independent_diagnostic["mean_pairwise_residual_correlation"],
            exact_diagnostic["mean_pairwise_residual_correlation"],
        )


if __name__ == "__main__":
    unittest.main()
