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

    def test_too_many_tasks_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            WorldConfig(teacher_primitives=2, program_length=2, tasks=5)


if __name__ == "__main__":
    unittest.main()

