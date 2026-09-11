import unittest
from dataclasses import replace

import numpy as np

from row.config import load_config
from row.curriculum_world import CurriculumWorldConfig, generate_curriculum_world


def config(length=1, tasks=60, seed=1):
    base = load_config("configs/v1.yaml")
    return CurriculumWorldConfig.from_world(replace(base.world, seed=seed), program_length=length, tasks=tasks)


class CurriculumWorldTests(unittest.TestCase):
    def test_programs_repeat_and_cover_every_primitive(self):
        cfg = config()
        world = generate_curriculum_world(cfg)
        self.assertEqual(len(world.tasks), 60)
        used = [int(t.program.primitive_ids[0]) for t in world.tasks]
        self.assertEqual(set(used), set(range(cfg.teacher_primitives)))
        self.assertLess(len(set(used)), len(used))  # repeats, unlike the distinct-program generator
        self.assertEqual(len({t.task_id for t in world.tasks}), 60)

    def test_targets_are_the_teacher_operation(self):
        cfg = config()
        world = generate_curriculum_world(cfg)
        for task in world.tasks[:5]:
            expected = task.program.execute(task.teacher_library, task.train_x)
            self.assertTrue(np.allclose(task.train_y, expected))
            primitive = task.teacher_library[int(task.program.primitive_ids[0])]
            self.assertTrue(np.allclose(task.eval_y, primitive(task.eval_x)))

    def test_deterministic_and_seed_sensitive(self):
        a = generate_curriculum_world(config())
        self.assertTrue(np.array_equal(a.tasks[0].train_x, generate_curriculum_world(config()).tasks[0].train_x))
        other = generate_curriculum_world(config(seed=2))
        self.assertFalse(np.array_equal(a.tasks[0].eval_y, other.tasks[0].eval_y))

    def test_length_two_stage_has_two_step_programs(self):
        world = generate_curriculum_world(config(length=2, tasks=40))
        self.assertTrue(all(len(t.program.primitive_ids) == 2 for t in world.tasks))


if __name__ == "__main__":
    unittest.main()
