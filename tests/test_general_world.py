"""The g1 generator must reproduce the current rotated worlds bitwise at gain 1."""
import unittest
from dataclasses import replace

import numpy as np

from row.config import load_config
from row.curriculum_world import CurriculumWorldConfig, generate_curriculum_world
from row.general_world import (GainPrimitive, general_library, generate_general_curriculum_world,
                               generate_general_world)
from row.rotated_world import generate_rotated_world


def _world_cfg(seed, **over):
    return replace(load_config('configs/v1.yaml').world, seed=seed, **over)


def _same(a, b):
    if len(a.tasks) != len(b.tasks):
        return False
    for s, t in zip(a.tasks, b.tasks):
        if s.task_id != t.task_id or s.program != t.program:
            return False
        for f in ('train_x', 'train_y', 'eval_x', 'eval_y'):
            if getattr(s, f).tobytes() != getattr(t, f).tobytes():
                return False
    return True


class TestGeneralWorld(unittest.TestCase):
    def test_gain_one_is_the_rotated_world_bitwise(self):
        for seed in (70, 57):
            cfg = _world_cfg(seed)
            self.assertTrue(_same(generate_general_world(cfg), generate_rotated_world(cfg)))

    def test_gain_one_is_the_curriculum_world_bitwise(self):
        base = load_config('configs/v1.yaml').world
        for length, tasks in ((1, 60), (2, 64)):
            cfg = CurriculumWorldConfig.from_world(replace(base, seed=71), program_length=length, tasks=tasks)
            self.assertTrue(_same(generate_general_curriculum_world(cfg), generate_curriculum_world(cfg)))

    def test_other_existing_knobs_pass_through(self):
        cfg = _world_cfg(72, teacher_primitives=8, alpha=1.0)
        self.assertTrue(_same(generate_general_world(cfg), generate_rotated_world(cfg)))

    def test_gain_changes_only_the_preactivation(self):
        cfg = _world_cfg(70)
        base, scaled = general_library(cfg), general_library(cfg, gain=2.0)
        self.assertIsInstance(scaled[0], GainPrimitive)
        z = np.random.default_rng(0).normal(size=(32, cfg.state_dim))
        p = base[0]
        manual = (z + p.alpha * (np.tanh(2.0 * (z @ p.V.T) + p.b) @ p.U.T)) @ p.Q.T
        np.testing.assert_array_equal(scaled[0](z), manual)
        self.assertFalse(np.allclose(scaled[0](z), base[0](z)))
        w = generate_general_world(cfg, gain=2.0)
        r = generate_rotated_world(cfg)
        self.assertEqual([t.program for t in w.tasks], [t.program for t in r.tasks])
        self.assertEqual(w.tasks[0].train_x.tobytes(), r.tasks[0].train_x.tobytes())
        self.assertNotEqual(w.tasks[0].train_y.tobytes(), r.tasks[0].train_y.tobytes())

    def test_shared_rotation_makes_primitives_similar(self):
        cfg = _world_cfg(70)
        base, mixed = general_library(cfg), general_library(cfg, shared=4.0)
        for p, q in zip(base, mixed):
            np.testing.assert_array_equal(p.U, q.U)
            np.testing.assert_array_equal(p.V, q.V)
            np.testing.assert_allclose(q.Q @ q.Q.T, np.eye(cfg.state_dim), atol=1e-10)

        def spread(lib):
            return np.mean([np.linalg.norm(a.Q - b.Q) for i, a in enumerate(lib) for b in lib[i + 1:]])

        self.assertLess(spread(mixed), 0.5 * spread(base))
        self.assertTrue(_same(generate_general_world(cfg, shared=0.0), generate_rotated_world(cfg)))


if __name__ == '__main__':
    unittest.main()
