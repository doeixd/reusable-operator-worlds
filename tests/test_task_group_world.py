"""Unit tests for the V3 task-group promotion testbed."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from row.mixed_world import CANONICAL_PROFILE, generate_mixed_world
from row.task_group_world import (
    TaskGroupSpec,
    generate_task_group_world,
    teacher_group_clustering,
)
from row.world import WorldConfig


def _config(seed: int = 0) -> WorldConfig:
    return WorldConfig(
        seed=seed,
        state_dim=8,
        teacher_rank=4,
        teacher_primitives=6,
        program_length=3,
        tasks=16,
        examples_per_task=8,
        evaluation_examples=32,
    )


class TaskGroupWorldTest(unittest.TestCase):
    def test_eta_zero_reproduces_the_canonical_mixed_world(self) -> None:
        config = _config()
        canonical = generate_mixed_world(config, CANONICAL_PROFILE)
        grouped = generate_task_group_world(
            config, CANONICAL_PROFILE, TaskGroupSpec(eta=0.0, future_tasks=0)
        )
        # The structureless control is the SAME generator at eta = 0, not a
        # separate one whose equivalence would have to be argued.
        for left, right in zip(canonical.tasks, grouped.tasks, strict=True):
            self.assertEqual(left.task_id, right.task_id)
            self.assertTrue(np.array_equal(left.train_y, right.train_y))
            self.assertTrue(np.array_equal(left.eval_y, right.eval_y))

    def test_group_structure_grows_with_eta(self) -> None:
        config = _config()
        separations = []
        for eta in (0.0, 0.5, 0.9):
            world = generate_task_group_world(
                config, CANONICAL_PROFILE, TaskGroupSpec(eta=eta)
            )
            separations.append(
                teacher_group_clustering(world, probe_examples=128)["separation"]
            )
        self.assertLess(abs(separations[0]), 0.05)
        self.assertLess(separations[0], separations[1])
        self.assertLess(separations[1], separations[2])

    def test_groups_are_balanced_and_hidden(self) -> None:
        config = _config()
        world = generate_task_group_world(
            config, CANONICAL_PROFILE, TaskGroupSpec(groups=2, eta=0.5, future_tasks=4)
        )
        counts = np.bincount(np.array(world.group_assignment))
        self.assertEqual(len(counts), 2)
        self.assertEqual(abs(int(counts[0]) - int(counts[1])), 0)
        # The assignment lives only on the world object and is shuffled, so
        # it is not recoverable from task order either.
        assignment = list(world.group_assignment)
        blocked = [0] * (len(assignment) // 2) + [1] * (len(assignment) - len(assignment) // 2)
        self.assertNotEqual(assignment, blocked)
        self.assertNotEqual(assignment, [i % 2 for i in range(len(assignment))])

    def test_future_block_is_held_out(self) -> None:
        config = _config()
        world = generate_task_group_world(
            config, CANONICAL_PROFILE, TaskGroupSpec(eta=0.5, future_tasks=4)
        )
        self.assertEqual(len(world.tasks), config.tasks)
        self.assertEqual(len(world.future_tasks), 4)
        lifetime_ids = {task.task_id for task in world.tasks}
        for task in world.future_tasks:
            self.assertNotIn(task.task_id, lifetime_ids)

    def test_regime_change_keeps_the_lifetime_identical(self) -> None:
        config = _config()
        stable = generate_task_group_world(
            config, CANONICAL_PROFILE, TaskGroupSpec(eta=0.7, future_tasks=4)
        )
        regime = generate_task_group_world(
            config,
            CANONICAL_PROFILE,
            TaskGroupSpec(eta=0.7, future_tasks=4, resample_future=True),
        )
        # Observationally indistinguishable during the lifetime, which is why
        # this world carries a same-decision prediction and never a refusal
        # requirement.
        for left, right in zip(stable.tasks, regime.tasks, strict=True):
            self.assertTrue(np.array_equal(left.train_y, right.train_y))
        differs = any(
            not np.array_equal(left.eval_y, right.eval_y)
            for left, right in zip(stable.future_tasks, regime.future_tasks, strict=True)
        )
        self.assertTrue(differs)

    def test_drifting_family_changes_direction_across_blocks(self) -> None:
        config = _config()
        drifting = generate_task_group_world(
            config,
            CANONICAL_PROFILE,
            TaskGroupSpec(eta=0.9, block_size=4, future_tasks=0),
        )
        stable = generate_task_group_world(
            config, CANONICAL_PROFILE, TaskGroupSpec(eta=0.9, future_tasks=0)
        )
        # Within-block structure is real in both, but the drifting control's
        # reusable direction is unstable across blocks, which is what makes
        # refusal inferable from sequential evidence rather than clairvoyant.
        stable_separation = teacher_group_clustering(stable, probe_examples=128)[
            "separation"
        ]
        drifting_separation = teacher_group_clustering(drifting, probe_examples=128)[
            "separation"
        ]
        self.assertLess(drifting_separation, stable_separation)

    def test_spec_validation(self) -> None:
        with self.assertRaises(ValueError):
            TaskGroupSpec(eta=1.5)
        with self.assertRaises(ValueError):
            TaskGroupSpec(groups=0)
        with self.assertRaises(ValueError):
            TaskGroupSpec(block_size=0)


if __name__ == "__main__":
    unittest.main()
