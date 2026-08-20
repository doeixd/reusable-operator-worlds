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


class DormancyWorldTest(unittest.TestCase):
    """The V4 real-options world: a regime goes quiet, then returns or does not."""

    def _world(self, returns: bool):
        config = _config()
        spec = TaskGroupSpec(
            groups=2,
            eta=0.9,
            future_tasks=0,
            family_onset=4,
            new_primitive_families=True,
            dormancy=(8, 12),
            dormancy_returns=returns,
        )
        return generate_task_group_world(config, CANONICAL_PROFILE, spec)

    def test_the_two_arms_are_identical_until_the_regime_would_return(self) -> None:
        returning = self._world(True)
        permanent = self._world(False)
        # Nothing observable before the end of dormancy separates them, so
        # retaining an abstraction through the gap must be an expectation
        # about the future rather than a reading of the past.
        for index in range(12):
            self.assertTrue(
                np.array_equal(
                    returning.tasks[index].train_y, permanent.tasks[index].train_y
                ),
                f"task {index} differs before the regime could return",
            )
        differs = any(
            not np.array_equal(
                returning.tasks[i].train_y, permanent.tasks[i].train_y
            )
            for i in range(12, len(returning.tasks))
        )
        self.assertTrue(differs)

    def test_the_family_primitive_is_absent_during_dormancy(self) -> None:
        world = self._world(True)
        spec = world.task_group_spec
        self.assertFalse(spec.family_active(2))    # before onset
        self.assertTrue(spec.family_active(5))     # active
        self.assertFalse(spec.family_active(9))    # dormant
        self.assertTrue(spec.family_active(13))    # returned
        gone = self._world(False).task_group_spec
        self.assertFalse(gone.family_active(13))   # never returns

    def test_dormancy_bounds_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            TaskGroupSpec(eta=0.9, dormancy=(12, 8))


class ReturnValueGainTest(unittest.TestCase):
    """V5 H19 s-arm (B1): move s_bar without moving the abstraction."""

    def _world(self, gain: float):
        spec = TaskGroupSpec(
            groups=2,
            eta=0.9,
            future_tasks=0,
            family_onset=4,
            new_primitive_families=True,
            dormancy=(8, 12),
            dormancy_returns=True,
            return_gain=gain,
        )
        return generate_task_group_world(_config(), CANONICAL_PROFILE, spec)

    def test_unit_gain_is_bit_exact(self) -> None:
        # The whole arm rests on this: g = 1.0 must reproduce every
        # existing artifact exactly, or the s-arm silently invalidates
        # the D-arm it is compared against.
        baseline = generate_task_group_world(
            _config(),
            CANONICAL_PROFILE,
            TaskGroupSpec(
                groups=2,
                eta=0.9,
                future_tasks=0,
                family_onset=4,
                new_primitive_families=True,
                dormancy=(8, 12),
                dormancy_returns=True,
            ),
        )
        gained = self._world(1.0)
        for index in range(len(baseline.tasks)):
            for field in ("train_x", "train_y", "eval_x", "eval_y"):
                self.assertTrue(
                    np.array_equal(
                        getattr(baseline.tasks[index], field),
                        getattr(gained.tasks[index], field),
                    ),
                    f"task {index} {field} moved at unit gain",
                )

    def test_the_gain_touches_returning_tasks_only(self) -> None:
        plain, loud = self._world(1.0), self._world(1.5)
        # Everything up to the moment the gap closes is identical, so the
        # abstraction's birth cannot depend on g.
        for index in range(12):
            self.assertTrue(
                np.array_equal(plain.tasks[index].train_y, loud.tasks[index].train_y),
                f"task {index} moved before the gap closed",
            )
        differs = [
            index
            for index in range(12, len(plain.tasks))
            if not np.array_equal(
                plain.tasks[index].train_y, loud.tasks[index].train_y
            )
        ]
        self.assertTrue(differs, "the gain changed no returning task")

    def test_inputs_never_move_with_the_gain(self) -> None:
        # Only the family's CONTRIBUTION may change. If the x streams
        # moved, s_bar and difficulty would be confounded.
        plain, loud = self._world(1.0), self._world(1.5)
        for index in range(len(plain.tasks)):
            self.assertTrue(
                np.array_equal(plain.tasks[index].train_x, loud.tasks[index].train_x)
            )
            self.assertTrue(
                np.array_equal(plain.tasks[index].eval_x, loud.tasks[index].eval_x)
            )

    def test_only_the_family_primitive_is_scaled(self) -> None:
        plain, loud = self._world(1.0), self._world(1.5)
        returning = 13
        base = plain.tasks[returning].teacher_library
        scaled = loud.tasks[returning].teacher_library
        self.assertEqual(len(base), len(scaled))
        for position in range(len(base) - 1):
            self.assertTrue(np.array_equal(base[position].U, scaled[position].U))
            self.assertTrue(np.array_equal(base[position].V, scaled[position].V))
            self.assertEqual(base[position].alpha, scaled[position].alpha)
        # The family primitive keeps its parameters and changes only the
        # residual scale, so it is the same operation worth more.
        self.assertTrue(np.array_equal(base[-1].U, scaled[-1].U))
        self.assertTrue(np.array_equal(base[-1].V, scaled[-1].V))
        self.assertTrue(np.array_equal(base[-1].b, scaled[-1].b))
        self.assertAlmostEqual(scaled[-1].alpha, base[-1].alpha * 1.5)

    def test_a_gain_needs_a_gap_that_closes(self) -> None:
        # Without returning tasks there is nothing for the gain to price.
        with self.assertRaises(ValueError):
            TaskGroupSpec(eta=0.9, dormancy=(8, 12), dormancy_returns=False,
                          return_gain=1.5)
        with self.assertRaises(ValueError):
            TaskGroupSpec(eta=0.9, return_gain=1.5)
        with self.assertRaises(ValueError):
            TaskGroupSpec(eta=0.9, dormancy=(8, 12), return_gain=0.0)

    def test_the_gain_is_recorded_in_provenance(self) -> None:
        self.assertEqual(self._world(1.5).task_group_spec.as_dict()["return_gain"], 1.5)
