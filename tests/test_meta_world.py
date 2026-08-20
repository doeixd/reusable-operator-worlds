"""Unit tests for the H20 meta-recurrence generator."""

from __future__ import annotations

import unittest

import numpy as np

from row.meta_world import MetaFamilySpec, family_operators, generate_meta_world
from row.world import WorldConfig


def _config(seed: int = 0, tasks: int = 72) -> WorldConfig:
    return WorldConfig(
        seed=seed,
        state_dim=8,
        teacher_rank=4,
        teacher_primitives=6,
        program_length=3,
        tasks=tasks,
        examples_per_task=8,
        evaluation_examples=32,
    )


def _spec(r_meta: float, families: int = 4) -> MetaFamilySpec:
    return MetaFamilySpec(
        families=families, tasks_per_family=16, r_meta=r_meta, subspace_rank=2,
        family_onset=8,
    )


class MetaFamilyGeneratorTest(unittest.TestCase):
    def test_hidden_features_are_shared_across_families(self) -> None:
        # This is what makes the mixture FUNCTIONAL rather than a
        # parameter-space one: with V and b common, the residual
        # contribution is linear in U, so mixing U mixes functions.
        operators = family_operators(_config(), _spec(0.5))
        for operator in operators[1:]:
            self.assertTrue(np.array_equal(operator.V, operators[0].V))
            self.assertTrue(np.array_equal(operator.b, operators[0].b))

    def test_operator_norm_is_exactly_invariant_to_r_meta(self) -> None:
        # Not "within tolerance" — the private part is projected out of
        # the shared subspace, so ||theta||^2 = r + (1 - r) exactly.
        norms = {}
        for r_meta in (0.0, 0.25, 0.5, 0.9, 1.0):
            operators = family_operators(_config(), _spec(r_meta))
            norms[r_meta] = [float(np.linalg.norm(o.U)) for o in operators]
        reference = norms[0.0]
        for r_meta, values in norms.items():
            for index, value in enumerate(values):
                self.assertAlmostEqual(
                    value, reference[index], places=10,
                    msg=f"family {index} norm moved at r_meta={r_meta}",
                )

    def test_r_meta_one_puts_every_family_in_one_subspace(self) -> None:
        operators = family_operators(_config(), _spec(1.0))
        flat = np.stack([o.U.ravel() for o in operators])
        # Rank of the family set collapses to the subspace rank.
        singular = np.linalg.svd(flat, compute_uv=False)
        self.assertGreater(singular[1], 1e-8)
        self.assertLess(singular[2], 1e-8 * singular[0])

    def test_r_meta_zero_gives_independent_operators(self) -> None:
        operators = family_operators(_config(), _spec(0.0))
        flat = np.stack([o.U.ravel() for o in operators])
        singular = np.linalg.svd(flat, compute_uv=False)
        # Four unrelated operators span four directions.
        self.assertGreater(singular[3], 1e-6 * singular[0])

    def test_family_assignment_is_contiguous_after_the_onset(self) -> None:
        spec = _spec(0.5)
        self.assertIsNone(spec.family_of(0))
        self.assertIsNone(spec.family_of(7))
        self.assertEqual(spec.family_of(8), 0)
        self.assertEqual(spec.family_of(23), 0)
        self.assertEqual(spec.family_of(24), 1)
        self.assertEqual(spec.total_tasks, 8 + 4 * 16)

    def test_world_uses_the_family_operator_at_a_fixed_position(self) -> None:
        config = _config()
        spec = _spec(0.9)
        world = generate_meta_world(config, spec)
        self.assertEqual(len(world.tasks), spec.total_tasks)
        early = world.tasks[0]
        self.assertEqual(len(early.teacher_library), config.teacher_primitives)
        late = world.tasks[8]
        self.assertEqual(len(late.teacher_library), config.teacher_primitives + 1)
        self.assertEqual(
            late.program.primitive_ids[config.program_length - 1],
            config.teacher_primitives,
        )

    def test_task_count_must_match_the_family_layout(self) -> None:
        with self.assertRaises(ValueError):
            generate_meta_world(_config(tasks=64), _spec(0.5))

    def test_spec_validation(self) -> None:
        for bad in (
            {"families": 0},
            {"tasks_per_family": 0},
            {"r_meta": 1.5},
            {"r_meta": -0.1},
            {"subspace_rank": 0},
            {"family_onset": -1},
        ):
            with self.assertRaises(ValueError):
                MetaFamilySpec(**bad)

    def test_provenance_records_the_knob(self) -> None:
        recorded = _spec(0.9).as_dict()
        self.assertEqual(recorded["r_meta"], 0.9)
        self.assertEqual(recorded["families"], 4)
        self.assertEqual(recorded["subspace_rank"], 2)


if __name__ == "__main__":
    unittest.main()
