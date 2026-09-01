"""Tests for the frozen G5R acquisition-localization instrument."""
from dataclasses import replace
import unittest

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_rotated_g5r_diagnosis import (
    ProjectedRotatedOperator,
    assign_slots,
    examples,
    householder_decompose,
    lbfgs_data_loss,
    oracle_forward,
    stage_a_cell,
    stage_b_cell,
    stage_c_world,
)
from row.experiments.audit_rotated_g5r_diagnosis_lbfgs_correction import (
    correction_protocol,
)
from row.models import HouseholderOrthogonal, RotatedDiscreteLibraryLearner
from row.models import RotatedLearnedOperator
from row.rotated_world import generate_rotated_world


class G5RDiagnosisTests(unittest.TestCase):
    def test_lbfgs_objective_has_no_unregistered_penalty(self):
        model = RotatedLearnedOperator(4, 2, 0.2, 19, learnable_alpha=True)
        x = torch.randn(8, 4)
        with torch.no_grad():
            target = model(x).clone()
        self.assertEqual(float(lbfgs_data_loss(model, x, target).detach()), 0.0)
        self.assertEqual(correction_protocol()["weight_penalty"], 0.0)

    def test_constructive_householder_decomposition_covers_both_components(self):
        for reflections in (3, 4):
            generator = torch.Generator().manual_seed(100 + reflections)
            raw = torch.randn(4, 4, generator=generator, dtype=torch.float64)
            target, upper = torch.linalg.qr(raw)
            target = target * torch.sign(torch.diag(upper))
            wanted = 1 if reflections % 2 == 0 else -1
            if (1 if float(torch.linalg.det(target)) > 0 else -1) != wanted:
                target[:, 0] *= -1
            vectors = householder_decompose(target, reflections)
            represented = HouseholderOrthogonal(4, reflections, seed=1).double()
            with torch.no_grad():
                represented.vectors.copy_(vectors)
            torch.testing.assert_close(
                represented.matrix(), target, atol=1e-10, rtol=1e-10
            )

    def test_assignment_is_unique_and_matches_determinant_component(self):
        base = load_config("configs/v1.yaml")
        config = replace(
            base.world,
            state_dim=4,
            teacher_rank=2,
            teacher_primitives=4,
            program_length=2,
            tasks=4,
            examples_per_task=4,
            evaluation_examples=4,
        )
        world = generate_rotated_world(config)
        assignment = assign_slots(world.library, 4, 5000, 8)
        self.assertEqual(len(set(assignment.values())), 4)
        for primitive, slot in assignment.items():
            teacher_sign = 1 if np.linalg.det(world.library[primitive].Q) > 0 else -1
            learner = RotatedLearnedOperator(
                4, 2, 0.2, 5000 + 997 * slot, learnable_alpha=True
            )
            learned_sign = 1 if float(
                torch.linalg.det(learner.rotation.matrix()).detach()
            ) > 0 else -1
            self.assertEqual(teacher_sign, learned_sign)

    def test_stage_a_constructs_the_teacher_to_registered_tolerance(self):
        base = load_config("configs/v1.yaml")
        config = replace(
            base.world,
            state_dim=4,
            teacher_rank=2,
            teacher_primitives=2,
            program_length=2,
            tasks=2,
            examples_per_task=4,
            evaluation_examples=4,
        )
        world = generate_rotated_world(config)
        assignment = assign_slots(world.library, 4, 5000, 4)
        teacher = world.library[0]
        seed = 5000 + 997 * assignment[0]
        result = stage_a_cell(teacher, seed, examples(0, 0, teacher))
        self.assertTrue(result["passes"])

    def test_projected_arm_starts_identically_and_retracts(self):
        reference = RotatedLearnedOperator(
            4, 2, 0.2, seed=11, learnable_alpha=True
        )
        projected = ProjectedRotatedOperator(reference)
        x = torch.randn(8, 4)
        torch.testing.assert_close(reference(x), projected(x))
        with torch.no_grad():
            projected.row_map.add_(0.1 * torch.randn_like(projected.row_map))
        projected.retract()
        torch.testing.assert_close(
            projected.row_map.T @ projected.row_map,
            torch.eye(4),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_short_stage_b_cells_are_finite_and_projected_q_stays_orthogonal(self):
        base = load_config("configs/v1.yaml")
        config = replace(
            base.world,
            state_dim=4,
            teacher_rank=2,
            teacher_primitives=2,
            program_length=2,
            tasks=2,
            examples_per_task=4,
            evaluation_examples=4,
        )
        world = generate_rotated_world(config)
        teacher = world.library[0]
        assignment = assign_slots(world.library, 4, 5000, 4)
        seed = 5000 + 997 * assignment[0]
        data = examples(0, 0, teacher)
        for arm in ("H-Adam", "H-LBFGS", "Q-Adam"):
            result = stage_b_cell(
                teacher, seed, data, arm, adam_steps=2, lbfgs_steps=2
            )
            self.assertTrue(result["finite"])
            self.assertLess(result["orthogonality_max_abs_error"], 1e-5)

    def test_tiny_stage_c_oracle_executor_is_finite(self):
        base = load_config("configs/v1.yaml")
        config = replace(
            base,
            world=replace(
                base.world,
                state_dim=4,
                teacher_rank=2,
                teacher_primitives=2,
                program_length=2,
                tasks=2,
                examples_per_task=4,
                evaluation_examples=4,
            ),
            discrete_model=replace(
                base.discrete_model,
                operator_slots=4,
                operator_rank=2,
                task_steps=2,
            ),
        )
        world = generate_rotated_world(config.world)
        assignment = assign_slots(world.library, 4, config.discrete_model.seed, 4)
        result = stage_c_world(
            config,
            world,
            assignment,
            steps=2,
            batch_size=2,
            checkpoints_requested=(0, 1, 2),
        )
        self.assertTrue(result["finite"])
        self.assertEqual(set(result["checkpoints"]), {"0", "1", "2"})

        model = RotatedDiscreteLibraryLearner(
            4, 4, 2, 2, 0.2, 1.0, 0.1, seed=5000
        )
        task = world.tasks[0]
        route = {
            task.task_id: tuple(
                assignment[int(p)] for p in task.program.primitive_ids
            )
        }
        x = torch.tensor(task.train_x, dtype=torch.float32)
        output = oracle_forward(model, x, [task.task_id] * len(x), route)
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.library.parameters()))


if __name__ == "__main__":
    unittest.main()
