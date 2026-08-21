"""Unit tests for the H39b parameterized-slot learner."""

from __future__ import annotations

import unittest

import torch

from row.models.prospective_models import ProspectiveLifecycleLearner
from row.models.pslot_models import ParameterizedSlotLearner

KW = dict(d=8, operator_slots=4, operator_rank=4, residual_rank=2,
          task_steps=3, alpha=0.2, seed=5)


class PSlotTests(unittest.TestCase):
    def test_frozen_args_is_bitwise_the_ordinary_learner(self):
        frozen = ParameterizedSlotLearner(slot_args=2, freeze_args=True, **KW)
        ordinary = ProspectiveLifecycleLearner(**KW)
        frozen.begin_task("t")
        ordinary.begin_task("t")
        with torch.no_grad():
            frozen.task_codes["t"].normal_()
            ordinary.task_codes["t"].copy_(frozen.task_codes["t"])
            frozen.task_residuals["t"].normal_()
            ordinary.task_residuals["t"].copy_(frozen.task_residuals["t"])
        x = torch.randn(32, 8)
        self.assertTrue(torch.equal(frozen(x, "t"), ordinary(x, "t")))
        self.assertFalse(frozen.task_alphas["t"].requires_grad)
        self.assertFalse(frozen.argument_matrices.requires_grad)
        self.assertEqual(len(frozen.shared_parameters()), len(ordinary.shared_parameters()))

    def test_alpha_changes_output_and_gradients_flow_at_zero(self):
        model = ParameterizedSlotLearner(slot_args=2, **KW)
        code, eps, alpha = model.begin_task("t")
        with torch.no_grad():
            # Put route mass on P at every step so alpha matters.
            code.view(3, 4)[:, model.pslot_index] = 5.0
        x = torch.randn(32, 8)
        loss = torch.mean(model(x, "t") ** 2)
        loss.backward()
        self.assertGreater(float(alpha.grad.abs().sum()), 0.0)
        # U_k receives alpha-weighted gradient: zero while alpha is zero,
        # nonzero as soon as any task's alpha has moved.
        self.assertEqual(float(model.argument_matrices.grad.abs().sum()), 0.0)
        before = model(x, "t").detach()
        with torch.no_grad():
            alpha.fill_(1.0)
        model.argument_matrices.grad = None
        torch.mean(model(x, "t") ** 2).backward()
        self.assertGreater(float(model.argument_matrices.grad.abs().sum()), 0.0)
        self.assertFalse(torch.allclose(before, model(x, "t")))

    def test_counts_and_forget(self):
        model = ParameterizedSlotLearner(slot_args=3, **KW)
        model.begin_task("a")
        model.begin_task("b")
        self.assertEqual(model.shared_parameter_count,
                         ProspectiveLifecycleLearner(**KW).shared_parameter_count + 3 * 8 * 4)
        self.assertEqual(model.task_state_scalar_count,
                         2 * (model.route_size + model.task_residuals["a"].numel() + 3))
        model.forget_task("a")
        self.assertNotIn("a", model.task_alphas)

    def test_state_dict_round_trip_exact_and_guard_can_fail(self):
        source = ParameterizedSlotLearner(slot_args=2, **KW)
        for tid in ("a", "b"):
            source.begin_task(tid)
        with torch.no_grad():
            for tid in ("a", "b"):
                source.task_alphas[tid].normal_()
                source.task_codes[tid].normal_()
            source.argument_matrices.normal_()
        state = {k: v.clone() for k, v in source.state_dict().items()}
        target = ParameterizedSlotLearner(slot_args=2, **KW)
        for key in state:
            if key.startswith("task_codes."):
                target.begin_task(key.split(".", 1)[1])
        target.load_state_dict(state)
        x = torch.randn(32, 8)
        for tid in ("a", "b"):
            self.assertTrue(torch.equal(source(x, tid), target(x, tid)))
        with torch.no_grad():
            target.task_alphas["a"].add_(1.0)
        self.assertFalse(torch.equal(source(x, "a"), target(x, "a")))


if __name__ == "__main__":
    unittest.main()


class FreezeMatricesTests(unittest.TestCase):
    def test_frozen_matrices_alpha_learns_but_directions_do_not(self):
        model = ParameterizedSlotLearner(slot_args=2, freeze_matrices=True, **KW)
        code, eps, alpha = model.begin_task("t")
        with torch.no_grad():
            code.view(3, 4)[:, model.pslot_index] = 5.0
        self.assertFalse(model.argument_matrices.requires_grad)
        self.assertTrue(alpha.requires_grad)
        self.assertNotIn(id(model.argument_matrices), {id(p) for p in model.shared_parameters()})
        torch.mean(model(torch.randn(16, 8), "t") ** 2).backward()
        self.assertGreater(float(alpha.grad.abs().sum()), 0.0)
        self.assertIsNone(model.argument_matrices.grad)


class MultiSlotTests(unittest.TestCase):
    def test_two_slots_at_zero_alpha_equal_ordinary_and_single_slot_layout_unchanged(self):
        multi = ParameterizedSlotLearner(slot_args=3, pslot_count=2, **KW)
        single = ParameterizedSlotLearner(slot_args=3, **KW)
        ordinary = ProspectiveLifecycleLearner(**KW)
        for m in (multi, single, ordinary):
            m.begin_task("t")
        with torch.no_grad():
            for m in (multi, single, ordinary):
                m.task_codes["t"].copy_(torch.linspace(-1, 1, multi.route_size))
        x = torch.randn(16, 8)
        self.assertTrue(torch.equal(multi(x, "t"), ordinary(x, "t")))
        self.assertEqual(tuple(multi.task_alphas["t"].shape), (2, 3))
        self.assertEqual(multi.pslot_indices, [3, 2])
        self.assertTrue(torch.equal(multi.argument_matrices, single.argument_matrices))
        self.assertEqual(set(single.state_dict()) - set(multi.state_dict()), set())

    def test_second_slot_argument_changes_output_and_gets_gradient(self):
        multi = ParameterizedSlotLearner(slot_args=2, pslot_count=2, **KW)
        code, eps, alpha = multi.begin_task("t")
        with torch.no_grad():
            code.view(3, 4)[:, 2] = 5.0  # route onto the second parameterized slot
        x = torch.randn(16, 8)
        before = multi(x, "t").detach()
        with torch.no_grad():
            alpha[1].fill_(1.0)
        after = multi(x, "t")
        self.assertFalse(torch.allclose(before, after))
        torch.mean(after ** 2).backward()
        self.assertGreater(float(multi.extra_argument_matrices.grad.abs().sum()), 0.0)
        self.assertEqual(len(multi.shared_parameters()), len(list(multi.basis.parameters())) + 2)
