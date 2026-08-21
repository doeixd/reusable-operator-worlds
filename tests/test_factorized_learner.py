"""Unit tests for the H39 pilot factorized learner."""

from __future__ import annotations

import unittest

import torch

from row.models.factorized_models import FactorizedLifecycleLearner
from row.models.prospective_models import ProspectiveLifecycleLearner


def _build(**kwargs) -> FactorizedLifecycleLearner:
    return FactorizedLifecycleLearner(
        d=8, operator_slots=4, operator_rank=4, residual_rank=2,
        task_steps=3, alpha=0.2, seed=5, schema_dim=2, schema_count=3,
        schema_seed=39001, world_seed=0, **kwargs,
    )


class FactorizedLearnerTests(unittest.TestCase):
    def test_effective_residual_is_eps_plus_schema_component(self):
        model = _build()
        code, eps, alpha = model.begin_task("t", schema_index=1)
        with torch.no_grad():
            alpha.copy_(torch.tensor([0.5, -1.0]))
        expected = eps + model.schemas[1] @ alpha
        self.assertTrue(torch.allclose(model.effective_residual("t"), expected))
        self.assertEqual(model.task_schema["t"], 1)

    def test_forward_changes_with_alpha_and_matches_ordinary_at_zero(self):
        model = _build()
        ordinary = ProspectiveLifecycleLearner(
            d=8, operator_slots=4, operator_rank=4, residual_rank=2,
            task_steps=3, alpha=0.2, seed=5,
        )
        model.begin_task("t")
        ordinary.begin_task("t")
        x = torch.randn(16, 8)
        # alpha = 0: identical to the ordinary learner with the same seed.
        self.assertTrue(torch.allclose(model(x, "t"), ordinary(x, "t")))
        with torch.no_grad():
            model.task_alphas["t"].fill_(3.0)
        self.assertFalse(torch.allclose(model(x, "t"), ordinary(x, "t")))

    def test_gradients_reach_alpha_eps_and_schema(self):
        model = _build()
        _, eps, alpha = model.begin_task("t", schema_index=0)
        x = torch.randn(16, 8)
        loss = torch.mean(model(x, "t") ** 2)
        loss.backward()
        self.assertGreater(float(alpha.grad.abs().sum()), 0.0)
        self.assertGreater(float(eps.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.schemas[0].grad)

    def test_shared_parameters_and_counts(self):
        model = _build()
        self.assertEqual(len(model.shared_parameters()),
                         len(list(model.basis.parameters())) + 3)
        frozen = _build(freeze_schema=True)
        self.assertEqual(len(frozen.shared_parameters()), len(list(frozen.basis.parameters())))
        self.assertFalse(frozen.schemas[0].requires_grad)
        model.begin_task("a")
        model.begin_task("b", schema_index=2)
        residual = model.task_residuals["a"].numel()
        self.assertEqual(model.task_state_scalar_count,
                         2 * (model.route_size + residual + 2))
        self.assertEqual(model.schema_diagnostics()["tasks_per_schema"], [1, 0, 1])

    def test_forget_task_removes_alpha_and_assignment(self):
        model = _build()
        model.begin_task("t", schema_index=2)
        model.forget_task("t")
        self.assertNotIn("t", model.task_alphas)
        self.assertNotIn("t", model.task_schema)
        self.assertNotIn("t", model.task_codes)

    def test_schema_init_is_seeded_and_scaled(self):
        a, b = _build(), _build()
        self.assertTrue(torch.equal(a.schemas[0], b.schemas[0]))
        self.assertFalse(torch.equal(a.schemas[0], a.schemas[1]))
        std = float(a.schemas[0].std())
        self.assertAlmostEqual(std, 1e-2 / (2 ** 0.5), delta=2e-3)


if __name__ == "__main__":
    unittest.main()
