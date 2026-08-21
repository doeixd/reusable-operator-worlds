"""Instrument tests for the H39 census gate C0."""

from __future__ import annotations

import types
import unittest

import numpy as np
import torch

from row.experiments.census_h39_schema import alpha_fit, schema_from_artifact
from row.models.prospective_models import ProspectiveLifecycleLearner


def _build() -> ProspectiveLifecycleLearner:
    return ProspectiveLifecycleLearner(
        d=8, operator_slots=4, operator_rank=4, residual_rank=2,
        task_steps=3, alpha=0.2, seed=5,
    )


def _task(seed: int, d: int = 8):
    rng = np.random.default_rng(seed)
    train_x = rng.normal(size=(16, d)).astype(np.float32)
    eval_x = rng.normal(size=(32, d)).astype(np.float32)
    w = rng.normal(size=(d, d)).astype(np.float32) / np.sqrt(d)
    return types.SimpleNamespace(
        task_id=f"task{seed}", train_x=train_x, train_y=np.tanh(train_x @ w),
        eval_x=eval_x, eval_y=np.tanh(eval_x @ w),
    )


class CensusTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _build()
        self.ids = []
        for i in range(6):
            tid = f"fam{i}"
            self.model.begin_task(tid)
            with torch.no_grad():
                self.model.task_residuals[tid].add_(0.1 * torch.randn_like(self.model.task_residuals[tid]))
            self.ids.append(tid)

    def test_schema_shapes_and_retired_exclusion(self):
        self.model.retired.add("fam0")
        mean, basis, info = schema_from_artifact(self.model, self.ids, 3)
        self.assertEqual(tuple(basis.shape), (self.model.task_residuals["fam1"].numel(), 3))
        self.assertEqual(info["trained_family_tasks_used"], 5)
        self.assertEqual(info["retired_family_tasks_skipped"], 1)
        self.assertEqual(info["fit_vectors"], 5)
        self.assertLessEqual(info["variance_explained"], 1.0 + 1e-6)
        # Orthonormal columns.
        gram = basis.T @ basis
        self.assertTrue(torch.allclose(gram, torch.eye(3), atol=1e-5))

    def test_rank_zero_means_maximum_available(self):
        _, basis, info = schema_from_artifact(self.model, self.ids, 0)
        self.assertEqual(info["schema_rank"], 5)
        self.assertEqual(basis.shape[1], 5)

    def test_alpha_fit_moves_alpha_and_reduces_support_loss(self):
        mean, basis, _ = schema_from_artifact(self.model, self.ids, 3)
        shared_before = [p.detach().clone() for p in self.model.shared_parameters()]
        task = _task(11)
        import row.experiments.census_h39_schema as census
        saved = census.B1_STEPS
        census.B1_STEPS = 50
        try:
            fit = alpha_fit(self.model, task, mean, basis, 16, "t")
        finally:
            census.B1_STEPS = saved
        self.assertTrue(fit["finite"])
        self.assertGreater(fit["alpha_norm"], 0.0)
        self.assertLess(fit["final_support_mse"], fit["initial_support_mse"])
        self.assertEqual(len(fit["alpha"]), 3)
        # The frozen representation and the base model are untouched.
        for before, after in zip(shared_before, self.model.shared_parameters()):
            self.assertTrue(torch.equal(before, after))
        self.assertNotIn(f"__h39census_t_{task.task_id}", self.model.task_codes)

    def test_fit_cannot_fail_to_detect_a_frozen_channel(self):
        # Companion guard: a zero basis must leave alpha unable to reduce the loss
        # beyond what the route code alone achieves, and alpha_norm stays
        # finite; the instrument is not vacuous about the channel it tests.
        mean, basis, _ = schema_from_artifact(self.model, self.ids, 2)
        zero = torch.zeros_like(basis)
        task = _task(12)
        import row.experiments.census_h39_schema as census
        saved = census.B1_STEPS
        census.B1_STEPS = 30
        try:
            with_channel = alpha_fit(self.model, task, mean, basis, 16, "a")
            without = alpha_fit(self.model, task, mean, zero, 16, "b")
        finally:
            census.B1_STEPS = saved
        self.assertEqual(without["alpha_norm"], 0.0)
        self.assertNotEqual(with_channel["final_support_mse"], without["final_support_mse"])


if __name__ == "__main__":
    unittest.main()
