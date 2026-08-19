"""Unit tests for the V3.1 PROMOTE operator."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from row.models import PromotingSharedResidualLearner


def _build() -> PromotingSharedResidualLearner:
    return PromotingSharedResidualLearner(
        d=8,
        operator_slots=4,
        operator_rank=4,
        residual_rank=2,
        task_steps=3,
        alpha=0.2,
        seed=5,
    )


def _probes(dim: int = 8):
    return (
        torch.as_tensor(
            np.random.default_rng(1).normal(size=(128, dim)), dtype=torch.float32
        ),
        torch.as_tensor(
            np.random.default_rng(2).normal(size=(128, dim)), dtype=torch.float32
        ),
    )


def _plant(model, ids, family_size, scale=0.3, noise=0.01, seed=3):
    generator = torch.Generator().manual_seed(seed)
    size = model.residual_u_size + model.residual_v_size + model.residual_b_size
    shared = torch.randn(size, generator=generator) * scale
    with torch.no_grad():
        for index, task_id in enumerate(ids):
            if index < family_size:
                value = shared + torch.randn(size, generator=generator) * noise
            else:
                value = torch.randn(size, generator=generator) * scale
            model.task_residuals[task_id].copy_(value)


class PromotingLearnerTest(unittest.TestCase):
    def test_promotes_a_planted_family_and_excludes_strangers(self) -> None:
        model = _build()
        ids = [f"t{i}" for i in range(8)]
        for task_id in ids:
            model.begin_task(task_id)
        _plant(model, ids, family_size=5)
        proposal, validation = _probes()
        record = model.sleep(ids, proposal, validation, lifetime_index=8)
        self.assertEqual(record["promoted"], 1)
        self.assertEqual(len(model.abstractions), 1)
        # The five planted members retire their private copies; the three
        # idiosyncratic tasks keep theirs.
        self.assertEqual(len(model.retired), 5)
        for task_id in ids[:5]:
            self.assertIn(task_id, model.retired)
        for task_id in ids[5:]:
            self.assertNotIn(task_id, model.retired)

    def test_migration_signature(self) -> None:
        model = _build()
        ids = [f"t{i}" for i in range(8)]
        for task_id in ids:
            model.begin_task(task_id)
        _plant(model, ids, family_size=5)
        shared_before = model.shared_parameter_count
        task_before = model.task_state_scalar_count
        model.sleep(ids, *_probes(), lifetime_index=8)
        # Task state falls, shared state rises, and the total falls: the
        # H11.1 three-sign pattern, here at the level of one sleep.
        self.assertLess(model.task_state_scalar_count, task_before)
        self.assertGreater(model.shared_parameter_count, shared_before)
        self.assertLess(
            model.shared_parameter_count + model.task_state_scalar_count,
            shared_before + task_before,
        )

    def test_refuses_when_there_is_no_recurring_structure(self) -> None:
        model = _build()
        ids = [f"t{i}" for i in range(8)]
        for task_id in ids:
            model.begin_task(task_id)
        # Every task idiosyncratic: nothing recurs, so nothing may fire.
        _plant(model, ids, family_size=0)
        record = model.sleep(ids, *_probes(), lifetime_index=8)
        self.assertEqual(record["promoted"], 0)
        self.assertEqual(len(model.abstractions), 0)
        self.assertEqual(len(model.retired), 0)

    def test_behavior_is_preserved_within_epsilon(self) -> None:
        model = _build()
        ids = [f"t{i}" for i in range(8)]
        for task_id in ids:
            model.begin_task(task_id)
        _plant(model, ids, family_size=5)
        proposal, validation = _probes()
        with torch.no_grad():
            before = {t: model(validation, t).clone() for t in ids}
        model.sleep(ids, proposal, validation, epsilon=0.02, lifetime_index=8)
        with torch.no_grad():
            for task_id in model.retired:
                after = model(validation, task_id)
                reference = before[task_id]
                denominator = float(
                    torch.mean(torch.square(reference - reference.mean(dim=0)))
                )
                deviation = float(torch.mean(torch.square(after - reference))) / denominator
                self.assertLessEqual(deviation, 0.02)

    def test_ledger_records_refusals_too(self) -> None:
        model = _build()
        ids = [f"t{i}" for i in range(6)]
        for task_id in ids:
            model.begin_task(task_id)
        _plant(model, ids, family_size=6, scale=0.3, noise=2.0)
        model.sleep(ids, *_probes(), lifetime_index=6)
        diagnostics = model.promotion_diagnostics()
        # Whatever the outcome, every candidate considered is on the record.
        self.assertEqual(
            diagnostics["candidates_considered"], len(diagnostics["ledger"])
        )
        for record in diagnostics["ledger"]:
            self.assertIn(record["decision"], {"promote", "refuse"})
            self.assertIn("value_retrospective_bits", record)

    def test_promotion_survives_a_deep_copy(self) -> None:
        import copy

        model = _build()
        ids = [f"t{i}" for i in range(8)]
        for task_id in ids:
            model.begin_task(task_id)
        _plant(model, ids, family_size=5)
        model.sleep(ids, *_probes(), lifetime_index=8)
        clone = copy.deepcopy(model)
        probe = torch.randn(16, 8)
        with torch.no_grad():
            for task_id in ids:
                self.assertTrue(torch.equal(model(probe, task_id), clone(probe, task_id)))


if __name__ == "__main__":
    unittest.main()
