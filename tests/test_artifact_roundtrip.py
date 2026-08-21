"""A reloaded model must compute what the saved model computed.

Review 55 found that `load_learner` restored promoted references but not
retirement state, so a reloaded learner added BOTH the shared
abstraction and the private residual that retirement had removed. Most
tasks in the affected artifacts were retired, so two audits were
measuring a model that never existed during training.

`model.pt` is tensor-only by project rule, which means every Python-side
container — `task_reference`, `retired`, lineage — travels separately
and can be forgotten silently. The only durable guard is a functional
one: save, reload, and require identical outputs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from row.models.prospective_models import ProspectiveLifecycleLearner


def _learner():
    return ProspectiveLifecycleLearner(
        d=8, operator_slots=4, operator_rank=4, residual_rank=1,
        task_steps=3, alpha=0.2, seed=0,
    )


def _rebuild(state, reference_table):
    """The reconstruction path the audits use."""

    model = _learner()
    count = sum(1 for k in state if k.startswith("abstractions."))
    for index in range(count):
        model.abstractions.append(
            torch.nn.Parameter(state[f"abstractions.{index}"].clone(),
                               requires_grad=False))
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    model.load_state_dict(state)
    model.eval()
    for task_id, reference in (reference_table.get("task_reference") or {}).items():
        model.task_reference[task_id] = int(reference)
    for task_id in reference_table.get("retired_task_ids") or []:
        model.retired.add(task_id)
    return model


class ArtifactRoundTripTest(unittest.TestCase):
    def _trained(self):
        # CLUSTERED residuals, not independent ones. Promotion only
        # fires when several tasks share a direction, and with random
        # residuals nothing is retired -- the first version of this
        # fixture promoted nothing and the guard would have passed
        # over an empty set.
        learner = _learner()
        generator = torch.Generator().manual_seed(0)
        ids = [f"task_{i}" for i in range(9)]
        shared = torch.randn(learner.task_residuals_size
                             if hasattr(learner, "task_residuals_size") else 1,
                             generator=generator)
        for index, task_id in enumerate(ids):
            learner.begin_task(task_id)
            with torch.no_grad():
                template = torch.randn(
                    learner.task_residuals[task_id].shape,
                    generator=torch.Generator().manual_seed(index % 3))
                jitter = 0.05 * torch.randn(
                    learner.task_residuals[task_id].shape, generator=generator)
                learner.task_residuals[task_id].copy_(template + jitter)
        probe = torch.randn(32, 8, generator=generator)
        learner.sleep(ids, probe, probe, minimum_cluster=2,
                      require_prospective=False)
        return learner, ids, probe

    def test_reloaded_model_is_functionally_identical(self) -> None:
        learner, ids, probe = self._trained()
        self.assertTrue(learner.retired, "no task was retired; the test would "
                                         "not exercise the state it guards")
        before = {t: learner(probe, t).detach().clone() for t in ids}

        table = {
            "task_reference": {k: int(v) for k, v in learner.task_reference.items()},
            "retired_task_ids": sorted(learner.retired),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model_state_dict": learner.state_dict()}, path)
            state = torch.load(path, weights_only=True)["model_state_dict"]
        rebuilt = _rebuild(state, table)

        for task_id in ids:
            self.assertTrue(
                torch.allclose(before[task_id], rebuilt(probe, task_id),
                               atol=1e-6),
                f"{task_id} computes differently after a save/load round trip",
            )

    def test_dropping_retirement_state_changes_the_computation(self) -> None:
        # The guard must be able to FAIL. If omitting retirement made no
        # difference, the test above would pass vacuously.
        learner, ids, probe = self._trained()
        before = {t: learner(probe, t).detach().clone() for t in ids}
        table = {
            "task_reference": {k: int(v) for k, v in learner.task_reference.items()},
            "retired_task_ids": [],          # the bug under review
        }
        state = learner.state_dict()
        broken = _rebuild(state, table)
        differs = [
            t for t in ids
            if not torch.allclose(before[t], broken(probe, t), atol=1e-6)
        ]
        self.assertTrue(
            differs,
            "omitting retirement state changed nothing, so this artifact "
            "cannot detect the review-55 loader bug",
        )
