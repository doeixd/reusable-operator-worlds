"""Unit tests for the V4.1 lifecycle operator (re-home, then retire orphans)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from row.models import LifecycleLibraryLearner


def _build() -> LifecycleLibraryLearner:
    return LifecycleLibraryLearner(
        d=8, operator_slots=4, operator_rank=4, residual_rank=2,
        task_steps=3, alpha=0.2, seed=5,
    )


def _probe(dim: int = 8) -> torch.Tensor:
    return torch.as_tensor(
        np.random.default_rng(1).normal(size=(128, dim)), dtype=torch.float32
    )


def _library_with_duplicates(model, ids, duplicate: bool):
    """Give the model two abstractions, identical or not, with dependents."""

    size = model.residual_u_size + model.residual_v_size + model.residual_b_size
    generator = torch.Generator().manual_seed(11)
    first = torch.randn(size, generator=generator) * 0.3
    second = first.clone() if duplicate else torch.randn(size, generator=generator) * 0.3
    with torch.no_grad():
        for value in (first, second):
            model.abstractions.append(torch.nn.Parameter(value, requires_grad=False))
    for position, task_id in enumerate(ids):
        model.task_reference[task_id] = 0 if position < len(ids) // 2 else 1
        model.retired.add(task_id)
    model.sync_lineage(0)


class LifecycleOperatorTest(unittest.TestCase):
    def test_redundant_abstraction_is_rehomed_and_retired(self) -> None:
        model = _build()
        ids = [f"t{i}" for i in range(8)]
        for task_id in ids:
            model.begin_task(task_id)
        _library_with_duplicates(model, ids, duplicate=True)
        report = model.consolidate(_probe(), task_index=24, grace=0)
        # Everyone can be served by one abstraction, so the other is
        # orphaned and collected.
        self.assertGreater(report["rehomed"], 0)
        self.assertEqual(report["deleted"], 1)
        self.assertEqual(report["live_library"], 1)

    def test_distinct_abstractions_are_both_kept(self) -> None:
        model = _build()
        ids = [f"t{i}" for i in range(8)]
        for task_id in ids:
            model.begin_task(task_id)
        _library_with_duplicates(model, ids, duplicate=False)
        report = model.consolidate(_probe(), task_index=24, grace=0)
        # Neither can serve the other's dependents, so nothing is orphaned.
        self.assertEqual(report["deleted"], 0)
        self.assertEqual(report["live_library"], 2)

    def test_load_bearing_abstractions_are_never_stranded(self) -> None:
        model = _build()
        ids = [f"t{i}" for i in range(8)]
        for task_id in ids:
            model.begin_task(task_id)
        _library_with_duplicates(model, ids, duplicate=False)
        model.consolidate(_probe(), task_index=24, grace=0)
        # Every retired task still points at a live abstraction.
        live = set(range(len(model.abstractions))) - model.retired_abstractions()
        for task_id in model.retired:
            self.assertIn(int(model.task_reference[task_id]), live)

    def test_grace_period_protects_newborns(self) -> None:
        model = _build()
        ids = [f"t{i}" for i in range(8)]
        for task_id in ids:
            model.begin_task(task_id)
        _library_with_duplicates(model, ids, duplicate=True)
        # Born at 0, consolidating at 4, grace 8: too young to retire.
        report = model.consolidate(_probe(), task_index=4, grace=8)
        self.assertEqual(report["deleted"], 0)

    def test_retirement_leaves_the_final_description(self) -> None:
        model = _build()
        ids = [f"t{i}" for i in range(8)]
        for task_id in ids:
            model.begin_task(task_id)
        _library_with_duplicates(model, ids, duplicate=True)
        before = model.shared_parameter_count
        model.consolidate(_probe(), task_index=24, grace=0)
        self.assertLess(model.shared_parameter_count, before)

    def test_every_edit_is_recorded(self) -> None:
        model = _build()
        ids = [f"t{i}" for i in range(8)]
        for task_id in ids:
            model.begin_task(task_id)
        _library_with_duplicates(model, ids, duplicate=True)
        model.consolidate(_probe(), task_index=24, grace=0)
        operations = {entry["operation"] for entry in model.migration_ledger}
        self.assertIn("rehome", operations)
        self.assertIn("delete", operations)
        self.assertTrue(model.decision_dataset)
        retired = model.retired_abstractions()
        for index in retired:
            self.assertIsNotNone(model.lineage[index].retirement_reason)


if __name__ == "__main__":
    unittest.main()


class ToleranceScalingTest(unittest.TestCase):
    """The substitution tolerance must be contribution-relative.

    Regression guard for the retracted V4.1 H14 result (PREDICTIONS.md).
    Normalizing deviation against TOTAL OUTPUT VARIANCE made every weak
    abstraction substitutable for every other, because an abstraction
    contributes a fraction of a percent of that variance. The symptom is
    that consolidation retires abstractions whose replacement changes
    behaviour by MORE than the abstraction itself was contributing.
    """

    def test_weak_but_distinct_abstractions_are_not_consolidated(self) -> None:
        model = _build()
        ids = [f"weak{i}" for i in range(8)]
        for task_id in ids:
            model.begin_task(task_id)
        size = model.residual_u_size + model.residual_v_size + model.residual_b_size
        generator = torch.Generator().manual_seed(23)
        with torch.no_grad():
            for _ in range(2):
                # Small scale: each contributes little to total output, but
                # the two are drawn independently, so they are DISTINCT.
                model.abstractions.append(
                    torch.nn.Parameter(
                        torch.randn(size, generator=generator) * 0.01,
                        requires_grad=False,
                    )
                )
        for position, task_id in enumerate(ids):
            model.task_reference[task_id] = 0 if position < 4 else 1
            model.retired.add(task_id)
        model.sync_lineage(0)

        report = model.consolidate(_probe(), task_index=32, epsilon=0.02)

        self.assertEqual(
            report["deleted"], 0,
            "weak-but-distinct abstractions were consolidated: the tolerance "
            "is being scaled against total output variance again",
        )
        self.assertEqual(report["rehomed"], 0)

    def test_null_edit_is_never_within_tolerance(self) -> None:
        """Dropping an abstraction costs exactly its own contribution.

        Under a contribution-relative denominator that ratio is 1.0, so no
        sane epsilon admits it. If this fails, the oracle's covers() would
        certify the empty set and every cover it reports is vacuous.
        """

        model = _build()
        ids = [f"t{i}" for i in range(6)]
        for task_id in ids:
            model.begin_task(task_id)
        _library_with_duplicates(model, ids, duplicate=False)
        probe = _probe()
        with torch.no_grad():
            for task_id in ids:
                current = model.task_reference[task_id]
                base = model.forward(probe, task_id)
                model.task_reference.pop(task_id)
                dropped = model.forward(probe, task_id)
                model.task_reference[task_id] = current
                contribution = float(torch.mean(torch.square(base - dropped)))
                self.assertAlmostEqual(
                    contribution / max(contribution, 1e-12), 1.0, places=6
                )


class PromotionSnapshotTest(unittest.TestCase):
    """H29 provenance: P_0 must survive the sleep that consumes it."""

    def _learner(self):
        return LifecycleLibraryLearner(
            d=8, operator_slots=4, operator_rank=4, residual_rank=1,
            task_steps=3, alpha=0.2, seed=0,
        )

    def test_snapshot_records_members_and_the_born_abstraction(self) -> None:
        import torch

        learner = self._learner()
        generator = torch.Generator().manual_seed(0)
        ids = [f"task_{i}" for i in range(8)]
        for task_id in ids:
            learner.begin_task(task_id)
            with torch.no_grad():
                learner.task_residuals[task_id].normal_(generator=generator)
        probe = torch.randn(32, 8, generator=generator)
        learner.sleep(ids, probe, probe, minimum_cluster=2, require_prospective=False)
        for index, snapshot in learner.promotion_snapshots.items():
            self.assertTrue(snapshot["members"], "promotion recorded no members")
            self.assertEqual(
                len(snapshot["member_residuals"]), len(snapshot["members"]),
                "a member's pre-promotion residual went unrecorded",
            )
            self.assertTrue(torch.equal(
                snapshot["born"], learner.abstractions[index].detach()),
                "P_1 does not match the abstraction as born",
            )

    def test_abstractions_never_train_so_p2_equals_p1(self) -> None:
        # H29's restructuring term is structurally zero here, and this
        # test is what makes that a property rather than an assumption.
        import torch

        learner = self._learner()
        generator = torch.Generator().manual_seed(1)
        ids = [f"task_{i}" for i in range(8)]
        for task_id in ids:
            learner.begin_task(task_id)
            with torch.no_grad():
                learner.task_residuals[task_id].normal_(generator=generator)
        probe = torch.randn(32, 8, generator=generator)
        learner.sleep(ids, probe, probe, minimum_cluster=2, require_prospective=False)
        for abstraction in learner.abstractions:
            self.assertFalse(
                abstraction.requires_grad,
                "a promoted abstraction is trainable; P_2 != P_1 and H29's "
                "restructuring term would no longer be structurally zero",
            )
