"""V4 infrastructure: a library whose abstractions have a life history.

V3 established that abstractions can be born. It also established that
birth is NOISY — 5.3 abstractions for two teacher families on development
worlds, 6.2 on sealed ones, and 2.9 even in structureless controls. V4's
premise is that this need not be fixed at birth:

    invent hypotheses cheaply, then make persistence expensive.

That reframing only works if the record needed to judge persistence
exists, so this module adds the accounting first and the operators after.
Every abstraction carries a lineage entry from the moment it is created,
because once MERGE and FORK start firing a library's state is otherwise
impossible to reconstruct after the fact.

DISCIPLINE: `PromotingSharedResidualLearner` is part of the frozen V3
sealed configuration (V3_CONFIRMATION_PLAN.md, commit bcc8319). It is
subclassed here and never edited, so every V3 artifact stays reproducible
from its own fingerprint.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch

from row.models.promoting_models import PromotingSharedResidualLearner

BITS_PER_SCALAR = 8
LN2 = math.log(2.0)


@dataclass
class AbstractionRecord:
    """Everything needed to reconstruct why the library looks as it does.

    Kept for the experimenter, not the learner: no field here may enter a
    promotion, retention, or merge decision unless that decision is
    computable from the learner's own observations.
    """

    abstraction_id: int
    born_at_task: int
    supporting_tasks: list[str]
    # Tasks currently pointing at this abstraction. Distinct from the
    # supporting set: dependents change as tasks arrive, are reassigned by
    # a merge, or are released by a delete.
    dependents: list[str] = field(default_factory=list)
    # Realized savings, accumulated as reuse actually happens, in bits.
    # This is the S(A) term the retention decision is built on.
    realized_savings_bits: float = 0.0
    reuse_count: int = 0
    parents: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)
    edits: list[dict[str, object]] = field(default_factory=list)
    retired_at_task: int | None = None
    retirement_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class LifecycleLibraryLearner(PromotingSharedResidualLearner):
    """PROMOTE plus the bookkeeping a lifecycle needs (V4.1 substrate).

    This class deliberately adds NO new decision rule. It records what
    happened so that RETAIN/DELETE can later be specified against evidence
    that already exists, rather than against evidence invented at the same
    time as the operator — which is the mistake that made V3's first
    refusal criterion untestable.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lineage: dict[int, AbstractionRecord] = {}
        self.migration_ledger: list[dict[str, object]] = []
        # (library state, candidate edit, delta J, outcome) tuples. Logged
        # from the first run because it is the training set a learned
        # restructuring policy would need later, and it costs nothing now.
        self.decision_dataset: list[dict[str, object]] = []

    # ---- lineage -----------------------------------------------------

    def record_birth(
        self, abstraction_id: int, task_index: int, supporting: list[str]
    ) -> AbstractionRecord:
        record = AbstractionRecord(
            abstraction_id=abstraction_id,
            born_at_task=task_index,
            supporting_tasks=list(supporting),
            dependents=list(supporting),
        )
        self.lineage[abstraction_id] = record
        return record

    def record_reuse(self, abstraction_id: int, task_id: str, savings_bits: float) -> None:
        record = self.lineage.get(abstraction_id)
        if record is None:
            return
        record.reuse_count += 1
        record.realized_savings_bits += float(savings_bits)
        if task_id not in record.dependents:
            record.dependents.append(task_id)

    def record_migration(
        self, operation: str, task_index: int, detail: dict[str, object]
    ) -> None:
        """Log what an edit COST, not only what it saved.

        Reference rewrites, re-validation, and refitting are real work. If
        they are treated as free, every lifecycle operation looks more
        attractive than it is.
        """

        self.migration_ledger.append(
            {"operation": operation, "task_index": task_index, **detail}
        )

    def record_decision(
        self,
        operation: str,
        task_index: int,
        delta_j_bits: float,
        applied: bool,
        detail: dict[str, object] | None = None,
    ) -> None:
        self.decision_dataset.append(
            {
                "operation": operation,
                "task_index": task_index,
                "library_size": len(self.abstractions),
                "delta_j_bits": float(delta_j_bits),
                "applied": bool(applied),
                **(detail or {}),
            }
        )

    # ---- value accounting --------------------------------------------

    def retention_value_bits(self, abstraction_id: int) -> dict[str, float]:
        """The standing account for one abstraction, in bits.

        Realized savings minus what it costs to keep. The PROSPECTIVE term
        is deliberately absent here: it is a modelling choice V4.1's spec
        must freeze (evidence window, decay, or change-point), and putting
        a placeholder in the substrate would let it be chosen after seeing
        results.
        """

        record = self.lineage.get(abstraction_id)
        if record is None:
            return {}
        retention_bits = BITS_PER_SCALAR * self.residual_scalars_per_task
        return {
            "abstraction_id": abstraction_id,
            "realized_savings_bits": record.realized_savings_bits,
            "retention_bits": float(retention_bits),
            "net_realized_bits": record.realized_savings_bits - retention_bits,
            "reuse_count": float(record.reuse_count),
            "dependents": float(len(record.dependents)),
        }

    @torch.no_grad()
    def lifecycle_diagnostics(self) -> dict[str, object]:
        return {
            "library_size": len(self.abstractions),
            "lineage": [record.as_dict() for record in self.lineage.values()],
            "accounts": [
                self.retention_value_bits(index) for index in sorted(self.lineage)
            ],
            "migration_ledger": self.migration_ledger,
            "decision_dataset": self.decision_dataset,
            # Survival table inputs (feedback-26): births, and for each the
            # reuse it actually attracted. Survivorship itself is a V4.1
            # outcome and is computed by the scorer, not here.
            "births": len(self.lineage),
            "births_with_reuse": sum(
                1 for record in self.lineage.values() if record.reuse_count > 0
            ),
        }
