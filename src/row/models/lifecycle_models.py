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
        # P_0 for H29: the member residuals as they stood at the sleep
        # that consumed them. A finished artifact otherwise holds only
        # the born abstraction and whatever the members' residuals
        # decayed to afterwards, so the "before" state of a promotion is
        # unrecoverable — the same class of provenance gap as the
        # missing `task_reference` table that once voided a coding
        # audit. Snapshotting here costs one tensor copy per promotion.
        self.promotion_snapshots: dict[int, dict[str, object]] = {}

    def sleep(self, task_ids, *args, **kwargs) -> dict[str, object]:
        """Promote as V3 does, and record what each promotion consumed.

        Overridden in the subclass so the frozen V3 learner is not
        touched. The snapshot is pure bookkeeping: it reads state before
        and after `super().sleep` and changes no decision.
        """

        before = {
            task_id: tensor.detach().clone()
            for task_id, tensor in self.task_residuals.items()
        }
        existing = len(self.abstractions)
        result = super().sleep(task_ids, *args, **kwargs)
        for index in range(existing, len(self.abstractions)):
            record = self.lineage.get(index)
            # The parent computes its clusters internally and the lineage
            # is synced later, so `supporting_tasks` is usually empty at
            # this moment -- the first version of this snapshot recorded
            # zero member residuals for exactly that reason. Fall back to
            # the whole pre-sleep residual population, which is the
            # honest P_0 for "was the structure there before promotion".
            members = list(record.supporting_tasks) if record else []
            residuals = [before[t] for t in members if t in before]
            if not residuals:
                members = sorted(before)
                residuals = [before[t] for t in members]
            self.promotion_snapshots[index] = {
                # P_0: what tasks privately held before promotion.
                "members": members,
                "member_residuals": residuals,
                # P_1: the abstraction as born. Promoted abstractions
                # carry requires_grad=False and sit in no optimizer
                # group, so P_2 == P_1 identically in this learner and
                # H29's restructuring term is structurally zero rather
                # than merely small.
                "born": self.abstractions[index].detach().clone(),
            }
        return result

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
    def sync_lineage(self, task_index: int) -> None:
        """Reconcile lineage with the live reference table.

        `task_reference` and `retired` are plain Python containers, so they
        are NOT in `state_dict` and do not survive a save/load round trip.
        V3 artifacts therefore cannot say which task depends on which
        abstraction, which silently turns any dependency-based analysis
        into an analysis of an empty library. Lineage is the durable
        record, and it is written to the artifact.
        """

        for index in range(len(self.abstractions)):
            if index not in self.lineage:
                supporting = [
                    task_id
                    for task_id, reference in self.task_reference.items()
                    if reference == index
                ]
                self.record_birth(index, task_index, supporting)
        for record in self.lineage.values():
            record.dependents = [
                task_id
                for task_id, reference in self.task_reference.items()
                if reference == record.abstraction_id
            ]

    # ---- the V4.1 operator: re-home, then retire orphans ---------------

    @torch.no_grad()
    def consolidate(
        self,
        probe: torch.Tensor,
        task_index: int,
        epsilon: float = 0.02,
        kappa: float = 0.0,
        tasks_total: int = 64,
        grace: int = 8,
    ) -> dict[str, object]:
        """One lifecycle pass: RE-HOME dependents, then DELETE orphans.

        The validity gate found that V4.1's opportunity is not deleting
        load-bearing abstractions — promotion fires only when it saves
        bits, so its inverse can never pay — but deleting REDUNDANT ones.
        Measured on the frozen testbed, every dependent of every
        abstraction could be served by some other abstraction within
        epsilon, so the four to six abstractions V3 creates are redundant
        estimates of one or two concepts.

        Re-homing is therefore the enabling step and deletion is the
        collection step. Consolidation is greedy and deterministic: each
        dependent migrates to the compatible abstraction with the most
        dependents, ties broken by lowest index, so edit order is not a
        hidden hyperparameter.

        AGE-NEUTRAL by construction (V4 spec H17): the only age term is the
        grace period, applied identically to every abstraction. A rule that
        made old abstractions harder to delete would build the hysteresis
        H17 is meant to discover.
        """

        if not self.abstractions:
            return {"task_index": task_index, "rehomed": 0, "deleted": 0}

        baselines = {
            task_id: self.forward(probe, task_id)
            for task_id in list(self.task_reference)
        }

        def _swap(task_id: str, reference: int | None) -> torch.Tensor:
            previous = self.task_reference.get(task_id)
            if reference is None:
                self.task_reference.pop(task_id, None)
            else:
                self.task_reference[task_id] = reference
            after = self.forward(probe, task_id)
            if previous is None:
                self.task_reference.pop(task_id, None)
            else:
                self.task_reference[task_id] = previous
            return after

        # What the current abstraction actually buys this task. This, not
        # total output variance, is the denominator: epsilon licenses the
        # loss of a CONTRIBUTION, so it must be measured against that
        # contribution. Normalizing against total scale made every
        # abstraction substitutable for every other (each contributes
        # ~0.2% of output variance) and admitted the null edit of deleting
        # the whole library. See PREDICTIONS.md, "V4.1 H14 — RETRACTED".
        contribution = {
            task_id: float(
                torch.mean(torch.square(baselines[task_id] - _swap(task_id, None)))
            )
            for task_id in baselines
        }

        def deviation(task_id: str, reference: int) -> float:
            after = _swap(task_id, reference)
            base = baselines[task_id]
            denominator = max(contribution[task_id], 1e-12)
            return float(torch.mean(torch.square(after - base))) / denominator

        # --- RE-HOME -----------------------------------------------------
        population: dict[int, int] = {}
        for reference in self.task_reference.values():
            population[int(reference)] = population.get(int(reference), 0) + 1
        order = sorted(
            range(len(self.abstractions)),
            key=lambda index: (-population.get(index, 0), index),
        )
        rehomed = 0
        for task_id in list(self.task_reference):
            current = int(self.task_reference[task_id])
            for candidate in order:
                if candidate == current:
                    break  # already on the most populous compatible target
                if deviation(task_id, candidate) <= epsilon:
                    self.task_reference[task_id] = candidate
                    self.record_migration(
                        "rehome",
                        task_index,
                        {"task": task_id, "from": current, "to": candidate},
                    )
                    rehomed += 1
                    break

        # --- DELETE ------------------------------------------------------
        self.sync_lineage(task_index)
        deleted = []
        for index in sorted(self.lineage, reverse=True):
            record = self.lineage[index]
            if record.retired_at_task is not None:
                continue
            if task_index - record.born_at_task < grace:
                continue
            if record.dependents:
                continue  # load-bearing; deleting it would strand dependents
            retention_bits = BITS_PER_SCALAR * self.residual_scalars_per_task
            # An orphan contributes its bits to the final description and
            # buys nothing, so retention value is negative for any positive
            # price of description or occupancy.
            value = -(
                retention_bits * LN2
                + kappa * retention_bits * max(0, tasks_total - task_index)
            )
            self.record_decision(
                "delete", task_index, value / LN2, applied=True,
                detail={"abstraction": index, "dependents": 0},
            )
            record.retired_at_task = task_index
            record.retirement_reason = "orphaned after re-homing"
            deleted.append(index)
            self.record_migration(
                "delete", task_index, {"abstraction": index, "reference_rewrites": 0}
            )
        return {
            "task_index": task_index,
            "rehomed": rehomed,
            "deleted": len(deleted),
            "deleted_ids": deleted,
            "live_library": len(self.abstractions) - len(self.retired_abstractions()),
        }

    def retired_abstractions(self) -> set[int]:
        return {
            index
            for index, record in self.lineage.items()
            if record.retired_at_task is not None
        }

    @property
    def shared_parameter_count(self) -> int:
        """Retired abstractions leave the final description."""

        retired = self.retired_abstractions()
        return sum(p.numel() for p in self.basis.parameters()) + sum(
            parameter.numel()
            for index, parameter in enumerate(self.abstractions)
            if index not in retired
        )

    @torch.no_grad()
    def lifecycle_diagnostics(self) -> dict[str, object]:
        return {
            "task_reference": {k: int(v) for k, v in self.task_reference.items()},
            "retired_task_ids": sorted(self.retired),
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

    # ---- the V4.2 operator: synthetic factorization ---------------------

    def factorize(
        self,
        probe: torch.Tensor,
        rank: int = 2,
        steps: int = 600,
        lr: float = 0.02,
    ) -> dict[str, object]:
        """Replace K distinct abstractions with ONE parameterized family.

        V4.1's gates established that this library holds no redundancy to
        eliminate: no abstraction substitutes for another, and compacting
        by re-homing is net negative. The abstractions are behaviorally
        DISTINCT. The V4.2 question is different and does not require
        redundancy:

            A_i(z)  ~=  A(z ; alpha_i)

        Is there a shared parent plus a small per-abstraction coordinate
        that reproduces each abstraction's function more cheaply than
        storing every abstraction outright? That is anti-unification, not
        deletion -- nothing is forgotten, the vocabulary changes from a
        set of atoms to one operator with arguments.

        FITTED IN BEHAVIOR SPACE, never parameter space. The innovation is
        nonlinear in (U, V), and V3 measured that a parameter mean recovers
        11.9% of behavioral value where a functional fit recovers 53.4%:
        gauge-inequivalent parameters do not average. So `C + alpha_i @ B`
        is a parameterization whose BEHAVIOR is matched by gradient
        descent, not an assertion that abstractions are linear in
        parameters.

        Returns the fit and its price. It does NOT mutate the library --
        pricing must precede adoption, which is the discipline the
        retracted V4.1 H14 result violated.
        """

        count = len(self.abstractions)
        if count < 2:
            return {"applied": False, "reason": "fewer than two abstractions"}

        with torch.enable_grad():
            stacked = torch.stack([p.detach() for p in self.abstractions])
            centre = torch.nn.Parameter(stacked.mean(dim=0).clone())
            spread = stacked - stacked.mean(dim=0)
            # Initialize the family directions from the observed spread so
            # the optimizer starts on the manifold rather than at zero.
            _, _, right = torch.linalg.svd(spread, full_matrices=False)
            basis = torch.nn.Parameter(right[:rank].clone())
            alpha = torch.nn.Parameter(spread @ right[:rank].T)

            with torch.no_grad():
                targets = torch.stack(
                    [
                        torch.stack(
                            [
                                self._innovation(
                                    probe, *self._split_residual(vector), step
                                )
                                for step in range(self.task_steps)
                            ]
                        )
                        for vector in stacked
                    ]
                )

            optimizer = torch.optim.Adam([centre, basis, alpha], lr=lr)
            for _ in range(steps):
                optimizer.zero_grad()
                rebuilt = centre.unsqueeze(0) + alpha @ basis
                predicted = torch.stack(
                    [
                        torch.stack(
                            [
                                self._innovation(
                                    probe, *self._split_residual(rebuilt[index]), step
                                )
                                for step in range(self.task_steps)
                            ]
                        )
                        for index in range(count)
                    ]
                )
                loss = torch.mean(torch.square(predicted - targets))
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            rebuilt = (centre + alpha @ basis).detach()
            scale = float(torch.mean(torch.square(targets)).clamp_min(1e-12))
            distortion = float(loss.detach()) / scale
            width = self.residual_u_size + self.residual_v_size + self.residual_b_size
            before = BITS_PER_SCALAR * width * count
            after = BITS_PER_SCALAR * (width * (1 + rank) + rank * count)
        return {
            "applied": False,
            "rank": rank,
            "abstractions": count,
            "relative_distortion": distortion,
            "bits_before": before,
            "bits_after": after,
            "bits_saved": before - after,
            "rebuilt": rebuilt,
        }

    @torch.no_grad()
    def quantize_to_budget(self, bits_total: float) -> list[torch.Tensor]:
        """Each abstraction stored privately at a MATCHED total bit budget.

        The independent-compression null for V4.2. If the atoms can simply
        be stored at coarser precision for the same bits the factorization
        costs, then any apparent gain is "each abstraction was
        individually overparameterized", not cross-abstraction reuse.
        Symmetric per-tensor quantization, the scheme already used for
        this project's retention proxies.
        """

        count = len(self.abstractions)
        width = self.residual_u_size + self.residual_v_size + self.residual_b_size
        per_scalar = max(1.0, bits_total / max(1, count * width))
        levels = max(2, int(2 ** min(16.0, per_scalar)))
        out = []
        for parameter in self.abstractions:
            vector = parameter.detach()
            scale = float(vector.abs().max().clamp_min(1e-12))
            step = 2 * scale / (levels - 1)
            out.append(torch.round(vector / step) * step)
        return out

    def fit_argument(
        self,
        target: torch.Tensor,
        centre: torch.Tensor,
        basis: torch.Tensor,
        probe: torch.Tensor,
        steps: int = 300,
        lr: float = 0.02,
    ) -> torch.Tensor:
        """Infer ONLY a held-out abstraction's arguments in a known family.

        The prospective test (V4.2 Gate 3, the analogue of V3's H11.3): if
        the family is real, a new abstraction costs `rank` scalars rather
        than a whole operator. Fitted in behavior space on `probe`; the
        caller must evaluate on DISJOINT probes, or this measures
        memorization of the proposal set.
        """

        with torch.enable_grad():
            alpha = torch.nn.Parameter(torch.zeros(basis.shape[0]))
            with torch.no_grad():
                wanted = torch.stack([
                    self._innovation(probe, *self._split_residual(target), step)
                    for step in range(self.task_steps)
                ])
            optimizer = torch.optim.Adam([alpha], lr=lr)
            for _ in range(steps):
                optimizer.zero_grad()
                vector = centre + alpha @ basis
                got = torch.stack([
                    self._innovation(probe, *self._split_residual(vector), step)
                    for step in range(self.task_steps)
                ])
                loss = torch.mean(torch.square(got - wanted))
                loss.backward()
                optimizer.step()
        return alpha.detach()

    @torch.no_grad()
    def prune_by_value(
        self,
        probe: torch.Tensor,
        task_index: int,
        tasks_total: int = 64,
        kappa: float = 0.0,
        grace: int = 0,
        sigma: float = 0.1,
        examples_per_task: int = 128,
    ) -> dict[str, object]:
        """Refuse promotions not worth their own storage price.

        Measured 2026-08-19, per-abstraction VALUE — the loss its
        dependents would pay without it — separates real abstractions
        from PROMOTE's false positives: in structureless controls where
        every promotion is spurious, 90% fall below one residual's carry
        cost, against 33% in structured worlds.

        The threshold is NOT tuned. It is the abstraction's own price,
        `8 bits x residual width` held to the end of the lifetime, so the
        rule is "an abstraction must be worth at least what it costs to
        keep" — the MDL criterion used everywhere else in this project.
        PROMOTE cannot apply it: its acceptance test is behavioral
        deviation plus a generalization check, and an abstraction can
        pass both while still not earning its bits.

        ONLINE-ADMISSIBLE. Value is measured against the dependents that
        exist NOW, on the learner's own probe, with no teacher quantity
        and no future information. It is therefore noisier than the
        post-hoc diagnostic, which is exactly what this operator is meant
        to test.

        AGE-NEUTRAL apart from the grace period, applied identically to
        every abstraction (V4 spec H17).
        """

        self.sync_lineage(task_index)
        width = self.residual_u_size + self.residual_v_size + self.residual_b_size
        carry = BITS_PER_SCALAR * width
        retired, kept = [], []
        for index in sorted(self.lineage, reverse=True):
            record = self.lineage[index]
            if record.retired_at_task is not None:
                continue
            if task_index - record.born_at_task < grace:
                continue
            dependents = [t for t, r in self.task_reference.items() if r == index]
            value = 0.0
            for task_id in dependents:
                with_ref = self.forward(probe, task_id)
                self.task_reference.pop(task_id)
                without = self.forward(probe, task_id)
                self.task_reference[task_id] = index
                # CONVERT TO NATS. A mean squared deviation on the probe is
                # not comparable to a description length; comparing them
                # directly makes every abstraction fail, because an MSE of
                # order 1e-2 is always below a price of order 1e3. Under
                # the project's Gaussian likelihood the per-example cost of
                # a squared deviation is `d * dev / (2 sigma^2)`, and the
                # abstraction buys that on every example of every
                # dependent task.
                deviation = float(torch.mean(torch.square(with_ref - without)))
                value += (
                    deviation * with_ref.shape[-1] * examples_per_task
                    / (2.0 * sigma * sigma)
                )
            # Price the deviation in the same currency as the bits: the
            # occupancy term makes an abstraction dearer the longer it
            # must be carried.
            price = carry * LN2 + kappa * carry * max(0, tasks_total - task_index)
            self.record_decision(
                "prune_by_value", task_index, (value - price) / LN2,
                applied=value < price,
                detail={"abstraction": index, "dependents": len(dependents),
                        "value": value, "price": price},
            )
            if value < price and record.born_at_task == task_index:
                # Reversal is free ONLY for a promotion made at THIS sleep:
                # the dependents' private residuals are still current, so
                # un-retiring them restores the pre-promotion state at no
                # cost. Reversing an OLD promotion instead pays n private
                # residuals to reclaim one shared one and can never pay --
                # the trap V4.1's compaction gate fell into. So this is a
                # promotion FILTER, not a garbage collector.
                for task_id in dependents:
                    self.task_reference.pop(task_id, None)
                    self.retired.discard(task_id)
                record.dependents = []
                record.retired_at_task = task_index
                record.retirement_reason = "value below carry cost at birth"
                retired.append(index)
                self.record_migration(
                    "prune_by_value", task_index, {"abstraction": index}
                )
            else:
                kept.append(index)
        return {
            "task_index": task_index,
            "pruned": len(retired),
            "pruned_ids": retired,
            "kept": len(kept),
            "live_library": len(self.abstractions) - len(self.retired_abstractions()),
        }

    def select_reference(self, task_id: str, x: torch.Tensor, y: torch.Tensor):
        """Reuse only from the LIVE library.

        V3's `select_reference` scans every abstraction ever created,
        because V3 has no notion of retirement. A retired abstraction that
        a new task can still adopt is not deleted in any meaningful sense,
        and measuring reacquisition cost against such a library measures
        nothing. Overridden here rather than edited there, so the frozen
        V3 configuration stays reproducible.
        """

        retired = self.retired_abstractions()
        if not retired:
            return super().select_reference(task_id, x, y)
        live = [i for i in range(len(self.abstractions)) if i not in retired]
        if not live:
            self.task_reference.pop(task_id, None)
            self.retired.discard(task_id)
            return None
        previous = self.task_reference.get(task_id)
        was_retired = task_id in self.retired
        best_index, best_loss = None, None
        with torch.no_grad():
            for index in [None, *live]:
                if index is None:
                    self.task_reference.pop(task_id, None)
                else:
                    self.task_reference[task_id] = index
                loss = float(torch.mean(torch.square(self.forward(x, task_id) - y)))
                if best_loss is None or loss < best_loss:
                    best_index, best_loss = index, loss
        if previous is None:
            self.task_reference.pop(task_id, None)
        else:
            self.task_reference[task_id] = previous
        if was_retired:
            self.retired.add(task_id)
        else:
            self.retired.discard(task_id)
        return best_index

    @torch.no_grad()
    def force_retire_all(
        self, task_index: int, only_largest: bool = False
    ) -> dict[str, object]:
        """Delete the whole live library, to price REACQUISITION.

        Not a policy — an intervention. Pairing a run against an identical
        run whose library survives the gap gives `C_reacquire` directly:
        the extra prequential loss and description the deleted arm pays
        while it rebuilds what the retained arm already had.
        """

        self.sync_lineage(task_index)
        killed = []
        candidates = list(self.lineage)
        if only_largest:
            # Delete just the most-depended-upon abstraction: the dormant
            # family's own object. Deleting the WHOLE library conflates the
            # carry cost of ~5 abstractions with the reacquisition cost of
            # one, which makes V_retain negative by construction.
            live = [i for i in candidates if self.lineage[i].retired_at_task is None]
            if not live:
                return {"task_index": task_index, "retired": []}
            candidates = [max(live, key=lambda i: (len(self.lineage[i].dependents), -i))]
        for index in candidates:
            record = self.lineage[index]
            if record.retired_at_task is None:
                record.retired_at_task = task_index
                record.retirement_reason = "forced (reacquisition probe)"
                killed.append(index)
        for task_id, reference in list(self.task_reference.items()):
            if reference in killed:
                self.task_reference.pop(task_id, None)
                self.retired.discard(task_id)
        self.record_migration("force_retire_all", task_index, {"abstractions": killed})
        return {"task_index": task_index, "retired": killed}
