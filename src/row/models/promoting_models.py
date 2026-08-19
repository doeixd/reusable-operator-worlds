"""V3.1 PROMOTE: task-local innovation becomes shared, addressable capacity.

The learner is the frozen H9 shared-residual architecture plus a growable
library of rank-2 ABSTRACTIONS. During sleep it looks for task innovations
that recur functionally, fits one shared abstraction to each recurring
cluster, and — if the substitution preserves its own behavior and shortens
the code — hands the member tasks a reference to that abstraction and
retires their private copies.

    before:  task i keeps its own rank-2 residual        (198 scalars each)
    after:   library keeps A once                        (198 scalars total)
             task i keeps a reference                    (a few bits each)

Three properties the V2 failures demand:
  * detection is FUNCTIONAL (behavioral distance on probe inputs), never
    parameter identity;
  * the accept test is behavioral SUBSTITUTABILITY on a probe set disjoint
    from the one used to propose and fit, so a promotion cannot be an
    overfit to its own evidence;
  * every candidate is logged with its value decomposition whether it fires
    or not, because refusal is part of the hypothesis.

Nothing here reads teacher structure. Behavior preservation is measured
against the learner's OWN current predictions, which is what makes the
operation legal during a lifetime.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from row.models.learned_models import SharedParentResidualLearner

BITS_PER_SCALAR = 8
LN2 = math.log(2.0)


class PromotingSharedResidualLearner(SharedParentResidualLearner):
    """Shared-residual learner whose library can grow by promotion."""

    def __init__(
        self,
        d: int,
        operator_slots: int,
        operator_rank: int,
        residual_rank: int,
        task_steps: int,
        alpha: float,
        seed: int,
        learnable_alpha: bool = True,
        activation: str = "tanh",
    ) -> None:
        super().__init__(
            d=d,
            operator_slots=operator_slots,
            operator_rank=operator_rank,
            residual_rank=residual_rank,
            task_steps=task_steps,
            alpha=alpha,
            seed=seed,
            learnable_alpha=learnable_alpha,
            activation=activation,
        )
        self.abstractions = nn.ParameterList()
        self.task_reference: dict[str, int] = {}
        # Tasks whose private residual has been retired in favour of a
        # reference; they contribute no residual scalars to the code.
        self.retired: set[str] = set()
        self.promotion_ledger: list[dict[str, object]] = []
        self.residual_scalars_per_task = (
            self.residual_u_size + self.residual_v_size + self.residual_b_size
        )

    # ---- forward -----------------------------------------------------

    def _split_residual(self, flat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        u, v, b = torch.split(
            flat,
            (self.residual_u_size, self.residual_v_size, self.residual_b_size),
        )
        return (
            u.reshape(self.task_steps, self.d, self.residual_rank),
            v.reshape(self.task_steps, self.residual_rank, self.d),
            b.reshape(self.task_steps, self.residual_rank),
        )

    def _innovation(self, z: Tensor, u: Tensor, v: Tensor, b: Tensor, step: int) -> Tensor:
        hidden = torch.tanh(torch.nn.functional.linear(z, v[step], b[step]))
        return torch.nn.functional.linear(hidden, u[step])

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        route, own_u, own_v, own_b = self._unpack(task_id)
        coefficients = torch.softmax(route, dim=-1)
        reference = self.task_reference.get(task_id)
        shared = (
            self._split_residual(self.abstractions[reference])
            if reference is not None
            else None
        )
        retired = task_id in self.retired
        z = x
        for step in range(self.task_steps):
            candidates = torch.stack([operator(z) for operator in self.basis], dim=0)
            parent = torch.sum(
                coefficients[step].view(self.operator_slots, 1, 1) * candidates, dim=0
            )
            residual = torch.zeros_like(parent)
            if shared is not None:
                residual = residual + self._innovation(z, *shared, step)
            if not retired:
                residual = residual + self._innovation(z, own_u, own_v, own_b, step)
            z = parent + residual
        return z

    # ---- sleep -------------------------------------------------------

    @torch.no_grad()
    def _residual_functions(self, task_ids: Sequence[str], probe: Tensor) -> Tensor:
        rows = []
        for task_id in task_ids:
            _, u, v, b = self._unpack(task_id)
            rows.append(
                torch.cat(
                    [self._innovation(probe, u, v, b, s).reshape(-1) for s in range(self.task_steps)]
                )
            )
        return torch.stack(rows)

    @torch.no_grad()
    def _cluster(self, functions: Tensor, candidate_ks: Sequence[int]) -> list[list[int]]:
        """Propose candidate groupings by k-means over several k.

        Threshold-based agglomeration was tried first and proposed nothing:
        it needs an absolute similarity scale, and after centering out the
        task-invariant component the within-family similarity in this world
        is a few hundredths. Clustering is therefore permissive and the
        BEHAVIORAL accept test does the filtering, with the selection charge
        log2(M) paying for the extra candidates this generates.
        """

        centered = functions - functions.mean(dim=0, keepdim=True)
        normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-12)
        data = normalized.cpu().numpy()
        proposals: list[list[int]] = []
        for k in candidate_ks:
            if k < 2 or k > len(data):
                continue
            generator = __import__("numpy").random.default_rng(k * 7919)
            centers = data[generator.choice(len(data), k, replace=False)]
            labels = None
            for _ in range(50):
                distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
                labels = distances.argmin(axis=1)
                for index in range(k):
                    if (labels == index).any():
                        centers[index] = data[labels == index].mean(axis=0)
            for index in range(k):
                members = [i for i in range(len(data)) if labels[i] == index]
                if members and members not in proposals:
                    proposals.append(members)
        return proposals

    def _fit_abstraction(
        self, member_ids: Sequence[str], probe: Tensor, steps: int = 300, lr: float = 0.02
    ) -> Tensor:
        """Best rank-2 FUNCTIONAL fit to a cluster's innovations.

        Averaging the members' parameters is not averaging their functions:
        two tasks can compute the same rank-2 innovation under different
        rotations and scalings of (U, V), so the mean of gauge-inequivalent
        parameters destroys the function. Measured on the promotion testbed,
        a parameter mean recovered 11.9% of the residuals' behavioral value
        while this functional fit recovers 53.4%.
        """

        with torch.enable_grad():
            initial = torch.stack(
                [self.task_residuals[t].detach() for t in member_ids]
            ).mean(dim=0)
            candidate = nn.Parameter(initial.clone())
            optimizer = torch.optim.Adam([candidate], lr=lr)
            with torch.no_grad():
                targets = torch.stack(
                    [
                        torch.stack(
                            [
                                self._innovation(
                                    probe,
                                    *self._split_residual(self.task_residuals[t].detach()),
                                    step,
                                )
                                for step in range(self.task_steps)
                            ]
                        )
                        for t in member_ids
                    ]
                )
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                u, v, b = self._split_residual(candidate)
                predicted = torch.stack(
                    [self._innovation(probe, u, v, b, step) for step in range(self.task_steps)]
                )
                torch.mean(torch.square(predicted.unsqueeze(0) - targets)).backward()
                optimizer.step()
        return candidate.detach()

    @torch.no_grad()
    def _behavior(self, task_ids: Sequence[str], probe: Tensor) -> dict[str, Tensor]:
        return {task_id: self.forward(probe, task_id) for task_id in task_ids}

    @torch.no_grad()
    def sleep(
        self,
        task_ids: Sequence[str],
        probe_proposal: Tensor,
        probe_validation: Tensor,
        epsilon: float = 0.02,
        minimum_cluster: int = 3,
        candidate_ks: Sequence[int] = (2, 3, 4),
        holdout: int = 8,
        require_prospective: bool = True,
        lifetime_index: int | None = None,
    ) -> dict[str, object]:
        """One consolidation pass over the tasks seen so far.

        Two value terms, kept separate because they behave differently:

        V_RETRO is the code the promotion retires now. At lambda = ln 2 a
        private rank-2 residual costs ~1,098 nats and buys only a few
        hundred, so retrospective value is positive for almost ANY grouping
        whose mean preserves behavior — a retrospective-only promoter is a
        vector quantizer and cannot refuse (measured: 26 abstractions for 2
        true families, and it fires in structureless controls too).

        V_FUTURE is estimated honestly from observed history alone: the
        candidate is fitted on all but the most recent `holdout` member
        tasks and must then fit those held-out members within epsilon. An
        abstraction that generalizes to member tasks it was not fitted on
        has prospective value; a quantization artifact does not. This is the
        term that lets the promoter refuse.
        """

        eligible = [
            task_id
            for task_id in task_ids
            if task_id not in self.retired and task_id in self.task_residuals
        ]
        if len(eligible) < minimum_cluster:
            return {"lifetime_index": lifetime_index, "candidates": [], "promoted": 0}

        functions = self._residual_functions(eligible, probe_proposal)
        partitions = {}
        for k in candidate_ks:
            clusters = self._cluster(functions, (k,))
            selected = [c for c in clusters if len(c) >= minimum_cluster]
            if selected:
                partitions[k] = selected
        if not partitions:
            return {"lifetime_index": lifetime_index, "candidates": [], "promoted": 0}
        # Selecting among partitions conveys information; charge it once.
        selection_bits = math.log2(len(partitions))

        before = self._behavior(eligible, probe_validation)
        # One global abstraction over ALL eligible tasks, fitted once per
        # sleep. A candidate must beat this on held-out members or it is
        # generic compression rather than a family-specific abstraction.
        global_fit = self._fit_abstraction(eligible, probe_proposal)

        def _deviation(task_id: str, reference: Tensor) -> float:
            after = self.forward(probe_validation, task_id)
            denominator = float(
                torch.mean(torch.square(reference - reference.mean(dim=0))).clamp_min(1e-12)
            )
            return float(torch.mean(torch.square(after - reference))) / denominator

        def _evaluate(cluster: list[int]) -> dict[str, object]:
            members = [eligible[index] for index in cluster]
            # Fit on all but the most recent members; those are the honest
            # stand-in for tasks the abstraction has not seen.
            held = members[-holdout:] if len(members) > holdout else members[-1:]
            fitted_on = [t for t in members if t not in held] or members
            candidate = self._fit_abstraction(fitted_on, probe_proposal)

            index = len(self.abstractions)
            self.abstractions.append(nn.Parameter(candidate.clone(), requires_grad=False))
            accepted, deviations = [], []
            for task_id in members:
                self.task_reference[task_id] = index
                self.retired.add(task_id)
                deviation = _deviation(task_id, before[task_id])
                deviations.append(deviation)
                if deviation <= epsilon:
                    accepted.append(task_id)
                del self.task_reference[task_id]
                self.retired.discard(task_id)
            self.abstractions = nn.ParameterList(list(self.abstractions)[:index])

            # V_transfer: on members it was NOT fitted on, does this
            # abstraction beat the single global abstraction? A cluster whose
            # candidate merely compresses its own members is a quantization
            # artifact; one that predicts a task it never saw is reusable
            # structure. This is the leave-one-out logic promotion claims,
            # applied as the promoter's own gate.
            generalizing = []
            for task_id in held:
                own, other = [], []
                for value, bucket in ((candidate, own), (global_fit, other)):
                    slot = len(self.abstractions)
                    self.abstractions.append(
                        nn.Parameter(value.clone(), requires_grad=False)
                    )
                    self.task_reference[task_id] = slot
                    self.retired.add(task_id)
                    bucket.append(_deviation(task_id, before[task_id]))
                    del self.task_reference[task_id]
                    self.retired.discard(task_id)
                    self.abstractions = nn.ParameterList(list(self.abstractions)[:slot])
                if own[0] < other[0] and own[0] <= epsilon:
                    generalizing.append(task_id)

            reference_bits = math.ceil(math.log2(len(self.abstractions) + 2))
            retro = (
                BITS_PER_SCALAR * self.residual_scalars_per_task * len(accepted)
                - BITS_PER_SCALAR * self.residual_scalars_per_task
                - reference_bits * len(accepted)
                - selection_bits
            )
            future = BITS_PER_SCALAR * self.residual_scalars_per_task * len(generalizing)
            return {
                "members": members,
                "candidate": candidate,
                "accepted": accepted,
                "held_out": held,
                "generalizing": generalizing,
                "mean_behavioral_deviation": float(
                    sum(deviations) / max(1, len(deviations))
                ),
                "value_retrospective_bits": retro,
                "value_future_bits": future,
            }

        # Score every partition, commit only the best one.
        scored = {}
        for k, clusters in partitions.items():
            evaluations = [_evaluate(cluster) for cluster in clusters]
            total = sum(
                e["value_retrospective_bits"] + e["value_future_bits"] for e in evaluations
            )
            scored[k] = (total, evaluations)
        best_k = max(scored, key=lambda k: scored[k][0])
        evaluations = scored[best_k][1]

        promoted = 0
        records = []
        for evaluation in evaluations:
            fires = (
                len(evaluation["accepted"]) >= minimum_cluster
                and evaluation["value_retrospective_bits"] > 0
                and (not require_prospective or evaluation["generalizing"])
            )
            record = {
                "lifetime_index": lifetime_index,
                "partition_k": best_k,
                "cluster_size": len(evaluation["members"]),
                "accepted_members": len(evaluation["accepted"]),
                "generalizing_members": len(evaluation["generalizing"]),
                "held_out_members": len(evaluation["held_out"]),
                "mean_behavioral_deviation": evaluation["mean_behavioral_deviation"],
                "epsilon": epsilon,
                "selection_charge_bits": selection_bits,
                "value_retrospective_bits": evaluation["value_retrospective_bits"],
                "value_future_bits": evaluation["value_future_bits"],
                "decision": "promote" if fires else "refuse",
            }
            if fires:
                index = len(self.abstractions)
                self.abstractions.append(
                    nn.Parameter(evaluation["candidate"].clone(), requires_grad=False)
                )
                for task_id in evaluation["accepted"]:
                    self.task_reference[task_id] = index
                    self.retired.add(task_id)
                promoted += 1
            records.append(record)
            self.promotion_ledger.append(record)
        return {
            "lifetime_index": lifetime_index,
            "partition_k": best_k,
            "candidates": records,
            "promoted": promoted,
            "library_size": len(self.abstractions),
        }

    # ---- accounting --------------------------------------------------

    @property
    def shared_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.basis.parameters()) + sum(
            parameter.numel() for parameter in self.abstractions
        )

    @property
    def task_state_scalar_count(self) -> int:
        routes = sum(parameter.numel() for parameter in self.task_codes.values())
        residuals = sum(
            parameter.numel()
            for task_id, parameter in self.task_residuals.items()
            if task_id not in self.retired
        )
        return routes + residuals

    @torch.no_grad()
    def promotion_diagnostics(self) -> dict[str, object]:
        reference_bits = (
            math.ceil(math.log2(len(self.abstractions) + 1)) if self.abstractions else 0
        )
        return {
            "library_size": len(self.abstractions),
            "retired_tasks": len(self.retired),
            "total_tasks": len(self.task_codes),
            "reference_bits_per_task": reference_bits,
            "reference_bits_total": reference_bits * len(self.retired),
            "candidates_considered": len(self.promotion_ledger),
            "candidates_promoted": sum(
                1 for record in self.promotion_ledger if record["decision"] == "promote"
            ),
            "ledger": self.promotion_ledger,
        }
