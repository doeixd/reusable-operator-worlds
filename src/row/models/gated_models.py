"""V3 wake candidate: task innovations with an exact null state.

The Gaussian task code was falsified as a wake substrate (P-2026-08-18-A)
because its zero-information state, q = p, is not the zero-perturbation
state: an unused coordinate still injects noise, so the optimizer buys
quiet with precision and pays for a coordinate carrying nothing. Here the
task innovation is

    R_tau,l = sum_k g_tau,l,k * u_k v_k^T

with g a relaxed Bernoulli over WHOLE RANK COMPONENTS, so g = 0 means
exactly "reuse the shared computation": no payload, no noise, no scalar.
Structure and cost agree, and rank(R) in {0, 1, 2} is also the state
PROMOTE manipulates.

Deliberately NOT changed from H9: the route machinery. Routes are the
reference channel that already works, and the Gaussian run showed that a
shared prior can collapse it (uniform mixtures in 2 of 3 worlds). Coupling
a reference code to an innovation prior would confound this hypothesis with
a second change; a proper categorical reference code is later work.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from row.models.learned_models import SharedParentResidualLearner

BITS_PER_SCALAR = 8


class GatedInnovationLearner(SharedParentResidualLearner):
    """H9 shared-residual with gated, rank-structured task innovations."""

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
        gate_temperature: float = 0.5,
        gate_logit_init: float = 2.0,
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
        if gate_temperature <= 0.0:
            raise ValueError("gate temperature must be positive")
        self.gate_temperature = float(gate_temperature)
        # Start ON, so the learner begins where H9 begins and the code
        # charge switches innovations OFF where they do not pay. Starting
        # neutral or off invites the symmetry-breaking trap that froze the
        # Gaussian task codes at zero for a whole lifetime.
        self.gate_logit_init = float(gate_logit_init)
        # Scalars per rank component: u (d) + v (d) + b (1).
        self.scalars_per_component = 2 * d + 1
        self.components_per_task = task_steps * residual_rank
        self.task_gate_logits = nn.ParameterDict()
        self._gate_generator = torch.Generator()
        self._gate_generator.manual_seed(int(seed) + 15486)

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        generator = state.pop("_gate_generator", None)
        if generator is not None:
            state["_gate_generator_seed"] = int(generator.initial_seed())
            state["_gate_generator_state"] = generator.get_state().clone()
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        state = dict(state)
        seed = state.pop("_gate_generator_seed", None)
        tensor_state = state.pop("_gate_generator_state", None)
        self.__dict__.update(state)
        generator = torch.Generator()
        generator.manual_seed(int(seed) if seed is not None else 0)
        if tensor_state is not None:
            generator.set_state(tensor_state)
        self._gate_generator = generator

    def begin_task(
        self, task_id: str
    ) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter]:
        route, residual = super().begin_task(task_id)
        self.task_gate_logits[task_id] = nn.Parameter(
            torch.full((self.task_steps, self.residual_rank), self.gate_logit_init)
        )
        return route, residual, self.task_gate_logits[task_id]

    def gate_probabilities(self, task_id: str) -> Tensor:
        return torch.sigmoid(self.task_gate_logits[task_id])

    def _gates(self, task_id: str) -> Tensor:
        logits = self.task_gate_logits[task_id]
        if not (self.training and torch.is_grad_enabled()):
            # Every scoring path sees the hard, discrete gate, so reported
            # behavior is the behavior of the representation actually
            # retained rather than of a relaxation.
            return (torch.sigmoid(logits) > 0.5).to(logits.dtype)
        uniform = torch.rand(
            logits.shape, generator=self._gate_generator, device=logits.device
        ).clamp(1e-6, 1.0 - 1e-6)
        logistic = torch.log(uniform) - torch.log1p(-uniform)
        return torch.sigmoid((logits + logistic) / self.gate_temperature)

    def _unpack(self, task_id: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        route, residual_u, residual_v, residual_b = super()._unpack(task_id)
        gates = self._gates(task_id)
        return (
            route,
            residual_u * gates.unsqueeze(1),
            residual_v * gates.unsqueeze(-1),
            residual_b * gates,
        )

    def description_penalty(self, task_ids: Sequence[str]) -> Tensor:
        """Expected structural code length in NATS for the tasks in a batch.

        The training signal is the literal sparse code the model will be
        scored under: one presence bit per rank component plus 8 bits per
        scalar of each ACTIVE component. Charging exactly what the two-part
        code charges is the point of this substrate — under a Gaussian
        precision code the training signal and the literal code measure
        different quantities.
        """

        unique = list(dict.fromkeys(task_ids))
        payload_bits = float(BITS_PER_SCALAR * self.scalars_per_component)
        totals = []
        for task_id in unique:
            probability = self.gate_probabilities(task_id)
            bits = probability.sum() * payload_bits + float(self.components_per_task)
            totals.append(bits * math.log(2.0))
        return torch.stack(totals).mean()

    def storage_penalty(self, task_ids: Sequence[str]) -> Tensor:
        device = self.task_residuals[next(iter(self.task_residuals))].device
        return torch.zeros((), device=device)

    @torch.no_grad()
    def gate_diagnostics(self) -> dict[str, object]:
        if not self.task_gate_logits:
            return {}
        probabilities = torch.stack(
            [self.gate_probabilities(task_id) for task_id in self.task_gate_logits]
        )
        active = (probabilities > 0.5).double()
        per_step_rank = active.sum(dim=-1)
        return {
            "mean_gate_probability": float(probabilities.mean()),
            "mean_active_components_per_task": float(active.sum(dim=(1, 2)).mean()),
            "components_per_task": self.components_per_task,
            "fraction_steps_rank0": float((per_step_rank == 0).double().mean()),
            "fraction_steps_rank1": float((per_step_rank == 1).double().mean()),
            "fraction_steps_rank2": float((per_step_rank == 2).double().mean()),
            "per_step_active_fraction": active.mean(dim=(0, 2)).tolist(),
        }

    @torch.no_grad()
    def structural_task_bits(self) -> dict[str, int]:
        """Literal sparse two-part cost of the retained task state."""

        active = 0
        for task_id in self.task_gate_logits:
            active += int((self.gate_probabilities(task_id) > 0.5).sum())
        tasks = max(1, len(self.task_gate_logits))
        route_bits = BITS_PER_SCALAR * self.route_size * tasks
        bitmap_bits = self.components_per_task * tasks
        payload_bits = BITS_PER_SCALAR * self.scalars_per_component * active
        dense = BITS_PER_SCALAR * (
            sum(parameter.numel() for parameter in self.task_codes.values())
            + sum(parameter.numel() for parameter in self.task_residuals.values())
        )
        return {
            "active_components": active,
            "total_components": self.components_per_task * tasks,
            "route_bits": route_bits,
            "presence_bitmap_bits": bitmap_bits,
            "payload_bits": payload_bits,
            "task_bits": route_bits + bitmap_bits + payload_bits,
            "dense_task_bits": dense,
        }
