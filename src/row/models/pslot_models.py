"""H39b pilot: a parameterized operator P(alpha) IN THE BASIS.

The H39 pilot showed family computation does not live in the private
residual (zeroing the whole residual channel moved live family-task NMSE
by ~2%; 56/64 family tasks were retired into promoted references). Family
structure is carried by ROUTES over the BASIS. So the fast argument is
moved into a basis slot, in the coordinate the generator mixes family
operators in -- the `U` matrix with `V, b` shared:

    P(alpha_i)(z) = tanh(z + a . (U_0 + sum_k alpha_{i,k} U_k) tanh(V z + b))

Slot 12 (index 11) becomes P. `U_0, V, b, a` are the ordinary slot's own
parameters with the ordinary seed, so at alpha = 0 the slot IS the ordinary
slot and the whole learner reproduces the ordinary learner bitwise (the
P2-frozen control in the plan). `U_k` are shared argument matrices
(spectral norm 1 at init, global optimizer group); `alpha_i in R^K` is the
per-task fast argument (zero-initialized, task learning rate). Nothing tells
the learner which tasks are family tasks or at which step the family fires.

`dL/dalpha_k = <dL/dU, U_k>` is nonzero at alpha = 0, so the stationary
point that bit the residual schema twice does not exist here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from row.models.prospective_models import ProspectiveLifecycleLearner


class ParameterizedSlotLearner(ProspectiveLifecycleLearner):
    def __init__(
        self,
        *args,
        slot_args: int = 2,
        freeze_args: bool = False,
        freeze_matrices: bool = False,
        pslot_index: int | None = None,
        pslot_count: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if slot_args < 1:
            raise ValueError("slot_args must be positive")
        self.slot_args = int(slot_args)
        self.freeze_args = bool(freeze_args)
        # G control: argument directions fixed at random init, alpha learns.
        self.freeze_matrices = bool(freeze_matrices) or self.freeze_args
        self.pslot_index = (
            self.operator_slots - 1 if pslot_index is None else int(pslot_index)
        )
        if not 0 <= self.pslot_index < self.operator_slots:
            raise ValueError("pslot_index out of range")
        # Multi-slot (H39d): additional parameterized slots descend from the
        # primary index (11, 10, ...). Slot s's matrices are seeded
        # seed + 997*s + 31*k, so the primary slot's layout is unchanged
        # and single-slot artifacts remain loadable.
        self.pslot_count = int(pslot_count)
        if not 1 <= self.pslot_count <= self.pslot_index + 1:
            raise ValueError("pslot_count out of range")
        self.pslot_indices = [self.pslot_index - i for i in range(self.pslot_count)]
        seed = int(kwargs.get("seed", 0))

        def _matrices(slot: int) -> Tensor:
            d, rank = self.basis[slot].U.shape
            out = []
            for k in range(1, self.slot_args + 1):
                generator = torch.Generator(device="cpu").manual_seed(seed + 997 * slot + 31 * k)
                U = torch.randn(d, rank, generator=generator)
                out.append(U / torch.linalg.matrix_norm(U, ord=2))
            return torch.stack(out)

        self.argument_matrices = nn.Parameter(
            _matrices(self.pslot_index), requires_grad=not self.freeze_matrices
        )
        self.register_buffer("initial_argument_matrices",
                             self.argument_matrices.detach().clone())
        if self.pslot_count > 1:
            self.extra_argument_matrices = nn.Parameter(
                torch.stack([_matrices(slot) for slot in self.pslot_indices[1:]]),
                requires_grad=not self.freeze_matrices,
            )
            self.register_buffer("initial_extra_argument_matrices",
                                 self.extra_argument_matrices.detach().clone())
        self.task_alphas = nn.ParameterDict()
        # H47 B1 route policies over the parameterized slots (multi-slot
        # only). `route_temperature` scales the two slots' logits when
        # forming their conditional (1.0 = plain softmax, bitwise M).
        # `task_mask` pins a task's parameterized-slot mass onto one slot.
        # Plain-slot mass is never touched by either.
        self.register_buffer("route_temperature", torch.tensor(1.0))
        self.task_mask: dict[str, int] = {}

    # ---- task state ---------------------------------------------------

    def begin_task(self, task_id: str, schema_index: int = 0):
        # `schema_index` is accepted for interface parity with the
        # factorized learner and ignored: there is one P slot.
        code, residual = super().begin_task(task_id)
        shape = (self.slot_args,) if self.pslot_count == 1 else (self.pslot_count, self.slot_args)
        alpha = nn.Parameter(torch.zeros(shape), requires_grad=not self.freeze_args)
        self.task_alphas[task_id] = alpha
        return code, residual, alpha

    def forget_task(self, task_id: str) -> None:
        super().forget_task(task_id)
        if task_id in self.task_alphas:
            del self.task_alphas[task_id]
        self.task_mask.pop(task_id, None)

    def set_route_temperature(self, value: float) -> None:
        with torch.no_grad():
            self.route_temperature.fill_(float(value))

    @torch.no_grad()
    def conditional_entropy_bits(self, task_id: str) -> list[float]:
        """Entropy of the policy-applied conditional over the P slots, per step."""
        route = self.task_codes[task_id].reshape(self.task_steps, self.operator_slots)
        coefficients = self._coefficients(route, task_id)
        idx = list(self.pslot_indices)
        cond = coefficients[:, idx] / coefficients[:, idx].sum(-1, keepdim=True).clamp_min(1e-12)
        return (-(cond * cond.clamp_min(1e-12).log()).sum(-1) / np.log(2)).tolist()

    # ---- forward ------------------------------------------------------

    def _slot_argument(self, task_id: str, position: int) -> tuple[Tensor, Tensor]:
        alpha = self.task_alphas[task_id]
        if self.pslot_count == 1:
            return alpha, self.argument_matrices
        if position == 0:
            return alpha[0], self.argument_matrices
        return alpha[position], self.extra_argument_matrices[position - 1]

    def _pslot(self, z: Tensor, task_id: str, position: int = 0) -> Tensor:
        base = self.basis[self.pslot_indices[position]]
        hidden = torch.nn.functional.linear(z, base.V, base.b)
        hidden = (torch.tanh(hidden) if base.activation == "tanh"
                  else torch.nn.functional.gelu(hidden))
        alpha, matrices = self._slot_argument(task_id, position)
        # U_0 + sum_k alpha_k U_k; exactly U_0 when alpha == 0.
        U = base.U + torch.einsum("k,kdr->dr", alpha, matrices)
        return torch.tanh(z + base.alpha * torch.nn.functional.linear(hidden, U))

    def _candidates(self, z: Tensor, task_id: str) -> Tensor:
        position = {slot: i for i, slot in enumerate(self.pslot_indices)}
        return torch.stack(
            [
                self._pslot(z, task_id, position[index]) if index in position else operator(z)
                for index, operator in enumerate(self.basis)
            ],
            dim=0,
        )

    def _coefficients(self, route: Tensor, task_id: str) -> Tensor:
        coefficients = torch.softmax(route, dim=-1)
        if self.pslot_count < 2:
            return coefficients
        mask_slot = self.task_mask.get(task_id)
        temperature = float(self.route_temperature)
        if mask_slot is None and temperature == 1.0:
            return coefficients
        idx = list(self.pslot_indices)
        total = coefficients[:, idx].sum(-1, keepdim=True)          # P mass per step
        if mask_slot is not None:
            cond = torch.zeros_like(coefficients[:, idx])
            cond[:, idx.index(mask_slot)] = 1.0
        else:
            cond = torch.softmax(route[:, idx] / temperature, dim=-1)
        new = coefficients.clone()
        new[:, idx] = total * cond
        return new

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        route, own_u, own_v, own_b = self._unpack(task_id)
        coefficients = self._coefficients(route, task_id)
        reference = self.task_reference.get(task_id)
        shared = (
            self._split_residual(self.abstractions[reference])
            if reference is not None
            else None
        )
        retired = task_id in self.retired
        z = x
        for step in range(self.task_steps):
            candidates = self._candidates(z, task_id)
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

    # ---- parameter groups and accounting -----------------------------

    def shared_parameters(self) -> list[nn.Parameter]:
        shared = super().shared_parameters()
        if not self.freeze_matrices:
            shared.append(self.argument_matrices)
            if self.pslot_count > 1:
                shared.append(self.extra_argument_matrices)
        return shared

    def all_argument_matrices(self) -> list[tuple[int, Tensor, Tensor]]:
        """(slot index, current, initial) for every parameterized slot."""
        out = [(self.pslot_index, self.argument_matrices, self.initial_argument_matrices)]
        for i, slot in enumerate(self.pslot_indices[1:]):
            out.append((slot, self.extra_argument_matrices[i], self.initial_extra_argument_matrices[i]))
        return out

    @property
    def shared_parameter_count(self) -> int:
        return super().shared_parameter_count + sum(m.numel() for _, m, _ in self.all_argument_matrices())

    @property
    def task_state_scalar_count(self) -> int:
        return super().task_state_scalar_count + sum(
            p.numel() for p in self.task_alphas.values()
        )

    @torch.no_grad()
    def pslot_diagnostics(self) -> dict[str, object]:
        moved_all = [float(torch.linalg.norm(m - init) / (torch.linalg.norm(init) or 1.0))
                     for _, m, init in self.all_argument_matrices()]
        moved = moved_all[0]
        alpha_norms = [float(torch.linalg.norm(a)) for a in self.task_alphas.values()]
        alpha_norms_by_slot = [
            [float(torch.linalg.norm(a if self.pslot_count == 1 else a[i]))
             for a in self.task_alphas.values()]
            for i in range(self.pslot_count)
        ]
        masses = []
        for code in self.task_codes.values():
            coefficients = torch.softmax(code.reshape(self.task_steps, self.operator_slots), dim=-1)
            masses.append(coefficients[:, self.pslot_index].tolist())
        mass = np.mean(masses, axis=0).tolist() if masses else []
        return {
            "slot_args": self.slot_args,
            "freeze_args": self.freeze_args,
            "freeze_matrices": self.freeze_matrices,
            "pslot_index": self.pslot_index,
            "pslot_count": self.pslot_count,
            "pslot_indices": list(self.pslot_indices),
            "route_temperature": float(self.route_temperature),
            "masked_tasks": len(self.task_mask),
            "argument_matrices_relative_movement": moved,
            "argument_matrices_relative_movement_by_slot": moved_all,
            "alpha_norm_mean_by_slot": [float(np.mean(v)) if v else 0.0 for v in alpha_norms_by_slot],
            "alpha_norm_mean": float(np.mean(alpha_norms)) if alpha_norms else 0.0,
            "alpha_nonzero_tasks": int(sum(1 for n in alpha_norms if n > 0)),
            "route_mass_on_P_by_step_all_tasks": mass,
            "argument_scalars": int(sum(m.numel() for _, m, _ in self.all_argument_matrices())),
            "alpha_scalars": sum(p.numel() for p in self.task_alphas.values()),
        }

    def save_extras(self, output: Path) -> None:
        (output / "pslot.json").write_text(
            json.dumps({"diagnostics": self.pslot_diagnostics(),
                        "task_mask": dict(self.task_mask)}, indent=2) + "\n",
            encoding="utf-8",
        )
