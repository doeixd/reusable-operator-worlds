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
        pslot_index: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if slot_args < 1:
            raise ValueError("slot_args must be positive")
        self.slot_args = int(slot_args)
        self.freeze_args = bool(freeze_args)
        self.pslot_index = (
            self.operator_slots - 1 if pslot_index is None else int(pslot_index)
        )
        if not 0 <= self.pslot_index < self.operator_slots:
            raise ValueError("pslot_index out of range")
        base = self.basis[self.pslot_index]
        d, rank = base.U.shape
        seed = int(kwargs.get("seed", 0))
        matrices = []
        for k in range(1, self.slot_args + 1):
            generator = torch.Generator(device="cpu").manual_seed(
                seed + 997 * self.pslot_index + 31 * k
            )
            U = torch.randn(d, rank, generator=generator)
            U = U / torch.linalg.matrix_norm(U, ord=2)
            matrices.append(U)
        self.argument_matrices = nn.Parameter(
            torch.stack(matrices), requires_grad=not self.freeze_args
        )
        self.register_buffer("initial_argument_matrices",
                             self.argument_matrices.detach().clone())
        self.task_alphas = nn.ParameterDict()

    # ---- task state ---------------------------------------------------

    def begin_task(self, task_id: str, schema_index: int = 0):
        # `schema_index` is accepted for interface parity with the
        # factorized learner and ignored: there is one P slot.
        code, residual = super().begin_task(task_id)
        alpha = nn.Parameter(torch.zeros(self.slot_args),
                             requires_grad=not self.freeze_args)
        self.task_alphas[task_id] = alpha
        return code, residual, alpha

    def forget_task(self, task_id: str) -> None:
        super().forget_task(task_id)
        if task_id in self.task_alphas:
            del self.task_alphas[task_id]

    # ---- forward ------------------------------------------------------

    def _pslot(self, z: Tensor, task_id: str) -> Tensor:
        base = self.basis[self.pslot_index]
        hidden = torch.nn.functional.linear(z, base.V, base.b)
        hidden = (torch.tanh(hidden) if base.activation == "tanh"
                  else torch.nn.functional.gelu(hidden))
        alpha = self.task_alphas[task_id]
        # U_0 + sum_k alpha_k U_k; exactly U_0 when alpha == 0.
        U = base.U + torch.einsum("k,kdr->dr", alpha, self.argument_matrices)
        return torch.tanh(z + base.alpha * torch.nn.functional.linear(hidden, U))

    def _candidates(self, z: Tensor, task_id: str) -> Tensor:
        return torch.stack(
            [
                self._pslot(z, task_id) if index == self.pslot_index else operator(z)
                for index, operator in enumerate(self.basis)
            ],
            dim=0,
        )

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
        if not self.freeze_args:
            shared.append(self.argument_matrices)
        return shared

    @property
    def shared_parameter_count(self) -> int:
        return super().shared_parameter_count + self.argument_matrices.numel()

    @property
    def task_state_scalar_count(self) -> int:
        return super().task_state_scalar_count + sum(
            p.numel() for p in self.task_alphas.values()
        )

    @torch.no_grad()
    def pslot_diagnostics(self) -> dict[str, object]:
        initial = self.initial_argument_matrices
        moved = float(torch.linalg.norm(self.argument_matrices - initial)
                      / (torch.linalg.norm(initial) or 1.0))
        alpha_norms = [float(torch.linalg.norm(a)) for a in self.task_alphas.values()]
        masses = []
        for code in self.task_codes.values():
            coefficients = torch.softmax(code.reshape(self.task_steps, self.operator_slots), dim=-1)
            masses.append(coefficients[:, self.pslot_index].tolist())
        mass = np.mean(masses, axis=0).tolist() if masses else []
        return {
            "slot_args": self.slot_args,
            "freeze_args": self.freeze_args,
            "pslot_index": self.pslot_index,
            "argument_matrices_relative_movement": moved,
            "alpha_norm_mean": float(np.mean(alpha_norms)) if alpha_norms else 0.0,
            "alpha_nonzero_tasks": int(sum(1 for n in alpha_norms if n > 0)),
            "route_mass_on_P_by_step_all_tasks": mass,
            "argument_scalars": int(self.argument_matrices.numel()),
            "alpha_scalars": sum(p.numel() for p in self.task_alphas.values()),
        }

    def save_extras(self, output: Path) -> None:
        (output / "pslot.json").write_text(
            json.dumps({"diagnostics": self.pslot_diagnostics()}, indent=2) + "\n",
            encoding="utf-8",
        )
