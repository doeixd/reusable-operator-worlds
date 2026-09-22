"""Per-task depth for the fast rotated discrete learner.

N1's `INTERLEAVED` arm pools length-1, length-2 and length-3 tasks into ONE
undifferentiated stream. The existing learners cannot express that: depth is a
property of the MODEL (`task_steps`), not of the task, and `begin_task`
allocates a code of shape `(task_steps, operator_slots)`. J1c sidesteps it by
building a fresh model per stage and carrying only the library.

This class makes depth a property of the TASK, and does so in the one way that
keeps N1's arms comparable: **when every task's depth equals `task_steps`, it
calls the parent's forward directly, so the reduction is bitwise BY
CONSTRUCTION rather than by tolerance.** `STAGED` and `SHAM` therefore run the
exact committed code path and `STAGED` keeps its J1c anchor; only `INTERLEAVED`
exercises the new branch.

Unused code rows for a shallow task receive no gradient and stay at their zero
initialisation, so they neither move nor contribute.
"""
from __future__ import annotations

import torch
from torch import Tensor

from row.models.learned_models import FastRotatedDiscreteLibraryLearner, batched_householder


class VariableDepthRotatedLearner(FastRotatedDiscreteLibraryLearner):
    """`FastRotatedDiscreteLibraryLearner` with an optional per-task depth."""

    implementation = "variable_depth_rotation_v1"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.task_depth: dict[str, int] = {}

    def begin_task(self, task_id: str, depth: int | None = None) -> torch.nn.Parameter:
        code = super().begin_task(task_id)
        if depth is not None:
            if not 1 <= depth <= self.task_steps:
                raise ValueError(f'depth {depth} outside 1..{self.task_steps}')
            self.task_depth[task_id] = int(depth)
        return code

    def depth_of(self, task_id: str) -> int:
        return self.task_depth.get(task_id, self.task_steps)

    def uniform_depth(self) -> bool:
        """True when no task asks for a depth other than `task_steps`."""
        return all(d == self.task_steps for d in self.task_depth.values())

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        depth = self.depth_of(task_id)
        if depth == self.task_steps:
            # The committed path, unmodified: bitwise identical by construction.
            return super().forward(x, task_id)

        library = list(self.library)
        V = torch.stack([op.V for op in library])
        U = torch.stack([op.U for op in library])
        b = torch.stack([op.b for op in library])
        alpha = torch.stack(
            [op.alpha if torch.is_tensor(op.alpha) else torch.tensor(float(op.alpha))
             for op in library]
        )
        Q = batched_householder([op.rotation.vectors for op in library])
        gelu = library[0].activation == 'gelu'
        coefficients = self._coefficients(task_id)
        z = x
        for step in range(depth):
            hidden = torch.einsum('bd,srd->bsr', z, V) + b
            hidden = torch.nn.functional.gelu(hidden) if gelu else torch.tanh(hidden)
            residual = z.unsqueeze(1) + alpha.view(1, -1, 1) * torch.einsum('bsr,sdr->bsd', hidden, U)
            candidates = torch.einsum('bsd,sde->bse', residual, Q)
            z = torch.sum(coefficients[step].view(1, -1, 1) * candidates, dim=1)
        return z

    def hard_routes(self) -> dict[str, list[int]]:
        """Truncate each route to its own depth; trailing rows are never applied."""
        return {task_id: route[: self.depth_of(task_id)]
                for task_id, route in super().hard_routes().items()}
