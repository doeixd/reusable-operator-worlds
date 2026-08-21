"""H39 pilot: SHARED SCHEMA + FAST ARGUMENT + PRIVATE INNOVATION.

The ordinary lifecycle learner gives every task a private 198-scalar
rank-2 residual `eps_i`. Census C0 showed that the affine span of the
residuals the finished ordinary learner retains does not contain the
held-out siblings at any rank. This learner changes only how a task's
residual vector is PRODUCED:

    residual_i = W_{s(i)} alpha_i + eps_i

    W_s      slow shared schema, one per schema index (198 x a), in the
             global optimizer group
    alpha_i  small fast per-task argument (a scalars), task learning rate
    eps_i    the private residual with the ordinary 1e-3 initialization
             (exact zero is a stationary point of the innovation and would
             freeze alpha too -- plan Amendment 1) under the unchanged L1
             storage penalty; never forced to stay zero

`s(i)` is supplied by the runner. In the pilot's primary arm it is the
ORACLE family index (explicit: the arm tests whether the substrate can
REPRESENT the decomposition, not discover the grouping); pre-onset tasks
share one extra schema. The pooled arm uses a single schema for every
task. No teacher operator values, family parameters, or subspaces enter.

Everything downstream reads `effective_residual`, so promotion fitting,
lifecycle snapshots, and the forward pass all see `W alpha + eps`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from row.models.prospective_models import ProspectiveLifecycleLearner


class FactorizedLifecycleLearner(ProspectiveLifecycleLearner):
    def __init__(
        self,
        *args,
        schema_dim: int = 2,
        schema_count: int = 1,
        schema_seed: int = 39001,
        schema_init_scale: float = 1e-2,
        freeze_schema: bool = False,
        world_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if schema_dim < 1 or schema_count < 1:
            raise ValueError("schema_dim and schema_count must be positive")
        self.schema_dim = int(schema_dim)
        self.schema_count = int(schema_count)
        self.schema_seed = int(schema_seed)
        self.schema_init_scale = float(schema_init_scale)
        self.freeze_schema = bool(freeze_schema)
        self.world_seed = int(world_seed)
        residual_size = int(self.initial_residual_state.numel())
        schemas = []
        for index in range(self.schema_count):
            rng = np.random.default_rng(
                np.random.SeedSequence([self.schema_seed, self.world_seed, index])
            )
            init = rng.normal(size=(residual_size, self.schema_dim)) * (
                self.schema_init_scale / np.sqrt(self.schema_dim)
            )
            schemas.append(nn.Parameter(
                torch.tensor(init, dtype=torch.float32),
                requires_grad=not self.freeze_schema,
            ))
        self.schemas = nn.ParameterList(schemas)
        self.register_buffer(
            "initial_schemas",
            torch.stack([p.detach().clone() for p in self.schemas]),
        )
        self.task_alphas = nn.ParameterDict()
        self.task_schema: dict[str, int] = {}

    # ---- task state ---------------------------------------------------

    def begin_task(self, task_id: str, schema_index: int = 0):
        if not 0 <= schema_index < self.schema_count:
            raise ValueError(f"schema_index {schema_index} out of range")
        # Amendment 1: eps keeps the ordinary 1e-3 initialization. Exact
        # zero is a stationary point of u.tanh(vz+b) and freezes both eps
        # and (through dL/dr) alpha; a literal null state is H40's gate.
        code, residual = super().begin_task(task_id)
        self.task_alphas[task_id] = nn.Parameter(torch.zeros(self.schema_dim))
        self.task_schema[task_id] = int(schema_index)
        return code, residual, self.task_alphas[task_id]

    def forget_task(self, task_id: str) -> None:
        super().forget_task(task_id)
        if task_id in self.task_alphas:
            del self.task_alphas[task_id]
        self.task_schema.pop(task_id, None)

    def schema_component(self, task_id: str) -> Tensor:
        return self.schemas[self.task_schema[task_id]] @ self.task_alphas[task_id]

    def effective_residual(self, task_id: str) -> Tensor:
        return self.task_residuals[task_id] + self.schema_component(task_id)

    def _unpack(self, task_id: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        route = self.task_codes[task_id]
        residual_u, residual_v, residual_b = torch.split(
            self.effective_residual(task_id),
            (self.residual_u_size, self.residual_v_size, self.residual_b_size),
        )
        return (
            route.reshape(self.task_steps, self.operator_slots),
            residual_u.reshape(self.task_steps, self.d, self.residual_rank),
            residual_v.reshape(self.task_steps, self.residual_rank, self.d),
            residual_b.reshape(self.task_steps, self.residual_rank),
        )

    # ---- parameter groups and accounting -----------------------------

    def shared_parameters(self) -> list[nn.Parameter]:
        shared = super().shared_parameters()
        if not self.freeze_schema:
            shared.extend(self.schemas)
        return shared

    @property
    def shared_parameter_count(self) -> int:
        return super().shared_parameter_count + sum(p.numel() for p in self.schemas)

    @property
    def task_state_scalar_count(self) -> int:
        return super().task_state_scalar_count + sum(
            p.numel() for p in self.task_alphas.values()
        )

    @torch.no_grad()
    def schema_diagnostics(self) -> dict[str, object]:
        moved = []
        for index, schema in enumerate(self.schemas):
            initial = self.initial_schemas[index]
            denom = float(torch.linalg.norm(initial)) or 1.0
            moved.append(float(torch.linalg.norm(schema - initial)) / denom)
        alpha_norms = [float(torch.linalg.norm(a)) for a in self.task_alphas.values()]
        eps_norms = [
            float(torch.linalg.norm(self.task_residuals[t]))
            for t in self.task_residuals if t not in self.retired
        ]
        return {
            "schema_dim": self.schema_dim,
            "schema_count": self.schema_count,
            "schema_seed": self.schema_seed,
            "schema_init_scale": self.schema_init_scale,
            "freeze_schema": self.freeze_schema,
            "schema_relative_movement": moved,
            "tasks_per_schema": [
                sum(1 for s in self.task_schema.values() if s == i)
                for i in range(self.schema_count)
            ],
            "alpha_norm_mean": float(np.mean(alpha_norms)) if alpha_norms else 0.0,
            "eps_norm_mean": float(np.mean(eps_norms)) if eps_norms else 0.0,
            "alpha_scalars": sum(p.numel() for p in self.task_alphas.values()),
            "schema_scalars": sum(p.numel() for p in self.schemas),
            "live_eps_scalars": sum(
                self.task_residuals[t].numel()
                for t in self.task_residuals if t not in self.retired
            ),
        }

    # ---- persistence -------------------------------------------------

    def save_extras(self, output: Path) -> None:
        (output / "factorized.json").write_text(
            json.dumps({
                "task_schema": dict(self.task_schema),
                "diagnostics": self.schema_diagnostics(),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
