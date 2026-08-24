"""H51 arm R_2: a decomposable innovation basis BESIDE the parameterized slots.

`ParameterizedSlotLearner` gives the basis a fast argument channel
(`U(alpha) = U_0 + sum_k alpha_k U_k`, H39's confirmed result). What it does
NOT change is how a task's PRIVATE innovation is stored: a dense 198-scalar
rank-2 residual per task, which is exactly the state H50 found unmigratable.

Review 72's H52 asks for the innovation itself to be represented in separately
addressable pieces, so that a later reorganization operates on components
rather than reverse-engineering a dense result:

    residual_i = W a_i + eps_i

`W` (198 x schema_dim) is one pooled, slowly-learned component basis in the
global optimizer group; `a_i` is the per-task coordinate at the task learning
rate; `eps_i` keeps the ordinary 1e-3 initialization and the ordinary L1
storage penalty. `schema_count = 1` and no task ever receives a group label:
this arm changes the STORAGE FORM of innovation, not the learner's information.

This is `FactorizedLifecycleLearner`'s residual factorization composed with the
parameterized slots; the two were separate `kind`s and did not compose. The
composition is by subclassing rather than multiple inheritance because both
parents name their fast argument `task_alphas`; here the slot argument keeps
that name and the component coordinate is `schema_alphas`.

Controls required by `H51_REORGANIZABILITY_PLAN.md`:
  * `freeze_schema=True` freezes `W` AND every `a_i` at zero, so the component
    contribution is identically zero and the learner is bitwise the ordinary
    `pslot` artifact.
  * `dL/da = W^T dL/dr` is nonzero at `a = 0` because `W` is randomly
    initialized, so the zero-stationary-point trap that bit the residual schema
    twice (AGENTS.md) does not exist here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from row.models.pslot_models import ParameterizedSlotLearner


class PslotFactorizedLearner(ParameterizedSlotLearner):
    def __init__(
        self,
        *args,
        schema_dim: int = 8,
        schema_count: int = 1,
        schema_seed: int = 51001,
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
        self.schema_alphas = nn.ParameterDict()
        self.task_schema: dict[str, int] = {}

    # ---- task state ---------------------------------------------------

    def begin_task(self, task_id: str, schema_index: int = 0):
        # `schema_index` selects the component basis; R_2 is pooled
        # (schema_count = 1) so it is 0 for every task and no group label
        # ever reaches the learner.
        index = int(schema_index) % self.schema_count
        code, residual, alpha = super().begin_task(task_id)
        self.schema_alphas[task_id] = nn.Parameter(
            torch.zeros(self.schema_dim), requires_grad=not self.freeze_schema
        )
        self.task_schema[task_id] = index
        return code, residual, [alpha, self.schema_alphas[task_id]]

    def forget_task(self, task_id: str) -> None:
        super().forget_task(task_id)
        if task_id in self.schema_alphas:
            del self.schema_alphas[task_id]
        self.task_schema.pop(task_id, None)

    # ---- the decomposable innovation ----------------------------------

    def schema_component(self, task_id: str) -> Tensor:
        return self.schemas[self.task_schema[task_id]] @ self.schema_alphas[task_id]

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
            p.numel() for p in self.schema_alphas.values()
        )

    @torch.no_grad()
    def schema_diagnostics(self) -> dict[str, object]:
        moved = []
        for index, schema in enumerate(self.schemas):
            initial = self.initial_schemas[index]
            denom = float(torch.linalg.norm(initial)) or 1.0
            moved.append(float(torch.linalg.norm(schema - initial)) / denom)
        alpha_norms = [float(torch.linalg.norm(a)) for a in self.schema_alphas.values()]
        component_norms, eps_norms = [], []
        for task_id in self.task_residuals:
            if task_id in self.retired or task_id not in self.schema_alphas:
                continue
            component_norms.append(float(torch.linalg.norm(self.schema_component(task_id))))
            eps_norms.append(float(torch.linalg.norm(self.task_residuals[task_id])))
        return {
            "schema_dim": self.schema_dim,
            "schema_count": self.schema_count,
            "schema_seed": self.schema_seed,
            "schema_init_scale": self.schema_init_scale,
            "freeze_schema": self.freeze_schema,
            "schema_relative_movement": moved,
            "schema_alpha_norm_mean": float(np.mean(alpha_norms)) if alpha_norms else 0.0,
            "schema_alpha_nonzero_tasks": int(sum(1 for n in alpha_norms if n > 0)),
            "component_norm_mean": float(np.mean(component_norms)) if component_norms else 0.0,
            "eps_norm_mean": float(np.mean(eps_norms)) if eps_norms else 0.0,
            "component_share_mean": (
                float(np.mean([c / (c + e) for c, e in zip(component_norms, eps_norms) if c + e > 0]))
                if component_norms else 0.0
            ),
            "schema_scalars": sum(p.numel() for p in self.schemas),
            "schema_alpha_scalars": sum(p.numel() for p in self.schema_alphas.values()),
        }

    # ---- persistence -------------------------------------------------

    def save_extras(self, output: Path) -> None:
        super().save_extras(output)
        (output / "pslot_factorized.json").write_text(
            json.dumps({
                "task_schema": dict(self.task_schema),
                "diagnostics": self.schema_diagnostics(),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
