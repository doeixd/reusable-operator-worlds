"""Non-oracle dense and continuous reusable ROW learners."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from row.models.torch_oracle import LearnedOperator


class ResidualStateBlock(nn.Module):
    def __init__(self, d: int, task_embedding_dim: int, hidden_width: int) -> None:
        super().__init__()
        self.linear_in = nn.Linear(d + task_embedding_dim, hidden_width)
        self.linear_out = nn.Linear(hidden_width, d)
        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, state: Tensor, task_code: Tensor) -> Tensor:
        features = torch.cat((state, task_code), dim=-1)
        delta = self.linear_out(torch.nn.functional.gelu(self.linear_in(features)))
        return torch.tanh(state + delta)


class DenseLearner(nn.Module):
    """Opaque task embedding concatenated with input and processed by a shared MLP."""

    def __init__(
        self,
        d: int,
        task_embedding_dim: int,
        hidden_width: int,
        residual_blocks: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.task_embedding_dim = task_embedding_dim
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.blocks = nn.ModuleList(
                ResidualStateBlock(d, task_embedding_dim, hidden_width)
                for _ in range(residual_blocks)
            )
        self.task_codes = nn.ParameterDict()

    def begin_task(self, task_id: str) -> nn.Parameter:
        if task_id in self.task_codes:
            raise ValueError(f"task already exists: {task_id}")
        self.task_codes[task_id] = nn.Parameter(torch.zeros(self.task_embedding_dim))
        return self.task_codes[task_id]

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        code = self.task_codes[task_id].expand(len(x), -1)
        state = x
        for block in self.blocks:
            state = block(state, code)
        return state

    def forward_tasks(self, x: Tensor, task_ids: Sequence[str]) -> Tensor:
        if len(x) != len(task_ids):
            raise ValueError("each input must have one task ID")
        return torch.cat(
            [self.forward(sample.unsqueeze(0), task_id) for sample, task_id in zip(x, task_ids, strict=True)],
            dim=0,
        )

    def shared_parameters(self) -> list[nn.Parameter]:
        return list(self.blocks.parameters())

    @property
    def shared_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.shared_parameters())

    @property
    def task_state_scalar_count(self) -> int:
        return sum(parameter.numel() for parameter in self.task_codes.values())


class ContinuousBasisLearner(nn.Module):
    """A shared operator basis mixed by per-task, per-step continuous codes."""

    def __init__(
        self,
        d: int,
        operator_slots: int,
        operator_rank: int,
        task_steps: int,
        alpha: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.operator_slots = operator_slots
        self.task_steps = task_steps
        self.basis = nn.ModuleList(
            LearnedOperator(d, operator_rank, alpha, seed + 997 * index)
            for index in range(operator_slots)
        )
        self.task_codes = nn.ParameterDict()

    def begin_task(self, task_id: str) -> nn.Parameter:
        if task_id in self.task_codes:
            raise ValueError(f"task already exists: {task_id}")
        # Identical zero logits enforce the same initialization policy for every task.
        self.task_codes[task_id] = nn.Parameter(
            torch.zeros(self.task_steps, self.operator_slots)
        )
        return self.task_codes[task_id]

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        z = x
        coefficients = torch.softmax(self.task_codes[task_id], dim=-1)
        for step in range(self.task_steps):
            candidates = torch.stack([operator(z) for operator in self.basis], dim=0)
            weights = coefficients[step].view(self.operator_slots, 1, 1)
            z = torch.sum(weights * candidates, dim=0)
        return z

    def forward_tasks(self, x: Tensor, task_ids: Sequence[str]) -> Tensor:
        if len(x) != len(task_ids):
            raise ValueError("each input must have one task ID")
        return torch.cat(
            [self.forward(sample.unsqueeze(0), task_id) for sample, task_id in zip(x, task_ids, strict=True)],
            dim=0,
        )

    def shared_parameters(self) -> list[nn.Parameter]:
        return list(self.basis.parameters())

    @property
    def shared_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.shared_parameters())

    @property
    def task_state_scalar_count(self) -> int:
        return sum(parameter.numel() for parameter in self.task_codes.values())

    def routing_diagnostics(self) -> dict[str, float]:
        if not self.task_codes:
            return {"mean_entropy_nats": 0.0, "mean_max_coefficient": 0.0}
        coefficients = torch.cat(
            [torch.softmax(code.detach(), dim=-1) for code in self.task_codes.values()], dim=0
        )
        entropy = -torch.sum(coefficients * torch.log(coefficients.clamp_min(1e-12)), dim=-1)
        return {
            "mean_entropy_nats": float(torch.mean(entropy)),
            "mean_max_coefficient": float(torch.mean(torch.max(coefficients, dim=-1).values)),
        }
