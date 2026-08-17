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


class DiscreteLibraryLearner(nn.Module):
    """Overcomplete operator library with relaxed training and hard evaluation routes."""

    def __init__(
        self,
        d: int,
        operator_slots: int,
        operator_rank: int,
        task_steps: int,
        alpha: float,
        initial_temperature: float,
        final_temperature: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.operator_slots = operator_slots
        self.task_steps = task_steps
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature = initial_temperature
        self.library = nn.ModuleList(
            LearnedOperator(d, operator_rank, alpha, seed + 997 * index)
            for index in range(operator_slots)
        )
        self.task_codes = nn.ParameterDict()

    def begin_task(self, task_id: str) -> nn.Parameter:
        if task_id in self.task_codes:
            raise ValueError(f"task already exists: {task_id}")
        self.task_codes[task_id] = nn.Parameter(
            torch.zeros(self.task_steps, self.operator_slots)
        )
        return self.task_codes[task_id]

    def set_training_progress(self, fraction: float) -> None:
        fraction = min(1.0, max(0.0, fraction))
        ratio = self.final_temperature / self.initial_temperature
        self.temperature = self.initial_temperature * ratio**fraction

    def _coefficients(self, task_id: str) -> Tensor:
        logits = self.task_codes[task_id]
        if self.training:
            return torch.softmax(logits / self.temperature, dim=-1)
        indices = torch.argmax(logits, dim=-1)
        return torch.nn.functional.one_hot(indices, self.operator_slots).to(logits.dtype)

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        z = x
        coefficients = self._coefficients(task_id)
        for step in range(self.task_steps):
            candidates = torch.stack([operator(z) for operator in self.library], dim=0)
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
        return list(self.library.parameters())

    @property
    def shared_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.shared_parameters())

    @property
    def task_state_scalar_count(self) -> int:
        return sum(parameter.numel() for parameter in self.task_codes.values())

    def hard_routes(self) -> dict[str, list[int]]:
        return {
            task_id: torch.argmax(code.detach(), dim=-1).cpu().tolist()
            for task_id, code in self.task_codes.items()
        }

    def routing_diagnostics(self) -> dict[str, object]:
        if not self.task_codes:
            return {
                "mean_entropy_nats": 0.0,
                "mean_max_coefficient": 0.0,
                "active_operators": 0,
                "usage_counts": [],
            }
        probabilities = torch.cat(
            [
                torch.softmax(code.detach() / self.temperature, dim=-1)
                for code in self.task_codes.values()
            ],
            dim=0,
        )
        entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-12)), dim=-1)
        routes = torch.cat(
            [torch.argmax(code.detach(), dim=-1) for code in self.task_codes.values()]
        )
        usage = torch.bincount(routes, minlength=self.operator_slots)
        position_usage = torch.zeros(self.task_steps, self.operator_slots, dtype=torch.int64)
        tasks_per_operator = torch.zeros(self.operator_slots, dtype=torch.int64)
        for code in self.task_codes.values():
            route = torch.argmax(code.detach(), dim=-1)
            for position, operator in enumerate(route):
                position_usage[position, operator] += 1
            for operator in torch.unique(route):
                tasks_per_operator[operator] += 1
        return {
            "mean_entropy_nats": float(torch.mean(entropy)),
            "mean_max_coefficient": float(torch.mean(torch.max(probabilities, dim=-1).values)),
            "active_operators": int(torch.count_nonzero(usage)),
            "usage_counts": usage.cpu().tolist(),
            "tasks_per_operator": tasks_per_operator.cpu().tolist(),
            "position_usage": position_usage.cpu().tolist(),
            "final_temperature": self.temperature,
        }
