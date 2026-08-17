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
        output = torch.zeros_like(x)
        for task_id in dict.fromkeys(task_ids):
            indices = torch.tensor(
                [index for index, value in enumerate(task_ids) if value == task_id],
                device=x.device,
            )
            output = output.index_copy(0, indices, self.forward(x.index_select(0, indices), task_id))
        return output

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
        learnable_alpha: bool = True,
        activation: str = "tanh",
        include_identity: bool = False,
    ) -> None:
        super().__init__()
        self.learned_operator_slots = operator_slots
        self.include_identity = include_identity
        self.operator_slots = operator_slots + int(include_identity)
        self.task_steps = task_steps
        self.basis = nn.ModuleList(
            LearnedOperator(
                d,
                operator_rank,
                alpha,
                seed + 997 * index,
                learnable_alpha=learnable_alpha,
                activation=activation,
            )
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
            if self.include_identity:
                candidates = torch.cat((candidates, z.unsqueeze(0)), dim=0)
            weights = coefficients[step].view(self.operator_slots, 1, 1)
            z = torch.sum(weights * candidates, dim=0)
        return z

    def forward_tasks(self, x: Tensor, task_ids: Sequence[str]) -> Tensor:
        if len(x) != len(task_ids):
            raise ValueError("each input must have one task ID")
        output = torch.zeros_like(x)
        for task_id in dict.fromkeys(task_ids):
            indices = torch.tensor(
                [index for index, value in enumerate(task_ids) if value == task_id],
                device=x.device,
            )
            output = output.index_copy(0, indices, self.forward(x.index_select(0, indices), task_id))
        return output

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


class SharedParentResidualLearner(nn.Module):
    """Continuous shared parent plus penalized rank-limited task residuals."""

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
        super().__init__()
        self.d = d
        self.operator_slots = operator_slots
        self.residual_rank = residual_rank
        self.task_steps = task_steps
        self.route_size = task_steps * operator_slots
        self.residual_u_size = task_steps * d * residual_rank
        self.residual_v_size = task_steps * residual_rank * d
        self.residual_b_size = task_steps * residual_rank
        self.basis = nn.ModuleList(
            LearnedOperator(
                d,
                operator_rank,
                alpha,
                seed + 997 * index,
                learnable_alpha=learnable_alpha,
                activation=activation,
            )
            for index in range(operator_slots)
        )
        with torch.random.fork_rng():
            torch.manual_seed(seed + 7919)
            initial_u = 1e-3 * torch.randn(task_steps, d, residual_rank)
            initial_v = 1e-3 * torch.randn(task_steps, residual_rank, d)
        self.register_buffer(
            "initial_task_state",
            torch.cat(
                (
                    torch.zeros(self.route_size),
                    initial_u.reshape(-1),
                    initial_v.reshape(-1),
                    torch.zeros(self.residual_b_size),
                )
            ),
        )
        self.task_states = nn.ParameterDict()

    def begin_task(self, task_id: str) -> nn.Parameter:
        if task_id in self.task_states:
            raise ValueError(f"task already exists: {task_id}")
        self.task_states[task_id] = nn.Parameter(self.initial_task_state.clone())
        return self.task_states[task_id]

    def _unpack(self, task_id: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state = self.task_states[task_id]
        route, residual_u, residual_v, residual_b = torch.split(
            state,
            (
                self.route_size,
                self.residual_u_size,
                self.residual_v_size,
                self.residual_b_size,
            ),
        )
        return (
            route.reshape(self.task_steps, self.operator_slots),
            residual_u.reshape(self.task_steps, self.d, self.residual_rank),
            residual_v.reshape(self.task_steps, self.residual_rank, self.d),
            residual_b.reshape(self.task_steps, self.residual_rank),
        )

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        route, residual_u, residual_v, residual_b = self._unpack(task_id)
        coefficients = torch.softmax(route, dim=-1)
        z = x
        for step in range(self.task_steps):
            candidates = torch.stack([operator(z) for operator in self.basis], dim=0)
            parent = torch.sum(
                coefficients[step].view(self.operator_slots, 1, 1) * candidates,
                dim=0,
            )
            hidden = torch.tanh(
                torch.nn.functional.linear(z, residual_v[step], residual_b[step])
            )
            residual = torch.nn.functional.linear(hidden, residual_u[step])
            z = parent + residual
        return z

    def forward_tasks(self, x: Tensor, task_ids: Sequence[str]) -> Tensor:
        if len(x) != len(task_ids):
            raise ValueError("each input must have one task ID")
        output = torch.zeros_like(x)
        for task_id in dict.fromkeys(task_ids):
            indices = torch.tensor(
                [index for index, value in enumerate(task_ids) if value == task_id],
                device=x.device,
            )
            output = output.index_copy(
                0, indices, self.forward(x.index_select(0, indices), task_id)
            )
        return output

    def storage_penalty(self, task_ids: Sequence[str]) -> Tensor:
        residuals = [
            self.task_states[task_id][self.route_size :]
            for task_id in dict.fromkeys(task_ids)
        ]
        return torch.mean(torch.abs(torch.cat(residuals)))

    def routing_diagnostics(self) -> dict[str, float]:
        if not self.task_states:
            return {"mean_entropy_nats": 0.0, "mean_max_coefficient": 0.0}
        coefficients = []
        for task_id in self.task_states:
            route, _, _, _ = self._unpack(task_id)
            coefficients.append(torch.softmax(route.detach(), dim=-1))
        combined = torch.cat(coefficients, dim=0)
        entropy = -torch.sum(
            combined * torch.log(combined.clamp_min(1e-12)), dim=-1
        )
        return {
            "mean_entropy_nats": float(torch.mean(entropy)),
            "mean_max_coefficient": float(
                torch.mean(torch.max(combined, dim=-1).values)
            ),
        }

    def shared_parameters(self) -> list[nn.Parameter]:
        return list(self.basis.parameters())

    @property
    def shared_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.shared_parameters())

    @property
    def task_state_scalar_count(self) -> int:
        return sum(parameter.numel() for parameter in self.task_states.values())

    @torch.no_grad()
    def residual_diagnostics(self, probe: Tensor) -> dict[str, object]:
        fractions = []
        residual_rms_values = []
        for task_id in self.task_states:
            route, residual_u, residual_v, residual_b = self._unpack(task_id)
            coefficients = torch.softmax(route, dim=-1)
            z = probe
            task_fractions = []
            for step in range(self.task_steps):
                candidates = torch.stack(
                    [operator(z) for operator in self.basis], dim=0
                )
                parent = torch.sum(
                    coefficients[step].view(self.operator_slots, 1, 1)
                    * candidates,
                    dim=0,
                )
                hidden = torch.tanh(
                    torch.nn.functional.linear(
                        z, residual_v[step], residual_b[step]
                    )
                )
                residual = torch.nn.functional.linear(hidden, residual_u[step])
                residual_rms = torch.sqrt(torch.mean(torch.square(residual)))
                parent_update_rms = torch.sqrt(torch.mean(torch.square(parent - z)))
                task_fractions.append(
                    float(residual_rms / parent_update_rms.clamp_min(1e-12))
                )
                residual_rms_values.append(float(residual_rms))
                z = parent + residual
            fractions.append(float(torch.tensor(task_fractions).mean()))
        return {
            "residual_rank": self.residual_rank,
            "mean_functional_residual_to_parent_update_ratio": float(
                torch.tensor(fractions).mean()
            ),
            "median_functional_residual_to_parent_update_ratio": float(
                torch.tensor(fractions).median()
            ),
            "maximum_task_functional_residual_to_parent_update_ratio": max(fractions),
            "mean_residual_output_rms": float(
                torch.tensor(residual_rms_values).mean()
            ),
            "per_task_functional_ratio": fractions,
        }


class HypernetworkLearner(nn.Module):
    """Generate per-step low-rank operators from opaque task codes without slots."""

    def __init__(
        self,
        d: int,
        step_code_dim: int,
        hypernetwork_hidden_dim: int,
        operator_rank: int,
        task_steps: int,
        alpha: float,
        seed: int,
        learnable_alpha: bool = True,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if activation not in {"tanh", "gelu"}:
            raise ValueError("activation must be 'tanh' or 'gelu'")
        self.d = d
        self.step_code_dim = step_code_dim
        self.operator_rank = operator_rank
        self.task_steps = task_steps
        self.activation = activation
        parameter_dim = 2 * d * operator_rank + operator_rank
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            V = torch.randn(operator_rank, d)
            U = torch.randn(d, operator_rank)
            self.V = nn.Parameter(V / torch.linalg.matrix_norm(V, ord=2))
            self.U = nn.Parameter(U / torch.linalg.matrix_norm(U, ord=2))
            self.b = nn.Parameter(torch.zeros(operator_rank))
            self.code_to_hidden = nn.Linear(
                step_code_dim, hypernetwork_hidden_dim, bias=False
            )
            self.hidden_to_parameters = nn.Linear(
                hypernetwork_hidden_dim, parameter_dim, bias=True
            )
            nn.init.normal_(self.hidden_to_parameters.weight, std=1e-3)
            nn.init.zeros_(self.hidden_to_parameters.bias)
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.alpha = float(alpha)
        self.task_codes = nn.ParameterDict()

    def begin_task(self, task_id: str) -> nn.Parameter:
        if task_id in self.task_codes:
            raise ValueError(f"task already exists: {task_id}")
        self.task_codes[task_id] = nn.Parameter(
            torch.zeros(self.task_steps, self.step_code_dim)
        )
        return self.task_codes[task_id]

    def _generated_parameters(self, code: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = torch.tanh(self.code_to_hidden(code))
        generated = self.hidden_to_parameters(hidden)
        u_size = self.d * self.operator_rank
        v_size = self.operator_rank * self.d
        delta_u, delta_v, delta_b = torch.split(
            generated, (u_size, v_size, self.operator_rank), dim=-1
        )
        return (
            self.U + delta_u.reshape(self.d, self.operator_rank),
            self.V + delta_v.reshape(self.operator_rank, self.d),
            self.b + delta_b,
        )

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        z = x
        for code in self.task_codes[task_id]:
            U, V, b = self._generated_parameters(code)
            hidden = torch.nn.functional.linear(z, V, b)
            hidden = (
                torch.tanh(hidden)
                if self.activation == "tanh"
                else torch.nn.functional.gelu(hidden)
            )
            z = torch.tanh(z + self.alpha * torch.nn.functional.linear(hidden, U))
        return z

    def forward_tasks(self, x: Tensor, task_ids: Sequence[str]) -> Tensor:
        if len(x) != len(task_ids):
            raise ValueError("each input must have one task ID")
        output = torch.zeros_like(x)
        for task_id in dict.fromkeys(task_ids):
            indices = torch.tensor(
                [index for index, value in enumerate(task_ids) if value == task_id],
                device=x.device,
            )
            output = output.index_copy(
                0, indices, self.forward(x.index_select(0, indices), task_id)
            )
        return output

    def shared_parameters(self) -> list[nn.Parameter]:
        task_ids = {id(parameter) for parameter in self.task_codes.values()}
        return [parameter for parameter in self.parameters() if id(parameter) not in task_ids]

    @property
    def shared_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.shared_parameters())

    @property
    def task_state_scalar_count(self) -> int:
        return sum(parameter.numel() for parameter in self.task_codes.values())


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
        learnable_alpha: bool = True,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.operator_slots = operator_slots
        self.task_steps = task_steps
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature = initial_temperature
        self.library = nn.ModuleList(
            LearnedOperator(
                d,
                operator_rank,
                alpha,
                seed + 997 * index,
                learnable_alpha=learnable_alpha,
                activation=activation,
            )
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
        output = torch.zeros_like(x)
        for task_id in dict.fromkeys(task_ids):
            indices = torch.tensor(
                [index for index, value in enumerate(task_ids) if value == task_id],
                device=x.device,
            )
            output = output.index_copy(0, indices, self.forward(x.index_select(0, indices), task_id))
        return output

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
