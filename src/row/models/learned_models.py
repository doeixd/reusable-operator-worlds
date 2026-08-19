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
            "initial_residual_state",
            torch.cat(
                (
                    initial_u.reshape(-1),
                    initial_v.reshape(-1),
                    torch.zeros(self.residual_b_size),
                )
            ),
        )
        self.task_codes = nn.ParameterDict()
        self.task_residuals = nn.ParameterDict()

    def begin_task(self, task_id: str) -> tuple[nn.Parameter, nn.Parameter]:
        if task_id in self.task_codes:
            raise ValueError(f"task already exists: {task_id}")
        self.task_codes[task_id] = nn.Parameter(torch.zeros(self.route_size))
        self.task_residuals[task_id] = nn.Parameter(
            self.initial_residual_state.clone()
        )
        return self.task_codes[task_id], self.task_residuals[task_id]

    def _unpack(self, task_id: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        route = self.task_codes[task_id]
        residual_u, residual_v, residual_b = torch.split(
            self.task_residuals[task_id],
            (
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
            self.task_residuals[task_id]
            for task_id in dict.fromkeys(task_ids)
        ]
        return torch.mean(torch.abs(torch.cat(residuals)))

    def routing_diagnostics(self) -> dict[str, float]:
        if not self.task_codes:
            return {"mean_entropy_nats": 0.0, "mean_max_coefficient": 0.0}
        coefficients = []
        for task_id in self.task_codes:
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
        return sum(parameter.numel() for parameter in self.task_codes.values()) + sum(
            parameter.numel() for parameter in self.task_residuals.values()
        )

    @torch.no_grad()
    def residual_diagnostics(self, probe: Tensor) -> dict[str, object]:
        fractions = []
        residual_rms_values = []
        for task_id in self.task_codes:
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


def _gaussian_kl(
    mu: Tensor, log_sigma: Tensor, prior_log_sigma: Tensor
) -> Tensor:
    """Per-coordinate KL(N(mu, sigma^2) || N(0, s^2)) in nats."""

    prior_variance = torch.exp(2.0 * prior_log_sigma)
    return (
        prior_log_sigma
        - log_sigma
        + (torch.exp(2.0 * log_sigma) + torch.square(mu)) / (2.0 * prior_variance)
        - 0.5
    )


class VariationalSharedResidualLearner(SharedParentResidualLearner):
    """Shared-residual learner whose task state is a variational code.

    Task-specific scalars are Gaussian posteriors q = N(mu, sigma^2) against a
    factorized prior N(0, s^2) whose scale is learned per task-state tensor
    TYPE and shared across all tasks, so description length enters the wake
    gradient as KL(q || p) rather than through an L1 surrogate. Retained task
    state is the posterior mean alone; the posterior scales are training
    state, discarded like optimizer state, so the two-part comparison against
    the frozen shared-residual learner stays scalar-for-scalar matched.
    """

    TASK_TENSOR_TYPES = ("route", "residual_u", "residual_v", "residual_b")

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
        prior_scale_init: float = 1.0,
        posterior_scale_init: float = 1e-3,
        prior_warmup_tasks: int = 8,
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
        if prior_scale_init <= 0.0 or posterior_scale_init <= 0.0:
            raise ValueError("prior and posterior scale initialization must be positive")
        self.posterior_log_scale_init = float(
            torch.log(torch.tensor(posterior_scale_init))
        )
        # The prior scale is learned by CLOSED-FORM empirical Bayes, not by
        # gradient descent. Gradient learning of a shared prior runs away:
        # whenever the posteriors are concentrated the gradient always says
        # "shrink", Adam's normalized step makes the move size independent of
        # the tiny gradient, and a collapsing prior then annihilates the task
        # state it is supposed to describe (measured: 1.0 -> 0.0034 over one
        # lifetime, after which a residual of 0.2 costs ~1,700 nats).
        self.prior_log_scales = nn.Parameter(
            torch.full(
                (len(self.TASK_TENSOR_TYPES),),
                float(torch.log(torch.tensor(prior_scale_init))),
            ),
            requires_grad=False,
        )
        self.minimum_prior_scale = 1e-6
        self.prior_warmup_tasks = int(prior_warmup_tasks)
        self.register_buffer(
            "residual_type_index",
            torch.cat(
                (
                    torch.full((self.residual_u_size,), 1, dtype=torch.long),
                    torch.full((self.residual_v_size,), 2, dtype=torch.long),
                    torch.full((self.residual_b_size,), 3, dtype=torch.long),
                )
            ),
        )
        self.task_code_log_sigma = nn.ParameterDict()
        self.task_residual_log_sigma = nn.ParameterDict()
        # Sampling uses a dedicated generator so reparameterization never
        # depends on (or perturbs) global torch RNG state; lifetimes stay
        # reproducible from the model seed alone.
        self._sample_generator = torch.Generator()
        self._sample_generator.manual_seed(int(seed) + 104729)

    def begin_task(
        self, task_id: str
    ) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter]:
        route_mu, residual_mu = super().begin_task(task_id)
        # The posterior starts near-deterministic and the prior starts wide:
        # sigma = prior would make the injected noise as large as the signal
        # it is supposed to carry, and a tight prior makes the KL gradient
        # (mu / s^2) swamp the data gradient. The code therefore starts at
        # high precision and RELAXES where precision turns out to be
        # unnecessary, which is the migration signature H11.1 looks for.
        self.task_code_log_sigma[task_id] = nn.Parameter(
            torch.full((self.route_size,), self.posterior_log_scale_init)
        )
        self.task_residual_log_sigma[task_id] = nn.Parameter(
            torch.full(
                (self.residual_type_index.numel(),), self.posterior_log_scale_init
            )
        )
        return (
            route_mu,
            self.task_code_log_sigma[task_id],
            residual_mu,
            self.task_residual_log_sigma[task_id],
        )

    def _sampled_state(self, task_id: str) -> tuple[Tensor, Tensor]:
        route_mu = self.task_codes[task_id]
        residual_mu = self.task_residuals[task_id]
        # Sample only inside a gradient-enabled training forward. Every
        # scoring path (online prediction, evaluation, diagnostics) runs
        # under no_grad and therefore uses the posterior mean, keeping
        # score-before-update and paired comparison rules intact.
        if not (self.training and torch.is_grad_enabled()):
            return route_mu, residual_mu
        route_sigma = torch.exp(self.task_code_log_sigma[task_id])
        residual_sigma = torch.exp(self.task_residual_log_sigma[task_id])
        route_noise = torch.randn(
            route_mu.shape, generator=self._sample_generator, device=route_mu.device
        )
        residual_noise = torch.randn(
            residual_mu.shape,
            generator=self._sample_generator,
            device=residual_mu.device,
        )
        return (
            route_mu + route_sigma * route_noise,
            residual_mu + residual_sigma * residual_noise,
        )

    def __getstate__(self) -> dict[str, object]:
        # torch.Generator is not deep-copyable; carry its seed and state so
        # checkpoint probes (which deep-copy the model) stay reproducible.
        state = dict(self.__dict__)
        generator = state.pop("_sample_generator", None)
        if generator is not None:
            state["_sample_generator_seed"] = int(generator.initial_seed())
            state["_sample_generator_state"] = generator.get_state().clone()
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        state = dict(state)
        seed = state.pop("_sample_generator_seed", None)
        tensor_state = state.pop("_sample_generator_state", None)
        self.__dict__.update(state)
        generator = torch.Generator()
        generator.manual_seed(int(seed) if seed is not None else 0)
        if tensor_state is not None:
            generator.set_state(tensor_state)
        self._sample_generator = generator

    def _unpack(self, task_id: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        route, residual = self._sampled_state(task_id)
        residual_u, residual_v, residual_b = torch.split(
            residual,
            (self.residual_u_size, self.residual_v_size, self.residual_b_size),
        )
        return (
            route.reshape(self.task_steps, self.operator_slots),
            residual_u.reshape(self.task_steps, self.d, self.residual_rank),
            residual_v.reshape(self.task_steps, self.residual_rank, self.d),
            residual_b.reshape(self.task_steps, self.residual_rank),
        )

    @torch.no_grad()
    def update_prior_scales(self, task_ids: Sequence[str] | None = None) -> None:
        """Closed-form empirical-Bayes M step for the shared prior scales.

        s^2 = mean over tasks and coordinates of (mu^2 + sigma^2) for each
        task-state tensor type. This is the exact maximizer, so the prior
        tracks the population spread of the task codes instead of racing to
        zero, and mean KL reduces to mean log(s / sigma) — "bits relative to
        what is typical across tasks", which is the semantics the promotion
        criterion needs.
        """

        selected_tasks = (
            [task_id for task_id in task_ids if task_id in self.task_codes]
            if task_ids is not None
            else list(self.task_codes)
        )
        # The prior describes the POPULATION of learned task codes, so it is
        # estimated from completed tasks only and not at all before a
        # population exists. Estimating it from an untrained code is a stable
        # degenerate fixed point: mu ~ 0 forces s ~ sigma_init, the resulting
        # mu / s^2 gradient pins every task code at zero, and the learner
        # never acquires task state at all (measured: prior 1.0 -> 0.002 on
        # the first update, uniform routes for the whole lifetime).
        if len(selected_tasks) < self.prior_warmup_tasks:
            return
        route_second_moment = torch.stack(
            [
                torch.mean(
                    torch.square(self.task_codes[task_id])
                    + torch.exp(2.0 * self.task_code_log_sigma[task_id])
                )
                for task_id in selected_tasks
            ]
        ).mean()
        scales = [torch.sqrt(route_second_moment)]
        residual_second_moments = torch.stack(
            [
                torch.square(self.task_residuals[task_id])
                + torch.exp(2.0 * self.task_residual_log_sigma[task_id])
                for task_id in selected_tasks
            ]
        ).mean(dim=0)
        for type_index in range(1, len(self.TASK_TENSOR_TYPES)):
            selected = residual_second_moments[self.residual_type_index == type_index]
            scales.append(torch.sqrt(selected.mean()))
        self.prior_log_scales.copy_(
            torch.log(
                torch.stack(scales).clamp_min(self.minimum_prior_scale)
            )
        )

    def coordinate_kl(self, task_id: str) -> Tensor:
        """Per-coordinate KL in nats for one task (route then residual)."""

        return torch.cat(
            (
                _gaussian_kl(
                    self.task_codes[task_id],
                    self.task_code_log_sigma[task_id],
                    self.prior_log_scales[0],
                ),
                _gaussian_kl(
                    self.task_residuals[task_id],
                    self.task_residual_log_sigma[task_id],
                    self.prior_log_scales[self.residual_type_index],
                ),
            )
        )

    def coordinate_mean_information(self, task_id: str) -> Tensor:
        """Nats each coordinate spends on having a NONZERO mean: mu^2/(2 s^2).

        Distinct from `coordinate_kl`, which also charges the precision term
        log(s / sigma). The full KL is the variational code length; this is
        the part a sparse code recovers by dropping the coordinate and paying
        one bitmap bit instead, so it is the right pruning criterion.
        """

        prior = torch.exp(
            torch.cat(
                (
                    self.prior_log_scales[0].expand(self.route_size),
                    self.prior_log_scales[self.residual_type_index],
                )
            )
        )
        means = torch.cat((self.task_codes[task_id], self.task_residuals[task_id]))
        return torch.square(means) / (2.0 * torch.square(prior))

    def description_penalty(self, task_ids: Sequence[str]) -> Tensor:
        """Mean per-task KL in nats over the tasks in a batch."""

        unique = list(dict.fromkeys(task_ids))
        return torch.stack(
            [torch.sum(self.coordinate_kl(task_id)) for task_id in unique]
        ).mean()

    def storage_penalty(self, task_ids: Sequence[str]) -> Tensor:
        # The KL charge replaces the L1 surrogate entirely; keeping this at
        # zero makes an accidental double charge impossible.
        return torch.zeros((), device=self.prior_log_scales.device)

    @property
    def task_state_scalar_count(self) -> int:
        # Retained task state is the posterior MEAN only; posterior scales are
        # training state, discarded like optimizer state.
        return sum(parameter.numel() for parameter in self.task_codes.values()) + sum(
            parameter.numel() for parameter in self.task_residuals.values()
        )

    @property
    def variational_training_state_scalar_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.task_code_log_sigma.values()
        ) + sum(
            parameter.numel() for parameter in self.task_residual_log_sigma.values()
        )

    def shared_parameters(self) -> list[nn.Parameter]:
        return [*self.basis.parameters(), self.prior_log_scales]

    @torch.no_grad()
    def variational_diagnostics(self, threshold_bits: float = 0.5) -> dict[str, object]:
        if not self.task_codes:
            return {}
        per_task_bits: list[float] = []
        informative_fractions: list[float] = []
        route_bits: list[float] = []
        residual_bits: list[float] = []
        for task_id in self.task_codes:
            bits = self.coordinate_kl(task_id) / float(torch.log(torch.tensor(2.0)))
            mean_bits = self.coordinate_mean_information(task_id) / float(
                torch.log(torch.tensor(2.0))
            )
            per_task_bits.append(float(torch.sum(bits)))
            informative_fractions.append(
                float(torch.mean((mean_bits >= threshold_bits).double()))
            )
            route_bits.append(float(torch.sum(bits[: self.route_size])))
            residual_bits.append(float(torch.sum(bits[self.route_size :])))
        return {
            "threshold_bits": threshold_bits,
            "total_task_kl_bits": float(sum(per_task_bits)),
            "mean_task_kl_bits": float(sum(per_task_bits) / len(per_task_bits)),
            "mean_route_kl_bits": float(sum(route_bits) / len(route_bits)),
            "mean_residual_kl_bits": float(sum(residual_bits) / len(residual_bits)),
            "mean_informative_coordinate_fraction": float(
                sum(informative_fractions) / len(informative_fractions)
            ),
            "prior_scales": [
                float(value) for value in torch.exp(self.prior_log_scales)
            ],
            "prior_scale_names": list(self.TASK_TENSOR_TYPES),
            "per_task_kl_bits": per_task_bits,
        }

    @torch.no_grad()
    def apply_information_prune(self, threshold_bits: float) -> dict[str, object]:
        """Zero every task coordinate carrying less than `threshold_bits`.

        A coordinate whose posterior has collapsed onto the prior carries no
        task information, so a sparse two-part code can drop it and pay one
        bitmap bit instead of a full quantized scalar. Callers must run this
        on a deep copy and validate behavior; it mutates the model.
        """

        retained = 0
        total = 0
        log2 = float(torch.log(torch.tensor(2.0)))
        for task_id in self.task_codes:
            bits = self.coordinate_mean_information(task_id) / log2
            keep = bits >= threshold_bits
            route_keep = keep[: self.route_size]
            residual_keep = keep[self.route_size :]
            self.task_codes[task_id].mul_(route_keep.to(self.task_codes[task_id].dtype))
            self.task_residuals[task_id].mul_(
                residual_keep.to(self.task_residuals[task_id].dtype)
            )
            retained += int(torch.sum(keep))
            total += int(keep.numel())
        return {
            "threshold_bits": threshold_bits,
            "retained_task_scalars": retained,
            "total_task_scalars": total,
            "retained_fraction": retained / total if total else 0.0,
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


class PresenceGatedDiscreteLibraryLearner(DiscreteLibraryLearner):
    """Discrete library with explicit learnable global slot-presence gates."""

    def __init__(
        self,
        d: int,
        operator_slots: int,
        operator_rank: int,
        task_steps: int,
        alpha: float,
        initial_temperature: float,
        final_temperature: float,
        presence_logit_init: float,
        presence_threshold: float,
        seed: int,
        learnable_alpha: bool = True,
        activation: str = "tanh",
    ) -> None:
        super().__init__(
            d=d,
            operator_slots=operator_slots,
            operator_rank=operator_rank,
            task_steps=task_steps,
            alpha=alpha,
            initial_temperature=initial_temperature,
            final_temperature=final_temperature,
            seed=seed,
            learnable_alpha=learnable_alpha,
            activation=activation,
        )
        if not 0.0 < presence_threshold < 1.0:
            raise ValueError("presence threshold must lie strictly between zero and one")
        self.presence_threshold = presence_threshold
        self.presence_logits = nn.Parameter(
            torch.full((operator_slots,), float(presence_logit_init))
        )

    def presence_probabilities(self) -> Tensor:
        return torch.sigmoid(self.presence_logits)

    def _active_mask(self) -> Tensor:
        probabilities = self.presence_probabilities()
        active = probabilities >= self.presence_threshold
        if not torch.any(active):
            active = torch.nn.functional.one_hot(
                torch.argmax(probabilities), self.operator_slots
            ).to(torch.bool)
        return active

    def _coefficients(self, task_id: str) -> Tensor:
        logits = self.task_codes[task_id]
        probabilities = self.presence_probabilities()
        if self.training:
            route = torch.softmax(logits / self.temperature, dim=-1)
            weighted = route * probabilities
            return weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scores = logits + torch.log(probabilities.clamp_min(1e-12))
        scores = scores.masked_fill(~self._active_mask().unsqueeze(0), -torch.inf)
        indices = torch.argmax(scores, dim=-1)
        return torch.nn.functional.one_hot(indices, self.operator_slots).to(logits.dtype)

    def shared_parameters(self) -> list[nn.Parameter]:
        return [*self.library.parameters(), self.presence_logits]

    def presence_penalty(self) -> Tensor:
        """Expected active-slot count under independent relaxed Bernoulli gates."""
        return self.presence_probabilities().sum()

    def route_entropy_penalty(self, task_ids: Sequence[str]) -> Tensor:
        unique = list(dict.fromkeys(task_ids))
        if not unique:
            return self.presence_logits.new_zeros(())
        entropies = []
        for task_id in unique:
            coefficients = self._coefficients(task_id)
            entropies.append(
                -torch.sum(
                    coefficients * torch.log(coefficients.clamp_min(1e-12)), dim=-1
                ).mean()
            )
        return torch.stack(entropies).mean()

    def hard_routes(self) -> dict[str, list[int]]:
        was_training = self.training
        self.eval()
        try:
            return {
                task_id: torch.argmax(self._coefficients(task_id), dim=-1)
                .detach()
                .cpu()
                .tolist()
                for task_id in self.task_codes
            }
        finally:
            self.train(was_training)

    def presence_diagnostics(self) -> dict[str, object]:
        probabilities = self.presence_probabilities().detach()
        active = self._active_mask().detach()
        routes = self.hard_routes()
        usage = torch.zeros(self.operator_slots, dtype=torch.int64)
        for route in routes.values():
            usage += torch.bincount(torch.as_tensor(route), minlength=self.operator_slots)
        return {
            "presence_probabilities": probabilities.cpu().tolist(),
            "expected_active_operators": float(probabilities.sum()),
            "active_operators_at_threshold": int(active.sum()),
            "presence_threshold": self.presence_threshold,
            "active_mask": active.cpu().tolist(),
            "active_but_unused_operators": int(torch.count_nonzero(active & (usage == 0))),
            "inactive_but_routed_operators": int(torch.count_nonzero((~active) & (usage > 0))),
        }

    def routing_diagnostics(self) -> dict[str, object]:
        base = super().routing_diagnostics()
        routes = self.hard_routes()
        usage = torch.zeros(self.operator_slots, dtype=torch.int64)
        position_usage = torch.zeros(self.task_steps, self.operator_slots, dtype=torch.int64)
        tasks_per_operator = torch.zeros(self.operator_slots, dtype=torch.int64)
        for route_values in routes.values():
            route = torch.as_tensor(route_values)
            usage += torch.bincount(route, minlength=self.operator_slots)
            for position, operator in enumerate(route):
                position_usage[position, operator] += 1
            for operator in torch.unique(route):
                tasks_per_operator[operator] += 1
        base.update(
            {
                "active_operators": int(torch.count_nonzero(usage)),
                "usage_counts": usage.tolist(),
                "tasks_per_operator": tasks_per_operator.tolist(),
                "position_usage": position_usage.tolist(),
            }
        )
        return base
