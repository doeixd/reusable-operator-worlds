"""Shared online protocol for non-oracle ROW learners."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment

from row.config import ExperimentConfig, load_config
from row.experiments.oracle_lifetime import _add_lifetime_transfer_summary, _functional_recovery
from row.experiments.scratch_difficulty import summarize
from row.metrics import examples_to_criterion, gaussian_nll, nmse
from row.models import (
    ContinuousBasisLearner,
    DenseLearner,
    DiscreteLibraryLearner,
    HypernetworkLearner,
    PresenceGatedDiscreteLibraryLearner,
    GatedInnovationLearner,
    LifecycleLibraryLearner,
    PromotingSharedResidualLearner,
    SharedParentResidualLearner,
    VariationalSharedResidualLearner,
)
from row.models.prospective_models import ProspectiveLifecycleLearner
from row.models.factorized_models import FactorizedLifecycleLearner
from row.models.pslot_models import ParameterizedSlotLearner

# H39b pilot knobs for kind="pslot"; set by the runner.
PSLOT_SETTINGS: dict[str, object] = {"slot_args": 2, "freeze_args": False, "freeze_matrices": False}
ARGUMENT_LEARNERS = (FactorizedLifecycleLearner, ParameterizedSlotLearner)

# H39 pilot: architecture knobs for kind="factorized". The runner sets
# these before calling `run`; they are recorded in the artifact's
# provenance by the runner and in `factorized.json` by the model.
FACTORIZED_SETTINGS: dict[str, object] = {
    "schema_dim": 2, "schema_count": 1, "schema_seed": 39001,
    "schema_init_scale": 1e-2, "freeze_schema": False,
}
from row.provenance import current_git_commit, write_fingerprint
from row.world import Program, Task, World

ModelKind = Literal[
    "dense",
    "continuous",
    "hypernetwork",
    "shared_residual",
    "variational",
    "gated",
    "promoting",
    "lifecycle",
    "prospective",
    "factorized",
    "pslot",
    "discrete",
    "mdl",
]
Learner = (
    DenseLearner
    | ContinuousBasisLearner
    | HypernetworkLearner
    | SharedParentResidualLearner
    | VariationalSharedResidualLearner
    | GatedInnovationLearner
    | PromotingSharedResidualLearner
    | LifecycleLibraryLearner
    | DiscreteLibraryLearner
    | PresenceGatedDiscreteLibraryLearner
)


class TaskReplayBuffer:
    def __init__(self, seed: int, sampling_seed: int | None = None) -> None:
        self.add_generator = np.random.default_rng(seed)
        self.sample_generator = (
            self.add_generator
            if sampling_seed is None
            else np.random.default_rng(sampling_seed)
        )
        self.items: list[tuple[np.ndarray, np.ndarray, str]] = []

    def add_task(self, task: Task, count: int) -> None:
        if count == 0:
            return
        indices = self.add_generator.choice(
            len(task.train_x), size=min(count, len(task.train_x)), replace=False
        )
        for index in indices:
            self.items.append((task.train_x[index].copy(), task.train_y[index].copy(), task.task_id))

    def sample(self, count: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
        if count == 0 or not self.items:
            return []
        indices = self.sample_generator.choice(
            len(self.items), size=min(count, len(self.items)), replace=False
        )
        return [self.items[int(index)] for index in np.atleast_1d(indices)]


def _update_batch_counts(batch_size: int, replay_ratio: float) -> tuple[int, int]:
    if batch_size <= 0:
        raise ValueError("update batch size must be positive")
    if replay_ratio < 0:
        raise ValueError("replay ratio must be nonnegative")
    current = max(1, int(round(batch_size / (1.0 + replay_ratio))))
    return current, batch_size - current


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32)


def _shared_optimizer(
    model: Learner,
    learning_rate: float,
    weight_decay: float,
    presence_learning_rate: float | None = None,
    scale_learning_rate: float | None = None,
) -> torch.optim.AdamW:
    alpha_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name == "alpha" or name.endswith(".alpha")
    ]
    alpha_ids = {id(parameter) for parameter in alpha_parameters}
    presence_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name == "presence_logits"
    ]
    presence_ids = {id(parameter) for parameter in presence_parameters}
    prior_scale_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name == "prior_log_scales"
    ]
    prior_scale_ids = {id(parameter) for parameter in prior_scale_parameters}
    decay_parameters = [
        parameter
        for parameter in model.shared_parameters()
        if id(parameter) not in alpha_ids
        and id(parameter) not in presence_ids
        and id(parameter) not in prior_scale_ids
    ]
    groups: list[dict[str, object]] = [
        {"params": decay_parameters, "weight_decay": weight_decay}
    ]
    if alpha_parameters:
        groups.append({"params": alpha_parameters, "weight_decay": 0.0})
    if presence_parameters:
        groups.append(
            {
                "params": presence_parameters,
                "lr": (
                    learning_rate
                    if presence_learning_rate is None
                    else presence_learning_rate
                ),
                "weight_decay": 0.0,
            }
        )
    return torch.optim.AdamW(groups, lr=learning_rate)


def _compute_accounting(config: ExperimentConfig, kind: ModelKind) -> dict[str, int | str]:
    d = config.world.state_dim
    if kind == "dense":
        selected = config.dense_model
        madds = selected.residual_blocks * (
            (d + selected.task_embedding_dim) * selected.hidden_width
            + selected.hidden_width * d
        )
        return {
            "training_forward_multiply_adds_per_sample": madds,
            "inference_multiply_adds_per_sample": madds,
            "note": "backward and optimizer operations excluded",
        }
    if kind == "continuous":
        selected = config.continuous_model
        learned = selected.task_steps * selected.operator_slots * (
            d * selected.operator_rank + selected.operator_rank * d
        )
        mixture_slots = selected.operator_slots + int(selected.include_identity)
        mixture = selected.task_steps * mixture_slots * d
        return {
            "training_forward_multiply_adds_per_sample": learned + mixture,
            "inference_multiply_adds_per_sample": learned + mixture,
            "note": "backward and optimizer operations excluded",
        }
    if kind == "hypernetwork":
        selected = config.hypernetwork_model
        parameter_dim = 2 * d * selected.operator_rank + selected.operator_rank
        generation = selected.task_steps * (
            selected.step_code_dim * selected.hypernetwork_hidden_dim
            + selected.hypernetwork_hidden_dim * parameter_dim
        )
        operators = selected.task_steps * (
            d * selected.operator_rank + selected.operator_rank * d
        )
        return {
            "training_forward_multiply_adds_per_sample": generation + operators,
            "inference_multiply_adds_per_sample": generation + operators,
            "note": "counts per-prediction operator generation; backward and optimizer operations excluded",
        }
    if kind in {"shared_residual", "variational", "gated", "promoting",
                "lifecycle", "prospective", "factorized", "pslot"}:
        selected = {
            "variational": config.variational_model,
            "gated": config.gated_model,
            "shared_residual": config.shared_residual_model,
            "promoting": config.shared_residual_model,
            "lifecycle": config.shared_residual_model,
        "prospective": config.shared_residual_model,
        "factorized": config.shared_residual_model,
        "pslot": config.shared_residual_model,
        }[kind]
        parent = selected.task_steps * selected.operator_slots * (
            d * selected.operator_rank + selected.operator_rank * d
        ) + selected.task_steps * selected.operator_slots * d
        residual = selected.task_steps * (
            d * selected.residual_rank + selected.residual_rank * d
        )
        return {
            "training_forward_multiply_adds_per_sample": parent + residual,
            "inference_multiply_adds_per_sample": parent + residual,
            "note": "rank-limited task residual included; backward and optimizer operations excluded",
        }
    selected = config.mdl_model if kind == "mdl" else config.discrete_model
    all_slots = selected.task_steps * selected.operator_slots * (
        d * selected.operator_rank + selected.operator_rank * d
    ) + selected.task_steps * selected.operator_slots * d
    hard = selected.task_steps * (d * selected.operator_rank + selected.operator_rank * d)
    return {
        "training_forward_multiply_adds_per_sample": all_slots,
        "inference_multiply_adds_per_sample": hard,
        "note": "training evaluates all relaxed slots; backward and optimizer operations excluded",
    }


def _build_model(config: ExperimentConfig, kind: ModelKind) -> Learner:
    if kind == "dense":
        model_config = config.dense_model
        return DenseLearner(
            d=config.world.state_dim,
            task_embedding_dim=model_config.task_embedding_dim,
            hidden_width=model_config.hidden_width,
            residual_blocks=model_config.residual_blocks,
            seed=model_config.seed,
        )
    if kind == "continuous":
        model_config = config.continuous_model
        return ContinuousBasisLearner(
            d=config.world.state_dim,
            operator_slots=model_config.operator_slots,
            operator_rank=model_config.operator_rank,
            task_steps=model_config.task_steps,
            alpha=model_config.operator_alpha_init,
            seed=model_config.seed,
            learnable_alpha=model_config.learnable_alpha,
            activation=model_config.operator_activation,
            include_identity=model_config.include_identity,
        )
    if kind == "hypernetwork":
        model_config = config.hypernetwork_model
        return HypernetworkLearner(
            d=config.world.state_dim,
            step_code_dim=model_config.step_code_dim,
            hypernetwork_hidden_dim=model_config.hypernetwork_hidden_dim,
            operator_rank=model_config.operator_rank,
            task_steps=model_config.task_steps,
            alpha=model_config.operator_alpha_init,
            seed=model_config.seed,
            learnable_alpha=model_config.learnable_alpha,
            activation=model_config.operator_activation,
        )
    if kind in {"promoting", "lifecycle", "prospective", "factorized", "pslot"}:
        model_config = config.shared_residual_model
        # `prospective` is a strict superset of `lifecycle`: it adds the
        # V6 adaptation penalty and no parameters, so a prospective run
        # with no arm reproduces a lifecycle run exactly. Kept as its own
        # kind rather than swapping the class under `lifecycle`, so V5's
        # artifacts stay reproducible from their own fingerprints.
        builder = {
            "lifecycle": LifecycleLibraryLearner,
            "prospective": ProspectiveLifecycleLearner,
            "promoting": PromotingSharedResidualLearner,
            "factorized": FactorizedLifecycleLearner,
            "pslot": ParameterizedSlotLearner,
        }[kind]
        extra = {}
        if kind == "pslot":
            extra = dict(PSLOT_SETTINGS)
        if kind == "factorized":
            # H39 pilot knobs, set by the runner; world seed fixes the
            # schema initialization stream.
            extra = dict(FACTORIZED_SETTINGS, world_seed=config.world.seed)
        return builder(
            d=config.world.state_dim,
            operator_slots=model_config.operator_slots,
            operator_rank=model_config.operator_rank,
            residual_rank=model_config.residual_rank,
            task_steps=model_config.task_steps,
            alpha=model_config.operator_alpha_init,
            seed=model_config.seed,
            learnable_alpha=model_config.learnable_alpha,
            activation=model_config.operator_activation,
            **extra,
        )
    if kind == "gated":
        model_config = config.gated_model
        return GatedInnovationLearner(
            d=config.world.state_dim,
            operator_slots=model_config.operator_slots,
            operator_rank=model_config.operator_rank,
            residual_rank=model_config.residual_rank,
            task_steps=model_config.task_steps,
            alpha=model_config.operator_alpha_init,
            seed=model_config.seed,
            learnable_alpha=model_config.learnable_alpha,
            activation=model_config.operator_activation,
            gate_temperature=model_config.gate_temperature,
            gate_logit_init=model_config.gate_logit_init,
        )
    if kind == "variational":
        model_config = config.variational_model
        return VariationalSharedResidualLearner(
            d=config.world.state_dim,
            operator_slots=model_config.operator_slots,
            operator_rank=model_config.operator_rank,
            residual_rank=model_config.residual_rank,
            task_steps=model_config.task_steps,
            alpha=model_config.operator_alpha_init,
            seed=model_config.seed,
            learnable_alpha=model_config.learnable_alpha,
            activation=model_config.operator_activation,
            prior_scale_init=model_config.prior_scale_init,
            posterior_scale_init=model_config.posterior_scale_init,
            prior_warmup_tasks=model_config.prior_warmup_tasks,
        )
    if kind == "shared_residual":
        model_config = config.shared_residual_model
        return SharedParentResidualLearner(
            d=config.world.state_dim,
            operator_slots=model_config.operator_slots,
            operator_rank=model_config.operator_rank,
            residual_rank=model_config.residual_rank,
            task_steps=model_config.task_steps,
            alpha=model_config.operator_alpha_init,
            seed=model_config.seed,
            learnable_alpha=model_config.learnable_alpha,
            activation=model_config.operator_activation,
        )
    model_config = config.mdl_model if kind == "mdl" else config.discrete_model
    model_class = (
        PresenceGatedDiscreteLibraryLearner
        if kind == "mdl"
        else DiscreteLibraryLearner
    )
    extra = (
        {
            "presence_logit_init": model_config.presence_logit_init,
            "presence_threshold": model_config.presence_threshold,
        }
        if kind == "mdl"
        else {}
    )
    return model_class(
        d=config.world.state_dim,
        operator_slots=model_config.operator_slots,
        operator_rank=model_config.operator_rank,
        task_steps=model_config.task_steps,
        alpha=model_config.operator_alpha_init,
        initial_temperature=model_config.initial_temperature,
        final_temperature=model_config.final_temperature,
        seed=model_config.seed,
        learnable_alpha=model_config.learnable_alpha,
        activation=model_config.operator_activation,
        **extra,
    )


def _training_values(
    config: ExperimentConfig, kind: ModelKind
) -> tuple[float, float, float, int, int, float, int]:
    selected = {
        "dense": config.dense_model,
        "continuous": config.continuous_model,
        "hypernetwork": config.hypernetwork_model,
        "shared_residual": config.shared_residual_model,
        "variational": config.variational_model,
        "gated": config.gated_model,
        "promoting": config.shared_residual_model,
        "lifecycle": config.shared_residual_model,
        "prospective": config.shared_residual_model,
        "factorized": config.shared_residual_model,
        "pslot": config.shared_residual_model,
        "discrete": config.discrete_model,
        "mdl": config.mdl_model,
    }[kind]
    return (
        selected.global_learning_rate,
        selected.task_learning_rate,
        selected.weight_decay,
        selected.updates_per_example,
        selected.replay_examples_per_task,
        selected.replay_ratio,
        selected.seed,
    )


@torch.no_grad()
def _evaluate(model: Learner, task: Task) -> tuple[float, np.ndarray]:
    model.eval()
    prediction = model(_tensor(task.eval_x), task.task_id).cpu().numpy()
    return nmse(prediction, task.eval_y), prediction


def run(
    config: ExperimentConfig,
    kind: ModelKind,
    order: str = "forward",
    task_id_scramble_seed: int | None = None,
    update_batch_size: int | None = None,
    freeze_shared_at: int | None = None,
    freeze_slots: int | None = None,
    sleeps: tuple[int, ...] = (),
    promotion_epsilon: float = 0.02,
    reuse_decision_at: int = 8,
    lifecycle_enabled: bool = False,
    lifecycle_filter: bool = False,
    force_retire_at: int | None = None,
    force_retire_one: bool = False,
    lifecycle_kappa: float = 0.0,
    lifecycle_grace: int = 8,
    prospective_hook=None,
    schema_index_hook=None,
    snapshot_history: bool = False,
) -> dict[str, object]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be 'forward' or 'reverse'")
    torch.set_num_threads(1)
    world = World.generate(config.world)
    if task_id_scramble_seed is not None:
        world = world.with_scrambled_task_ids(task_id_scramble_seed)
    model = _build_model(config, kind)
    global_lr, task_lr, weight_decay, update_count, replay_per_task, replay_ratio, seed = (
        _training_values(config, kind)
    )
    optimizer = _shared_optimizer(
        model,
        global_lr,
        weight_decay,
        presence_learning_rate=(
            config.mdl_model.presence_learning_rate if kind == "mdl" else None
        ),
        scale_learning_rate=(
            config.variational_model.scale_learning_rate
            if kind == "variational"
            else None
        ),
    )
    if update_batch_size is None:
        replay = TaskReplayBuffer(seed + 1)
    else:
        replay_seed = int(
            np.random.SeedSequence([config.world.seed, 96]).generate_state(1)[0]
        )
        replay = TaskReplayBuffer(replay_seed, sampling_seed=replay_seed + 1)
    task_indices = list(range(len(world.tasks)))
    if order == "reverse":
        task_indices.reverse()
    support = set(config.evaluation.support_points)
    rows: list[dict[str, object]] = []
    cumulative_nll = 0.0
    cumulative_mass_log_loss = 0.0
    checkpoint_results: list[dict[str, object]] = []
    observed_update_batch_sizes: list[int] = []
    completed_task_ids: list[str] = []
    sleep_records: list[dict[str, object]] = []
    history_residuals: dict[str, torch.Tensor] = {}
    history_codes: dict[str, torch.Tensor] = {}
    history_eps: dict[str, torch.Tensor] = {}

    for lifetime_index, world_task_index in enumerate(task_indices):
        task = world.tasks[world_task_index]
        if freeze_shared_at is not None and lifetime_index == freeze_shared_at:
            # Saturated-library condition: the shared operators stop learning,
            # so new recurring structure has nowhere to go but task-local
            # innovations. This is what makes an explicit promotion operator
            # necessary rather than optional.
            #
            # `freeze_slots` freezes only the FIRST N basis slots, leaving the
            # rest trainable. That is the oracle for promotion: a library
            # that grew by (slots - N) operators at exactly the right moment,
            # against which a real promoter can be judged.
            if freeze_slots is None:
                for parameter in model.shared_parameters():
                    parameter.requires_grad_(False)
            else:
                for slot_index, operator in enumerate(model.basis):
                    if slot_index < freeze_slots:
                        for parameter in operator.parameters():
                            parameter.requires_grad_(False)
        if isinstance(model, ARGUMENT_LEARNERS):
            schema_index = (
                schema_index_hook(world_task_index) if schema_index_hook else 0
            )
            task_parameter = model.begin_task(task.task_id, schema_index=schema_index)
        else:
            task_parameter = model.begin_task(task.task_id)
        if isinstance(model, ARGUMENT_LEARNERS):
            route_parameter, residual_parameter, alpha_parameter = task_parameter
            optimizer.add_param_group(
                {"params": [route_parameter], "lr": task_lr, "weight_decay": 0.0}
            )
            optimizer.add_param_group(
                {
                    "params": [residual_parameter],
                    "lr": config.shared_residual_model.residual_learning_rate,
                    "weight_decay": 0.0,
                }
            )
            optimizer.add_param_group(
                {"params": [alpha_parameter], "lr": task_lr, "weight_decay": 0.0}
            )
        elif isinstance(model, GatedInnovationLearner):
            route_parameter, residual_parameter, gate_parameter = task_parameter
            gated_config = config.gated_model
            for parameter, parameter_lr in (
                (route_parameter, task_lr),
                (residual_parameter, gated_config.residual_learning_rate),
                (gate_parameter, gated_config.gate_learning_rate),
            ):
                optimizer.add_param_group(
                    {"params": [parameter], "lr": parameter_lr, "weight_decay": 0.0}
                )
        elif isinstance(model, VariationalSharedResidualLearner):
            (
                route_parameter,
                route_scale_parameter,
                residual_parameter,
                residual_scale_parameter,
            ) = task_parameter
            variational_config = config.variational_model
            for parameter, parameter_lr in (
                (route_parameter, task_lr),
                (route_scale_parameter, variational_config.scale_learning_rate),
                (residual_parameter, variational_config.residual_learning_rate),
                (residual_scale_parameter, variational_config.scale_learning_rate),
            ):
                optimizer.add_param_group(
                    {"params": [parameter], "lr": parameter_lr, "weight_decay": 0.0}
                )
        elif isinstance(model, SharedParentResidualLearner):
            route_parameter, residual_parameter = task_parameter
            optimizer.add_param_group(
                {"params": [route_parameter], "lr": task_lr, "weight_decay": 0.0}
            )
            optimizer.add_param_group(
                {
                    "params": [residual_parameter],
                    "lr": config.shared_residual_model.residual_learning_rate,
                    "weight_decay": 0.0,
                }
            )
        else:
            optimizer.add_param_group(
                {"params": [task_parameter], "lr": task_lr, "weight_decay": 0.0}
            )
        curve: dict[int, float] = {}
        for n_seen in range(config.world.examples_per_task + 1):
            if n_seen in support:
                score, prediction = _evaluate(model, task)
                curve[n_seen] = score
                rows.append(
                    {
                        "record_type": "evaluation",
                        "task_index": lifetime_index,
                        "world_task_index": world_task_index,
                        "task_id": task.task_id,
                        "n_seen": n_seen,
                        "nmse": score,
                        "gaussian_nll": gaussian_nll(
                            prediction, task.eval_y, config.evaluation.gaussian_sigma
                        ),
                    }
                )
            if n_seen == config.world.examples_per_task:
                break

            x = _tensor(task.train_x[n_seen : n_seen + 1])
            model.eval()
            with torch.no_grad():
                online_prediction = model(x, task.task_id).cpu().numpy()
            online_nll = gaussian_nll(
                online_prediction, task.train_y[n_seen : n_seen + 1], config.evaluation.gaussian_sigma
            )
            cumulative_nll += online_nll
            online_mass_log_loss = online_nll - config.world.state_dim * np.log(
                config.evaluation.target_precision
            )
            cumulative_mass_log_loss += online_mass_log_loss
            rows.append(
                {
                    "record_type": "prequential",
                    "task_index": lifetime_index,
                    "world_task_index": world_task_index,
                    "task_id": task.task_id,
                    "n_seen": n_seen,
                    "nll": online_nll,
                    "cumulative_nll": cumulative_nll,
                    "quantized_target_log_loss": online_mass_log_loss,
                    "cumulative_quantized_target_log_loss": cumulative_mass_log_loss,
                }
            )

            if update_batch_size is None:
                current_indices = [n_seen]
                replay_count = int(round(replay_ratio))
            else:
                current_count, replay_count = _update_batch_counts(
                    update_batch_size, replay_ratio
                )
                prior_count = min(current_count - 1, n_seen)
                prior_generator = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.world.seed, 97, world_task_index, n_seen]
                    )
                )
                prior_indices = prior_generator.choice(
                    n_seen, size=prior_count, replace=False
                )
                current_indices = [
                    n_seen,
                    *(int(index) for index in np.atleast_1d(prior_indices)),
                ]
            if (
                isinstance(model, PromotingSharedResidualLearner)
                and n_seen == reuse_decision_at
            ):
                model.select_reference(
                    task.task_id,
                    _tensor(task.train_x[:n_seen]),
                    _tensor(task.train_y[:n_seen]),
                )
            replay_items = replay.sample(replay_count)
            batch_x = [
                *(task.train_x[index] for index in current_indices),
                *(item[0] for item in replay_items),
            ]
            batch_y = [
                *(task.train_y[index] for index in current_indices),
                *(item[1] for item in replay_items),
            ]
            task_ids = [
                *(task.task_id for _ in current_indices),
                *(item[2] for item in replay_items),
            ]
            observed_update_batch_sizes.append(len(batch_x))
            for _ in range(update_count):
                if isinstance(model, DiscreteLibraryLearner):
                    global_example = lifetime_index * config.world.examples_per_task + n_seen
                    total_examples = len(world.tasks) * config.world.examples_per_task
                    discrete_config = (
                        config.mdl_model if kind == "mdl" else config.discrete_model
                    )
                    if discrete_config.temperature_schedule == "per_task":
                        progress = n_seen / max(1, config.world.examples_per_task - 1)
                    else:
                        progress = global_example / max(1, total_examples - 1)
                    model.set_training_progress(progress)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_tasks(_tensor(np.stack(batch_x)), task_ids)
                loss = torch.nn.functional.mse_loss(prediction, _tensor(np.stack(batch_y)))
                if isinstance(model, VariationalSharedResidualLearner):
                    # Description length in the wake gradient (V3 spec 3.1):
                    # KL(q || p), scaled by beta. The L1 surrogate is not
                    # applied — storage_penalty is identically zero here.
                    #
                    # Charged as the MEAN over the unique tasks present, which
                    # matches the data term's own weighting: MSE is a mean over
                    # batch elements, so a task holding 1 of 2 examples carries
                    # data weight 1/2 and KL weight 1/2. The KL-to-likelihood
                    # PRESSURE RATIO is therefore exactly 1.000 for every task
                    # regardless of arrival order (audited in
                    # row.experiments.audit_kl_charge), which is the quantity
                    # that fixes the implied rate-distortion tradeoff.
                    #
                    # The raw integrated KL coefficient is tilted by replay
                    # (0.50x-3.38x, early tasks highest), but that is a
                    # difference in how many optimization STEPS a task's code
                    # receives toward the same objective, not in the price it
                    # pays. Charging only the current task instead would make
                    # the pressure ratio position-dependent (2.00 for the last
                    # tasks against 0.55 for the first) and would let replayed
                    # tasks accumulate retained information with no code charge
                    # at all -- coherent only if task state froze after
                    # acquisition, which is a protocol change, not a bug fix.
                    loss = loss + (
                        _variational_kl_scale(config, config.world.examples_per_task)
                        * model.description_penalty(task_ids)
                    )
                elif isinstance(model, SharedParentResidualLearner):
                    loss = loss + (
                        config.shared_residual_model.residual_penalty
                        * model.storage_penalty(task_ids)
                    )
                if isinstance(model, PresenceGatedDiscreteLibraryLearner):
                    loss = loss + (
                        config.mdl_model.library_presence_penalty
                        * model.presence_penalty()
                        + config.mdl_model.route_entropy_penalty
                        * model.route_entropy_penalty(task_ids)
                    )
                loss.backward()
                optimizer.step()

        summary_row: dict[str, object] = {
            "record_type": "task_summary",
            "task_index": lifetime_index,
            "world_task_index": world_task_index,
            "task_id": task.task_id,
            "zero_shot_nmse": curve[0],
            "final_nmse": curve[max(curve)],
        }
        for threshold in config.evaluation.nmse_thresholds:
            summary_row[f"examples_to_{threshold:g}"] = examples_to_criterion(
                curve, threshold, config.world.examples_per_task
            )
        rows.append(summary_row)
        # H39 pilot: read-only snapshot of the task's residual as it stands
        # when the task completes, BEFORE any sleep can retire it.
        if snapshot_history and isinstance(model, SharedParentResidualLearner):
            with torch.no_grad():
                history_residuals[task.task_id] = (
                    model.effective_residual(task.task_id).detach().clone()
                    if hasattr(model, "effective_residual")
                    else model.task_residuals[task.task_id].detach().clone()
                )
                history_codes[task.task_id] = model.task_codes[task.task_id].detach().clone()
                if isinstance(model, ARGUMENT_LEARNERS):
                    history_eps[task.task_id] = model.task_residuals[task.task_id].detach().clone()
        # V6: prospective pressure. Called after the task is learned and
        # before any sleep, so it shapes the representation the next task
        # inherits rather than second-guessing a promotion.
        if prospective_hook is not None:
            record = prospective_hook(model, lifetime_index, world_task_index)
            if record:
                rows.append({"record_type": "prospective", **record})
        if isinstance(model, PromotingSharedResidualLearner) and (
            lifetime_index + 1
        ) in sleeps:
            # Disjoint proposal and validation probes (V3 spec 3.2): a
            # promotion may not be validated on the evidence that proposed
            # it. Both are deterministic in the world seed.
            proposal_rng = np.random.default_rng(
                np.random.SeedSequence([config.world.seed, 71])
            )
            validation_rng = np.random.default_rng(
                np.random.SeedSequence([config.world.seed, 72])
            )
            record = model.sleep(
                [world.tasks[i].task_id for i in task_indices[: lifetime_index + 1]],
                _tensor(proposal_rng.normal(size=(256, config.world.state_dim))),
                _tensor(validation_rng.normal(size=(256, config.world.state_dim))),
                epsilon=promotion_epsilon,
                lifetime_index=lifetime_index + 1,
            )
            sleep_records.append(record)
            if isinstance(model, LifecycleLibraryLearner):
                model.sync_lineage(lifetime_index + 1)
                if (
                    force_retire_at is not None
                    and lifetime_index + 1 >= force_retire_at
                    and not model.retired_abstractions()
                ):
                    record["forced_retirement"] = model.force_retire_all(
                        lifetime_index + 1, only_largest=force_retire_one
                    )
                if lifecycle_filter:
                    filter_rng = np.random.default_rng(
                        np.random.SeedSequence([config.world.seed, 91])
                    )
                    record["value_filter"] = model.prune_by_value(
                        _tensor(
                            filter_rng.normal(size=(256, config.world.state_dim))
                        ),
                        task_index=lifetime_index + 1,
                        tasks_total=len(world.tasks),
                        kappa=lifecycle_kappa,
                    )
                if lifecycle_enabled:
                    consolidation_rng = np.random.default_rng(
                        np.random.SeedSequence([config.world.seed, 73])
                    )
                    record["lifecycle"] = model.consolidate(
                        _tensor(
                            consolidation_rng.normal(
                                size=(256, config.world.state_dim)
                            )
                        ),
                        task_index=lifetime_index + 1,
                        epsilon=promotion_epsilon,
                        kappa=lifecycle_kappa,
                        tasks_total=len(world.tasks),
                        grace=lifecycle_grace,
                    )
        if isinstance(model, VariationalSharedResidualLearner):
            completed_task_ids.append(task.task_id)
            model.update_prior_scales(completed_task_ids)
        replay.add_task(task, replay_per_task)
        tasks_completed = lifetime_index + 1
        if tasks_completed in config.evaluation.lifetime_checkpoints:
            checkpoint = _novel_checkpoint(
                model,
                world,
                config,
                task_lr,
                tasks_completed,
                config.evaluation.checkpoint_novel_tasks,
            )
            if config.world.reuse_rho == 1.0 and isinstance(
                model, (ContinuousBasisLearner, DiscreteLibraryLearner)
            ):
                checkpoint["true_route_operator_quality"] = (
                    _true_route_operator_quality(
                        model, world, config, tasks_completed
                    )
                )
            checkpoint_results.append(checkpoint)

    summary = summarize(rows, config.world.examples_per_task)
    _add_lifetime_transfer_summary(summary, rows)
    summary.update(
        {
            "model": kind,
            "order": order,
            "cumulative_prequential_nll": cumulative_nll,
            "cumulative_prequential_gaussian_log_loss": cumulative_nll,
            "cumulative_prequential_quantized_target_log_loss": cumulative_mass_log_loss,
            "target_precision": config.evaluation.target_precision,
            "prequential_gaussian_log_loss_per_online_example": (
                cumulative_nll
                / (len(world.tasks) * config.world.examples_per_task)
            ),
            "prequential_gaussian_log_loss_per_target_scalar": (
                cumulative_nll
                / (
                    len(world.tasks)
                    * config.world.examples_per_task
                    * config.world.state_dim
                )
            ),
            "prequential_quantized_target_bits_per_online_example": (
                cumulative_mass_log_loss
                / (len(world.tasks) * config.world.examples_per_task)
                / np.log(2.0)
            ),
            "compute_accounting": _compute_accounting(config, kind),
            "shared_parameter_count": model.shared_parameter_count,
            "task_state_scalar_count": model.task_state_scalar_count,
            "world_functional_reuse": world.functional_reuse_diagnostics(),
        }
    )
    if isinstance(model, PromotingSharedResidualLearner):
        if isinstance(model, LifecycleLibraryLearner):
            model.sync_lineage(len(world.tasks))
            summary["lifecycle"] = model.lifecycle_diagnostics()
        # PROVENANCE: `task_reference` and `retired` are plain Python
        # containers and are NOT in `state_dict`, so an artifact that
        # omits them cannot say afterwards which task depended on which
        # abstraction. Every analysis of the promoted-versus-private
        # split then silently reads an empty library. This cost a voided
        # coding-frontier audit and a full 30-world re-run (PROGRESS.md,
        # 2026-08-19). Recorded for EVERY promoting model, not only the
        # lifecycle subclass; it is a summary field, so no frozen model
        # behaviour and no resolved-config fingerprint changes.
        summary["reference_table"] = {
            "task_reference": {k: int(v) for k, v in model.task_reference.items()},
            "retired_task_ids": sorted(model.retired),
        }
        summary["promotion"] = model.promotion_diagnostics()
        summary["sleeps"] = sleep_records
    if isinstance(model, ContinuousBasisLearner):
        summary["routing"] = model.routing_diagnostics()
        if config.evaluation.extended_diagnostics:
            summary["functional_recovery"] = _functional_recovery(model.basis, world, config)
    elif isinstance(model, SharedParentResidualLearner):
        if isinstance(model, GatedInnovationLearner):
            summary["gates"] = model.gate_diagnostics()
            summary["structural_task_bits"] = model.structural_task_bits()
        if isinstance(model, VariationalSharedResidualLearner):
            summary["variational"] = _variational_summary(model, world, config)
        summary["routing"] = model.routing_diagnostics()
        if config.evaluation.extended_diagnostics:
            summary["functional_recovery"] = _functional_recovery(
                model.basis, world, config
            )
        diagnostic_generator = np.random.default_rng(
            np.random.SeedSequence([config.world.seed, 98])
        )
        summary["residual_diagnostics"] = model.residual_diagnostics(
            _tensor(
                diagnostic_generator.normal(
                    size=(2048, config.world.state_dim)
                )
            )
        )
    elif isinstance(model, DiscreteLibraryLearner):
        summary["routing"] = model.routing_diagnostics()
        if isinstance(model, PresenceGatedDiscreteLibraryLearner):
            summary["presence"] = model.presence_diagnostics()
        if config.evaluation.extended_diagnostics:
            summary["functional_recovery"] = _functional_recovery(model.library, world, config)
            summary["route_recovery"] = _route_recovery(
                model, world, summary["functional_recovery"]
            )
            summary["operator_specialization"] = _operator_specialization(model, config)
    summary["novel_composition"] = _adapt_novel_composition(
        model, world, config, task_lr
    )
    summary["novel_composition_checkpoints"] = checkpoint_results
    if task_id_scramble_seed is not None:
        summary["task_id_scramble_seed"] = task_id_scramble_seed
    if update_batch_size is not None:
        summary["update_batch"] = {
            "target_size": update_batch_size,
            "mean_observed_size": float(np.mean(observed_update_batch_sizes)),
            "minimum_observed_size": min(observed_update_batch_sizes),
            "maximum_observed_size": max(observed_update_batch_sizes),
            "current_to_replay_ratio": f"1:{replay_ratio:g}",
            "sampling_seed_policy": "world-paired",
        }
    _write_artifacts(
        config,
        world,
        model,
        rows,
        summary,
        kind,
        order,
        task_id_scramble_seed,
        update_batch_size,
        history=(
            {
                "residuals": history_residuals,
                "codes": history_codes,
                "eps": history_eps,
                "order": list(history_residuals),
            }
            if snapshot_history else None
        ),
    )
    return summary


def _gated_code_scale(config: ExperimentConfig, examples: int) -> float:
    """Same MDL conversion as the variational learner, with the gated beta."""

    sigma = config.evaluation.gaussian_sigma
    return (
        config.gated_model.description_beta
        * 2.0
        * sigma
        * sigma
        / (examples * config.world.state_dim)
    )


def _variational_kl_scale(config: ExperimentConfig, examples: int) -> float:
    """KL coefficient that makes `description_beta` a dimensionless MDL dial.

    The wake objective is the description length L_preq + beta * KL, but the
    optimizer minimizes MSE (a mean over batch and dimensions) while KL is a
    sum of nats over a task's whole code. Converting between them:

        NLL_total(task) = N * d / (2 * sigma^2) * MSE

    so dividing the per-task objective by that factor puts the KL charge in
    MSE units as beta * 2 * sigma^2 / (N * d). beta = 1 is then the literal
    MDL point rather than an arbitrary scale, and the tuning grid is a
    genuine rate-distortion dial around it. Charging raw KL against MSE
    instead (the first implementation) is ~4 orders of magnitude too strong
    and collapses every posterior onto the prior.
    """

    sigma = config.evaluation.gaussian_sigma
    return (
        config.variational_model.description_beta
        * 2.0
        * sigma
        * sigma
        / (examples * config.world.state_dim)
    )


def _variational_summary(
    model: VariationalSharedResidualLearner,
    world: World,
    config: ExperimentConfig,
) -> dict[str, object]:
    """Variational-currency report plus the sparse two-part pruning probe.

    Three quantities the V3 spec keeps strictly separate (section 4.2):
    L_mean (behavior, from the posterior mean), the variational code
    E_q[L] + KL estimated with a fixed-seed MC scheme, and the literal
    two-part code. The pruning probe is what lets literal bits fall at all:
    a coordinate whose posterior has collapsed onto the prior carries no
    task information, so a sparse code drops it for one bitmap bit.
    """

    selected = config.variational_model
    diagnostics = dict(model.variational_diagnostics(selected.prune_threshold_bits))
    tasks = [task for task in world.tasks if task.task_id in model.task_codes]
    log2 = float(np.log(2.0))

    model.eval()
    mean_nll = 0.0
    with torch.no_grad():
        for task in tasks:
            prediction = model(_tensor(task.eval_x), task.task_id).cpu().numpy()
            mean_nll += gaussian_nll(
                prediction, task.eval_y, config.evaluation.gaussian_sigma
            )

    # E_q[L]: fixed-seed Monte Carlo over the posterior, deliberately NOT
    # the L_mean + KL hybrid (which is not a codelength).
    sample_generator = torch.Generator()
    sample_generator.manual_seed(selected.seed + 7)
    expected_nll = 0.0
    with torch.no_grad():
        for _ in range(selected.variational_samples):
            for task in tasks:
                route_mu = model.task_codes[task.task_id]
                residual_mu = model.task_residuals[task.task_id]
                route_sigma = torch.exp(model.task_code_log_sigma[task.task_id])
                residual_sigma = torch.exp(
                    model.task_residual_log_sigma[task.task_id]
                )
                route_backup = route_mu.detach().clone()
                residual_backup = residual_mu.detach().clone()
                route_mu.add_(
                    route_sigma
                    * torch.randn(
                        route_mu.shape, generator=sample_generator
                    )
                )
                residual_mu.add_(
                    residual_sigma
                    * torch.randn(
                        residual_mu.shape, generator=sample_generator
                    )
                )
                prediction = model(_tensor(task.eval_x), task.task_id).cpu().numpy()
                expected_nll += gaussian_nll(
                    prediction, task.eval_y, config.evaluation.gaussian_sigma
                )
                route_mu.copy_(route_backup)
                residual_mu.copy_(residual_backup)
    expected_nll /= selected.variational_samples

    kl_bits = float(diagnostics.get("total_task_kl_bits", 0.0))
    diagnostics.update(
        {
            "evaluation_loss_posterior_mean": mean_nll,
            "evaluation_expected_loss_under_q": expected_nll,
            "variational_code_nats": expected_nll + kl_bits * log2,
            "variational_task_bits": kl_bits,
            "note": (
                "variational code is E_q[L] + KL on held-out evaluation sets; "
                "L_mean + KL is never reported as a codelength"
            ),
        }
    )

    # Sparse two-part probe: prune uninformative coordinates on a deep copy
    # and measure the behavioral cost of doing so.
    pruned = copy.deepcopy(model)
    prune = pruned.apply_information_prune(selected.prune_threshold_bits)
    pruned.eval()
    pruned_nmse: list[float] = []
    base_nmse: list[float] = []
    with torch.no_grad():
        for task in tasks:
            base_nmse.append(
                nmse(model(_tensor(task.eval_x), task.task_id).cpu().numpy(), task.eval_y)
            )
            pruned_nmse.append(
                nmse(
                    pruned(_tensor(task.eval_x), task.task_id).cpu().numpy(),
                    task.eval_y,
                )
            )
    retained = int(prune["retained_task_scalars"])
    total = int(prune["total_task_scalars"])
    diagnostics["sparse_two_part"] = {
        **prune,
        # 8 bits per retained scalar plus a one-bit-per-coordinate presence
        # bitmap; the dense code charges 8 bits for every coordinate.
        "sparse_task_bits": 8 * retained + total,
        "dense_task_bits": 8 * total,
        "mean_final_nmse": float(np.mean(base_nmse)) if base_nmse else 0.0,
        "mean_pruned_final_nmse": float(np.mean(pruned_nmse)) if pruned_nmse else 0.0,
        "mean_nmse_increase": (
            float(np.mean(pruned_nmse) - np.mean(base_nmse)) if base_nmse else 0.0
        ),
        "maximum_task_nmse_increase": (
            float(np.max(np.array(pruned_nmse) - np.array(base_nmse)))
            if base_nmse
            else 0.0
        ),
    }
    return diagnostics


def _edit_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


@torch.no_grad()
def _route_recovery(
    model: DiscreteLibraryLearner,
    world: World,
    functional_recovery: object,
) -> dict[str, float]:
    assert isinstance(functional_recovery, dict)
    explanations = functional_recovery["best_depth_1_to_3_explanations"]
    assert isinstance(explanations, list)
    slot_to_teacher = {
        int(item["learned_operator"]): tuple(int(x) for x in item["teacher_route"])
        for item in explanations
    }
    routes = model.hard_routes()
    exact = 0
    position_matches = 0
    position_total = 0
    edit_distances = []
    behavioral_scores = []
    model.eval()
    for task in world.tasks:
        learned_slots = routes[task.task_id]
        expanded = tuple(
            primitive
            for slot in learned_slots
            for primitive in slot_to_teacher[int(slot)]
        )
        teacher = task.program.primitive_ids
        exact += expanded == teacher
        edit_distances.append(_edit_distance(expanded, teacher))
        for position, slot in enumerate(learned_slots):
            explanation = slot_to_teacher[int(slot)]
            if len(explanation) == 1:
                position_total += 1
                position_matches += explanation[0] == teacher[position]
        prediction = model(
            torch.as_tensor(task.eval_x, dtype=torch.float32), task.task_id
        ).cpu().numpy()
        behavioral_scores.append(nmse(prediction, task.eval_y))
    return {
        "exact_explained_route_fraction": exact / len(world.tasks),
        "per_position_match_fraction": (
            position_matches / position_total if position_total else 0.0
        ),
        "mean_explained_route_edit_distance": float(np.mean(edit_distances)),
        "final_lifetime_behavioral_nmse_mean": float(np.mean(behavioral_scores)),
    }


@torch.no_grad()
def _operator_specialization(
    model: DiscreteLibraryLearner, config: ExperimentConfig
) -> dict[str, object]:
    generator = np.random.default_rng(np.random.SeedSequence([config.world.seed, 94]))
    probe = torch.as_tensor(
        generator.normal(size=(2048, config.world.state_dim)), dtype=torch.float32
    )
    outputs = [operator(probe).cpu().numpy() for operator in model.library]
    distances = np.zeros((len(outputs), len(outputs)), dtype=np.float64)
    close_001: list[list[int]] = []
    close_01: list[list[int]] = []
    nonzero = []
    for first in range(len(outputs)):
        for second in range(first + 1, len(outputs)):
            denominator = 0.5 * (float(np.var(outputs[first])) + float(np.var(outputs[second])))
            distance = float(np.mean(np.square(outputs[first] - outputs[second])) / denominator)
            distances[first, second] = distances[second, first] = distance
            nonzero.append(distance)
            if distance < 0.001:
                close_001.append([first, second])
            if distance < 0.01:
                close_01.append([first, second])
    routing = model.routing_diagnostics()
    usage = np.asarray(routing["usage_counts"], dtype=np.float64)
    return {
        "pairwise_functional_distance": distances.tolist(),
        "minimum_pairwise_distance": float(np.min(nonzero)),
        "mean_pairwise_distance": float(np.mean(nonzero)),
        "duplicate_pairs_below_0.001": close_001,
        "near_duplicate_pairs_below_0.01": close_01,
        "top_operator_usage_fraction": float(np.max(usage) / np.sum(usage)),
        "active_operator_fraction": float(np.count_nonzero(usage) / len(usage)),
    }


def _novel_data(
    world: World, config: ExperimentConfig, novel_index: int = 0
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    used = {task.program.primitive_ids for task in world.tasks}
    candidates = [
        tuple(route)
        for route in product(
            range(config.world.teacher_primitives), repeat=config.world.program_length
        )
        if tuple(route) not in used
    ]
    route_generator = np.random.default_rng(np.random.SeedSequence([config.world.seed, 91]))
    route_order = route_generator.permutation(len(candidates))
    route = candidates[int(route_order[novel_index % len(candidates)])]
    generator = np.random.default_rng(
        np.random.SeedSequence([config.world.seed, 93, novel_index])
    )
    train_x = generator.normal(size=(32, config.world.state_dim))
    eval_x = generator.normal(size=(config.world.evaluation_examples, config.world.state_dim))
    program = Program(route)
    novel_library = world.library_for_task(config.world.tasks + novel_index)
    return (
        route,
        train_x,
        program.execute(novel_library, train_x),
        eval_x,
        program.execute(novel_library, eval_x),
    )


def _adapt_novel_composition(
    model: Learner,
    world: World,
    config: ExperimentConfig,
    task_lr: float,
    novel_index: int = 0,
) -> dict[str, object]:
    route, train_x, train_y, eval_x, eval_y = _novel_data(world, config, novel_index)
    for parameter in model.shared_parameters():
        parameter.requires_grad_(False)
    novel_id = f"task_novel_composition_{novel_index}"
    novel_code = (
        model.begin_task(novel_id, schema_index=getattr(model, "schema_count", 1) - 1)
        if isinstance(model, ARGUMENT_LEARNERS)
        else model.begin_task(novel_id)
    )
    if isinstance(model, GatedInnovationLearner):
        route_parameter, residual_parameter, gate_parameter = novel_code
        gated_config = config.gated_model
        optimizer = torch.optim.Adam(
            [
                {"params": [route_parameter], "lr": task_lr},
                {"params": [residual_parameter], "lr": gated_config.residual_learning_rate},
                {"params": [gate_parameter], "lr": gated_config.gate_learning_rate},
            ]
        )
    elif isinstance(model, VariationalSharedResidualLearner):
        (
            route_parameter,
            route_scale_parameter,
            residual_parameter,
            residual_scale_parameter,
        ) = novel_code
        variational_config = config.variational_model
        optimizer = torch.optim.Adam(
            [
                {"params": [route_parameter], "lr": task_lr},
                {
                    "params": [route_scale_parameter],
                    "lr": variational_config.scale_learning_rate,
                },
                {
                    "params": [residual_parameter],
                    "lr": variational_config.residual_learning_rate,
                },
                {
                    "params": [residual_scale_parameter],
                    "lr": variational_config.scale_learning_rate,
                },
            ]
        )
    elif isinstance(model, ARGUMENT_LEARNERS):
        route_parameter, residual_parameter, alpha_parameter = novel_code
        optimizer = torch.optim.Adam(
            [
                {"params": [route_parameter], "lr": task_lr},
                {
                    "params": [residual_parameter],
                    "lr": config.shared_residual_model.residual_learning_rate,
                },
                {"params": [alpha_parameter], "lr": task_lr},
            ]
        )
    elif isinstance(model, SharedParentResidualLearner):
        route_parameter, residual_parameter = novel_code
        optimizer = torch.optim.Adam(
            [
                {"params": [route_parameter], "lr": task_lr},
                {
                    "params": [residual_parameter],
                    "lr": config.shared_residual_model.residual_learning_rate,
                },
            ]
        )
    else:
        optimizer = torch.optim.Adam([novel_code], lr=task_lr)
    curve: dict[str, float] = {}
    supports = {0, 1, 2, 4, 8, 16, 32}
    for n_seen in range(33):
        if n_seen in supports:
            model.eval()
            with torch.no_grad():
                prediction = model(_tensor(eval_x), novel_id).cpu().numpy()
            curve[str(n_seen)] = nmse(prediction, eval_y)
        if n_seen == 32:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(_tensor(train_x[n_seen : n_seen + 1]), novel_id)
        loss = torch.nn.functional.mse_loss(prediction, _tensor(train_y[n_seen : n_seen + 1]))
        if isinstance(model, GatedInnovationLearner):
            loss = loss + (
                _gated_code_scale(config, 32) * model.description_penalty([novel_id])
            )
        elif isinstance(model, VariationalSharedResidualLearner):
            loss = loss + (
                # 32 is the novel-adaptation example budget below.
                _variational_kl_scale(config, 32)
                * model.description_penalty([novel_id])
            )
        elif isinstance(model, SharedParentResidualLearner):
            loss = loss + (
                config.shared_residual_model.residual_penalty
                * model.storage_penalty([novel_id])
            )
        loss.backward()
        optimizer.step()
    return {"teacher_route": list(route), "nmse_by_support": curve}


def _novel_checkpoint(
    model: Learner,
    world: World,
    config: ExperimentConfig,
    task_lr: float,
    tasks_completed: int,
    novel_tasks: int,
) -> dict[str, object]:
    checkpoint_model = copy.deepcopy(model)
    individuals = [
        _adapt_novel_composition(
            checkpoint_model,
            world,
            config,
            task_lr,
            novel_index=index,
        )
        for index in range(novel_tasks)
    ]
    support_points = individuals[0]["nmse_by_support"].keys()
    mean_curve = {
        support: float(
            np.mean(
                [float(result["nmse_by_support"][support]) for result in individuals]
            )
        )
        for support in support_points
    }
    return {
        "tasks_completed": tasks_completed,
        "novel_tasks": novel_tasks,
        "mean_nmse_by_support": mean_curve,
        "individuals": individuals,
    }


@torch.no_grad()
def _true_route_operator_quality(
    model: ContinuousBasisLearner | DiscreteLibraryLearner,
    world: World,
    config: ExperimentConfig,
    tasks_completed: int,
) -> dict[str, object]:
    if config.world.reuse_rho != 1.0:
        raise ValueError("true-route operator checkpoints currently require exact reuse")
    operators = model.basis if isinstance(model, ContinuousBasisLearner) else model.library
    generator = np.random.default_rng(
        np.random.SeedSequence([config.world.seed, 95, tasks_completed])
    )
    probe_x = generator.normal(size=(2048, config.world.state_dim))
    learned_outputs = [operator(_tensor(probe_x)).cpu().numpy() for operator in operators]
    teacher_outputs = [primitive(probe_x) for primitive in world.library]
    distances = np.empty(
        (len(teacher_outputs), len(learned_outputs)), dtype=np.float64
    )
    for teacher_index, teacher_output in enumerate(teacher_outputs):
        denominator = float(np.var(teacher_output))
        for learned_index, learned_output in enumerate(learned_outputs):
            distances[teacher_index, learned_index] = float(
                np.mean(np.square(teacher_output - learned_output)) / denominator
            )
    teacher_indices, learned_indices = linear_sum_assignment(distances)
    teacher_to_learned = {
        int(teacher): int(learned)
        for teacher, learned in zip(teacher_indices, learned_indices, strict=True)
    }

    def true_route_prediction(task: Task) -> np.ndarray:
        state = _tensor(task.eval_x)
        for teacher_operator in task.program.primitive_ids:
            state = operators[teacher_to_learned[int(teacher_operator)]](state)
        return state.cpu().numpy()

    was_training = model.training
    model.eval()
    try:
        true_route_scores = [
            nmse(true_route_prediction(task), task.eval_y) for task in world.tasks
        ]
        learned_route_scores = [
            nmse(model(_tensor(task.eval_x), task.task_id).cpu().numpy(), task.eval_y)
            for task in world.tasks[:tasks_completed]
        ]
    finally:
        model.train(was_training)
    future_scores = true_route_scores[tasks_completed:]
    return {
        "tasks_completed": tasks_completed,
        "probe_examples": len(probe_x),
        "teacher_to_learned_operator": teacher_to_learned,
        "one_to_one_mean_primitive_distance": float(
            np.mean(distances[teacher_indices, learned_indices])
        ),
        "true_route_all_programs_nmse_mean": float(np.mean(true_route_scores)),
        "true_route_completed_programs_nmse_mean": float(
            np.mean(true_route_scores[:tasks_completed])
        ),
        "true_route_future_programs_nmse_mean": (
            float(np.mean(future_scores)) if future_scores else None
        ),
        "learned_route_completed_programs_nmse_mean": float(
            np.mean(learned_route_scores)
        ),
        "learned_minus_true_route_completed_nmse": float(
            np.mean(learned_route_scores)
            - np.mean(true_route_scores[:tasks_completed])
        ),
    }


def resolved_learned_config(
    config: ExperimentConfig,
    kind: ModelKind,
    order: str,
    task_id_scramble_seed: int | None = None,
    update_batch_size: int | None = None,
) -> dict[str, object]:
    model_config = {
        "dense": config.dense_model,
        "continuous": config.continuous_model,
        "hypernetwork": config.hypernetwork_model,
        "shared_residual": config.shared_residual_model,
        "variational": config.variational_model,
        "gated": config.gated_model,
        "promoting": config.shared_residual_model,
        "lifecycle": config.shared_residual_model,
        "prospective": config.shared_residual_model,
        "factorized": config.shared_residual_model,
        "pslot": config.shared_residual_model,
        "discrete": config.discrete_model,
        "mdl": config.mdl_model,
    }[kind]
    resolved: dict[str, object] = {
        "world": asdict(config.world),
        f"{kind}_model": asdict(model_config),
        "evaluation": asdict(config.evaluation),
        "order": order,
        "output": {"directory": str(config.output_directory)},
    }
    if task_id_scramble_seed is not None:
        resolved["task_id_scramble_seed"] = task_id_scramble_seed
    if update_batch_size is not None:
        resolved["update_batch_size"] = update_batch_size
    return resolved


def _write_artifacts(
    config: ExperimentConfig,
    world: World,
    model: Learner,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    kind: ModelKind,
    order: str,
    task_id_scramble_seed: int | None,
    update_batch_size: int | None,
    history: dict[str, object] | None = None,
) -> None:
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    model_config = {
        "dense": config.dense_model,
        "continuous": config.continuous_model,
        "hypernetwork": config.hypernetwork_model,
        "shared_residual": config.shared_residual_model,
        "variational": config.variational_model,
        "gated": config.gated_model,
        "promoting": config.shared_residual_model,
        "lifecycle": config.shared_residual_model,
        "prospective": config.shared_residual_model,
        "factorized": config.shared_residual_model,
        "pslot": config.shared_residual_model,
        "discrete": config.discrete_model,
        "mdl": config.mdl_model,
    }[kind]
    resolved = resolved_learned_config(
        config, kind, order, task_id_scramble_seed, update_batch_size
    )
    (output / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "world_programs.json").write_text(
        json.dumps(world.programs_json(), indent=2), encoding="utf-8"
    )
    (output / "world_functional_reuse.json").write_text(
        json.dumps(summary["world_functional_reuse"], indent=2), encoding="utf-8"
    )
    (output / "world_seed.txt").write_text(f"{config.world.seed}\n", encoding="utf-8")
    (output / "model_seed.txt").write_text(f"{model_config.seed}\n", encoding="utf-8")
    # Summary data is already stored as JSON. Keeping model.pt tensor-only lets
    # downstream tools use PyTorch's restricted weights-only loader.
    torch.save({"model_state_dict": model.state_dict()}, output / "model.pt")
    if hasattr(model, "save_extras"):
        model.save_extras(output)
    if history is not None:
        torch.save(history, output / "history.pt")
    if isinstance(model, DiscreteLibraryLearner):
        (output / "hard_routes.json").write_text(
            json.dumps(model.hard_routes(), indent=2), encoding="utf-8"
        )
    # H29's P_0: the member residuals a promotion consumed, as they
    # stood at that sleep. Written to its own file rather than into
    # model.pt, which stays tensor-only for the restricted loader. An
    # in-memory snapshot that never reaches disk is not provenance --
    # the whole point is that a finished artifact can be re-audited.
    snapshots = getattr(model, "promotion_snapshots", None)
    if snapshots:
        payload = {}
        for index, record in snapshots.items():
            payload[f"born_{index}"] = record["born"].cpu().numpy()
            if record["member_residuals"]:
                payload[f"members_{index}"] = torch.stack(
                    record["member_residuals"]
                ).cpu().numpy()
        np.savez_compressed(output / "promotion_snapshots.npz", **payload)
        (output / "promotion_members.json").write_text(
            json.dumps(
                {str(k): list(v["members"]) for k, v in snapshots.items()}, indent=2
            ),
            encoding="utf-8",
        )
    commit = current_git_commit()
    (output / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    write_fingerprint(output, resolved, kind, commit)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    (output / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--model",
        choices=(
            "dense",
            "continuous",
            "hypernetwork",
            "shared_residual",
            "variational",
            "gated",
            "promoting",
            "lifecycle",
            "discrete",
            "mdl",
        ),
        required=True,
    )
    parser.add_argument("--world-seed", type=int)
    parser.add_argument("--reuse-rho", type=float)
    parser.add_argument("--teacher-rank", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--task-id-scramble-seed", type=int)
    parser.add_argument("--update-batch-size", type=int)
    parser.add_argument("--global-learning-rate", type=float)
    parser.add_argument("--task-learning-rate", type=float)
    parser.add_argument("--updates-per-example", type=int)
    parser.add_argument("--hidden-width", type=int)
    parser.add_argument("--task-embedding-dim", type=int)
    parser.add_argument("--step-code-dim", type=int)
    parser.add_argument("--hypernetwork-hidden-dim", type=int)
    parser.add_argument("--operator-activation", choices=("tanh", "gelu"))
    parser.add_argument("--operator-alpha-init", type=float)
    parser.add_argument("--residual-penalty", type=float)
    parser.add_argument("--residual-learning-rate", type=float)
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument("--temperature-schedule", choices=("global", "per_task"))
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fast-tuning", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config = replace(
        config,
        world=replace(
            config.world,
            seed=config.world.seed if args.world_seed is None else args.world_seed,
            reuse_rho=config.world.reuse_rho if args.reuse_rho is None else args.reuse_rho,
            teacher_rank=(
                config.world.teacher_rank if args.teacher_rank is None else args.teacher_rank
            ),
        ),
        output_directory=config.output_directory if args.output is None else args.output,
    )
    if args.fast_tuning:
        config = replace(
            config,
            evaluation=replace(
                config.evaluation,
                lifetime_checkpoints=(),
                checkpoint_novel_tasks=1,
                extended_diagnostics=False,
            ),
        )
    selected = {
        "dense": config.dense_model,
        "continuous": config.continuous_model,
        "hypernetwork": config.hypernetwork_model,
        "shared_residual": config.shared_residual_model,
        "variational": config.variational_model,
        "gated": config.gated_model,
        "promoting": config.shared_residual_model,
        "lifecycle": config.shared_residual_model,
        "prospective": config.shared_residual_model,
        "factorized": config.shared_residual_model,
        "pslot": config.shared_residual_model,
        "discrete": config.discrete_model,
        "mdl": config.mdl_model,
    }[args.model]
    selected = replace(
        selected,
        seed=selected.seed if args.model_seed is None else args.model_seed,
        global_learning_rate=(
            selected.global_learning_rate
            if args.global_learning_rate is None
            else args.global_learning_rate
        ),
        task_learning_rate=(
            selected.task_learning_rate
            if args.task_learning_rate is None
            else args.task_learning_rate
        ),
        updates_per_example=(
            selected.updates_per_example
            if args.updates_per_example is None
            else args.updates_per_example
        ),
        **(
            {"hidden_width": args.hidden_width}
            if args.model == "dense" and args.hidden_width is not None
            else {}
        ),
        **(
            {"task_embedding_dim": args.task_embedding_dim}
            if args.model == "dense" and args.task_embedding_dim is not None
            else {}
        ),
        **(
            {"step_code_dim": args.step_code_dim}
            if args.model == "hypernetwork" and args.step_code_dim is not None
            else {}
        ),
        **(
            {"hypernetwork_hidden_dim": args.hypernetwork_hidden_dim}
            if args.model == "hypernetwork" and args.hypernetwork_hidden_dim is not None
            else {}
        ),
        **(
            {"operator_activation": args.operator_activation}
            if args.model in {
                "continuous",
                "hypernetwork",
                "shared_residual",
                "variational",
                "gated",
                "discrete",
                "mdl",
            }
            and args.operator_activation is not None
            else {}
        ),
        **(
            {"operator_alpha_init": args.operator_alpha_init}
            if args.model in {
                "continuous",
                "hypernetwork",
                "shared_residual",
                "variational",
                "discrete",
                "mdl",
            }
            and args.operator_alpha_init is not None
            else {}
        ),
        **(
            {"residual_penalty": args.residual_penalty}
            if args.model == "shared_residual" and args.residual_penalty is not None
            else {}
        ),
        **(
            {"residual_learning_rate": args.residual_learning_rate}
            if args.model == "shared_residual"
            and args.residual_learning_rate is not None
            else {}
        ),
        **(
            {"include_identity": True}
            if args.model == "continuous" and args.include_identity
            else {}
        ),
        **(
            {"temperature_schedule": args.temperature_schedule}
            if args.model in {"discrete", "mdl"}
            and args.temperature_schedule is not None
            else {}
        ),
    )
    config = replace(
        config,
        dense_model=selected if args.model == "dense" else config.dense_model,
        continuous_model=(
            selected if args.model == "continuous" else config.continuous_model
        ),
        hypernetwork_model=(
            selected if args.model == "hypernetwork" else config.hypernetwork_model
        ),
        shared_residual_model=(
            selected
            if args.model == "shared_residual"
            else config.shared_residual_model
        ),
        variational_model=(
            selected if args.model == "variational" else config.variational_model
        ),
        gated_model=selected if args.model == "gated" else config.gated_model,
        discrete_model=selected if args.model == "discrete" else config.discrete_model,
        mdl_model=selected if args.model == "mdl" else config.mdl_model,
    )
    summary = run(
        config,
        kind=args.model,
        order=args.order,
        task_id_scramble_seed=args.task_id_scramble_seed,
        update_batch_size=args.update_batch_size,
    )
    final = summary["final_nmse"]
    assert isinstance(final, dict)
    novel = summary["novel_composition"]
    assert isinstance(novel, dict)
    novel_curve = novel["nmse_by_support"]
    assert isinstance(novel_curve, dict)
    print(
        f"{args.model} {args.order}: final median NMSE={final['median']:.4f}; "
        f"prequential Gaussian log loss={summary['cumulative_prequential_gaussian_log_loss']:.1f}; "
        f"novel NMSE 0/32={novel_curve['0']:.4f}/{novel_curve['32']:.4f}"
    )


if __name__ == "__main__":
    main()
