"""Configuration loading for ROW experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from row.world import WorldConfig


@dataclass(frozen=True)
class ScratchModelConfig:
    hidden_dim: int = 64
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    updates_per_example: int = 1
    seed: int = 1000

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.batch_size <= 0 or self.updates_per_example <= 0:
            raise ValueError("scratch dimensions, batch size, and update count must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")


@dataclass(frozen=True)
class EvaluationConfig:
    support_points: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128)
    nmse_thresholds: tuple[float, ...] = (0.1, 0.05, 0.02)
    gaussian_sigma: float = 0.1


@dataclass(frozen=True)
class OracleModelConfig:
    operator_rank: int = 8
    alpha: float = 0.35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    updates_per_example: int = 1
    replay_examples_per_task: int = 4
    replay_ratio: float = 1.0
    seed: int = 2000

    def __post_init__(self) -> None:
        if self.operator_rank <= 0 or self.updates_per_example <= 0:
            raise ValueError("operator rank and update count must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.replay_examples_per_task < 0 or self.replay_ratio < 0.0:
            raise ValueError("replay settings must be nonnegative")


@dataclass(frozen=True)
class DenseModelConfig:
    task_embedding_dim: int = 32
    hidden_width: int = 128
    residual_blocks: int = 3
    global_learning_rate: float = 1e-3
    task_learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    updates_per_example: int = 1
    replay_examples_per_task: int = 4
    replay_ratio: float = 1.0
    seed: int = 3000

    def __post_init__(self) -> None:
        if min(self.task_embedding_dim, self.hidden_width, self.residual_blocks) <= 0:
            raise ValueError("dense model dimensions and block count must be positive")
        if min(self.global_learning_rate, self.task_learning_rate) <= 0.0:
            raise ValueError("dense learning rates must be positive")
        if self.updates_per_example <= 0 or self.replay_examples_per_task < 0:
            raise ValueError("dense update count must be positive and replay size nonnegative")


@dataclass(frozen=True)
class ContinuousModelConfig:
    operator_slots: int = 8
    operator_rank: int = 8
    task_steps: int = 3
    global_learning_rate: float = 1e-3
    task_learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    updates_per_example: int = 1
    replay_examples_per_task: int = 4
    replay_ratio: float = 1.0
    seed: int = 4000

    def __post_init__(self) -> None:
        if min(self.operator_slots, self.operator_rank, self.task_steps) <= 0:
            raise ValueError("continuous model dimensions and step count must be positive")
        if min(self.global_learning_rate, self.task_learning_rate) <= 0.0:
            raise ValueError("continuous learning rates must be positive")
        if self.updates_per_example <= 0 or self.replay_examples_per_task < 0:
            raise ValueError("continuous update count must be positive and replay size nonnegative")


@dataclass(frozen=True)
class DiscreteModelConfig:
    operator_slots: int = 12
    operator_rank: int = 8
    task_steps: int = 3
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    global_learning_rate: float = 1e-3
    task_learning_rate: float = 5e-2
    weight_decay: float = 1e-4
    updates_per_example: int = 1
    replay_examples_per_task: int = 4
    replay_ratio: float = 1.0
    seed: int = 5000

    def __post_init__(self) -> None:
        if min(self.operator_slots, self.operator_rank, self.task_steps) <= 0:
            raise ValueError("discrete model dimensions and step count must be positive")
        if self.initial_temperature <= 0.0 or self.final_temperature <= 0.0:
            raise ValueError("routing temperatures must be positive")
        if min(self.global_learning_rate, self.task_learning_rate) <= 0.0:
            raise ValueError("discrete learning rates must be positive")
        if self.updates_per_example <= 0 or self.replay_examples_per_task < 0:
            raise ValueError("discrete update count must be positive and replay size nonnegative")


@dataclass(frozen=True)
class ExperimentConfig:
    world: WorldConfig
    scratch_model: ScratchModelConfig
    oracle_model: OracleModelConfig
    dense_model: DenseModelConfig
    continuous_model: ContinuousModelConfig
    discrete_model: DiscreteModelConfig
    evaluation: EvaluationConfig
    output_directory: Path


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a mapping")
    return data


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = _require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "config")
    world_raw = _require_mapping(raw.get("world", {}), "world")
    model_raw = _require_mapping(raw.get("scratch_model", {}), "scratch_model")
    oracle_raw = _require_mapping(raw.get("oracle_model", {}), "oracle_model")
    dense_raw = _require_mapping(raw.get("dense_model", {}), "dense_model")
    continuous_raw = _require_mapping(raw.get("continuous_model", {}), "continuous_model")
    discrete_raw = _require_mapping(raw.get("discrete_model", {}), "discrete_model")
    eval_raw = _require_mapping(raw.get("evaluation", {}), "evaluation")
    output_raw = _require_mapping(raw.get("output", {}), "output")

    world = WorldConfig(
        seed=int(world_raw.get("seed", 0)),
        state_dim=int(world_raw.get("state_dim", 16)),
        teacher_rank=int(world_raw.get("teacher_rank", 8)),
        teacher_primitives=int(world_raw.get("teacher_primitives", 6)),
        program_length=int(world_raw.get("program_length", 3)),
        tasks=int(world_raw.get("tasks", 64)),
        examples_per_task=int(world_raw.get("examples_per_task", 128)),
        evaluation_examples=int(world_raw.get("evaluation_examples", 256)),
        reuse_rho=float(world_raw.get("reuse_rho", 1.0)),
        alpha=float(world_raw.get("alpha", 0.35)),
    )
    scratch_model = ScratchModelConfig(
        hidden_dim=int(model_raw.get("hidden_dim", 64)),
        learning_rate=float(model_raw.get("learning_rate", 3e-3)),
        weight_decay=float(model_raw.get("weight_decay", 1e-4)),
        batch_size=int(model_raw.get("batch_size", 8)),
        updates_per_example=int(model_raw.get("updates_per_example", 1)),
        seed=int(model_raw.get("seed", 1000)),
    )
    oracle_model = OracleModelConfig(
        operator_rank=int(oracle_raw.get("operator_rank", 8)),
        alpha=float(oracle_raw.get("alpha", 0.35)),
        learning_rate=float(oracle_raw.get("learning_rate", 1e-3)),
        weight_decay=float(oracle_raw.get("weight_decay", 1e-4)),
        updates_per_example=int(oracle_raw.get("updates_per_example", 1)),
        replay_examples_per_task=int(oracle_raw.get("replay_examples_per_task", 4)),
        replay_ratio=float(oracle_raw.get("replay_ratio", 1.0)),
        seed=int(oracle_raw.get("seed", 2000)),
    )
    dense_model = DenseModelConfig(
        task_embedding_dim=int(dense_raw.get("task_embedding_dim", 32)),
        hidden_width=int(dense_raw.get("hidden_width", 128)),
        residual_blocks=int(dense_raw.get("residual_blocks", 3)),
        global_learning_rate=float(dense_raw.get("global_learning_rate", 1e-3)),
        task_learning_rate=float(dense_raw.get("task_learning_rate", 5e-3)),
        weight_decay=float(dense_raw.get("weight_decay", 1e-4)),
        updates_per_example=int(dense_raw.get("updates_per_example", 1)),
        replay_examples_per_task=int(dense_raw.get("replay_examples_per_task", 4)),
        replay_ratio=float(dense_raw.get("replay_ratio", 1.0)),
        seed=int(dense_raw.get("seed", 3000)),
    )
    continuous_model = ContinuousModelConfig(
        operator_slots=int(continuous_raw.get("operator_slots", 8)),
        operator_rank=int(continuous_raw.get("operator_rank", 8)),
        task_steps=int(continuous_raw.get("task_steps", 3)),
        global_learning_rate=float(continuous_raw.get("global_learning_rate", 1e-3)),
        task_learning_rate=float(continuous_raw.get("task_learning_rate", 5e-3)),
        weight_decay=float(continuous_raw.get("weight_decay", 1e-4)),
        updates_per_example=int(continuous_raw.get("updates_per_example", 1)),
        replay_examples_per_task=int(continuous_raw.get("replay_examples_per_task", 4)),
        replay_ratio=float(continuous_raw.get("replay_ratio", 1.0)),
        seed=int(continuous_raw.get("seed", 4000)),
    )
    discrete_model = DiscreteModelConfig(
        operator_slots=int(discrete_raw.get("operator_slots", 12)),
        operator_rank=int(discrete_raw.get("operator_rank", 8)),
        task_steps=int(discrete_raw.get("task_steps", 3)),
        initial_temperature=float(discrete_raw.get("initial_temperature", 1.0)),
        final_temperature=float(discrete_raw.get("final_temperature", 0.1)),
        global_learning_rate=float(discrete_raw.get("global_learning_rate", 1e-3)),
        task_learning_rate=float(discrete_raw.get("task_learning_rate", 5e-2)),
        weight_decay=float(discrete_raw.get("weight_decay", 1e-4)),
        updates_per_example=int(discrete_raw.get("updates_per_example", 1)),
        replay_examples_per_task=int(discrete_raw.get("replay_examples_per_task", 4)),
        replay_ratio=float(discrete_raw.get("replay_ratio", 1.0)),
        seed=int(discrete_raw.get("seed", 5000)),
    )
    evaluation = EvaluationConfig(
        support_points=tuple(int(n) for n in eval_raw.get("support_points", EvaluationConfig.support_points)),
        nmse_thresholds=tuple(float(x) for x in eval_raw.get("nmse_thresholds", EvaluationConfig.nmse_thresholds)),
        gaussian_sigma=float(eval_raw.get("gaussian_sigma", 0.1)),
    )
    output_directory = Path(output_raw.get("directory", "artifacts/scratch_difficulty"))
    return ExperimentConfig(
        world=world,
        scratch_model=scratch_model,
        oracle_model=oracle_model,
        dense_model=dense_model,
        continuous_model=continuous_model,
        discrete_model=discrete_model,
        evaluation=evaluation,
        output_directory=output_directory,
    )
