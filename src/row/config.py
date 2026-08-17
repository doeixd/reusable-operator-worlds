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
class ExperimentConfig:
    world: WorldConfig
    scratch_model: ScratchModelConfig
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
    evaluation = EvaluationConfig(
        support_points=tuple(int(n) for n in eval_raw.get("support_points", EvaluationConfig.support_points)),
        nmse_thresholds=tuple(float(x) for x in eval_raw.get("nmse_thresholds", EvaluationConfig.nmse_thresholds)),
        gaussian_sigma=float(eval_raw.get("gaussian_sigma", 0.1)),
    )
    output_directory = Path(output_raw.get("directory", "artifacts/scratch_difficulty"))
    return ExperimentConfig(world, scratch_model, evaluation, output_directory)
