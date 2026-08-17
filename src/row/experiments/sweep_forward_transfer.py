"""Run same-architecture fresh-task baselines for explicit forward transfer."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

from row.config import ExperimentConfig, load_config
from row.experiments.learned_lifetime import (
    _build_model,
    _shared_optimizer,
    _tensor,
    _training_values,
)
from row.metrics import gaussian_nll
from row.provenance import current_git_commit, validate_artifact, write_fingerprint
from row.world import Task, World


Model = Literal["continuous", "dense"]
PROTOCOL = {
    "name": "same_architecture_fresh_per_task",
    "score_before_update": True,
    "prior_tasks": 0,
    "replay_ratio": 0.0,
    "shared_and_task_parameters_updated": True,
    "reset_model_and_optimizer_for_every_task": True,
}


def _resolved(config: ExperimentConfig, model: Model) -> dict[str, Any]:
    selected = (
        config.continuous_model if model == "continuous" else config.dense_model
    )
    return {
        "world": asdict(config.world),
        f"{model}_model": asdict(selected),
        "evaluation": asdict(config.evaluation),
        "protocol": PROTOCOL,
        "output": {"directory": str(config.output_directory)},
    }


def _fresh_task_loss(
    config: ExperimentConfig, model_kind: Model, task: Task
) -> float:
    model = _build_model(config, model_kind)
    global_lr, task_lr, weight_decay, updates, _, _, _ = _training_values(
        config, model_kind
    )
    optimizer = _shared_optimizer(model, global_lr, weight_decay)
    task_parameter = model.begin_task(task.task_id)
    optimizer.add_param_group(
        {"params": [task_parameter], "lr": task_lr, "weight_decay": 0.0}
    )
    cumulative = 0.0
    for index in range(config.world.examples_per_task):
        x = _tensor(task.train_x[index : index + 1])
        y = _tensor(task.train_y[index : index + 1])
        model.eval()
        with torch.no_grad():
            prediction = model(x, task.task_id)
        cumulative += gaussian_nll(
            prediction.cpu().numpy(),
            task.train_y[index : index + 1],
            config.evaluation.gaussian_sigma,
        )
        for _ in range(updates):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, task.task_id)
            loss = torch.nn.functional.mse_loss(prediction, y)
            loss.backward()
            optimizer.step()
    return cumulative


def run(config: ExperimentConfig, model: Model) -> dict[str, Any]:
    torch.set_num_threads(1)
    world = World.generate(config.world)
    rows = []
    for task_index, task in enumerate(world.tasks):
        loss = _fresh_task_loss(config, model, task)
        rows.append(
            {
                "task_index": task_index,
                "world_task_index": task_index,
                "task_id": task.task_id,
                "fresh_prequential_gaussian_log_loss": loss,
            }
        )
    total = float(sum(row["fresh_prequential_gaussian_log_loss"] for row in rows))
    summary = {
        "model": model,
        "protocol": PROTOCOL,
        "tasks": len(rows),
        "total_fresh_prequential_gaussian_log_loss": total,
        "mean_fresh_task_gaussian_log_loss": total / len(rows),
    }
    _write_artifacts(config, model, world, rows, summary)
    return summary


def _write_artifacts(
    config: ExperimentConfig,
    model: Model,
    world: World,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    resolved = _resolved(config, model)
    (output / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "world_programs.json").write_text(
        json.dumps(world.programs_json(), indent=2) + "\n", encoding="utf-8"
    )
    selected = (
        config.continuous_model if model == "continuous" else config.dense_model
    )
    (output / "world_seed.txt").write_text(
        f"{config.world.seed}\n", encoding="utf-8"
    )
    (output / "model_seed.txt").write_text(f"{selected.seed}\n", encoding="utf-8")
    commit = current_git_commit()
    (output / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    write_fingerprint(output, resolved, model, commit)
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
        "--output", type=Path, default=Path("artifacts/forward_transfer/fresh")
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--models", choices=("continuous", "dense"), nargs="+", default=["continuous", "dense"]
    )
    args = parser.parse_args()
    base = load_config(args.config)
    total = len(args.worlds) * len(args.models)
    completed = 0
    for world_seed in args.worlds:
        for model in args.models:
            completed += 1
            output = args.output / f"world_{world_seed}" / model
            config = replace(
                base,
                world=replace(base.world, seed=world_seed),
                output_directory=output,
                evaluation=replace(
                    base.evaluation,
                    lifetime_checkpoints=(),
                    checkpoint_novel_tasks=1,
                    extended_diagnostics=False,
                ),
            )
            print(f"[{completed}/{total}] world_{world_seed}/{model}", flush=True)
            if (output / "summary.json").exists():
                validate_artifact(output, _resolved(config, model), model)
            else:
                run(config, model)


if __name__ == "__main__":
    main()
