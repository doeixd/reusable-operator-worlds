"""Shared online protocol for dense and continuous-basis non-oracle learners."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml

from row.config import ExperimentConfig, load_config
from row.experiments.oracle_lifetime import _add_lifetime_transfer_summary, _functional_recovery
from row.experiments.scratch_difficulty import summarize
from row.metrics import examples_to_criterion, gaussian_nll, nmse
from row.models import ContinuousBasisLearner, DenseLearner
from row.world import Program, Task, World

ModelKind = Literal["dense", "continuous"]
Learner = DenseLearner | ContinuousBasisLearner


class TaskReplayBuffer:
    def __init__(self, seed: int) -> None:
        self.generator = np.random.default_rng(seed)
        self.items: list[tuple[np.ndarray, np.ndarray, str]] = []

    def add_task(self, task: Task, count: int) -> None:
        if count == 0:
            return
        indices = self.generator.choice(len(task.train_x), size=min(count, len(task.train_x)), replace=False)
        for index in indices:
            self.items.append((task.train_x[index].copy(), task.train_y[index].copy(), task.task_id))

    def sample(self, count: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
        if count == 0 or not self.items:
            return []
        indices = self.generator.choice(len(self.items), size=min(count, len(self.items)), replace=False)
        return [self.items[int(index)] for index in np.atleast_1d(indices)]


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32)


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
    model_config = config.continuous_model
    return ContinuousBasisLearner(
        d=config.world.state_dim,
        operator_slots=model_config.operator_slots,
        operator_rank=model_config.operator_rank,
        task_steps=model_config.task_steps,
        alpha=config.world.alpha,
        seed=model_config.seed,
    )


def _training_values(
    config: ExperimentConfig, kind: ModelKind
) -> tuple[float, float, float, int, int, float, int]:
    selected = config.dense_model if kind == "dense" else config.continuous_model
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


def run(config: ExperimentConfig, kind: ModelKind, order: str = "forward") -> dict[str, object]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be 'forward' or 'reverse'")
    torch.set_num_threads(1)
    world = World.generate(config.world)
    model = _build_model(config, kind)
    global_lr, task_lr, weight_decay, update_count, replay_per_task, replay_ratio, seed = (
        _training_values(config, kind)
    )
    optimizer = torch.optim.AdamW(
        model.shared_parameters(), lr=global_lr, weight_decay=weight_decay
    )
    replay = TaskReplayBuffer(seed + 1)
    task_indices = list(range(len(world.tasks)))
    if order == "reverse":
        task_indices.reverse()
    support = set(config.evaluation.support_points)
    rows: list[dict[str, object]] = []
    cumulative_nll = 0.0

    for lifetime_index, world_task_index in enumerate(task_indices):
        task = world.tasks[world_task_index]
        task_parameter = model.begin_task(task.task_id)
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
            rows.append(
                {
                    "record_type": "prequential",
                    "task_index": lifetime_index,
                    "world_task_index": world_task_index,
                    "task_id": task.task_id,
                    "n_seen": n_seen,
                    "nll": online_nll,
                    "cumulative_nll": cumulative_nll,
                }
            )

            replay_items = replay.sample(int(round(replay_ratio)))
            batch_x = [task.train_x[n_seen], *(item[0] for item in replay_items)]
            batch_y = [task.train_y[n_seen], *(item[1] for item in replay_items)]
            task_ids = [task.task_id, *(item[2] for item in replay_items)]
            for _ in range(update_count):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_tasks(_tensor(np.stack(batch_x)), task_ids)
                loss = torch.nn.functional.mse_loss(prediction, _tensor(np.stack(batch_y)))
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
        replay.add_task(task, replay_per_task)

    summary = summarize(rows, config.world.examples_per_task)
    _add_lifetime_transfer_summary(summary, rows)
    summary.update(
        {
            "model": kind,
            "order": order,
            "cumulative_prequential_nll": cumulative_nll,
            "shared_parameter_count": model.shared_parameter_count,
            "task_state_scalar_count": model.task_state_scalar_count,
        }
    )
    if isinstance(model, ContinuousBasisLearner):
        summary["routing"] = model.routing_diagnostics()
        summary["functional_recovery"] = _functional_recovery(model.basis, world, config)
    summary["novel_composition"] = _adapt_novel_composition(
        model, world, config, task_lr
    )
    _write_artifacts(config, world, model, rows, summary, kind, order)
    return summary


def _novel_data(world: World, config: ExperimentConfig) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    used = {task.program.primitive_ids for task in world.tasks}
    candidates = [
        tuple(route)
        for route in product(
            range(config.world.teacher_primitives), repeat=config.world.program_length
        )
        if tuple(route) not in used
    ]
    generator = np.random.default_rng(np.random.SeedSequence([config.world.seed, 91]))
    route = candidates[int(generator.integers(len(candidates)))]
    train_x = generator.normal(size=(32, config.world.state_dim))
    eval_x = generator.normal(size=(config.world.evaluation_examples, config.world.state_dim))
    program = Program(route)
    return (
        route,
        train_x,
        program.execute(world.library, train_x),
        eval_x,
        program.execute(world.library, eval_x),
    )


def _adapt_novel_composition(
    model: Learner, world: World, config: ExperimentConfig, task_lr: float
) -> dict[str, object]:
    route, train_x, train_y, eval_x, eval_y = _novel_data(world, config)
    for parameter in model.shared_parameters():
        parameter.requires_grad_(False)
    novel_id = "task_novel_composition"
    novel_code = model.begin_task(novel_id)
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
        loss.backward()
        optimizer.step()
    return {"teacher_route": list(route), "nmse_by_support": curve}


def _write_artifacts(
    config: ExperimentConfig,
    world: World,
    model: Learner,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    kind: ModelKind,
    order: str,
) -> None:
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    model_config = config.dense_model if kind == "dense" else config.continuous_model
    resolved = {
        "world": world.config_dict(),
        f"{kind}_model": asdict(model_config),
        "evaluation": asdict(config.evaluation),
        "order": order,
        "output": {"directory": str(output)},
    }
    (output / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "world_programs.json").write_text(
        json.dumps(world.programs_json(), indent=2), encoding="utf-8"
    )
    (output / "world_seed.txt").write_text(f"{config.world.seed}\n", encoding="utf-8")
    (output / "model_seed.txt").write_text(f"{model_config.seed}\n", encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "summary": summary}, output / "model.pt")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        commit = "uncommitted"
    (output / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
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
    parser.add_argument("--model", choices=("dense", "continuous"), required=True)
    parser.add_argument("--world-seed", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--global-learning-rate", type=float)
    parser.add_argument("--task-learning-rate", type=float)
    parser.add_argument("--updates-per-example", type=int)
    parser.add_argument("--hidden-width", type=int)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    config = replace(
        config,
        world=config.world if args.world_seed is None else replace(config.world, seed=args.world_seed),
        output_directory=config.output_directory if args.output is None else args.output,
    )
    selected = config.dense_model if args.model == "dense" else config.continuous_model
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
    )
    config = replace(
        config,
        dense_model=selected if args.model == "dense" else config.dense_model,
        continuous_model=(
            selected if args.model == "continuous" else config.continuous_model
        ),
    )
    summary = run(config, kind=args.model, order=args.order)
    final = summary["final_nmse"]
    assert isinstance(final, dict)
    novel = summary["novel_composition"]
    assert isinstance(novel, dict)
    novel_curve = novel["nmse_by_support"]
    assert isinstance(novel_curve, dict)
    print(
        f"{args.model} {args.order}: final median NMSE={final['median']:.4f}; "
        f"prequential NLL={summary['cumulative_prequential_nll']:.1f}; "
        f"novel NMSE 0/32={novel_curve['0']:.4f}/{novel_curve['32']:.4f}"
    )


if __name__ == "__main__":
    main()
