"""Shared online protocol for dense and continuous-basis non-oracle learners."""

from __future__ import annotations

import argparse
import copy
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
from row.models import ContinuousBasisLearner, DenseLearner, DiscreteLibraryLearner
from row.world import Program, Task, World

ModelKind = Literal["dense", "continuous", "discrete"]
Learner = DenseLearner | ContinuousBasisLearner | DiscreteLibraryLearner


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
    if kind == "continuous":
        model_config = config.continuous_model
        return ContinuousBasisLearner(
            d=config.world.state_dim,
            operator_slots=model_config.operator_slots,
            operator_rank=model_config.operator_rank,
            task_steps=model_config.task_steps,
            alpha=config.world.alpha,
            seed=model_config.seed,
        )
    model_config = config.discrete_model
    return DiscreteLibraryLearner(
        d=config.world.state_dim,
        operator_slots=model_config.operator_slots,
        operator_rank=model_config.operator_rank,
        task_steps=model_config.task_steps,
        alpha=config.world.alpha,
        initial_temperature=model_config.initial_temperature,
        final_temperature=model_config.final_temperature,
        seed=model_config.seed,
    )


def _training_values(
    config: ExperimentConfig, kind: ModelKind
) -> tuple[float, float, float, int, int, float, int]:
    selected = {
        "dense": config.dense_model,
        "continuous": config.continuous_model,
        "discrete": config.discrete_model,
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
    cumulative_mass_log_loss = 0.0
    checkpoint_results: list[dict[str, object]] = []

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

            replay_items = replay.sample(int(round(replay_ratio)))
            batch_x = [task.train_x[n_seen], *(item[0] for item in replay_items)]
            batch_y = [task.train_y[n_seen], *(item[1] for item in replay_items)]
            task_ids = [task.task_id, *(item[2] for item in replay_items)]
            for _ in range(update_count):
                if isinstance(model, DiscreteLibraryLearner):
                    global_example = lifetime_index * config.world.examples_per_task + n_seen
                    total_examples = len(world.tasks) * config.world.examples_per_task
                    model.set_training_progress(global_example / max(1, total_examples - 1))
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
        tasks_completed = lifetime_index + 1
        if tasks_completed in config.evaluation.lifetime_checkpoints:
            checkpoint_results.append(
                _novel_checkpoint(
                    model,
                    world,
                    config,
                    task_lr,
                    tasks_completed,
                    config.evaluation.checkpoint_novel_tasks,
                )
            )

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
            "shared_parameter_count": model.shared_parameter_count,
            "task_state_scalar_count": model.task_state_scalar_count,
            "world_functional_reuse": world.functional_reuse_diagnostics(),
        }
    )
    if isinstance(model, ContinuousBasisLearner):
        summary["routing"] = model.routing_diagnostics()
        summary["functional_recovery"] = _functional_recovery(model.basis, world, config)
    elif isinstance(model, DiscreteLibraryLearner):
        summary["routing"] = model.routing_diagnostics()
        summary["functional_recovery"] = _functional_recovery(model.library, world, config)
        summary["route_recovery"] = _route_recovery(
            model, world, summary["functional_recovery"]
        )
        summary["operator_specialization"] = _operator_specialization(model, config)
    summary["novel_composition"] = _adapt_novel_composition(
        model, world, config, task_lr
    )
    summary["novel_composition_checkpoints"] = checkpoint_results
    _write_artifacts(config, world, model, rows, summary, kind, order)
    return summary


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
    model_config = {
        "dense": config.dense_model,
        "continuous": config.continuous_model,
        "discrete": config.discrete_model,
    }[kind]
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
    (output / "world_functional_reuse.json").write_text(
        json.dumps(summary["world_functional_reuse"], indent=2), encoding="utf-8"
    )
    (output / "world_seed.txt").write_text(f"{config.world.seed}\n", encoding="utf-8")
    (output / "model_seed.txt").write_text(f"{model_config.seed}\n", encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "summary": summary}, output / "model.pt")
    if isinstance(model, DiscreteLibraryLearner):
        (output / "hard_routes.json").write_text(
            json.dumps(model.hard_routes(), indent=2), encoding="utf-8"
        )
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
    parser.add_argument("--model", choices=("dense", "continuous", "discrete"), required=True)
    parser.add_argument("--world-seed", type=int)
    parser.add_argument("--reuse-rho", type=float)
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
        world=replace(
            config.world,
            seed=config.world.seed if args.world_seed is None else args.world_seed,
            reuse_rho=config.world.reuse_rho if args.reuse_rho is None else args.reuse_rho,
        ),
        output_directory=config.output_directory if args.output is None else args.output,
    )
    selected = {
        "dense": config.dense_model,
        "continuous": config.continuous_model,
        "discrete": config.discrete_model,
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
    )
    config = replace(
        config,
        dense_model=selected if args.model == "dense" else config.dense_model,
        continuous_model=(
            selected if args.model == "continuous" else config.continuous_model
        ),
        discrete_model=selected if args.model == "discrete" else config.discrete_model,
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
        f"prequential Gaussian log loss={summary['cumulative_prequential_gaussian_log_loss']:.1f}; "
        f"novel NMSE 0/32={novel_curve['0']:.4f}/{novel_curve['32']:.4f}"
    )


if __name__ == "__main__":
    main()
