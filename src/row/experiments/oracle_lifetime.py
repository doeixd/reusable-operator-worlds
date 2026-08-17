"""Online lifetime experiment for the true-route oracle positive control."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment

from row.config import ExperimentConfig, load_config
from row.experiments.scratch_difficulty import summarize
from row.metrics import examples_to_criterion, gaussian_nll, nmse
from row.models import OracleCompositor
from row.world import Program, Task, World


class ReplayBuffer:
    """Fixed per-completed-task exemplars with deterministic random sampling."""

    def __init__(self, seed: int) -> None:
        self.generator = np.random.default_rng(seed)
        self.items: list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]] = []

    def add_task(self, task: Task, count: int) -> None:
        if count == 0:
            return
        indices = self.generator.choice(len(task.train_x), size=min(count, len(task.train_x)), replace=False)
        for index in indices:
            self.items.append(
                (task.train_x[index].copy(), task.train_y[index].copy(), task.program.primitive_ids)
            )

    def sample(self, count: int) -> list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
        if count == 0 or not self.items:
            return []
        indices = self.generator.choice(len(self.items), size=min(count, len(self.items)), replace=False)
        return [self.items[int(index)] for index in np.atleast_1d(indices)]


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32)


@torch.no_grad()
def _evaluate(model: OracleCompositor, task: Task) -> tuple[float, np.ndarray]:
    model.eval()
    prediction = model(_tensor(task.eval_x), task.program.primitive_ids).cpu().numpy()
    return nmse(prediction, task.eval_y), prediction


def run(config: ExperimentConfig, order: str = "forward") -> dict[str, object]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be 'forward' or 'reverse'")
    torch.set_num_threads(1)
    world = World.generate(config.world)
    model = OracleCompositor(
        d=config.world.state_dim,
        rank=config.oracle_model.operator_rank,
        operators=config.world.teacher_primitives,
        alpha=config.oracle_model.alpha,
        seed=config.oracle_model.seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.oracle_model.learning_rate,
        weight_decay=config.oracle_model.weight_decay,
    )
    replay = ReplayBuffer(seed=config.oracle_model.seed + 1)
    task_indices = list(range(len(world.tasks)))
    if order == "reverse":
        task_indices.reverse()
    support = set(config.evaluation.support_points)
    rows: list[dict[str, object]] = []
    cumulative_nll = 0.0

    for lifetime_index, world_task_index in enumerate(task_indices):
        task = world.tasks[world_task_index]
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
            y = _tensor(task.train_y[n_seen : n_seen + 1])
            model.eval()
            with torch.no_grad():
                online_prediction = model(x, task.program.primitive_ids).cpu().numpy()
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

            replay_count = int(round(config.oracle_model.replay_ratio))
            replay_items = replay.sample(replay_count)
            batch_x = [task.train_x[n_seen], *(item[0] for item in replay_items)]
            batch_y = [task.train_y[n_seen], *(item[1] for item in replay_items)]
            routes = [task.program.primitive_ids, *(item[2] for item in replay_items)]
            for _ in range(config.oracle_model.updates_per_example):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_routes(_tensor(np.stack(batch_x)), routes)
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
        replay.add_task(task, config.oracle_model.replay_examples_per_task)

    novel = _evaluate_novel_composition(model, world, config)
    summary = summarize(rows, config.world.examples_per_task)
    _add_lifetime_transfer_summary(summary, rows)
    summary["order"] = order
    summary["cumulative_prequential_nll"] = cumulative_nll
    summary["parameter_count"] = model.parameter_count
    summary["world_functional_reuse"] = world.functional_reuse_diagnostics()
    summary["novel_composition"] = novel
    summary["functional_recovery"] = _functional_recovery(model.operators, world, config)
    _write_artifacts(config, world, model, rows, summary, order)
    return summary


def _add_lifetime_transfer_summary(
    summary: dict[str, object], rows: list[dict[str, object]]
) -> None:
    tasks = sorted(
        (row for row in rows if row["record_type"] == "task_summary"),
        key=lambda row: int(row["task_index"]),
    )
    quarter = max(1, len(tasks) // 4)
    early = tasks[:quarter]
    late = tasks[-quarter:]
    zero_shot = np.array([float(row["zero_shot_nmse"]) for row in tasks])
    indices = np.arange(len(tasks), dtype=np.float64)
    summary["zero_shot_nmse"] = {
        "early_quarter_mean": float(np.mean(zero_shot[:quarter])),
        "late_quarter_mean": float(np.mean(zero_shot[-quarter:])),
        "slope_per_task": float(np.polyfit(indices, zero_shot, deg=1)[0]),
        "task_index_correlation": float(np.corrcoef(indices, zero_shot)[0, 1]),
    }
    criteria = summary["criteria"]
    assert isinstance(criteria, dict)
    for threshold, values in criteria.items():
        assert isinstance(values, dict)
        key = f"examples_to_{threshold}"
        values["early_quarter_mean"] = float(np.mean([float(row[key]) for row in early]))
        values["late_quarter_mean"] = float(np.mean([float(row[key]) for row in late]))


@torch.no_grad()
def _evaluate_novel_composition(
    model: OracleCompositor, world: World, config: ExperimentConfig
) -> dict[str, object]:
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
    # Consume the same 32 adaptation inputs used by non-oracle novel-task runs so
    # all models share the identical evaluation set.
    generator.normal(size=(32, config.world.state_dim))
    x = generator.normal(size=(config.world.evaluation_examples, config.world.state_dim))
    y = Program(route).execute(world.library_for_task(config.world.tasks), x)
    prediction = model(_tensor(x), route).cpu().numpy()
    return {"route": list(route), "zero_shot_nmse": nmse(prediction, y)}


@torch.no_grad()
def _functional_recovery(
    operators: torch.nn.ModuleList, world: World, config: ExperimentConfig
) -> dict[str, object]:
    generator = np.random.default_rng(np.random.SeedSequence([config.world.seed, 92]))
    probe_x = generator.normal(size=(4096, config.world.state_dim))
    learned = [
        operator(_tensor(probe_x)).cpu().numpy()
        for operator in operators
    ]
    teacher = [primitive(probe_x) for primitive in world.library]
    distances = np.empty((len(teacher), len(learned)), dtype=np.float64)
    for teacher_index, teacher_output in enumerate(teacher):
        denominator = float(np.var(teacher_output))
        for learned_index, learned_output in enumerate(learned):
            distances[teacher_index, learned_index] = float(
                np.mean(np.square(teacher_output - learned_output)) / denominator
            )

    teacher_indices, learned_indices = linear_sum_assignment(distances)
    best_assignment = tuple(int(index) for index in learned_indices)
    best_cost = float(distances[teacher_indices, learned_indices].sum())

    candidate_explanations: list[tuple[tuple[int, ...], np.ndarray, float]] = []
    for depth in range(1, config.world.program_length + 1):
        for route in product(range(config.world.teacher_primitives), repeat=depth):
            output = Program(tuple(route)).execute(world.library, probe_x)
            candidate_explanations.append((tuple(route), output, float(np.var(output))))
    short_explanations: list[dict[str, object]] = []
    for learned_index, learned_output in enumerate(learned):
        best_route: tuple[int, ...] | None = None
        best_distance = float("inf")
        for route, teacher_output, denominator in candidate_explanations:
            distance = float(np.mean(np.square(teacher_output - learned_output)) / denominator)
            if distance < best_distance:
                best_distance = distance
                best_route = route
        assert best_route is not None
        short_explanations.append(
            {
                "learned_operator": learned_index,
                "teacher_route": list(best_route),
                "normalized_distance": best_distance,
            }
        )

    return {
        "probe_examples": len(probe_x),
        "distance_matrix": distances.tolist(),
        "one_to_one_matches": [
            {
                "teacher_operator": teacher_index,
                "learned_operator": learned_index,
                "normalized_distance": float(distances[teacher_index, learned_index]),
            }
            for teacher_index, learned_index in enumerate(best_assignment)
        ],
        "one_to_one_mean_distance": best_cost / len(teacher),
        "best_depth_1_to_3_explanations": short_explanations,
    }


def _write_artifacts(
    config: ExperimentConfig,
    world: World,
    model: OracleCompositor,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    order: str,
) -> None:
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        "world": world.config_dict(),
        "oracle_model": asdict(config.oracle_model),
        "evaluation": asdict(config.evaluation),
        "order": order,
        "output": {"directory": str(output)},
    }
    (output / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "functional_recovery.json").write_text(
        json.dumps(summary["functional_recovery"], indent=2), encoding="utf-8"
    )
    (output / "world_programs.json").write_text(
        json.dumps(world.programs_json(), indent=2), encoding="utf-8"
    )
    (output / "world_functional_reuse.json").write_text(
        json.dumps(summary["world_functional_reuse"], indent=2), encoding="utf-8"
    )
    (output / "world_seed.txt").write_text(f"{config.world.seed}\n", encoding="utf-8")
    (output / "model_seed.txt").write_text(f"{config.oracle_model.seed}\n", encoding="utf-8")
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
    parser.add_argument("--world-seed", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--updates-per-example", type=int)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    config = replace(
        config,
        world=(
            config.world if args.world_seed is None else replace(config.world, seed=args.world_seed)
        ),
        oracle_model=replace(
            config.oracle_model,
            seed=config.oracle_model.seed if args.model_seed is None else args.model_seed,
            learning_rate=(
                config.oracle_model.learning_rate
                if args.learning_rate is None
                else args.learning_rate
            ),
            updates_per_example=(
                config.oracle_model.updates_per_example
                if args.updates_per_example is None
                else args.updates_per_example
            ),
        ),
        output_directory=config.output_directory if args.output is None else args.output,
    )
    summary = run(config, order=args.order)
    final = summary["final_nmse"]
    assert isinstance(final, dict)
    criteria = summary["criteria"]
    assert isinstance(criteria, dict)
    criterion = criteria.get("0.1", {})
    print(
        f"oracle {args.order}: final median NMSE={final['median']:.4f}; "
        f"N@0.1 median={criterion.get('median')}; "
        f"novel zero-shot NMSE={summary['novel_composition']['zero_shot_nmse']:.4f}"
    )


if __name__ == "__main__":
    main()
