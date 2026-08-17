"""Train an independent model per hidden task to test difficulty stationarity."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml

from row.config import ExperimentConfig, load_config
from row.metrics import examples_to_criterion, gaussian_nll, nmse
from row.models import ScratchResidualMLP
from row.world import World


def run(config: ExperimentConfig) -> list[dict[str, object]]:
    world = World.generate(config.world)
    rows: list[dict[str, object]] = []
    support = set(config.evaluation.support_points)
    max_examples = config.world.examples_per_task

    for task_index, task in enumerate(world.tasks):
        model = ScratchResidualMLP(
            input_dim=config.world.state_dim,
            hidden_dim=config.scratch_model.hidden_dim,
            seed=config.scratch_model.seed + task_index,
            learning_rate=config.scratch_model.learning_rate,
            weight_decay=config.scratch_model.weight_decay,
        )
        batch_rng = np.random.default_rng(config.scratch_model.seed + 100_000 + task_index)
        curve: dict[int, float] = {}
        for n_seen in range(max_examples + 1):
            if n_seen in support:
                prediction = model.predict(task.eval_x)
                score = nmse(prediction, task.eval_y)
                curve[n_seen] = score
                rows.append(
                    {
                        "record_type": "evaluation",
                        "task_index": task_index,
                        "task_id": task.task_id,
                        "n_seen": n_seen,
                        "nmse": score,
                        "gaussian_nll": gaussian_nll(
                            prediction, task.eval_y, config.evaluation.gaussian_sigma
                        ),
                    }
                )
            if n_seen == max_examples:
                break
            pool_size = n_seen + 1
            batch_size = min(config.scratch_model.batch_size, pool_size)
            for _ in range(config.scratch_model.updates_per_example):
                indices = batch_rng.choice(pool_size, size=batch_size, replace=False)
                model.update(task.train_x[indices], task.train_y[indices])

        summary: dict[str, object] = {
            "record_type": "task_summary",
            "task_index": task_index,
            "task_id": task.task_id,
            "final_nmse": curve[max(curve)],
        }
        for threshold in config.evaluation.nmse_thresholds:
            summary[f"examples_to_{threshold:g}"] = examples_to_criterion(
                curve, threshold, max_examples
            )
        rows.append(summary)

    _write_artifacts(config, world, rows, summarize(rows, config.world.examples_per_task))
    return rows


def summarize(rows: list[dict[str, object]], max_examples: int) -> dict[str, object]:
    tasks = sorted(
        (row for row in rows if row["record_type"] == "task_summary"),
        key=lambda row: int(row["task_index"]),
    )
    indices = np.arange(len(tasks), dtype=np.float64)

    def trend(values: np.ndarray) -> dict[str, float]:
        if len(values) < 2 or float(np.std(values)) == 0.0:
            correlation = 0.0
            slope = 0.0
        else:
            correlation = float(np.corrcoef(indices, values)[0, 1])
            slope = float(np.polyfit(indices, values, deg=1)[0])
        return {"slope_per_task": slope, "task_index_correlation": correlation}

    final_nmse = np.array([float(row["final_nmse"]) for row in tasks])
    result: dict[str, object] = {
        "tasks": len(tasks),
        "final_nmse": {
            "mean": float(np.mean(final_nmse)),
            "median": float(np.median(final_nmse)),
            "minimum": float(np.min(final_nmse)),
            "maximum": float(np.max(final_nmse)),
            "forward_order": trend(final_nmse),
            "reverse_order": trend(final_nmse[::-1]),
        },
        "criteria": {},
    }
    criteria = result["criteria"]
    assert isinstance(criteria, dict)
    criterion_keys = sorted(key for key in tasks[0] if key.startswith("examples_to_"))
    for key in criterion_keys:
        values = np.array([float(row[key]) for row in tasks])
        criteria[key.removeprefix("examples_to_")] = {
            "median": float(np.median(values)),
            "censored_fraction": float(np.mean(values > max_examples)),
            "forward_order": trend(values),
            "reverse_order": trend(values[::-1]),
        }
    return result


def _write_artifacts(
    config: ExperimentConfig,
    world: World,
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        "world": world.config_dict(),
        "scratch_model": asdict(config.scratch_model),
        "evaluation": {
            **asdict(config.evaluation),
            "support_points": list(config.evaluation.support_points),
            "nmse_thresholds": list(config.evaluation.nmse_thresholds),
        },
        "output": {"directory": str(output)},
    }
    (output / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "world_programs.json").write_text(
        json.dumps(world.programs_json(), indent=2), encoding="utf-8"
    )
    (output / "world_seed.txt").write_text(f"{config.world.seed}\n", encoding="utf-8")
    (output / "model_seed.txt").write_text(f"{config.scratch_model.seed}\n", encoding="utf-8")
    (output / "world_diagnostics.json").write_text(
        json.dumps(world.diagnostics(), indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    }
    (output / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--world-seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--updates-per-example", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    world = config.world if args.world_seed is None else replace(config.world, seed=args.world_seed)
    model = replace(
        config.scratch_model,
        learning_rate=(
            config.scratch_model.learning_rate
            if args.learning_rate is None
            else args.learning_rate
        ),
        updates_per_example=(
            config.scratch_model.updates_per_example
            if args.updates_per_example is None
            else args.updates_per_example
        ),
    )
    config = replace(
        config,
        world=world,
        scratch_model=model,
        output_directory=config.output_directory if args.output is None else args.output,
    )
    rows = run(config)
    summaries = [row for row in rows if row["record_type"] == "task_summary"]
    median_final = float(np.median([float(row["final_nmse"]) for row in summaries]))
    result = summarize(
        rows,
        max_examples=max(
            int(row["n_seen"]) for row in rows if row["record_type"] == "evaluation"
        ),
    )
    final_summary = result["final_nmse"]
    assert isinstance(final_summary, dict)
    forward = final_summary["forward_order"]
    assert isinstance(forward, dict)
    print(
        f"completed {len(summaries)} scratch tasks; median final NMSE={median_final:.4f}; "
        f"task-index correlation={forward['task_index_correlation']:.4f}"
    )


if __name__ == "__main__":
    main()
