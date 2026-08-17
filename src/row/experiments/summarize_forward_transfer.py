"""Validate fresh-task baselines and summarize explicit forward transfer."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from row.config import load_config
from row.experiments.learned_lifetime import resolved_learned_config
from row.experiments.summarize_robustness import _effect_summary
from row.experiments.sweep_forward_transfer import _resolved
from row.provenance import validate_artifact
from row.world import World


WORLDS = tuple(range(10))
MODELS = ("continuous", "dense")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _lifetime_task_losses(path: Path) -> dict[int, float]:
    losses: defaultdict[int, float] = defaultdict(float)
    for row in _jsonl(path / "metrics.jsonl"):
        if row["record_type"] == "prequential":
            losses[int(row["world_task_index"])] += float(row["nll"])
    return dict(losses)


def _fresh_task_losses(path: Path) -> dict[int, float]:
    return {
        int(row["world_task_index"]): float(
            row["fresh_prequential_gaussian_log_loss"]
        )
        for row in _jsonl(path / "metrics.jsonl")
    }


def _route_context(world: World, task_index: int) -> tuple[int, float]:
    route = world.tasks[task_index].program.primitive_ids
    prior = [task.program.primitive_ids for task in world.tasks[:task_index]]
    encountered = len({primitive for program in prior for primitive in program})
    similarity = (
        max(
            sum(left == right for left, right in zip(route, program, strict=True))
            / len(route)
            for program in prior
        )
        if prior
        else 0.0
    )
    return encountered, similarity


def _slope(rows: list[dict[str, Any]], key: str) -> float:
    x = np.asarray([float(row["task_index"]) for row in rows])
    y = np.asarray([float(row[key]) for row in rows])
    return float(np.polyfit(x, y, deg=1)[0])


def _strata(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(float(row["forward_transfer_gaussian_log_loss"]))
    return [
        {key: group, "tasks": len(values), "mean_forward_transfer": fmean(values)}
        for group, values in sorted(groups.items(), key=lambda item: float(item[0]))
    ]


def _task_index_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        for start in range(0, 64, 8):
            world_means = []
            for world in WORLDS:
                values = [
                    float(row["forward_transfer_gaussian_log_loss"])
                    for row in selected
                    if row["world_seed"] == world
                    and start <= int(row["task_index"]) < start + 8
                ]
                world_means.append(fmean(values))
            result.append(
                {
                    "model": model,
                    "task_index_start": start,
                    "task_index_end": start + 7,
                    "mean_forward_transfer": fmean(world_means),
                    "world_means": world_means,
                }
            )
    return result


def summarize(
    config_path: Path, fresh_root: Path, lifetime_root: Path
) -> dict[str, Any]:
    base = load_config(config_path)
    task_rows = []
    for world_seed in WORLDS:
        world = World.generate(replace(base.world, seed=world_seed))
        for model in MODELS:
            fresh_output = fresh_root / f"world_{world_seed}" / model
            fresh_config = replace(
                base,
                world=replace(base.world, seed=world_seed),
                output_directory=fresh_output,
                evaluation=replace(
                    base.evaluation,
                    lifetime_checkpoints=(),
                    checkpoint_novel_tasks=1,
                    extended_diagnostics=False,
                ),
            )
            validate_artifact(
                fresh_output, _resolved(fresh_config, model), model
            )
            lifetime_output = lifetime_root / f"world_{world_seed}" / model
            lifetime_config = replace(
                base,
                world=replace(base.world, seed=world_seed),
                output_directory=lifetime_output,
            )
            validate_artifact(
                lifetime_output,
                resolved_learned_config(lifetime_config, model, "forward"),
                model,
            )
            fresh = _fresh_task_losses(fresh_output)
            lifetime = _lifetime_task_losses(lifetime_output)
            if tuple(sorted(fresh)) != tuple(range(64)) or tuple(sorted(lifetime)) != tuple(range(64)):
                raise ValueError(f"incomplete task losses for world {world_seed} {model}")
            for task_index in range(64):
                exposure, similarity = _route_context(world, task_index)
                task_rows.append(
                    {
                        "world_seed": world_seed,
                        "model": model,
                        "task_index": task_index,
                        "fresh_gaussian_log_loss": fresh[task_index],
                        "lifetime_gaussian_log_loss": lifetime[task_index],
                        "forward_transfer_gaussian_log_loss": (
                            fresh[task_index] - lifetime[task_index]
                        ),
                        "teacher_primitives_encountered_before_task": exposure,
                        "maximum_prior_route_position_similarity": similarity,
                    }
                )

    model_summaries = []
    world_model_means: dict[tuple[int, str], float] = {}
    for model_index, model in enumerate(MODELS):
        selected = [row for row in task_rows if row["model"] == model]
        world_means = []
        world_slopes = []
        for world in WORLDS:
            world_rows = [row for row in selected if row["world_seed"] == world]
            mean = fmean(
                float(row["forward_transfer_gaussian_log_loss"])
                for row in world_rows
            )
            world_model_means[(world, model)] = mean
            world_means.append(mean)
            world_slopes.append(_slope(world_rows, "forward_transfer_gaussian_log_loss"))
        model_summaries.append(
            {
                "model": model,
                "mean_task_forward_transfer_by_world": _effect_summary(
                    world_means, 10100 + model_index
                ),
                "forward_transfer_positive_task_fraction": (
                    sum(
                        float(row["forward_transfer_gaussian_log_loss"]) > 0.0
                        for row in selected
                    )
                    / len(selected)
                ),
                "task_index_slope_by_world": _effect_summary(
                    world_slopes, 10200 + model_index
                ),
                "by_prior_primitive_exposure": _strata(
                    selected, "teacher_primitives_encountered_before_task"
                ),
                "by_prior_route_similarity": _strata(
                    selected, "maximum_prior_route_position_similarity"
                ),
            }
        )

    continuous_minus_dense = [
        world_model_means[(world, "continuous")]
        - world_model_means[(world, "dense")]
        for world in WORLDS
    ]
    return {
        "scope": "development worlds 0-9; same-architecture fresh-task baselines",
        "forward_transfer_definition": "fresh task prequential loss minus lifetime task prequential loss; positive is beneficial transfer",
        "replication_unit": "world",
        "model_summaries": model_summaries,
        "continuous_minus_dense_mean_forward_transfer": _effect_summary(
            continuous_minus_dense, 10300
        ),
        "task_index_bins": _task_index_bins(task_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--fresh",
        type=Path,
        default=Path("artifacts/forward_transfer/fresh"),
    )
    parser.add_argument(
        "--lifetimes",
        type=Path,
        default=Path("artifacts/checkpoints_development"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/forward_transfer/forward-transfer.json"),
    )
    args = parser.parse_args()
    report = summarize(args.config, args.fresh, args.lifetimes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["model_summaries"], indent=2))


if __name__ == "__main__":
    main()
