"""Validate and summarize corrected-architecture stage-two tuning artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from row.config import load_config


DEVELOPMENT_WORLDS = tuple(range(3, 10))
DEFAULT_CANDIDATES = (
    ("continuous", 0.001, Path("artifacts/tuning/stage2_current/continuous_1em03")),
    ("continuous", 0.003, Path("artifacts/tuning/stage2_current/continuous_3em03")),
    ("dense", 0.001, Path("artifacts/tuning/stage2_current/dense_1em03")),
    ("dense", 0.003, Path("artifacts/tuning/stage2_current/dense_3em03")),
)


def _validated_candidate(
    model: str, global_lr: float, artifact: Path
) -> dict[str, Any]:
    report = json.loads((artifact / "ranking.json").read_text(encoding="utf-8"))
    runs = report["runs"]
    worlds = tuple(sorted(int(run["world_seed"]) for run in runs))
    if worlds != DEVELOPMENT_WORLDS:
        raise ValueError(
            f"{artifact} contains worlds {worlds}; expected {DEVELOPMENT_WORLDS}"
        )
    if any(str(run["model"]) != model for run in runs):
        raise ValueError(f"{artifact} contains a mismatched model")
    if any(abs(float(run["global_learning_rate"]) - global_lr) > 1e-12 for run in runs):
        raise ValueError(f"{artifact} contains a mismatched global learning rate")
    if any(abs(float(run["task_learning_rate"]) - 0.05) > 1e-12 for run in runs):
        raise ValueError(f"{artifact} contains a mismatched task learning rate")

    normalized_runs = [
        {
            "world_seed": int(run["world_seed"]),
            "gaussian_log_loss": float(run["gaussian_log_loss"]),
            "novel_32_shot_nmse": float(run["novel_32_shot_nmse"]),
            "final_median_nmse": float(run["final_median_nmse"]),
        }
        for run in sorted(runs, key=lambda item: int(item["world_seed"]))
    ]
    return {
        "model": model,
        "global_learning_rate": global_lr,
        "task_learning_rate": 0.05,
        "worlds": list(DEVELOPMENT_WORLDS),
        "mean_gaussian_log_loss": fmean(
            run["gaussian_log_loss"] for run in normalized_runs
        ),
        "mean_novel_32_shot_nmse": fmean(
            run["novel_32_shot_nmse"] for run in normalized_runs
        ),
        "runs": normalized_runs,
    }


def summarize(
    candidates: list[tuple[str, float, Path]], config_path: Path
) -> dict[str, Any]:
    rows = [_validated_candidate(*candidate) for candidate in candidates]
    selected: dict[str, dict[str, Any]] = {}
    for model in ("continuous", "dense"):
        model_rows = [row for row in rows if row["model"] == model]
        if len(model_rows) != 2:
            raise ValueError(f"expected two {model} finalists, found {len(model_rows)}")
        selected[model] = min(
            model_rows, key=lambda row: float(row["mean_gaussian_log_loss"])
        )

    config = load_config(config_path)
    frozen = {
        "continuous": {
            "global_learning_rate": config.continuous_model.global_learning_rate,
            "task_learning_rate": config.continuous_model.task_learning_rate,
        },
        "dense": {
            "global_learning_rate": config.dense_model.global_learning_rate,
            "task_learning_rate": config.dense_model.task_learning_rate,
        },
    }
    for model, winner in selected.items():
        for key in ("global_learning_rate", "task_learning_rate"):
            if abs(float(winner[key]) - float(frozen[model][key])) > 1e-12:
                raise ValueError(
                    f"{config_path} does not freeze selected {model} {key}"
                )

    continuous_by_world = {
        run["world_seed"]: run for run in selected["continuous"]["runs"]
    }
    dense_by_world = {run["world_seed"]: run for run in selected["dense"]["runs"]}
    paired = []
    for world in DEVELOPMENT_WORLDS:
        continuous = continuous_by_world[world]
        dense = dense_by_world[world]
        paired.append(
            {
                "world_seed": world,
                "dense_minus_continuous_gaussian_log_loss": (
                    dense["gaussian_log_loss"] - continuous["gaussian_log_loss"]
                ),
                "dense_minus_continuous_novel_32_shot_nmse": (
                    dense["novel_32_shot_nmse"] - continuous["novel_32_shot_nmse"]
                ),
            }
        )
    effects = [row["dense_minus_continuous_gaussian_log_loss"] for row in paired]
    novel_effects = [
        row["dense_minus_continuous_novel_32_shot_nmse"] for row in paired
    ]
    return {
        "scope": "development worlds 3-9; corrected learnable-alpha architecture",
        "selection_criterion": "lowest mean cumulative prequential Gaussian log loss",
        "candidates": sorted(
            [
                {key: value for key, value in row.items() if key != "runs"}
                for row in rows
            ],
            key=lambda row: (str(row["model"]), float(row["global_learning_rate"])),
        ),
        "selected": {
            model: {key: value for key, value in row.items() if key != "runs"}
            for model, row in selected.items()
        },
        "frozen_config": frozen,
        "selected_model_comparison": {
            "sign_convention": "positive Dense-minus-Continuous favors Continuous",
            "continuous_wins": sum(effect > 0 for effect in effects),
            "worlds": len(effects),
            "mean_gaussian_log_loss_advantage": fmean(effects),
            "minimum_gaussian_log_loss_advantage": min(effects),
            "maximum_gaussian_log_loss_advantage": max(effects),
            "mean_novel_32_shot_nmse_advantage": fmean(novel_effects),
            "paired_world_effects": paired,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("MODEL", "GLOBAL_LR", "ARTIFACT"),
        help="candidate root; repeat four times to replace canonical inputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/stage2_current/selection.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    candidates = (
        [(model, float(global_lr), Path(path)) for model, global_lr, path in args.candidate]
        if args.candidate
        else list(DEFAULT_CANDIDATES)
    )
    report = summarize(candidates, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
