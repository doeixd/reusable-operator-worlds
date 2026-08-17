"""Validate and summarize the generic-hypernetwork and Dense-24 controls."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

from row.config import ExperimentConfig, load_config
from row.experiments.learned_lifetime import resolved_learned_config
from row.experiments.tune_development import _configured_run, _label
from row.provenance import validate_artifact


DEVELOPMENT_WORLDS = tuple(range(10))
STAGE_ONE_WORLDS = (0, 1, 2)
STAGE_TWO_WORLDS = tuple(range(3, 10))


def _summary_row(output: Path, world: int, model: str) -> dict[str, Any]:
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    return {
        "world_seed": world,
        "model": model,
        "gaussian_log_loss": float(
            summary["cumulative_prequential_gaussian_log_loss"]
        ),
        "novel_32_shot_nmse": float(
            summary["novel_composition"]["nmse_by_support"]["32"]
        ),
        "final_median_nmse": float(summary["final_nmse"]["median"]),
        "shared_parameter_count": int(summary["shared_parameter_count"]),
        "task_state_scalar_count": int(summary["task_state_scalar_count"]),
        "compute_accounting": summary["compute_accounting"],
    }


def _collect_tuned(
    base: ExperimentConfig,
    root: Path,
    model: str,
    worlds: tuple[int, ...],
    global_lrs: tuple[float, ...],
    task_lrs: tuple[float, ...],
    *,
    dense_task_embedding_dim: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fast_base = replace(
        base,
        evaluation=replace(
            base.evaluation,
            lifetime_checkpoints=(),
            checkpoint_novel_tasks=1,
            extended_diagnostics=False,
        ),
    )
    rows = []
    configurations = []
    for global_lr in global_lrs:
        for task_lr in task_lrs:
            selected = []
            for world in worlds:
                output = (
                    root
                    / model
                    / f"global_{_label(global_lr)}_task_{_label(task_lr)}"
                    / f"world_{world}"
                )
                config = _configured_run(
                    fast_base,
                    model,
                    world,
                    global_lr,
                    task_lr,
                    output,
                    dense_hidden_width=32 if model == "dense" else None,
                    dense_task_embedding_dim=dense_task_embedding_dim,
                )
                validate_artifact(
                    output,
                    resolved_learned_config(config, model, "forward"),
                    model,
                )
                row = _summary_row(output, world, model)
                row.update(
                    {
                        "global_learning_rate": global_lr,
                        "task_learning_rate": task_lr,
                    }
                )
                rows.append(row)
                selected.append(row)
            configurations.append(
                {
                    "model": model,
                    "global_learning_rate": global_lr,
                    "task_learning_rate": task_lr,
                    "worlds": list(worlds),
                    "mean_gaussian_log_loss": fmean(
                        row["gaussian_log_loss"] for row in selected
                    ),
                    "mean_novel_32_shot_nmse": fmean(
                        row["novel_32_shot_nmse"] for row in selected
                    ),
                }
            )
    return rows, configurations


def _collect_baseline(
    base: ExperimentConfig, root: Path, model: str
) -> list[dict[str, Any]]:
    rows = []
    for world in DEVELOPMENT_WORLDS:
        output = root / f"world_{world}" / model
        config = replace(
            base,
            world=replace(base.world, seed=world),
            output_directory=output,
        )
        validate_artifact(
            output,
            resolved_learned_config(config, model, "forward"),
            model,
        )
        rows.append(_summary_row(output, world, model))
    return rows


def _by_world(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(row["world_seed"]): row for row in rows}
    if tuple(sorted(result)) != DEVELOPMENT_WORLDS:
        raise ValueError(f"expected development worlds {DEVELOPMENT_WORLDS}")
    return result


def _effect_summary(
    rows: list[dict[str, float | int]],
    loss_key: str,
    novel_key: str,
) -> dict[str, Any]:
    loss = [float(row[loss_key]) for row in rows]
    novel = [float(row[novel_key]) for row in rows]
    return {
        "worlds": len(rows),
        "loss_wins": sum(value > 0.0 for value in loss),
        "mean_gaussian_log_loss_advantage": fmean(loss),
        "minimum_gaussian_log_loss_advantage": min(loss),
        "maximum_gaussian_log_loss_advantage": max(loss),
        "novel_32_shot_wins": sum(value > 0.0 for value in novel),
        "mean_novel_32_shot_nmse_advantage": fmean(novel),
    }


def summarize(
    config_path: Path,
    hyper_stage_one_root: Path,
    hyper_stage_two_root: Path,
    dense24_root: Path,
    baseline_root: Path,
) -> dict[str, Any]:
    base = load_config(config_path)
    hyper_stage_one_rows, hyper_stage_one = _collect_tuned(
        base,
        hyper_stage_one_root,
        "hypernetwork",
        STAGE_ONE_WORLDS,
        (3e-4, 1e-3, 3e-3),
        (5e-3, 5e-2),
    )
    hyper_stage_two_rows, hyper_stage_two = _collect_tuned(
        base,
        hyper_stage_two_root,
        "hypernetwork",
        STAGE_TWO_WORLDS,
        (1e-3, 3e-3),
        (5e-2,),
    )
    dense24_rows, dense24_configs = _collect_tuned(
        base,
        dense24_root,
        "dense",
        DEVELOPMENT_WORLDS,
        (1e-3,),
        (5e-2,),
        dense_task_embedding_dim=24,
    )
    continuous_rows = _collect_baseline(base, baseline_root, "continuous")
    dense32_rows = _collect_baseline(base, baseline_root, "dense")

    stage_one_winner = min(
        hyper_stage_one, key=lambda row: float(row["mean_gaussian_log_loss"])
    )
    stage_two_winner = min(
        hyper_stage_two, key=lambda row: float(row["mean_gaussian_log_loss"])
    )
    selected_global_lr = float(stage_two_winner["global_learning_rate"])
    selected_task_lr = float(stage_two_winner["task_learning_rate"])
    if (
        selected_global_lr != float(stage_one_winner["global_learning_rate"])
        or selected_task_lr != float(stage_one_winner["task_learning_rate"])
    ):
        raise ValueError("hypernetwork stage-one and stage-two winners disagree")
    hyper_rows = [
        row
        for row in hyper_stage_one_rows + hyper_stage_two_rows
        if float(row["global_learning_rate"]) == selected_global_lr
        and float(row["task_learning_rate"]) == selected_task_lr
    ]

    hyper = _by_world(hyper_rows)
    continuous = _by_world(continuous_rows)
    dense32 = _by_world(dense32_rows)
    dense24 = _by_world(dense24_rows)
    paired = []
    for world in DEVELOPMENT_WORLDS:
        paired.append(
            {
                "world_seed": world,
                "continuous_advantage_over_hyper_loss": (
                    hyper[world]["gaussian_log_loss"]
                    - continuous[world]["gaussian_log_loss"]
                ),
                "continuous_advantage_over_hyper_novel_32": (
                    hyper[world]["novel_32_shot_nmse"]
                    - continuous[world]["novel_32_shot_nmse"]
                ),
                "hyper_advantage_over_dense32_loss": (
                    dense32[world]["gaussian_log_loss"]
                    - hyper[world]["gaussian_log_loss"]
                ),
                "hyper_advantage_over_dense32_novel_32": (
                    dense32[world]["novel_32_shot_nmse"]
                    - hyper[world]["novel_32_shot_nmse"]
                ),
                "dense24_advantage_over_dense32_loss": (
                    dense32[world]["gaussian_log_loss"]
                    - dense24[world]["gaussian_log_loss"]
                ),
                "dense24_advantage_over_dense32_novel_32": (
                    dense32[world]["novel_32_shot_nmse"]
                    - dense24[world]["novel_32_shot_nmse"]
                ),
            }
        )

    return {
        "scope": "development worlds 0-9; exact reuse; descriptive, not confirmatory",
        "sign_convention": "positive named-model advantage favors the first named model",
        "hypernetwork_selection": {
            "stage_one": hyper_stage_one,
            "stage_two": hyper_stage_two,
            "selected_global_learning_rate": selected_global_lr,
            "selected_task_learning_rate": selected_task_lr,
        },
        "architecture_accounting": {
            "continuous": {
                key: continuous[0][key]
                for key in (
                    "shared_parameter_count",
                    "task_state_scalar_count",
                    "compute_accounting",
                )
            },
            "hypernetwork": {
                key: hyper[0][key]
                for key in (
                    "shared_parameter_count",
                    "task_state_scalar_count",
                    "compute_accounting",
                )
            },
            "dense24": {
                key: dense24[0][key]
                for key in (
                    "shared_parameter_count",
                    "task_state_scalar_count",
                    "compute_accounting",
                )
            },
            "dense32": {
                key: dense32[0][key]
                for key in (
                    "shared_parameter_count",
                    "task_state_scalar_count",
                    "compute_accounting",
                )
            },
        },
        "comparisons": {
            "continuous_over_hypernetwork": _effect_summary(
                paired,
                "continuous_advantage_over_hyper_loss",
                "continuous_advantage_over_hyper_novel_32",
            ),
            "hypernetwork_over_dense32": _effect_summary(
                paired,
                "hyper_advantage_over_dense32_loss",
                "hyper_advantage_over_dense32_novel_32",
            ),
            "dense24_over_dense32": _effect_summary(
                paired,
                "dense24_advantage_over_dense32_loss",
                "dense24_advantage_over_dense32_novel_32",
            ),
        },
        "dense24_configuration": dense24_configs[0],
        "paired_world_effects": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--hyper-stage-one",
        type=Path,
        default=Path("artifacts/tuning/hypernetwork_stage1"),
    )
    parser.add_argument(
        "--hyper-stage-two",
        type=Path,
        default=Path("artifacts/tuning/hypernetwork_stage2"),
    )
    parser.add_argument(
        "--dense24",
        type=Path,
        default=Path("artifacts/controls/dense24_worlds0_9"),
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("artifacts/checkpoints_development"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/structural_controls/structural-controls.json"),
    )
    args = parser.parse_args()
    report = summarize(
        args.config,
        args.hyper_stage_one,
        args.hyper_stage_two,
        args.dense24,
        args.baselines,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["comparisons"], indent=2))


if __name__ == "__main__":
    main()
