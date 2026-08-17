"""Validate and summarize task-order and replay robustness artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

from row.config import load_config
from row.experiments.learned_lifetime import resolved_learned_config
from row.experiments.sweep_robustness import CONDITIONS, _condition_config
from row.provenance import validate_artifact


WORLDS = tuple(range(10))
MODELS = ("continuous", "dense")


def _bootstrap_mean_ci(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(10_000, len(array)), replace=True)
    return [float(value) for value in np.quantile(np.mean(samples, axis=1), (0.025, 0.975))]


def _effect_summary(values: list[float], seed: int) -> dict[str, Any]:
    return {
        "wins": sum(value > 0.0 for value in values),
        "worlds": len(values),
        "mean": fmean(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
        "bootstrap_95_percent_ci_of_mean": _bootstrap_mean_ci(values, seed),
    }


def _summary_values(path: Path) -> dict[str, float]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return {
        "gaussian_log_loss": float(
            summary["cumulative_prequential_gaussian_log_loss"]
        ),
        "novel_32_shot_nmse": float(
            summary["novel_composition"]["nmse_by_support"]["32"]
        ),
        "final_median_nmse": float(summary["final_nmse"]["median"]),
    }


def summarize(
    config_path: Path, robustness_root: Path, baseline_root: Path
) -> dict[str, Any]:
    base = load_config(config_path)
    records: dict[tuple[str, int, str], dict[str, float]] = {}
    for condition in CONDITIONS:
        for world in WORLDS:
            for model in MODELS:
                output = robustness_root / condition / f"world_{world}" / model
                config, order = _condition_config(
                    base, condition, model, world, output
                )
                validate_artifact(
                    output,
                    resolved_learned_config(config, model, order),
                    model,
                )
                records[(condition, world, model)] = _summary_values(output)

    for world in WORLDS:
        for model in MODELS:
            output = baseline_root / f"world_{world}" / model
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
            records[("replay1", world, model)] = _summary_values(output)

    condition_summaries = []
    paired_worlds = []
    condition_order = ("replay0", "replay1", "replay4", "reverse")
    for condition_index, condition in enumerate(condition_order):
        loss_effects = []
        novel_effects = []
        for world in WORLDS:
            continuous = records[(condition, world, "continuous")]
            dense = records[(condition, world, "dense")]
            loss_effect = dense["gaussian_log_loss"] - continuous["gaussian_log_loss"]
            novel_effect = dense["novel_32_shot_nmse"] - continuous["novel_32_shot_nmse"]
            loss_effects.append(loss_effect)
            novel_effects.append(novel_effect)
            paired_worlds.append(
                {
                    "condition": condition,
                    "world_seed": world,
                    "dense_minus_continuous_gaussian_log_loss": loss_effect,
                    "dense_minus_continuous_novel_32_shot_nmse": novel_effect,
                    "continuous_gaussian_log_loss": continuous["gaussian_log_loss"],
                    "dense_gaussian_log_loss": dense["gaussian_log_loss"],
                }
            )
        order, replay_ratio = (
            CONDITIONS[condition] if condition in CONDITIONS else ("forward", 1.0)
        )
        condition_summaries.append(
            {
                "condition": condition,
                "order": order,
                "replay_ratio": replay_ratio,
                "lifetime_loss_effect": _effect_summary(
                    loss_effects, 8100 + condition_index
                ),
                "novel_32_shot_effect": _effect_summary(
                    novel_effects, 8200 + condition_index
                ),
            }
        )

    order_sensitivity = {}
    for model_index, model in enumerate(MODELS):
        values = [
            records[("reverse", world, model)]["gaussian_log_loss"]
            - records[("replay1", world, model)]["gaussian_log_loss"]
            for world in WORLDS
        ]
        order_sensitivity[model] = {
            "sign_convention": "positive means reverse order has higher/worse loss",
            **_effect_summary(values, 8300 + model_index),
            "paired_reverse_minus_forward": values,
        }

    return {
        "scope": "development worlds 0-9; selected exact-reuse models",
        "primary_sign_convention": "positive Dense-minus-Continuous favors Continuous",
        "condition_summaries": condition_summaries,
        "order_sensitivity": order_sensitivity,
        "paired_world_effects": paired_worlds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--robustness", type=Path, default=Path("artifacts/robustness")
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("artifacts/checkpoints_development"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/robustness/robustness.json"),
    )
    args = parser.parse_args()
    report = summarize(args.config, args.robustness, args.baselines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["condition_summaries"], indent=2))


if __name__ == "__main__":
    main()
