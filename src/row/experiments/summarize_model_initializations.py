"""Validate and summarize the two-initialization ROW development pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

from row.config import load_config
from row.experiments.learned_lifetime import resolved_learned_config
from row.experiments.summarize_robustness import _effect_summary
from row.experiments.sweep_model_initializations import SECOND_SEEDS, _seeded_config
from row.provenance import validate_artifact


WORLDS = tuple(range(10))
MODELS = ("continuous", "dense")


def _values(path: Path) -> dict[str, float]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return {
        "gaussian_log_loss": float(
            summary["cumulative_prequential_gaussian_log_loss"]
        ),
        "novel_32_shot_nmse": float(
            summary["novel_composition"]["nmse_by_support"]["32"]
        ),
    }


def summarize(
    config_path: Path, baseline_root: Path, second_root: Path
) -> dict[str, Any]:
    base = load_config(config_path)
    records: dict[tuple[int, int, str], dict[str, float]] = {}
    initialization_specs = (
        (0, {"continuous": base.continuous_model.seed, "dense": base.dense_model.seed}),
        (1, SECOND_SEEDS),
    )
    for initialization, seeds in initialization_specs:
        for world in WORLDS:
            for model in MODELS:
                if initialization == 0:
                    output = baseline_root / f"world_{world}" / model
                    config = replace(
                        base,
                        world=replace(base.world, seed=world),
                        output_directory=output,
                    )
                else:
                    output = second_root / f"world_{world}" / model
                    config = _seeded_config(
                        base, model, world, int(seeds[model]), output
                    )
                validate_artifact(
                    output,
                    resolved_learned_config(config, model, "forward"),
                    model,
                )
                records[(initialization, world, model)] = _values(output)

    initialization_summaries = []
    paired = []
    per_initialization_effects: dict[int, list[float]] = {}
    per_initialization_novel: dict[int, list[float]] = {}
    for initialization, seeds in initialization_specs:
        loss_effects = []
        novel_effects = []
        for world in WORLDS:
            continuous = records[(initialization, world, "continuous")]
            dense = records[(initialization, world, "dense")]
            loss = dense["gaussian_log_loss"] - continuous["gaussian_log_loss"]
            novel = dense["novel_32_shot_nmse"] - continuous["novel_32_shot_nmse"]
            loss_effects.append(loss)
            novel_effects.append(novel)
            paired.append(
                {
                    "initialization": initialization,
                    "world_seed": world,
                    "continuous_model_seed": int(seeds["continuous"]),
                    "dense_model_seed": int(seeds["dense"]),
                    "dense_minus_continuous_gaussian_log_loss": loss,
                    "dense_minus_continuous_novel_32_shot_nmse": novel,
                }
            )
        per_initialization_effects[initialization] = loss_effects
        per_initialization_novel[initialization] = novel_effects
        initialization_summaries.append(
            {
                "initialization": initialization,
                "continuous_model_seed": int(seeds["continuous"]),
                "dense_model_seed": int(seeds["dense"]),
                "lifetime_loss_effect": _effect_summary(
                    loss_effects, 9100 + initialization
                ),
                "novel_32_shot_effect": _effect_summary(
                    novel_effects, 9200 + initialization
                ),
            }
        )

    world_mean_loss = [
        fmean(
            per_initialization_effects[initialization][world]
            for initialization in (0, 1)
        )
        for world in WORLDS
    ]
    world_mean_novel = [
        fmean(
            per_initialization_novel[initialization][world]
            for initialization in (0, 1)
        )
        for world in WORLDS
    ]
    initialization_difference = [
        per_initialization_effects[1][world]
        - per_initialization_effects[0][world]
        for world in WORLDS
    ]
    return {
        "scope": "development worlds 0-9; two model initializations per architecture",
        "replication_unit": "world; initialization effects are averaged within world",
        "sign_convention": "positive Dense-minus-Continuous favors Continuous",
        "initialization_summaries": initialization_summaries,
        "world_averaged_across_initializations": {
            "lifetime_loss_effect": _effect_summary(world_mean_loss, 9300),
            "novel_32_shot_effect": _effect_summary(world_mean_novel, 9301),
        },
        "second_minus_first_initialization_loss_effect": {
            "mean": fmean(initialization_difference),
            "paired_by_world": initialization_difference,
        },
        "paired_world_effects": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("artifacts/checkpoints_development"),
    )
    parser.add_argument(
        "--second", type=Path, default=Path("artifacts/model_initializations")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/model_initializations/model-initializations.json"
        ),
    )
    args = parser.parse_args()
    report = summarize(args.config, args.baselines, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["initialization_summaries"], indent=2))


if __name__ == "__main__":
    main()
