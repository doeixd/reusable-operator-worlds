"""Run and aggregate replicated exact-reuse learning-to-learn checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

from row.config import load_config
from row.experiments.learned_lifetime import resolved_learned_config, run
from row.provenance import validate_artifact


CHECKPOINTS = (8, 16, 32, 64)


def _checkpoint_curve(summary: dict[str, Any]) -> dict[str, float]:
    checkpoints = summary["novel_composition_checkpoints"]
    observed = tuple(int(item["tasks_completed"]) for item in checkpoints)
    if observed != CHECKPOINTS:
        raise ValueError(f"checkpoint sequence {observed} does not match {CHECKPOINTS}")
    return {
        str(item["tasks_completed"]): float(item["mean_nmse_by_support"]["32"])
        for item in checkpoints
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    worlds = sorted({int(record["world_seed"]) for record in records})
    by_key = {
        (int(record["world_seed"]), str(record["model"])): record
        for record in records
    }
    for world in worlds:
        if {(world, "continuous"), (world, "dense")} - set(by_key):
            raise ValueError(f"world {world} is missing a paired checkpoint model")

    checkpoint_summaries = []
    for checkpoint in CHECKPOINTS:
        key = str(checkpoint)
        continuous = [
            float(by_key[(world, "continuous")]["checkpoint_32_shot_nmse"][key])
            for world in worlds
        ]
        dense = [
            float(by_key[(world, "dense")]["checkpoint_32_shot_nmse"][key])
            for world in worlds
        ]
        effects = [dense_value - continuous_value for continuous_value, dense_value in zip(continuous, dense, strict=True)]
        checkpoint_summaries.append(
            {
                "tasks_completed": checkpoint,
                "mean_continuous_32_shot_nmse": fmean(continuous),
                "mean_dense_c_32_shot_nmse": fmean(dense),
                "mean_dense_minus_continuous_32_shot_nmse": fmean(effects),
                "continuous_wins": sum(effect > 0 for effect in effects),
                "worlds": len(worlds),
                "paired_world_effects": [
                    {"world_seed": world, "dense_minus_continuous_32_shot_nmse": effect}
                    for world, effect in zip(worlds, effects, strict=True)
                ],
            }
        )

    learning_gain = {}
    for model in ("continuous", "dense"):
        ratios = []
        reductions = []
        for world in worlds:
            curve = by_key[(world, model)]["checkpoint_32_shot_nmse"]
            early = float(curve["8"])
            late = float(curve["64"])
            ratios.append(early / late)
            reductions.append(early - late)
        learning_gain[model] = {
            "mean_8_to_64_ratio": fmean(ratios),
            "mean_absolute_reduction": fmean(reductions),
            "improves_in_all_worlds": all(reduction > 0 for reduction in reductions),
            "per_world_ratio": [
                {"world_seed": world, "ratio": ratio}
                for world, ratio in zip(worlds, ratios, strict=True)
            ],
        }

    return {
        "scope": f"exact-reuse development worlds {worlds}; descriptive",
        "primary_metric": "mean novel-composition NMSE after 32 code-only examples",
        "records": sorted(records, key=lambda row: (int(row["world_seed"]), str(row["model"]))),
        "checkpoint_summaries": checkpoint_summaries,
        "learning_gain": learning_gain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/checkpoints_development")
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--models",
        choices=("continuous", "dense"),
        nargs="+",
        default=["continuous", "dense"],
    )
    args = parser.parse_args()

    base = load_config(args.config)
    if base.world.reuse_rho != 1.0:
        raise ValueError("checkpoint sweep requires exact reuse (rho=1)")
    if base.evaluation.lifetime_checkpoints != CHECKPOINTS:
        raise ValueError("config does not contain the frozen checkpoint sequence")
    if base.evaluation.checkpoint_novel_tasks != 4:
        raise ValueError("checkpoint sweep requires four fixed novel compositions")

    records: list[dict[str, Any]] = []
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
            )
            summary_path = output / "summary.json"
            print(f"[{completed}/{total}] world={world_seed} model={model}", flush=True)
            if summary_path.exists():
                validate_artifact(
                    output, resolved_learned_config(config, model, "forward"), model
                )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                summary = run(config, kind=model)
                gc.collect()
            records.append(
                {
                    "world_seed": world_seed,
                    "model": model,
                    "checkpoint_32_shot_nmse": _checkpoint_curve(summary),
                    "cumulative_prequential_gaussian_log_loss": summary[
                        "cumulative_prequential_gaussian_log_loss"
                    ],
                }
            )

    report = _aggregate(records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sweep.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["checkpoint_summaries"], indent=2), flush=True)
    print(json.dumps(report["learning_gain"], indent=2), flush=True)


if __name__ == "__main__":
    main()
