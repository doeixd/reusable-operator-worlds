"""Ablate the effective online update batch under paired 1:1 replay."""

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


BATCH_SIZES = (2, 8)
MODELS = ("continuous", "dense")


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    worlds = sorted({int(record["world_seed"]) for record in records})
    by_key = {
        (int(record["world_seed"]), str(record["model"]), int(record["batch_size"])): record
        for record in records
    }
    expected = {
        (world, model, batch_size)
        for world in worlds
        for model in MODELS
        for batch_size in BATCH_SIZES
    }
    if set(by_key) != expected:
        raise ValueError("batch-size sweep is not fully paired")

    architecture_effects = []
    for batch_size in BATCH_SIZES:
        loss_effects = [
            float(by_key[(world, "dense", batch_size)]["gaussian_log_loss"])
            - float(by_key[(world, "continuous", batch_size)]["gaussian_log_loss"])
            for world in worlds
        ]
        novel_effects = [
            float(by_key[(world, "dense", batch_size)]["novel_32_shot_nmse"])
            - float(by_key[(world, "continuous", batch_size)]["novel_32_shot_nmse"])
            for world in worlds
        ]
        architecture_effects.append(
            {
                "batch_size": batch_size,
                "mean_dense_minus_continuous_gaussian_log_loss": fmean(loss_effects),
                "continuous_lifetime_wins": sum(effect > 0 for effect in loss_effects),
                "mean_dense_minus_continuous_novel_32_shot_nmse": fmean(novel_effects),
                "continuous_novel_wins": sum(effect > 0 for effect in novel_effects),
                "worlds": len(worlds),
                "per_world_loss_effect": [
                    {"world_seed": world, "value": effect}
                    for world, effect in zip(worlds, loss_effects, strict=True)
                ],
                "per_world_novel_effect": [
                    {"world_seed": world, "value": effect}
                    for world, effect in zip(worlds, novel_effects, strict=True)
                ],
            }
        )

    size_effects = []
    for model in MODELS:
        loss_changes = [
            float(by_key[(world, model, 8)]["gaussian_log_loss"])
            - float(by_key[(world, model, 2)]["gaussian_log_loss"])
            for world in worlds
        ]
        novel_changes = [
            float(by_key[(world, model, 8)]["novel_32_shot_nmse"])
            - float(by_key[(world, model, 2)]["novel_32_shot_nmse"])
            for world in worlds
        ]
        size_effects.append(
            {
                "model": model,
                "mean_batch8_minus_batch2_gaussian_log_loss": fmean(loss_changes),
                "batch8_lifetime_improves": sum(change < 0 for change in loss_changes),
                "mean_batch8_minus_batch2_novel_32_shot_nmse": fmean(novel_changes),
                "batch8_novel_improves": sum(change < 0 for change in novel_changes),
                "worlds": len(worlds),
                "per_world_loss_change": [
                    {"world_seed": world, "value": change}
                    for world, change in zip(worlds, loss_changes, strict=True)
                ],
                "per_world_novel_change": [
                    {"world_seed": world, "value": change}
                    for world, change in zip(worlds, novel_changes, strict=True)
                ],
            }
        )

    return {
        "scope": f"exact-reuse development worlds {worlds}; fixed selected learning rates",
        "protocol": (
            "Batch 2 uses one current plus one replay example. Batch 8 uses the "
            "new current example, up to three prior current-task examples, and "
            "four replay examples. Sampling is paired across models."
        ),
        "records": sorted(
            records,
            key=lambda row: (
                int(row["world_seed"]),
                int(row["batch_size"]),
                str(row["model"]),
            ),
        ),
        "architecture_effects": architecture_effects,
        "batch_size_effects": size_effects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/batch_sizes")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/batch_sizes/batch-sizes.json"),
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    parser.add_argument("--batch-sizes", type=int, choices=BATCH_SIZES, nargs="+", default=list(BATCH_SIZES))
    args = parser.parse_args()

    base = load_config(args.config)
    base = replace(
        base,
        evaluation=replace(
            base.evaluation,
            lifetime_checkpoints=(),
            checkpoint_novel_tasks=1,
            extended_diagnostics=False,
        ),
    )
    records = []
    total = len(args.worlds) * len(args.models) * len(args.batch_sizes)
    completed = 0
    for world_seed in args.worlds:
        for batch_size in args.batch_sizes:
            for model in args.models:
                completed += 1
                output = (
                    args.output
                    / f"world_{world_seed}"
                    / f"batch_{batch_size}"
                    / model
                )
                config = replace(
                    base,
                    world=replace(base.world, seed=world_seed),
                    output_directory=output,
                )
                summary_path = output / "summary.json"
                print(
                    f"[{completed}/{total}] world={world_seed} batch={batch_size} model={model}",
                    flush=True,
                )
                if summary_path.exists():
                    validate_artifact(
                        output,
                        resolved_learned_config(
                            config,
                            model,
                            "forward",
                            update_batch_size=batch_size,
                        ),
                        model,
                    )
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                else:
                    summary = run(config, kind=model, update_batch_size=batch_size)
                    gc.collect()
                novel = summary["novel_composition"]["nmse_by_support"]
                records.append(
                    {
                        "world_seed": world_seed,
                        "model": model,
                        "batch_size": batch_size,
                        "gaussian_log_loss": summary[
                            "cumulative_prequential_gaussian_log_loss"
                        ],
                        "novel_32_shot_nmse": novel["32"],
                        "update_batch": summary["update_batch"],
                    }
                )

    report = _aggregate(records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["architecture_effects"], indent=2), flush=True)
    print(json.dumps(report["batch_size_effects"], indent=2), flush=True)


if __name__ == "__main__":
    main()
