"""Run a resumable paired recurrence sweep on development worlds."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

from row.config import load_config
from row.experiments.learned_lifetime import run


def _rho_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _validated_summary(path: Path, model: str, rho: float) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary["model"] != model:
        raise ValueError(f"{path} reports model={summary['model']}, expected {model}")
    actual_rho = float(summary["world_functional_reuse"]["configured_rho"])
    if abs(actual_rho - rho) > 1e-12:
        raise ValueError(f"{path} reports rho={actual_rho}, expected {rho}")
    return summary


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    rho_summaries = []
    for rho in sorted({float(record["configured_rho"]) for record in records}):
        selected = [record for record in records if record["configured_rho"] == rho]
        worlds = sorted({int(record["world_seed"]) for record in selected})
        paired_effects = []
        novel_effects = []
        for world in worlds:
            by_model = {
                str(record["model"]): record
                for record in selected
                if record["world_seed"] == world
            }
            if set(by_model) != {"continuous", "dense"}:
                continue
            paired_effects.append(
                float(by_model["dense"]["gaussian_log_loss"])
                - float(by_model["continuous"]["gaussian_log_loss"])
            )
            novel_effects.append(
                float(by_model["dense"]["novel_32_shot_nmse"])
                - float(by_model["continuous"]["novel_32_shot_nmse"])
            )
        rho_summaries.append(
            {
                "configured_rho": rho,
                "worlds": worlds,
                "paired_worlds": len(paired_effects),
                "mean_measured_residual_correlation": fmean(
                    float(record["measured_residual_correlation"])
                    for record in selected
                    if record["model"] == "continuous"
                ),
                "mean_dense_minus_continuous_gaussian_log_loss": (
                    fmean(paired_effects) if paired_effects else None
                ),
                "continuous_wins": sum(effect > 0 for effect in paired_effects),
                "mean_dense_minus_continuous_novel_32_shot_nmse": (
                    fmean(novel_effects) if novel_effects else None
                ),
            }
        )
    return {
        "sign_convention": "positive Dense-minus-Continuous favors Continuous",
        "records": sorted(
            records,
            key=lambda row: (
                float(row["configured_rho"]),
                int(row["world_seed"]),
                str(row["model"]),
            ),
        ),
        "rho_summaries": rho_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/rho_development")
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--rhos", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 0.9, 1]
    )
    parser.add_argument(
        "--models",
        choices=("continuous", "dense"),
        nargs="+",
        default=["continuous", "dense"],
    )
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
    records: list[dict[str, Any]] = []
    total = len(args.worlds) * len(args.rhos) * len(args.models)
    completed = 0
    for world_seed in args.worlds:
        for rho in args.rhos:
            for model in args.models:
                completed += 1
                output = (
                    args.output
                    / f"rho_{_rho_label(rho)}"
                    / f"world_{world_seed}"
                    / model
                )
                summary_path = output / "summary.json"
                print(
                    f"[{completed}/{total}] world={world_seed} rho={rho:g} model={model}",
                    flush=True,
                )
                if summary_path.exists():
                    summary = _validated_summary(summary_path, model, rho)
                else:
                    config = replace(
                        base,
                        world=replace(base.world, seed=world_seed, reuse_rho=rho),
                        output_directory=output,
                    )
                    summary = run(config, kind=model)
                    gc.collect()
                records.append(
                    {
                        "world_seed": world_seed,
                        "configured_rho": rho,
                        "model": model,
                        "measured_residual_correlation": summary[
                            "world_functional_reuse"
                        ]["mean_pairwise_residual_correlation"],
                        "gaussian_log_loss": summary[
                            "cumulative_prequential_gaussian_log_loss"
                        ],
                        "final_median_nmse": summary["final_nmse"]["median"],
                        "novel_32_shot_nmse": summary["novel_composition"][
                            "nmse_by_support"
                        ]["32"],
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    report = _aggregate(records)
    (args.output / "sweep.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["rho_summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
