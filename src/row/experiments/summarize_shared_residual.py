"""Compare the shared-parent residual control with the fixed-model envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt


def _baseline_records(
    seed_zero: dict[str, Any], development: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    for point in seed_zero["points"]:
        records.extend(
            [
                {
                    "world_seed": 0,
                    "configured_rho": float(point["configured_rho"]),
                    "model": "continuous",
                    "gaussian_log_loss": float(
                        point["continuous_prequential_gaussian_log_loss"]
                    ),
                    "novel_32_shot_nmse": float(
                        point["continuous_novel_32_shot_nmse"]
                    ),
                },
                {
                    "world_seed": 0,
                    "configured_rho": float(point["configured_rho"]),
                    "model": "dense",
                    "gaussian_log_loss": float(
                        point["dense_c_prequential_gaussian_log_loss"]
                    ),
                    "novel_32_shot_nmse": float(
                        point["dense_c_novel_32_shot_nmse"]
                    ),
                },
            ]
        )
    records.extend(development["records"])
    return records


def summarize(
    shared: dict[str, Any], baseline_records: list[dict[str, Any]]
) -> dict[str, Any]:
    baselines: dict[tuple[int, float], dict[str, dict[str, Any]]] = {}
    for record in baseline_records:
        key = (int(record["world_seed"]), float(record["configured_rho"]))
        baselines.setdefault(key, {})[str(record["model"])] = record

    comparisons = []
    for record in shared["records"]:
        world = int(record["world_seed"])
        rho = float(record["configured_rho"])
        by_model = baselines.get((world, rho), {})
        if set(by_model) != {"continuous", "dense"}:
            raise ValueError(f"missing fixed-model pair for world={world}, rho={rho}")
        continuous = by_model["continuous"]
        dense = by_model["dense"]
        best_loss_model = min(
            ("continuous", "dense"),
            key=lambda model: float(by_model[model]["gaussian_log_loss"]),
        )
        best_novel_model = min(
            ("continuous", "dense"),
            key=lambda model: float(by_model[model]["novel_32_shot_nmse"]),
        )
        shared_loss = float(record["gaussian_log_loss"])
        shared_novel = float(record["novel_32_shot_nmse"])
        comparisons.append(
            {
                **record,
                "continuous_gaussian_log_loss": float(
                    continuous["gaussian_log_loss"]
                ),
                "dense_gaussian_log_loss": float(dense["gaussian_log_loss"]),
                "best_fixed_loss_model": best_loss_model,
                "best_fixed_minus_shared_gaussian_log_loss": float(
                    by_model[best_loss_model]["gaussian_log_loss"]
                )
                - shared_loss,
                "continuous_minus_shared_gaussian_log_loss": float(
                    continuous["gaussian_log_loss"]
                )
                - shared_loss,
                "dense_minus_shared_gaussian_log_loss": float(
                    dense["gaussian_log_loss"]
                )
                - shared_loss,
                "best_fixed_novel_model": best_novel_model,
                "best_fixed_minus_shared_novel_32_shot_nmse": float(
                    by_model[best_novel_model]["novel_32_shot_nmse"]
                )
                - shared_novel,
            }
        )

    worlds = sorted({int(row["world_seed"]) for row in comparisons})
    rhos = sorted({float(row["configured_rho"]) for row in comparisons})
    expected = {(world, rho) for world in worlds for rho in rhos}
    observed = {
        (int(row["world_seed"]), float(row["configured_rho"]))
        for row in comparisons
    }
    if observed != expected:
        raise ValueError("shared-residual sweep is not rectangular")

    rho_summaries = []
    for rho in rhos:
        rows = [row for row in comparisons if row["configured_rho"] == rho]
        rho_summaries.append(
            {
                "configured_rho": rho,
                "mean_measured_residual_correlation": fmean(
                    float(row["measured_residual_correlation"]) for row in rows
                ),
                "mean_best_fixed_minus_shared_gaussian_log_loss": fmean(
                    float(row["best_fixed_minus_shared_gaussian_log_loss"])
                    for row in rows
                ),
                "shared_loss_wins_over_fixed_envelope": sum(
                    float(row["best_fixed_minus_shared_gaussian_log_loss"]) > 0
                    for row in rows
                ),
                "mean_best_fixed_minus_shared_novel_32_shot_nmse": fmean(
                    float(row["best_fixed_minus_shared_novel_32_shot_nmse"])
                    for row in rows
                ),
                "shared_novel_wins_over_fixed_envelope": sum(
                    float(row["best_fixed_minus_shared_novel_32_shot_nmse"]) > 0
                    for row in rows
                ),
                "mean_functional_ratio": fmean(
                    float(row["mean_functional_ratio"]) for row in rows
                ),
                "mean_maximum_task_functional_ratio": fmean(
                    float(row["maximum_task_functional_ratio"]) for row in rows
                ),
                "world_effects": rows,
            }
        )

    low_rho = min(rhos)
    high_rho = max(rhos)
    low_by_world = {
        int(row["world_seed"]): row
        for row in comparisons
        if row["configured_rho"] == low_rho
    }
    high_by_world = {
        int(row["world_seed"]): row
        for row in comparisons
        if row["configured_rho"] == high_rho
    }
    return {
        "scope": shared["scope"],
        "sign_convention": (
            "positive fixed-envelope-minus-shared means the shared residual "
            "model has lower loss or NMSE"
        ),
        "selected_configuration": shared["selected_configuration"],
        "escape_hatch_max_ratio": shared["escape_hatch_max_ratio"],
        "worlds": worlds,
        "rho_summaries": rho_summaries,
        "comparisons": comparisons,
        "predictions": {
            "residual_ratio_lower_at_high_reuse_worlds": sum(
                float(high_by_world[world]["mean_functional_ratio"])
                < float(low_by_world[world]["mean_functional_ratio"])
                for world in worlds
            ),
            "world_count": len(worlds),
            "low_configured_rho": low_rho,
            "high_configured_rho": high_rho,
            "mean_low_reuse_functional_ratio": fmean(
                float(low_by_world[world]["mean_functional_ratio"])
                for world in worlds
            ),
            "mean_high_reuse_functional_ratio": fmean(
                float(high_by_world[world]["mean_functional_ratio"])
                for world in worlds
            ),
            "shared_loss_wins_over_fixed_envelope_total": sum(
                float(row["best_fixed_minus_shared_gaussian_log_loss"]) > 0
                for row in comparisons
            ),
            "comparison_count": len(comparisons),
        },
        "resource_tradeoff": {
            "shared_parameter_count": comparisons[0]["shared_parameter_count"],
            "task_state_scalar_count": comparisons[0]["task_state_scalar_count"],
            "training_forward_multiply_adds_per_sample": comparisons[0][
                "training_forward_multiply_adds_per_sample"
            ],
            "inference_multiply_adds_per_sample": comparisons[0][
                "inference_multiply_adds_per_sample"
            ],
            "note": (
                "the shared-residual control stores rank-two residual factors "
                "for every task; its loss is not directly storage-matched to the "
                "fixed baselines"
            ),
        },
    }


def plot_report(report: dict[str, Any], destination: Path) -> None:
    summaries = report["rho_summaries"]
    worlds = report["worlds"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    axes[0].axhline(0, color="0.3", linewidth=1)
    for world in worlds:
        rows = [
            row for row in report["comparisons"] if row["world_seed"] == world
        ]
        axes[0].plot(
            [row["configured_rho"] for row in rows],
            [row["best_fixed_minus_shared_gaussian_log_loss"] for row in rows],
            marker="o",
            linewidth=1,
            alpha=0.35,
            color="0.5",
            label="World curves" if world == worlds[0] else None,
        )
    axes[0].plot(
        [row["configured_rho"] for row in summaries],
        [row["mean_best_fixed_minus_shared_gaussian_log_loss"] for row in summaries],
        marker="D",
        linewidth=2.5,
        color="tab:blue",
        label=f"{len(worlds)}-world mean",
    )
    axes[0].set_xlabel("Configured recurrence (rho)")
    axes[0].set_ylabel("Best fixed model minus shared residual log loss")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].plot(
        [row["configured_rho"] for row in summaries],
        [row["mean_functional_ratio"] for row in summaries],
        marker="o",
        linewidth=2.5,
        color="tab:orange",
        label="Mean task residual / parent update",
    )
    axes[1].plot(
        [row["configured_rho"] for row in summaries],
        [row["mean_maximum_task_functional_ratio"] for row in summaries],
        marker="s",
        linewidth=1.8,
        color="tab:red",
        label="Mean of per-run maxima",
    )
    axes[1].axhline(
        report["escape_hatch_max_ratio"],
        color="0.4",
        linestyle="--",
        linewidth=1,
        label="Escape guard",
    )
    axes[1].set_xlabel("Configured recurrence (rho)")
    axes[1].set_ylabel("Functional residual ratio")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    figure.suptitle("Shared-parent residual control across recurrence")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared",
        type=Path,
        default=Path("artifacts/shared_residual_rho/sweep.json"),
    )
    parser.add_argument(
        "--seed-zero",
        type=Path,
        default=Path("reports/rho_seed0/rho-comparison.json"),
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("artifacts/rho_development/sweep.json"),
    )
    parser.add_argument(
        "--tuning",
        type=Path,
        default=Path("artifacts/shared_residual_tuning/tuning.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/shared_residual")
    )
    args = parser.parse_args()

    shared = json.loads(args.shared.read_text(encoding="utf-8"))
    seed_zero = json.loads(args.seed_zero.read_text(encoding="utf-8"))
    development = json.loads(args.development.read_text(encoding="utf-8"))
    tuning = json.loads(args.tuning.read_text(encoding="utf-8"))
    report = summarize(shared, _baseline_records(seed_zero, development))
    report["tuning"] = tuning
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "shared-residual.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    plot_report(report, args.output / "shared-residual.png")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
