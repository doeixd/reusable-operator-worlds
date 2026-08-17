"""Combine the canonical seed-0 curve with resumable development sweeps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt


def _crossing(rows: list[dict[str, float]], x_key: str, y_key: str) -> float | None:
    rows = sorted(rows, key=lambda row: row[x_key])
    for left, right in zip(rows, rows[1:], strict=False):
        left_y = left[y_key]
        right_y = right[y_key]
        if left_y == 0:
            return left[x_key]
        if left_y * right_y < 0:
            fraction = -left_y / (right_y - left_y)
            return left[x_key] + fraction * (right[x_key] - left[x_key])
    return None


def _seed_zero_rows(report: dict[str, Any]) -> list[dict[str, float | int]]:
    return [
        {
            "world_seed": 0,
            "configured_rho": float(point["configured_rho"]),
            "measured_residual_correlation": float(
                point["measured_residual_correlation"]
            ),
            "dense_minus_continuous_gaussian_log_loss": float(
                point["dense_minus_continuous_prequential_advantage"]
            ),
            "dense_minus_continuous_novel_32_shot_nmse": float(
                point["dense_minus_continuous_novel_32_shot_advantage"]
            ),
        }
        for point in report["points"]
    ]


def _sweep_rows(report: dict[str, Any]) -> list[dict[str, float | int]]:
    by_point: dict[tuple[int, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in report["records"]:
        key = (int(record["world_seed"]), float(record["configured_rho"]))
        by_point[key][str(record["model"])] = record
    rows = []
    for (world, rho), by_model in sorted(by_point.items()):
        if set(by_model) != {"continuous", "dense"}:
            raise ValueError(f"world={world}, rho={rho} is not a complete pair")
        continuous = by_model["continuous"]
        dense = by_model["dense"]
        measured_continuous = float(continuous["measured_residual_correlation"])
        measured_dense = float(dense["measured_residual_correlation"])
        if abs(measured_continuous - measured_dense) > 1e-9:
            raise ValueError(f"world={world}, rho={rho} has mismatched recurrence")
        rows.append(
            {
                "world_seed": world,
                "configured_rho": rho,
                "measured_residual_correlation": measured_continuous,
                "dense_minus_continuous_gaussian_log_loss": float(
                    dense["gaussian_log_loss"]
                )
                - float(continuous["gaussian_log_loss"]),
                "dense_minus_continuous_novel_32_shot_nmse": float(
                    dense["novel_32_shot_nmse"]
                )
                - float(continuous["novel_32_shot_nmse"]),
            }
        )
    return rows


def summarize(seed_zero: dict[str, Any], sweep: dict[str, Any]) -> dict[str, Any]:
    rows = _seed_zero_rows(seed_zero) + _sweep_rows(sweep)
    worlds = sorted({int(row["world_seed"]) for row in rows})
    rhos = sorted({float(row["configured_rho"]) for row in rows})
    expected = {(world, rho) for world in worlds for rho in rhos}
    observed = {
        (int(row["world_seed"]), float(row["configured_rho"])) for row in rows
    }
    if observed != expected:
        raise ValueError("combined development curve is not rectangular")

    summaries = []
    for rho in rhos:
        selected = [row for row in rows if row["configured_rho"] == rho]
        loss_effects = [
            float(row["dense_minus_continuous_gaussian_log_loss"])
            for row in selected
        ]
        novel_effects = [
            float(row["dense_minus_continuous_novel_32_shot_nmse"])
            for row in selected
        ]
        summaries.append(
            {
                "configured_rho": rho,
                "mean_measured_residual_correlation": fmean(
                    float(row["measured_residual_correlation"]) for row in selected
                ),
                "mean_dense_minus_continuous_gaussian_log_loss": fmean(loss_effects),
                "continuous_loss_wins": sum(effect > 0 for effect in loss_effects),
                "mean_dense_minus_continuous_novel_32_shot_nmse": fmean(
                    novel_effects
                ),
                "continuous_novel_wins": sum(effect > 0 for effect in novel_effects),
                "world_effects": selected,
            }
        )

    per_world_crossings = []
    for world in worlds:
        selected = [row for row in rows if row["world_seed"] == world]
        per_world_crossings.append(
            {
                "world_seed": world,
                "configured_rho": _crossing(
                    selected,
                    "configured_rho",
                    "dense_minus_continuous_gaussian_log_loss",
                ),
                "measured_residual_correlation": _crossing(
                    selected,
                    "measured_residual_correlation",
                    "dense_minus_continuous_gaussian_log_loss",
                ),
            }
        )
    mean_rows = [
        {
            "configured_rho": float(row["configured_rho"]),
            "measured_residual_correlation": float(
                row["mean_measured_residual_correlation"]
            ),
            "effect": float(row["mean_dense_minus_continuous_gaussian_log_loss"]),
        }
        for row in summaries
    ]
    return {
        "scope": f"development worlds {worlds}; descriptive, not confirmatory",
        "sign_convention": "positive Dense-minus-Continuous favors Continuous",
        "worlds": worlds,
        "rho_summaries": summaries,
        "linear_interpolated_zero_crossing": {
            "configured_rho": _crossing(mean_rows, "configured_rho", "effect"),
            "measured_residual_correlation": _crossing(
                mean_rows, "measured_residual_correlation", "effect"
            ),
            "per_world": per_world_crossings,
            "note": "descriptive linear interpolation between adjacent development points",
        },
        "secondary_hypothesis": {
            "claim": (
                "Continuous improves 32-shot transfer before it wins cumulative loss "
                "at intermediate recurrence"
            ),
            "status": "not replicated across worlds 0-2",
        },
    }


def plot_report(report: dict[str, Any], destination: Path) -> None:
    summaries = report["rho_summaries"]
    worlds = report["worlds"]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    panels = (
        (axes[0], "configured_rho", "Configured recurrence (rho)"),
        (
            axes[1],
            "measured_residual_correlation",
            "Measured residual-function correlation",
        ),
    )
    for axis, x_key, xlabel in panels:
        axis.axhline(0, color="0.3", linewidth=1)
        for world in worlds:
            points = [
                effect
                for summary in summaries
                for effect in summary["world_effects"]
                if effect["world_seed"] == world
            ]
            x = [float(point[x_key]) for point in points]
            y = [
                float(point["dense_minus_continuous_gaussian_log_loss"])
                for point in points
            ]
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=1,
                alpha=0.38,
                label=f"World {world}",
            )
        mean_x_key = (
            "configured_rho"
            if x_key == "configured_rho"
            else "mean_measured_residual_correlation"
        )
        mean_x = [float(summary[mean_x_key]) for summary in summaries]
        mean_y = [
            float(summary["mean_dense_minus_continuous_gaussian_log_loss"])
            for summary in summaries
        ]
        axis.plot(mean_x, mean_y, marker="D", linewidth=2.5, label="Three-world mean")
        if x_key == "configured_rho":
            for x_value, effect in zip(mean_x, mean_y, strict=True):
                axis.annotate(
                    f"{effect:+,.0f}",
                    (x_value, effect),
                    xytext=(0, 7 if effect >= 0 else -15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        crossing_key = (
            "configured_rho"
            if x_key == "configured_rho"
            else "measured_residual_correlation"
        )
        crossing = report["linear_interpolated_zero_crossing"][crossing_key]
        if crossing is not None:
            axis.axvline(crossing, color="0.45", linestyle="--", linewidth=1)
            axis.annotate(
                f"mean crossing ~{crossing:.2f}",
                (crossing, 0),
                xytext=(5, 10),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", alpha=0.2)
        axis.margins(x=0.06, y=0.14)
    axes[0].set_ylabel("Dense-C minus Continuous cumulative log loss")
    axes[1].legend(frameon=False, loc="upper left")
    figure.suptitle("Replicated specialization-to-reuse crossover")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-zero",
        type=Path,
        default=Path("reports/rho_seed0/rho-comparison.json"),
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("artifacts/rho_development/sweep.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/rho_worlds_0_2")
    )
    args = parser.parse_args()
    seed_zero = json.loads(args.seed_zero.read_text(encoding="utf-8"))
    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
    report = summarize(seed_zero, sweep)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rho-replication.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    plot_report(report, args.output / "rho-replication.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
