"""Aggregate and plot paired Continuous/Dense-C recurrence controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_POINTS = (
    (0.0, Path("artifacts/sparse_rho/0/continuous"), Path("artifacts/sparse_rho/0/dense_c")),
    (
        0.25,
        Path("artifacts/sparse_rho/0.25/continuous"),
        Path("artifacts/sparse_rho/0.25/dense_c"),
    ),
    (
        0.5,
        Path("artifacts/sparse_rho/0.5/continuous"),
        Path("artifacts/sparse_rho/0.5/dense_c"),
    ),
    (
        0.75,
        Path("artifacts/sparse_rho/0.75/continuous"),
        Path("artifacts/sparse_rho/0.75/dense_c"),
    ),
    (
        0.9,
        Path("artifacts/sparse_rho/0.9/continuous"),
        Path("artifacts/sparse_rho/0.9/dense_c"),
    ),
    (
        1.0,
        Path("artifacts/family_controls/decoupled_tanh"),
        Path("artifacts/tuning/stage1/dense/global_1em03_task_5em02/world_0"),
    ),
)


def _summary(artifact: Path) -> dict[str, Any]:
    return json.loads((artifact / "summary.json").read_text(encoding="utf-8"))


def summarize_point(
    configured_rho: float, continuous_artifact: Path, dense_artifact: Path
) -> dict[str, float]:
    continuous = _summary(continuous_artifact)
    dense = _summary(dense_artifact)
    continuous_reuse = continuous["world_functional_reuse"]
    dense_reuse = dense["world_functional_reuse"]
    for label, summary, artifact in (
        ("continuous", continuous, continuous_artifact),
        ("dense", dense, dense_artifact),
    ):
        actual_rho = float(summary["world_functional_reuse"]["configured_rho"])
        if abs(actual_rho - configured_rho) > 1e-9:
            raise ValueError(
                f"{label} artifact {artifact} has rho={actual_rho}, "
                f"expected {configured_rho}"
            )
    measured_continuous = float(
        continuous_reuse["mean_pairwise_residual_correlation"]
    )
    measured_dense = float(dense_reuse["mean_pairwise_residual_correlation"])
    if abs(measured_continuous - measured_dense) > 1e-9:
        raise ValueError("paired artifacts report different functional recurrence")

    continuous_loss = float(continuous["cumulative_prequential_gaussian_log_loss"])
    dense_loss = float(dense["cumulative_prequential_gaussian_log_loss"])
    continuous_novel = float(
        continuous["novel_composition"]["nmse_by_support"]["32"]
    )
    dense_novel = float(dense["novel_composition"]["nmse_by_support"]["32"])
    return {
        "configured_rho": configured_rho,
        "measured_residual_correlation": measured_continuous,
        "continuous_prequential_gaussian_log_loss": continuous_loss,
        "dense_c_prequential_gaussian_log_loss": dense_loss,
        "dense_minus_continuous_prequential_advantage": dense_loss
        - continuous_loss,
        "continuous_novel_32_shot_nmse": continuous_novel,
        "dense_c_novel_32_shot_nmse": dense_novel,
        "dense_minus_continuous_novel_32_shot_advantage": dense_novel
        - continuous_novel,
        "continuous_final_median_nmse": float(continuous["final_nmse"]["median"]),
        "dense_c_final_median_nmse": float(dense["final_nmse"]["median"]),
    }


def _interpolated_crossing(
    rows: list[dict[str, float]], x_key: str, y_key: str
) -> float | None:
    for left, right in zip(rows, rows[1:], strict=False):
        left_y = left[y_key]
        right_y = right[y_key]
        if left_y == 0:
            return left[x_key]
        if left_y * right_y < 0:
            fraction = -left_y / (right_y - left_y)
            return left[x_key] + fraction * (right[x_key] - left[x_key])
    return None


def aggregate(points: list[tuple[float, Path, Path]]) -> dict[str, Any]:
    rows = [summarize_point(*point) for point in sorted(points)]
    y_key = "dense_minus_continuous_prequential_advantage"
    return {
        "scope": "development world 0; descriptive, not confirmatory",
        "sign_convention": (
            "positive Dense-minus-Continuous advantage means Continuous has "
            "lower cumulative Gaussian log loss"
        ),
        "points": rows,
        "linear_interpolated_zero_crossing": {
            "configured_rho": _interpolated_crossing(
                rows, "configured_rho", y_key
            ),
            "measured_residual_correlation": _interpolated_crossing(
                rows, "measured_residual_correlation", y_key
            ),
            "note": "descriptive interpolation between adjacent seed-0 points",
        },
    }


def plot_comparison(report: dict[str, Any], destination: Path) -> None:
    rows = report["points"]
    effects = [row["dense_minus_continuous_prequential_advantage"] for row in rows]
    configured = [row["configured_rho"] for row in rows]
    measured = [row["measured_residual_correlation"] for row in rows]
    crossing = report["linear_interpolated_zero_crossing"]

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for panel_index, (axis, x, xlabel, crossing_key) in enumerate((
        (axes[0], configured, "Configured recurrence (rho)", "configured_rho"),
        (
            axes[1],
            measured,
            "Measured residual-function correlation",
            "measured_residual_correlation",
        ),
    )):
        axis.axhline(0, color="0.35", linewidth=1)
        axis.plot(x, effects, marker="o", linewidth=2)
        for point_index, (x_value, effect) in enumerate(zip(x, effects, strict=True)):
            if panel_index == 0:
                label = f"{effect:+,.0f}"
                y_offset = 7 if effect >= 0 else -15
            else:
                label = f"rho={configured[point_index]:g}"
                y_offset = 7 if point_index % 2 == 0 else -15
            axis.annotate(
                label,
                (x_value, effect),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        crossing_value = crossing[crossing_key]
        if crossing_value is not None:
            axis.axvline(crossing_value, color="0.5", linestyle="--", linewidth=1)
            axis.annotate(
                f"linear crossing ~{crossing_value:.2f}",
                (crossing_value, 0),
                xytext=(5, 10),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", alpha=0.2)
        axis.margins(x=0.06, y=0.12)
    axes[0].set_ylabel("Dense-C minus Continuous cumulative log loss")
    figure.suptitle("Seed-0 specialization-to-reuse crossover")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point",
        action="append",
        nargs=3,
        metavar=("RHO", "CONTINUOUS_ARTIFACT", "DENSE_ARTIFACT"),
        help="paired point; repeat to replace the canonical seed-0 inputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rho_seed0"),
        help="directory for the machine-readable table and figure",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    points = (
        [(float(rho), Path(continuous), Path(dense)) for rho, continuous, dense in args.point]
        if args.point
        else list(DEFAULT_POINTS)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = aggregate(points)
    (args.output / "rho-comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    plot_comparison(report, args.output / "rho-crossover.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
