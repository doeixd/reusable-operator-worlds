"""Analyze measured-recurrence smoothness and truncated-lifetime crossovers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from row.experiments.summarize_rho_replication import _crossing


TRUNCATIONS = (2048, 4096, 8192)
RHO_LABELS = {
    0.0: "rho_0",
    0.25: "rho_0p25",
    0.5: "rho_0p5",
    0.75: "rho_0p75",
    0.9: "rho_0p9",
    1.0: "rho_1",
}


def _artifact_path(root: Path, world: int, rho: float, model: str) -> Path:
    if world > 0:
        return root / "rho_development" / RHO_LABELS[rho] / f"world_{world}" / model
    if rho < 1.0:
        seed_zero_model = "dense_c" if model == "dense" else model
        return root / "sparse_rho" / f"{rho:g}" / seed_zero_model
    if model == "continuous":
        return root / "family_controls" / "decoupled_tanh"
    return root / "checkpoints_development" / "world_0" / "dense"


def _cumulative_nll(path: Path) -> dict[int, float]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["record_type"] == "prequential":
                values.append(float(row["nll"]))
    if len(values) != TRUNCATIONS[-1]:
        raise ValueError(f"{path} has {len(values)} prequential rows")
    cumulative = np.cumsum(np.asarray(values, dtype=np.float64))
    return {truncation: float(cumulative[truncation - 1]) for truncation in TRUNCATIONS}


def _linear_fit_quality(x: list[float], y: list[float]) -> dict[str, float]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    prediction = slope * x_values + intercept
    residual = y_values - prediction
    denominator = float(np.sum(np.square(y_values - np.mean(y_values))))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": 1.0 - float(np.sum(np.square(residual))) / denominator,
        "root_mean_square_residual": float(np.sqrt(np.mean(np.square(residual)))),
    }


def _summarize(rows: list[dict[str, float | int]]) -> dict[str, Any]:
    worlds = sorted({int(row["world_seed"]) for row in rows})
    rhos = sorted({float(row["configured_rho"]) for row in rows})
    truncations = sorted({int(row["online_examples"]) for row in rows})
    expected = {
        (world, rho, truncation)
        for world in worlds
        for rho in rhos
        for truncation in truncations
    }
    observed = {
        (
            int(row["world_seed"]),
            float(row["configured_rho"]),
            int(row["online_examples"]),
        )
        for row in rows
    }
    if observed != expected:
        raise ValueError("truncated rho analysis is not rectangular")

    curves = []
    crossings = []
    for truncation in truncations:
        mean_curve = []
        for rho in rhos:
            selected = [
                row
                for row in rows
                if row["online_examples"] == truncation
                and row["configured_rho"] == rho
            ]
            mean_curve.append(
                {
                    "configured_rho": rho,
                    "measured_residual_correlation": fmean(
                        float(row["measured_residual_correlation"])
                        for row in selected
                    ),
                    "mean_dense_minus_continuous_gaussian_log_loss": fmean(
                        float(row["dense_minus_continuous_gaussian_log_loss"])
                        for row in selected
                    ),
                    "continuous_wins": sum(
                        float(row["dense_minus_continuous_gaussian_log_loss"]) > 0
                        for row in selected
                    ),
                    "world_effects": selected,
                }
            )
        curves.append(
            {"online_examples": truncation, "mean_curve": mean_curve}
        )
        per_world = []
        for world in worlds:
            selected = [
                row
                for row in rows
                if row["online_examples"] == truncation
                and row["world_seed"] == world
            ]
            per_world.append(
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
        mean_crossing = {
            "configured_rho": _crossing(
                mean_curve,
                "configured_rho",
                "mean_dense_minus_continuous_gaussian_log_loss",
            ),
            "measured_residual_correlation": _crossing(
                mean_curve,
                "measured_residual_correlation",
                "mean_dense_minus_continuous_gaussian_log_loss",
            ),
        }
        crossings.append(
            {
                "online_examples": truncation,
                "tasks_completed": truncation // 128,
                "mean_curve_crossing": mean_crossing,
                "per_world": per_world,
            }
        )

    full_curve = curves[-1]["mean_curve"]
    y = [
        float(row["mean_dense_minus_continuous_gaussian_log_loss"])
        for row in full_curve
    ]
    configured_fit = _linear_fit_quality(
        [float(row["configured_rho"]) for row in full_curve], y
    )
    measured_fit = _linear_fit_quality(
        [float(row["measured_residual_correlation"]) for row in full_curve], y
    )
    configured_values = [
        float(row["mean_curve_crossing"]["configured_rho"])
        for row in crossings
        if row["mean_curve_crossing"]["configured_rho"] is not None
    ]
    h5a_status = (
        "supported"
        if len(configured_values) == len(crossings)
        and all(
            later < earlier
            for earlier, later in zip(configured_values, configured_values[1:])
        )
        else "not supported"
    )
    return {
        "scope": f"development worlds {worlds}; existing logs only",
        "sign_convention": "positive Dense-minus-Continuous favors Continuous",
        "rows": rows,
        "truncated_mean_curves": curves,
        "crossings_by_lifetime": crossings,
        "h5a_amortization_prediction": {
            "prediction": "crossover decreases as lifetime length grows",
            "status": h5a_status,
        },
        "h5b_coordinate_test": {
            "configured_rho_linear_fit": configured_fit,
            "measured_recurrence_linear_fit": measured_fit,
            "measured_coordinate_is_more_linear": (
                measured_fit["r_squared"] > configured_fit["r_squared"]
            ),
            "note": (
                "Linearity is a compact elbow diagnostic on the six-point mean "
                "curve; both coordinates and raw world curves remain reported."
            ),
        },
    }


def analyze(root: Path, rho_report_path: Path) -> dict[str, Any]:
    rho_report = json.loads(rho_report_path.read_text(encoding="utf-8"))
    expected_effect = {
        (int(effect["world_seed"]), float(summary["configured_rho"])): float(
            effect["dense_minus_continuous_gaussian_log_loss"]
        )
        for summary in rho_report["rho_summaries"]
        for effect in summary["world_effects"]
    }
    measured = {
        (int(effect["world_seed"]), float(summary["configured_rho"])): float(
            effect["measured_residual_correlation"]
        )
        for summary in rho_report["rho_summaries"]
        for effect in summary["world_effects"]
    }
    rows = []
    for world, rho in sorted(expected_effect):
        losses = {
            model: _cumulative_nll(
                _artifact_path(root, world, rho, model) / "metrics.jsonl"
            )
            for model in ("continuous", "dense")
        }
        final_effect = losses["dense"][TRUNCATIONS[-1]] - losses["continuous"][TRUNCATIONS[-1]]
        if abs(final_effect - expected_effect[(world, rho)]) > 1e-6:
            raise ValueError(
                f"artifact mismatch at world={world}, rho={rho}: {final_effect}"
            )
        for truncation in TRUNCATIONS:
            rows.append(
                {
                    "world_seed": world,
                    "configured_rho": rho,
                    "measured_residual_correlation": measured[(world, rho)],
                    "online_examples": truncation,
                    "tasks_completed": truncation // 128,
                    "dense_minus_continuous_gaussian_log_loss": (
                        losses["dense"][truncation]
                        - losses["continuous"][truncation]
                    ),
                }
            )
    return _summarize(rows)


def plot_report(report: dict[str, Any], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    styles = {2048: ("16 tasks", "o"), 4096: ("32 tasks", "s"), 8192: ("64 tasks", "D")}
    for curve in report["truncated_mean_curves"]:
        truncation = int(curve["online_examples"])
        label, marker = styles[truncation]
        points = curve["mean_curve"]
        axes[0].plot(
            [point["measured_residual_correlation"] for point in points],
            [point["mean_dense_minus_continuous_gaussian_log_loss"] for point in points],
            marker=marker,
            linewidth=2.1,
            label=label,
        )
    axes[0].axhline(0, color="0.3", linewidth=1)
    axes[0].set_title("Crossover evolves over the lifetime")
    axes[0].set_xlabel("Measured residual-function correlation")
    axes[0].set_ylabel("Dense-C minus Continuous log loss")
    axes[0].legend(frameon=False)

    crossings = report["crossings_by_lifetime"]
    tasks = [row["tasks_completed"] for row in crossings]
    configured = [row["mean_curve_crossing"]["configured_rho"] for row in crossings]
    measured = [row["mean_curve_crossing"]["measured_residual_correlation"] for row in crossings]
    axes[1].plot(tasks, configured, marker="o", linewidth=2.3, label="Configured rho")
    axes[1].plot(tasks, measured, marker="s", linewidth=2.3, label="Measured recurrence")
    axes[1].set_title("More tasks lower the reuse threshold")
    axes[1].set_xlabel("Lifetime tasks completed")
    axes[1].set_ylabel("Interpolated mean crossover")
    axes[1].set_xticks(tasks)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Existing logs support an amortization-dependent crossover")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--rho-report",
        type=Path,
        default=Path("reports/rho_worlds_0_9/rho-replication.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/rho_bridge"))
    args = parser.parse_args()
    report = analyze(args.artifact_root, args.rho_report)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rho-bridge.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    plot_report(report, args.output / "rho-bridge.png")
    print(json.dumps(report["crossings_by_lifetime"], indent=2), flush=True)
    print(json.dumps(report["h5b_coordinate_test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
