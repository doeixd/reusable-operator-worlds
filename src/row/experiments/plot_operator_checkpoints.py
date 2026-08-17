"""Validate and plot post-hoc operator-quality checkpoint diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from row.experiments.sweep_operator_checkpoints import CHECKPOINTS, MODELS


MODEL_STYLE = {
    "continuous": ("Continuous", "#31688e", "o"),
    "discrete": ("Discrete", "#b65f22", "s"),
}


def validate_report(report: dict[str, Any]) -> None:
    keys = {
        (int(record["world_seed"]), str(record["model"]))
        for record in report["records"]
    }
    expected = {(world, model) for world in range(10) for model in MODELS}
    if keys != expected:
        raise ValueError("operator checkpoint report requires both models for worlds 0-9")
    observed = tuple(
        int(summary["tasks_completed"])
        for summary in report["checkpoint_summaries"]
    )
    if observed != CHECKPOINTS:
        raise ValueError(f"operator checkpoint report has sequence {observed}")


def plot_report(report: dict[str, Any], destination: Path) -> None:
    summaries = report["checkpoint_summaries"]
    checkpoints = np.asarray(CHECKPOINTS)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)

    for model, (label, color, marker) in MODEL_STYLE.items():
        primitive = np.asarray(
            [
                row["models"][model]["mean_one_to_one_mean_primitive_distance"]
                for row in summaries
            ],
            dtype=float,
        )
        axes[0].plot(
            checkpoints,
            primitive,
            color=color,
            marker=marker,
            linewidth=2.3,
            label=label,
        )
    axes[0].set_title("Individual slots recover primitives")
    axes[0].set_ylabel("Normalized functional distance")
    axes[0].legend(frameon=False)

    for axis, model in zip(axes[1:], MODELS, strict=True):
        label, color, marker = MODEL_STYLE[model]
        model_rows = [row["models"][model] for row in summaries]
        learned = np.asarray(
            [row["mean_learned_route_completed_programs_nmse_mean"] for row in model_rows]
        )
        matched = np.asarray(
            [row["mean_true_route_completed_programs_nmse_mean"] for row in model_rows]
        )
        future = np.asarray(
            [
                np.nan
                if row["mean_true_route_future_programs_nmse_mean"] is None
                else row["mean_true_route_future_programs_nmse_mean"]
                for row in model_rows
            ]
        )
        axis.plot(
            checkpoints,
            learned,
            color=color,
            marker=marker,
            linewidth=2.4,
            label="Learner's route, completed tasks",
        )
        axis.plot(
            checkpoints,
            matched,
            color="#555555",
            marker="D",
            linewidth=1.9,
            label="Matched-slot teacher route, completed",
        )
        axis.plot(
            checkpoints,
            future,
            color="#888888",
            marker="^",
            linewidth=1.7,
            linestyle="--",
            label="Matched-slot teacher route, future",
        )
        axis.set_title(label)
        axis.set_ylabel("Program NMSE")
        axis.legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xlabel("Lifetime tasks experienced")
        axis.set_xticks(checkpoints)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(bottom=0)
    figure.suptitle("Operator recovery improves throughout the exact-reuse lifetime")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("artifacts/operator_checkpoints/sweep.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/operator_checkpoints")
    )
    args = parser.parse_args()
    report = json.loads(args.sweep.read_text(encoding="utf-8"))
    validate_report(report)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "operator-checkpoints.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    plot_report(report, args.output / "operator-checkpoints.png")


if __name__ == "__main__":
    main()
