"""Plot acquired same-architecture forward transfer over the lifetime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODEL_STYLE = {
    "continuous": ("Continuous", "#31688e", "o"),
    "dense": ("Dense-C", "#35b779", "s"),
}


def _model_bins(report: dict[str, Any], model: str) -> list[dict[str, Any]]:
    rows = sorted(
        [row for row in report["task_index_bins"] if row["model"] == model],
        key=lambda row: row["task_index_start"],
    )
    if len(rows) != 8 or any(len(row["world_means"]) != 10 for row in rows):
        raise ValueError(f"incomplete binned forward-transfer data for {model}")
    return rows


def plot(report_path: Path, output: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    for model, (label, color, marker) in MODEL_STYLE.items():
        bins = _model_bins(report, model)
        x = np.asarray(
            [0.5 * (row["task_index_start"] + row["task_index_end"]) for row in bins]
        )
        matrix = np.asarray([row["world_means"] for row in bins], dtype=float).T
        for world_values in matrix:
            axes[0].plot(x, world_values, color=color, linewidth=0.7, alpha=0.18)
        mean = np.mean(matrix, axis=0)
        axes[0].plot(
            x,
            mean,
            color=color,
            marker=marker,
            linewidth=2.4,
            markersize=5,
            label=label,
        )

        summary = next(row for row in report["model_summaries"] if row["model"] == model)
        similarity = summary["by_prior_route_similarity"]
        similarity_x = [float(row["maximum_prior_route_position_similarity"]) for row in similarity]
        similarity_y = [float(row["mean_forward_transfer"]) for row in similarity]
        axes[1].plot(
            similarity_x,
            similarity_y,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6,
            label=label,
        )

    axes[0].axhline(0.0, color="#222222", linewidth=0.9)
    axes[0].set_title("Forward transfer grows over the lifetime")
    axes[0].set_xlabel("Task index (8-task bins)")
    axes[0].set_ylabel("Fresh minus lifetime log loss")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, color="#222222", linewidth=0.9)
    axes[1].set_title("Transfer increases with prior-route similarity")
    axes[1].set_xlabel("Maximum fraction of route positions matched")
    axes[1].set_ylabel("Mean fresh minus lifetime log loss")
    axes[1].set_xticks((0.0, 1 / 3, 2 / 3), ("0/3", "1/3", "2/3"))
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Lifetime experience creates explicit forward transfer")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/forward_transfer/forward-transfer.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/forward_transfer/forward-transfer.png"),
    )
    args = parser.parse_args()
    plot(args.report, args.output)


if __name__ == "__main__":
    main()
