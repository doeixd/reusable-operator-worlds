"""Plot paired ROW task-order and replay robustness effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt


CONDITIONS = ("replay0", "replay1", "replay4", "reverse")
LABELS = ("No replay", "Replay 1:1", "Replay 1:4", "Reverse 1:1")


def _effect_matrix(
    report: dict[str, Any], key: str
) -> list[list[float]]:
    rows = report["paired_world_effects"]
    matrix = []
    for world in range(10):
        values = []
        for condition in CONDITIONS:
            matches = [
                row
                for row in rows
                if int(row["world_seed"]) == world
                and row["condition"] == condition
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one {condition} record for world {world}"
                )
            values.append(float(matches[0][key]))
        matrix.append(values)
    return matrix


def plot(report_path: Path, output: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    loss = _effect_matrix(
        report, "dense_minus_continuous_gaussian_log_loss"
    )
    novel = _effect_matrix(
        report, "dense_minus_continuous_novel_32_shot_nmse"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    x = list(range(len(CONDITIONS)))
    for axis, values, title, ylabel in (
        (
            axes[0],
            loss,
            "Lifetime-loss advantage survives every condition",
            "Dense-C minus Continuous log loss",
        ),
        (
            axes[1],
            novel,
            "Novel transfer weakens without replay",
            "Dense-C minus Continuous 32-shot NMSE",
        ),
    ):
        for world_values in values:
            axis.plot(
                x,
                world_values,
                color="#aaaaaa",
                marker="o",
                linewidth=0.8,
                markersize=3,
                alpha=0.65,
            )
        means = [fmean(row[index] for row in values) for index in x]
        axis.plot(
            x,
            means,
            color="#31688e",
            marker="D",
            linewidth=2.4,
            markersize=6,
            label="10-world mean",
        )
        axis.axhline(0.0, color="#222222", linewidth=0.9)
        axis.set_xticks(x, LABELS, rotation=18, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="best")
    fig.suptitle("Exact-reuse advantage is robust to task order and replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/robustness/robustness.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/robustness/robustness.png"),
    )
    args = parser.parse_args()
    plot(args.report, args.output)


if __name__ == "__main__":
    main()
