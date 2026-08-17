"""Plot paired development-world effects for the ROW structural controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt


SERIES = (
    (
        "Continuous over Hypernetwork",
        "continuous_advantage_over_hyper_loss",
        "continuous_advantage_over_hyper_novel_32",
        "#31688e",
        "o",
    ),
    (
        "Hypernetwork over Dense-C",
        "hyper_advantage_over_dense32_loss",
        "hyper_advantage_over_dense32_novel_32",
        "#35b779",
        "s",
    ),
    (
        "Dense-24 over Dense-32",
        "dense24_advantage_over_dense32_loss",
        "dense24_advantage_over_dense32_novel_32",
        "#777777",
        "^",
    ),
)


def _validated_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = sorted(report["paired_world_effects"], key=lambda row: row["world_seed"])
    worlds = tuple(int(row["world_seed"]) for row in rows)
    if worlds != tuple(range(10)):
        raise ValueError(f"expected worlds 0-9, found {worlds}")
    return rows


def plot(report_path: Path, output: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _validated_rows(report)
    worlds = [int(row["world_seed"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for label, loss_key, novel_key, color, marker in SERIES:
        loss = [float(row[loss_key]) for row in rows]
        novel = [float(row[novel_key]) for row in rows]
        axes[0].plot(
            worlds,
            loss,
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=5,
            label=f"{label} (mean {fmean(loss):,.0f})",
        )
        axes[1].plot(
            worlds,
            novel,
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=5,
            label=f"{label} (mean {fmean(novel):.4f})",
        )

    axes[0].set_title("Cumulative Gaussian log-loss advantage")
    axes[0].set_ylabel("Advantage (lower loss favored)")
    axes[1].set_title("Novel 32-shot NMSE advantage")
    axes[1].set_ylabel("Advantage (lower NMSE favored)")
    for axis in axes:
        axis.axhline(0.0, color="#222222", linewidth=0.9)
        axis.set_xlabel("Development world seed")
        axis.set_xticks(worlds)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Structural controls preserve a three-level lifetime-loss ordering")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/structural_controls/structural-controls.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/structural_controls/structural-controls.png"),
    )
    args = parser.parse_args()
    plot(args.report, args.output)


if __name__ == "__main__":
    main()
