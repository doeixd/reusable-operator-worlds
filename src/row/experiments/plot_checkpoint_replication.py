"""Validate and plot the replicated exact-reuse checkpoint sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from row.experiments.sweep_checkpoints import CHECKPOINTS


def validate_report(report: dict[str, Any]) -> None:
    worlds = sorted(int(record["world_seed"]) for record in report["records"])
    if worlds != [world for world in range(10) for _ in range(2)]:
        raise ValueError("checkpoint report must contain both models for worlds 0-9")
    observed = tuple(
        int(summary["tasks_completed"]) for summary in report["checkpoint_summaries"]
    )
    if observed != CHECKPOINTS:
        raise ValueError(f"checkpoint report has sequence {observed}")


def plot_report(report: dict[str, Any], destination: Path) -> None:
    summaries = report["checkpoint_summaries"]
    checkpoints = np.array([int(row["tasks_completed"]) for row in summaries])
    continuous = np.array(
        [float(row["mean_continuous_32_shot_nmse"]) for row in summaries]
    )
    dense = np.array([float(row["mean_dense_c_32_shot_nmse"]) for row in summaries])

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    axes[0].plot(checkpoints, continuous, marker="o", linewidth=2.3, label="Continuous")
    axes[0].plot(checkpoints, dense, marker="s", linewidth=2.3, label="Dense-C")
    for index, checkpoint in enumerate(checkpoints):
        axes[0].annotate(
            f"{continuous[index]:.4f}",
            (checkpoint, continuous[index]),
            xytext=(0, -15 if index == 0 else -13),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
        axes[0].annotate(
            f"{dense[index]:.4f}",
            (checkpoint, dense[index]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[0].set_xlabel("Lifetime tasks experienced")
    axes[0].set_ylabel("Fresh-composition NMSE after 32 examples")
    axes[0].set_xticks(checkpoints)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    axes[0].margins(y=0.18)

    positions = np.arange(len(checkpoints))
    mean_effects = []
    for position, summary in zip(positions, summaries, strict=True):
        effects = np.array(
            [
                float(row["dense_minus_continuous_32_shot_nmse"])
                for row in summary["paired_world_effects"]
            ]
        )
        offsets = np.linspace(-0.09, 0.09, len(effects))
        axes[1].scatter(
            np.full(len(effects), position) + offsets,
            effects,
            color="0.55",
            alpha=0.55,
            s=25,
            label="World-level effects" if position == 0 else None,
        )
        mean_effects.append(float(np.mean(effects)))
    axes[1].axhline(0, color="0.3", linewidth=1)
    axes[1].plot(
        positions,
        mean_effects,
        marker="D",
        linewidth=2.3,
        label="10-world mean",
    )
    for position, effect, summary in zip(positions, mean_effects, summaries, strict=True):
        axes[1].annotate(
            f"{int(summary['continuous_wins'])}/10 wins",
            (position, effect),
            xytext=(0, 8 if effect >= 0 else -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[1].set_xlabel("Lifetime tasks experienced")
    axes[1].set_ylabel("Dense-C minus Continuous 32-shot NMSE")
    axes[1].set_xticks(positions, checkpoints)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    axes[1].margins(y=0.18)

    figure.suptitle("Reusable computation increasingly lowers fresh-task learning cost")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("artifacts/checkpoints_development/sweep.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/checkpoints_worlds_0_9")
    )
    args = parser.parse_args()
    report = json.loads(args.sweep.read_text(encoding="utf-8"))
    validate_report(report)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoint-replication.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    plot_report(report, args.output / "checkpoint-replication.png")


if __name__ == "__main__":
    main()
