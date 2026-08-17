"""Generate the required exact-reuse ROW pilot figures from saved artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _rows(artifact: Path, record_type: str) -> list[dict[str, object]]:
    with (artifact / "metrics.jsonl").open(encoding="utf-8") as handle:
        return [
            row
            for line in handle
            if (row := json.loads(line))["record_type"] == record_type
        ]


def _summary(artifact: Path) -> dict[str, object]:
    return json.loads((artifact / "summary.json").read_text(encoding="utf-8"))


def _rolling(values: np.ndarray, window: int = 8) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    valid = np.convolve(values, kernel, mode="valid")
    return np.concatenate((np.full(window - 1, np.nan), valid))


def _finish(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_scratch(scratch: Path, output: Path) -> None:
    tasks = sorted(_rows(scratch, "task_summary"), key=lambda row: row["task_index"])
    indices = np.array([int(row["task_index"]) + 1 for row in tasks])
    final = np.array([float(row["final_nmse"]) for row in tasks])
    plt.figure(figsize=(7.2, 4.2))
    plt.scatter(indices, final, s=14, alpha=0.55, label="Task final NMSE")
    slope, intercept = np.polyfit(indices, final, 1)
    plt.plot(indices, slope * indices + intercept, linewidth=2, label=f"Linear trend ({slope:+.2e}/task)")
    plt.xlabel("Task index")
    plt.ylabel("Final scratch NMSE")
    plt.title("Scratch task difficulty is stationary")
    plt.legend(frameon=False)
    _finish(output / "01-scratch-task-difficulty.png")


def plot_examples_to_criterion(
    oracle: Path, continuous: Path, dense: Path, discrete: Path, output: Path
) -> None:
    artifacts = {
        "Oracle": oracle,
        "Continuous": continuous,
        "Dense-C": dense,
        "Discrete": discrete,
    }
    plt.figure(figsize=(7.2, 4.2))
    for label, artifact in artifacts.items():
        tasks = sorted(_rows(artifact, "task_summary"), key=lambda row: row["task_index"])
        values = np.array([float(row["examples_to_0.02"]) for row in tasks])
        plt.plot(np.arange(1, len(values) + 1), _rolling(values), linewidth=2, label=label)
    plt.xlabel("Lifetime task index")
    plt.ylabel("Examples to NMSE 0.02 (8-task rolling mean)")
    plt.title("Reusable learners become faster on later tasks")
    plt.legend(frameon=False)
    _finish(output / "02-examples-to-criterion.png")


def plot_prequential(continuous: Path, dense: Path, discrete: Path, output: Path) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for label, artifact in {
        "Continuous": continuous,
        "Dense-C": dense,
        "Discrete": discrete,
    }.items():
        rows = _rows(artifact, "prequential")
        cumulative = np.array([float(row["cumulative_nll"]) for row in rows])
        plt.plot(np.arange(1, len(cumulative) + 1), cumulative, linewidth=2, label=label)
    plt.xlabel("Online examples")
    plt.ylabel("Cumulative Gaussian log loss")
    plt.title("Exact-reuse lifetime predictive cost")
    plt.legend(frameon=False)
    _finish(output / "03-cumulative-prequential-loss.png")


def plot_checkpoint_adaptation(continuous: Path, dense: Path, output: Path) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for label, artifact, marker in (
        ("Continuous", continuous, "o"),
        ("Dense-C", dense, "s"),
    ):
        checkpoints = _summary(artifact)["novel_composition_checkpoints"]
        tasks = [int(item["tasks_completed"]) for item in checkpoints]
        values = [float(item["mean_nmse_by_support"]["32"]) for item in checkpoints]
        plt.plot(tasks, values, marker=marker, linewidth=2, label=label)
        for x, y in zip(tasks, values, strict=True):
            plt.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    plt.xlabel("Tasks experienced")
    plt.ylabel("Novel-composition NMSE after 32 code-only examples")
    plt.title("A reusable basis progressively lowers future learning cost")
    plt.xticks([8, 16, 32, 64])
    plt.margins(y=0.16)
    plt.legend(frameon=False)
    _finish(output / "04-checkpoint-novel-adaptation.png")


def plot_novel_curve(continuous: Path, dense: Path, output: Path) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for label, artifact, marker in (
        ("Continuous", continuous, "o"),
        ("Dense-C", dense, "s"),
    ):
        curve = _summary(artifact)["novel_composition"]["nmse_by_support"]
        support = np.array([int(value) for value in curve])
        values = np.array([float(curve[str(value)]) for value in support])
        plt.plot(support, values, marker=marker, linewidth=2, label=label)
    plt.xlabel("Fresh task-code adaptation examples")
    plt.ylabel("Novel-composition NMSE")
    plt.title("Frozen-library novel-composition adaptation")
    plt.xticks([0, 1, 2, 4, 8, 16, 32])
    plt.legend(frameon=False)
    _finish(output / "05-novel-composition-curve.png")


def plot_functional_heatmap(continuous: Path, output: Path) -> None:
    matrix = np.asarray(_summary(continuous)["functional_recovery"]["distance_matrix"])
    plt.figure(figsize=(7.2, 4.2))
    image = plt.imshow(matrix, aspect="auto", cmap="viridis_r")
    plt.colorbar(image, label="Normalized functional distance")
    plt.xlabel("Learned basis operator")
    plt.ylabel("Teacher primitive")
    plt.title("Teacher/learner functional operator distance")
    plt.xticks(range(matrix.shape[1]))
    plt.yticks(range(matrix.shape[0]))
    _finish(output / "06-functional-distance-heatmap.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--discrete", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_scratch(args.scratch, args.output)
    plot_examples_to_criterion(
        args.oracle, args.continuous, args.dense, args.discrete, args.output
    )
    plot_prequential(args.continuous, args.dense, args.discrete, args.output)
    plot_checkpoint_adaptation(args.continuous, args.dense, args.output)
    plot_novel_curve(args.continuous, args.dense, args.output)
    plot_functional_heatmap(args.continuous, args.output)
    print(f"wrote six figures to {args.output}")


if __name__ == "__main__":
    main()
