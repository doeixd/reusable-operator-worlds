"""Run and resume the symmetric ROW development hyperparameter grid."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from row.config import load_config
from row.experiments.learned_lifetime import run


def _label(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tuning/stage1"))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--global-lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--task-lrs", type=float, nargs="+", default=[5e-3, 5e-2])
    parser.add_argument("--models", choices=("continuous", "dense"), nargs="+", default=["continuous", "dense"])
    args = parser.parse_args()
    base = load_config(args.config)
    base = replace(
        base,
        evaluation=replace(
            base.evaluation,
            lifetime_checkpoints=(),
            checkpoint_novel_tasks=1,
            extended_diagnostics=False,
        ),
    )
    records: list[dict[str, object]] = []
    total = len(args.models) * len(args.global_lrs) * len(args.task_lrs) * len(args.worlds)
    completed = 0
    for kind in args.models:
        for global_lr in args.global_lrs:
            for task_lr in args.task_lrs:
                for world_seed in args.worlds:
                    completed += 1
                    name = (
                        f"{kind}/global_{_label(global_lr)}_task_{_label(task_lr)}"
                        f"/world_{world_seed}"
                    )
                    output = args.output / name
                    summary_path = output / "summary.json"
                    print(f"[{completed}/{total}] {name}", flush=True)
                    if summary_path.exists():
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    else:
                        config = replace(base, world=replace(base.world, seed=world_seed), output_directory=output)
                        if kind == "continuous":
                            config = replace(
                                config,
                                continuous_model=replace(
                                    config.continuous_model,
                                    global_learning_rate=global_lr,
                                    task_learning_rate=task_lr,
                                ),
                            )
                        else:
                            config = replace(
                                config,
                                dense_model=replace(
                                    config.dense_model,
                                    hidden_width=32,
                                    global_learning_rate=global_lr,
                                    task_learning_rate=task_lr,
                                ),
                            )
                        summary = run(config, kind=kind)
                        gc.collect()
                    records.append(
                        {
                            "model": kind,
                            "global_learning_rate": global_lr,
                            "task_learning_rate": task_lr,
                            "world_seed": world_seed,
                            "gaussian_log_loss": summary[
                                "cumulative_prequential_gaussian_log_loss"
                            ],
                            "final_median_nmse": summary["final_nmse"]["median"],
                            "novel_32_shot_nmse": summary["novel_composition"][
                                "nmse_by_support"
                            ]["32"],
                        }
                    )

    configurations: list[dict[str, object]] = []
    for kind in args.models:
        for global_lr in args.global_lrs:
            for task_lr in args.task_lrs:
                selected = [
                    row
                    for row in records
                    if row["model"] == kind
                    and row["global_learning_rate"] == global_lr
                    and row["task_learning_rate"] == task_lr
                ]
                configurations.append(
                    {
                        "model": kind,
                        "global_learning_rate": global_lr,
                        "task_learning_rate": task_lr,
                        "worlds": len(selected),
                        "mean_gaussian_log_loss": float(
                            np.mean([float(row["gaussian_log_loss"]) for row in selected])
                        ),
                        "mean_novel_32_shot_nmse": float(
                            np.mean([float(row["novel_32_shot_nmse"]) for row in selected])
                        ),
                    }
                )
    configurations.sort(key=lambda row: (str(row["model"]), float(row["mean_gaussian_log_loss"])))
    result = {"runs": records, "configurations": configurations}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ranking.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(configurations, indent=2), flush=True)


if __name__ == "__main__":
    main()
