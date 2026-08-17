"""Run and resume the symmetric ROW development hyperparameter grid."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from row.config import ExperimentConfig, load_config
from row.experiments.learned_lifetime import (
    ModelKind,
    resolved_learned_config,
    run,
)
from row.provenance import validate_artifact


def _label(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def _configured_run(
    base: ExperimentConfig,
    kind: ModelKind,
    world_seed: int,
    global_lr: float,
    task_lr: float,
    output: Path,
    *,
    dense_hidden_width: int | None = None,
    dense_task_embedding_dim: int | None = None,
) -> ExperimentConfig:
    config = replace(
        base,
        world=replace(base.world, seed=world_seed),
        output_directory=output,
    )
    if kind == "dense":
        selected = replace(
            config.dense_model,
            hidden_width=(
                config.dense_model.hidden_width
                if dense_hidden_width is None
                else dense_hidden_width
            ),
            task_embedding_dim=(
                config.dense_model.task_embedding_dim
                if dense_task_embedding_dim is None
                else dense_task_embedding_dim
            ),
            global_learning_rate=global_lr,
            task_learning_rate=task_lr,
        )
        return replace(config, dense_model=selected)
    if kind == "continuous":
        return replace(
            config,
            continuous_model=replace(
                config.continuous_model,
                global_learning_rate=global_lr,
                task_learning_rate=task_lr,
            ),
        )
    if kind == "hypernetwork":
        return replace(
            config,
            hypernetwork_model=replace(
                config.hypernetwork_model,
                global_learning_rate=global_lr,
                task_learning_rate=task_lr,
            ),
        )
    return replace(
        config,
        discrete_model=replace(
            config.discrete_model,
            global_learning_rate=global_lr,
            task_learning_rate=task_lr,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tuning/stage1"))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--global-lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--task-lrs", type=float, nargs="+", default=[5e-3, 5e-2])
    parser.add_argument(
        "--models",
        choices=("continuous", "dense", "hypernetwork", "discrete"),
        nargs="+",
        default=["continuous", "dense"],
    )
    parser.add_argument("--dense-hidden-width", type=int)
    parser.add_argument("--dense-task-embedding-dim", type=int)
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
                    config = _configured_run(
                        base,
                        kind,
                        world_seed,
                        global_lr,
                        task_lr,
                        output,
                        dense_hidden_width=args.dense_hidden_width,
                        dense_task_embedding_dim=args.dense_task_embedding_dim,
                    )
                    if summary_path.exists():
                        validate_artifact(
                            output,
                            resolved_learned_config(config, kind, "forward"),
                            kind,
                        )
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    else:
                        summary = run(config, kind=kind)
                        gc.collect()
                    records.append(
                        {
                            "model": kind,
                            "global_learning_rate": global_lr,
                            "task_learning_rate": task_lr,
                            "world_seed": world_seed,
                            "shared_parameter_count": summary["shared_parameter_count"],
                            "task_state_scalar_count": summary["task_state_scalar_count"],
                            "compute_accounting": summary["compute_accounting"],
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
