"""Run resumable task-order and replay robustness conditions."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from row.config import ExperimentConfig, load_config
from row.experiments.learned_lifetime import resolved_learned_config, run
from row.provenance import validate_artifact


Condition = Literal["reverse", "replay0", "replay4"]
Model = Literal["continuous", "dense"]
CONDITIONS: dict[Condition, tuple[str, float]] = {
    "reverse": ("reverse", 1.0),
    "replay0": ("forward", 0.0),
    "replay4": ("forward", 4.0),
}


def _condition_config(
    base: ExperimentConfig,
    condition: Condition,
    model: Model,
    world_seed: int,
    output: Path,
) -> tuple[ExperimentConfig, str]:
    order, replay_ratio = CONDITIONS[condition]
    config = replace(
        base,
        world=replace(base.world, seed=world_seed),
        output_directory=output,
        evaluation=replace(
            base.evaluation,
            lifetime_checkpoints=(),
            checkpoint_novel_tasks=1,
            extended_diagnostics=False,
        ),
    )
    if model == "continuous":
        config = replace(
            config,
            continuous_model=replace(
                config.continuous_model, replay_ratio=replay_ratio
            ),
        )
    else:
        config = replace(
            config,
            dense_model=replace(config.dense_model, replay_ratio=replay_ratio),
        )
    return config, order


def _record(
    condition: Condition,
    model: Model,
    world_seed: int,
    order: str,
    replay_ratio: float,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "condition": condition,
        "model": model,
        "world_seed": world_seed,
        "order": order,
        "replay_ratio": replay_ratio,
        "gaussian_log_loss": float(
            summary["cumulative_prequential_gaussian_log_loss"]
        ),
        "final_median_nmse": float(summary["final_nmse"]["median"]),
        "novel_32_shot_nmse": float(
            summary["novel_composition"]["nmse_by_support"]["32"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/robustness")
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--conditions",
        choices=tuple(CONDITIONS),
        nargs="+",
        default=list(CONDITIONS),
    )
    parser.add_argument(
        "--models", choices=("continuous", "dense"), nargs="+", default=["continuous", "dense"]
    )
    args = parser.parse_args()
    base = load_config(args.config)
    records = []
    total = len(args.conditions) * len(args.models) * len(args.worlds)
    completed = 0
    for condition in args.conditions:
        for world_seed in args.worlds:
            for model in args.models:
                completed += 1
                output = args.output / condition / f"world_{world_seed}" / model
                config, order = _condition_config(
                    base, condition, model, world_seed, output
                )
                print(
                    f"[{completed}/{total}] {condition}/world_{world_seed}/{model}",
                    flush=True,
                )
                summary_path = output / "summary.json"
                if summary_path.exists():
                    validate_artifact(
                        output,
                        resolved_learned_config(config, model, order),
                        model,
                    )
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                else:
                    summary = run(config, kind=model, order=order)
                    gc.collect()
                records.append(
                    _record(
                        condition,
                        model,
                        world_seed,
                        order,
                        CONDITIONS[condition][1],
                        summary,
                    )
                )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "development-world task-order and replay robustness",
        "records": records,
    }
    (args.output / "robustness.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
