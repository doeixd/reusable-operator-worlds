"""Run a resumable second-initialization sweep for selected ROW models."""

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


Model = Literal["continuous", "dense"]
SECOND_SEEDS: dict[Model, int] = {"continuous": 4001, "dense": 3001}


def _seeded_config(
    base: ExperimentConfig,
    model: Model,
    world_seed: int,
    model_seed: int,
    output: Path,
) -> ExperimentConfig:
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
        return replace(
            config,
            continuous_model=replace(config.continuous_model, seed=model_seed),
        )
    return replace(config, dense_model=replace(config.dense_model, seed=model_seed))


def _record(
    model: Model, world_seed: int, model_seed: int, summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model": model,
        "world_seed": world_seed,
        "model_seed": model_seed,
        "gaussian_log_loss": float(
            summary["cumulative_prequential_gaussian_log_loss"]
        ),
        "novel_32_shot_nmse": float(
            summary["novel_composition"]["nmse_by_support"]["32"]
        ),
        "final_median_nmse": float(summary["final_nmse"]["median"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/model_initializations")
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--models", choices=("continuous", "dense"), nargs="+", default=["continuous", "dense"]
    )
    parser.add_argument("--continuous-seed", type=int, default=4001)
    parser.add_argument("--dense-seed", type=int, default=3001)
    args = parser.parse_args()
    seeds: dict[Model, int] = {
        "continuous": args.continuous_seed,
        "dense": args.dense_seed,
    }
    base = load_config(args.config)
    records = []
    total = len(args.worlds) * len(args.models)
    completed = 0
    for world_seed in args.worlds:
        for model in args.models:
            completed += 1
            model_seed = seeds[model]
            output = args.output / f"world_{world_seed}" / model
            config = _seeded_config(
                base, model, world_seed, model_seed, output
            )
            print(f"[{completed}/{total}] world_{world_seed}/{model}", flush=True)
            summary_path = output / "summary.json"
            if summary_path.exists():
                validate_artifact(
                    output,
                    resolved_learned_config(config, model, "forward"),
                    model,
                )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                summary = run(config, kind=model)
                gc.collect()
            records.append(_record(model, world_seed, model_seed, summary))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "initializations.json").write_text(
        json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
