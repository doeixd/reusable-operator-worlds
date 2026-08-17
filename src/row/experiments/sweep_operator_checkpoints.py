"""Run exact-reuse operator-quality checkpoints for reusable learners."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

from row.config import load_config
from row.experiments.learned_lifetime import resolved_learned_config, run
from row.provenance import validate_artifact


CHECKPOINTS = (8, 16, 32, 64)
MODELS = ("continuous", "discrete")
METRICS = (
    "one_to_one_mean_primitive_distance",
    "true_route_all_programs_nmse_mean",
    "true_route_completed_programs_nmse_mean",
    "true_route_future_programs_nmse_mean",
    "learned_route_completed_programs_nmse_mean",
    "learned_minus_true_route_completed_nmse",
)


def _operator_curve(summary: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoints = summary["novel_composition_checkpoints"]
    observed = tuple(int(item["tasks_completed"]) for item in checkpoints)
    if observed != CHECKPOINTS:
        raise ValueError(f"checkpoint sequence {observed} does not match {CHECKPOINTS}")
    curve = []
    for item in checkpoints:
        if "true_route_operator_quality" not in item:
            raise ValueError("artifact predates true-route operator diagnostics")
        curve.append(dict(item["true_route_operator_quality"]))
    return curve


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    worlds = sorted({int(record["world_seed"]) for record in records})
    by_key = {
        (int(record["world_seed"]), str(record["model"])): record
        for record in records
    }
    for world in worlds:
        missing = {(world, model) for model in MODELS} - set(by_key)
        if missing:
            raise ValueError(f"world {world} is missing models: {sorted(missing)}")

    summaries = []
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        by_model = {}
        for model in MODELS:
            model_summary: dict[str, Any] = {}
            for metric in METRICS:
                values = [
                    by_key[(world, model)]["operator_checkpoints"][checkpoint_index][
                        metric
                    ]
                    for world in worlds
                ]
                finite = [float(value) for value in values if value is not None]
                model_summary[f"mean_{metric}"] = fmean(finite) if finite else None
                model_summary[f"per_world_{metric}"] = [
                    {"world_seed": world, "value": value}
                    for world, value in zip(worlds, values, strict=True)
                ]
            by_model[model] = model_summary
        summaries.append(
            {"tasks_completed": checkpoint, "models": by_model}
        )

    return {
        "scope": f"exact-reuse development worlds {worlds}; post-hoc analysis only",
        "method": (
            "Hungarian-match learned slots to hidden teacher primitives on held-out "
            "probes, then execute programs through the matched slots using hidden "
            "routes. Hidden identities and routes never enter training."
        ),
        "records": sorted(
            records, key=lambda row: (int(row["world_seed"]), str(row["model"]))
        ),
        "checkpoint_summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/operator_checkpoints")
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--models", choices=MODELS, nargs="+", default=list(MODELS)
    )
    args = parser.parse_args()

    base = load_config(args.config)
    if base.world.reuse_rho != 1.0:
        raise ValueError("operator checkpoint sweep requires exact reuse (rho=1)")
    if base.evaluation.lifetime_checkpoints != CHECKPOINTS:
        raise ValueError("config does not contain the frozen checkpoint sequence")
    base = replace(
        base,
        evaluation=replace(base.evaluation, checkpoint_novel_tasks=1),
        discrete_model=replace(base.discrete_model, temperature_schedule="per_task"),
    )

    records: list[dict[str, Any]] = []
    total = len(args.worlds) * len(args.models)
    completed = 0
    for world_seed in args.worlds:
        for model in args.models:
            completed += 1
            output = args.output / f"world_{world_seed}" / model
            config = replace(
                base,
                world=replace(base.world, seed=world_seed),
                output_directory=output,
            )
            summary_path = output / "summary.json"
            print(f"[{completed}/{total}] world={world_seed} model={model}", flush=True)
            if summary_path.exists():
                validate_artifact(
                    output, resolved_learned_config(config, model, "forward"), model
                )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                summary = run(config, kind=model)
                gc.collect()
            records.append(
                {
                    "world_seed": world_seed,
                    "model": model,
                    "operator_checkpoints": _operator_curve(summary),
                }
            )

    report = _aggregate(records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sweep.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["checkpoint_summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
