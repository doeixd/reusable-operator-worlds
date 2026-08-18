"""Post-hoc operator recovery across the reuse continuum (V2 bridge analysis 5).

Rebuilds saved continuous models from the development recurrence sweep and
measures functional primitive recovery against each world's base library,
asking whether partial recurrence yields a shared manifold without
identifiable primitives while exact recurrence crystallizes them.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
from pathlib import Path

import torch
import yaml

from row.experiments.oracle_lifetime import _functional_recovery
from row.experiments.quantize_artifact import _build_from_artifact
from row.models import ContinuousBasisLearner
from row.world import World, WorldConfig


def _load_continuous(artifact: Path) -> tuple[ContinuousBasisLearner, World, object]:
    raw = yaml.safe_load((artifact / "config.yaml").read_text(encoding="utf-8"))
    world_raw = raw["world"]
    assert isinstance(world_raw, dict)
    world = World.generate(WorldConfig(**world_raw))
    model = _build_from_artifact(raw, "continuous")
    assert isinstance(model, ContinuousBasisLearner)
    for task in world.tasks:
        model.begin_task(task.task_id)
    try:
        checkpoint = torch.load(artifact / "model.pt", map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(artifact / "model.pt", map_location="cpu", weights_only=False)
    for key in checkpoint["model_state_dict"]:
        if key.startswith("task_codes.task_novel_composition"):
            model.begin_task(key.removeprefix("task_codes."))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class _ConfigView:
        pass

    view = _ConfigView()
    view.world = WorldConfig(**world_raw)
    return model, world, view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=Path("artifacts/rho_development"))
    parser.add_argument("--output", type=Path, default=Path("reports/rho_operator_recovery"))
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for artifact in sorted(args.sweep.glob("rho_*/world_*/continuous")):
        if not (artifact / "summary.json").exists():
            continue
        rho = float(artifact.parent.parent.name.replace("rho_", "").replace("p", "."))
        world_seed = int(artifact.parent.name.split("_")[1])
        model, world, config = _load_continuous(artifact)
        with torch.no_grad():
            recovery = _functional_recovery(model.basis, world, config)
        explanations = recovery["best_depth_1_to_3_explanations"]
        assert isinstance(explanations, list)
        records.append(
            {
                "world_seed": world_seed,
                "configured_rho": rho,
                "one_to_one_mean_primitive_distance": recovery["one_to_one_mean_distance"],
                "mean_short_explanation_distance": statistics.mean(
                    float(item["normalized_distance"]) for item in explanations
                ),
                "depth_one_explanations": sum(
                    1 for item in explanations if len(item["teacher_route"]) == 1
                ),
            }
        )
        print(
            f"rho={rho:g} world={world_seed}: one-to-one "
            f"{records[-1]['one_to_one_mean_primitive_distance']:.4f}",
            flush=True,
        )

    by_rho: dict[float, list[dict[str, object]]] = {}
    for record in records:
        by_rho.setdefault(float(record["configured_rho"]), []).append(record)
    summaries = [
        {
            "configured_rho": rho,
            "worlds": len(rows),
            "mean_one_to_one_distance": statistics.mean(
                float(row["one_to_one_mean_primitive_distance"]) for row in rows
            ),
            "median_one_to_one_distance": statistics.median(
                float(row["one_to_one_mean_primitive_distance"]) for row in rows
            ),
            "mean_short_explanation_distance": statistics.mean(
                float(row["mean_short_explanation_distance"]) for row in rows
            ),
            "mean_depth_one_explanations": statistics.mean(
                float(row["depth_one_explanations"]) for row in rows
            ),
        }
        for rho, rows in sorted(by_rho.items())
    ]
    report = {
        "scope": "development recurrence sweep continuous artifacts; post-hoc; "
        "recovery measured against each world's base (shared) library",
        "prediction_under_test": "partial recurrence yields useful sharing without "
        "identifiable primitives (mediocre recovery at rho=0.9); exact recurrence "
        "crystallizes identifiable primitives (sharp recovery at rho=1.0)",
        "rho_summaries": summaries,
        "records": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "operator-recovery.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
