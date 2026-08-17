"""Verify that opaque task-token reassignment leaves learned runs unchanged."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from row.config import load_config
from row.experiments.learned_lifetime import resolved_learned_config, run
from row.provenance import validate_artifact


MODELS = ("continuous", "dense")
SCRAMBLE_SEED = 1701


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row.pop("task_id", None)
        rows.append(row)
    return rows


def _normalized_summary(path: Path) -> dict[str, Any]:
    summary = _read_json(path)
    summary.pop("task_id_scramble_seed", None)
    return summary


def _normalized_state(
    path: Path, programs: list[dict[str, Any]]
) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    task_index = {str(row["task_id"]): int(row["task_index"]) for row in programs}
    normalized = {}
    for key, value in state.items():
        if key.startswith("task_codes."):
            task_id = key.removeprefix("task_codes.")
            key = f"task_codes.task_index_{task_index[task_id]}"
        normalized[key] = value
    return normalized


def _compare_artifacts(canonical: Path, scrambled: Path) -> dict[str, Any]:
    canonical_programs = _read_json(canonical / "world_programs.json")
    scrambled_programs = _read_json(scrambled / "world_programs.json")
    canonical_ids = {str(row["task_id"]) for row in canonical_programs}
    scrambled_ids = {str(row["task_id"]) for row in scrambled_programs}
    canonical_hidden = [
        {key: value for key, value in row.items() if key != "task_id"}
        for row in canonical_programs
    ]
    scrambled_hidden = [
        {key: value for key, value in row.items() if key != "task_id"}
        for row in scrambled_programs
    ]

    canonical_state = _normalized_state(canonical / "model.pt", canonical_programs)
    scrambled_state = _normalized_state(scrambled / "model.pt", scrambled_programs)
    state_keys_equal = canonical_state.keys() == scrambled_state.keys()
    unequal_tensors = (
        []
        if not state_keys_equal
        else [
            key
            for key in canonical_state
            if not torch.equal(canonical_state[key], scrambled_state[key])
        ]
    )
    result = {
        "opaque_id_sets_disjoint": canonical_ids.isdisjoint(scrambled_ids),
        "hidden_programs_and_order_exact": canonical_hidden == scrambled_hidden,
        "metrics_rows_exact_after_removing_task_id": (
            _normalized_rows(canonical / "metrics.jsonl")
            == _normalized_rows(scrambled / "metrics.jsonl")
        ),
        "summary_exact_after_removing_scramble_seed": (
            _normalized_summary(canonical / "summary.json")
            == _normalized_summary(scrambled / "summary.json")
        ),
        "model_state_keys_exact_after_task_index_relabel": state_keys_equal,
        "unequal_model_state_tensors": unequal_tensors,
    }
    result["exact_invariance"] = all(
        (
            result["opaque_id_sets_disjoint"],
            result["hidden_programs_and_order_exact"],
            result["metrics_rows_exact_after_removing_task_id"],
            result["summary_exact_after_removing_scramble_seed"],
            result["model_state_keys_exact_after_task_index_relabel"],
            not unequal_tensors,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/scrambled_ids")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/scrambled_ids/scrambled-ids.json"),
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=[0])
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    parser.add_argument("--scramble-seed", type=int, default=SCRAMBLE_SEED)
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
    records = []
    for world_seed in args.worlds:
        for model in args.models:
            roots = {}
            for condition, scramble_seed in (
                ("canonical", None),
                ("scrambled", args.scramble_seed),
            ):
                output = args.output / f"world_{world_seed}" / model / condition
                config = replace(
                    base,
                    world=replace(base.world, seed=world_seed),
                    output_directory=output,
                )
                summary_path = output / "summary.json"
                print(
                    f"world={world_seed} model={model} condition={condition}",
                    flush=True,
                )
                if summary_path.exists():
                    validate_artifact(
                        output,
                        resolved_learned_config(
                            config, model, "forward", scramble_seed
                        ),
                        model,
                    )
                else:
                    run(
                        config,
                        kind=model,
                        task_id_scramble_seed=scramble_seed,
                    )
                    gc.collect()
                roots[condition] = output
            comparison = _compare_artifacts(roots["canonical"], roots["scrambled"])
            if not comparison["exact_invariance"]:
                raise RuntimeError(
                    f"scrambled-ID invariance failed for world {world_seed} {model}"
                )
            records.append(
                {"world_seed": world_seed, "model": model, **comparison}
            )

    report = {
        "scope": "selected exact-reuse development runs; task token is the only intervention",
        "scramble_seed": args.scramble_seed,
        "models": list(args.models),
        "worlds": list(args.worlds),
        "records": records,
        "all_exact": all(record["exact_invariance"] for record in records),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
