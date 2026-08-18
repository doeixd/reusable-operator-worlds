"""Tune explicit presence-gate strength for the MDL-pruned library."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from row.config import load_config
from row.experiments.learned_lifetime import resolved_learned_config, run
from row.provenance import validate_artifact


PRESENCE_LEARNING_RATES = (1e-4, 1e-3)
LIBRARY_PENALTIES = (1e-6, 1e-5, 1e-4)
FINAL_MEDIAN_NMSE_LIMIT = 0.02
NOVEL_32_SHOT_NMSE_LIMIT = 0.02


def _label(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def _select(records: list[dict[str, Any]], operator_slots: int) -> dict[str, Any]:
    sufficient = [
        record
        for record in records
        if int(record["active_operators_at_threshold"]) < operator_slots
        and float(record["final_median_nmse"]) <= FINAL_MEDIAN_NMSE_LIMIT
        and float(record["novel_32_shot_nmse"]) <= NOVEL_32_SHOT_NMSE_LIMIT
    ]
    if not sufficient:
        raise ValueError("no tuning candidate produced a shorter sufficient library")
    selected = min(
        sufficient,
        key=lambda row: (
            int(row["active_operators_at_threshold"]),
            float(row["gaussian_log_loss"]),
        ),
    )
    return {
        "criterion": (
            "fewest threshold-active operators among candidates with final median "
            "and novel 32-shot NMSE <= 0.02; cumulative Gaussian log loss breaks ties"
        ),
        "sufficiency_limits": {
            "final_median_nmse": FINAL_MEDIAN_NMSE_LIMIT,
            "novel_32_shot_nmse": NOVEL_32_SHOT_NMSE_LIMIT,
        },
        "selected": selected,
        "records": sorted(
            records,
            key=lambda row: (
                float(row["presence_learning_rate"]),
                float(row["library_presence_penalty"]),
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mdl_tuning"))
    parser.add_argument("--world-seed", type=int, default=0)
    parser.add_argument(
        "--presence-learning-rates",
        type=float,
        nargs="+",
        default=list(PRESENCE_LEARNING_RATES),
    )
    parser.add_argument(
        "--library-penalties",
        type=float,
        nargs="+",
        default=list(LIBRARY_PENALTIES),
    )
    args = parser.parse_args()

    base = load_config(args.config)
    base = replace(
        base,
        world=replace(base.world, seed=args.world_seed, reuse_rho=1.0),
        evaluation=replace(
            base.evaluation,
            lifetime_checkpoints=(),
            checkpoint_novel_tasks=1,
            extended_diagnostics=False,
        ),
    )
    records = []
    total = len(args.presence_learning_rates) * len(args.library_penalties)
    completed = 0
    for presence_lr in args.presence_learning_rates:
        for library_penalty in args.library_penalties:
            completed += 1
            output = (
                args.output
                / f"presence_lr_{_label(presence_lr)}"
                / f"library_penalty_{_label(library_penalty)}"
            )
            config = replace(
                base,
                mdl_model=replace(
                    base.mdl_model,
                    presence_learning_rate=presence_lr,
                    library_presence_penalty=library_penalty,
                ),
                output_directory=output,
            )
            summary_path = output / "summary.json"
            print(
                f"[{completed}/{total}] presence_lr={presence_lr:g} "
                f"library_penalty={library_penalty:g}",
                flush=True,
            )
            if summary_path.exists():
                validate_artifact(
                    output,
                    resolved_learned_config(config, "mdl", "forward"),
                    "mdl",
                )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                summary = run(config, kind="mdl")
                gc.collect()
            presence = summary["presence"]
            records.append(
                {
                    "world_seed": args.world_seed,
                    "presence_learning_rate": presence_lr,
                    "library_presence_penalty": library_penalty,
                    "route_entropy_penalty": config.mdl_model.route_entropy_penalty,
                    "gaussian_log_loss": summary[
                        "cumulative_prequential_gaussian_log_loss"
                    ],
                    "final_median_nmse": summary["final_nmse"]["median"],
                    "novel_32_shot_nmse": summary["novel_composition"][
                        "nmse_by_support"
                    ]["32"],
                    "expected_active_operators": presence[
                        "expected_active_operators"
                    ],
                    "active_operators_at_threshold": presence[
                        "active_operators_at_threshold"
                    ],
                    "routed_operators": summary["routing"]["active_operators"],
                    "active_but_unused_operators": presence[
                        "active_but_unused_operators"
                    ],
                }
            )

    report = _select(records, base.mdl_model.operator_slots)
    report["scope"] = f"development world {args.world_seed}, exact reuse; tuning only"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tuning.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
