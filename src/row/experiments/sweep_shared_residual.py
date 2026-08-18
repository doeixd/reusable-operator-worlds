"""Run the selected shared-parent residual control across recurrence levels."""

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


ESCAPE_HATCH_MAX_RATIO = 1.0
TUNING_ARTIFACT = Path(
    "artifacts/shared_residual_tuning/residual_lr_0p01/penalty_0p01"
)


def _rho_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _record(summary: dict[str, Any], world_seed: int, rho: float) -> dict[str, Any]:
    diagnostic = summary["residual_diagnostics"]
    maximum_ratio = float(
        diagnostic["maximum_task_functional_residual_to_parent_update_ratio"]
    )
    if maximum_ratio >= ESCAPE_HATCH_MAX_RATIO:
        raise ValueError(
            f"world={world_seed}, rho={rho:g} violates the residual escape guard: "
            f"maximum functional ratio={maximum_ratio:.6g}"
        )
    return {
        "world_seed": world_seed,
        "configured_rho": rho,
        "measured_residual_correlation": summary["world_functional_reuse"][
            "mean_pairwise_residual_correlation"
        ],
        "gaussian_log_loss": summary[
            "cumulative_prequential_gaussian_log_loss"
        ],
        "final_median_nmse": summary["final_nmse"]["median"],
        "novel_32_shot_nmse": summary["novel_composition"]["nmse_by_support"][
            "32"
        ],
        "mean_functional_ratio": diagnostic[
            "mean_functional_residual_to_parent_update_ratio"
        ],
        "maximum_task_functional_ratio": maximum_ratio,
        "mean_residual_output_rms": diagnostic["mean_residual_output_rms"],
        "mean_route_entropy_nats": summary["routing"]["mean_entropy_nats"],
        "mean_route_max_coefficient": summary["routing"][
            "mean_max_coefficient"
        ],
        "shared_parameter_count": summary["shared_parameter_count"],
        "task_state_scalar_count": summary["task_state_scalar_count"],
        "training_forward_multiply_adds_per_sample": summary[
            "compute_accounting"
        ]["training_forward_multiply_adds_per_sample"],
        "inference_multiply_adds_per_sample": summary["compute_accounting"][
            "inference_multiply_adds_per_sample"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/shared_residual_rho")
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--rhos", type=float, nargs="+", default=[0.5, 0.75, 0.9, 1.0]
    )
    parser.add_argument(
        "--reuse-tuning",
        action="store_true",
        help="reuse the validated selected world-0 rho-0.75 tuning artifact",
    )
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
    total = len(args.worlds) * len(args.rhos)
    completed = 0
    for world_seed in args.worlds:
        for rho in args.rhos:
            completed += 1
            output = (
                args.output / f"rho_{_rho_label(rho)}" / f"world_{world_seed}"
            )
            config = replace(
                base,
                world=replace(base.world, seed=world_seed, reuse_rho=rho),
                output_directory=output,
            )
            summary_path = output / "summary.json"
            print(
                f"[{completed}/{total}] world={world_seed} rho={rho:g}", flush=True
            )
            if summary_path.exists():
                validate_artifact(
                    output,
                    resolved_learned_config(
                        config, "shared_residual", "forward"
                    ),
                    "shared_residual",
                )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            elif args.reuse_tuning and world_seed == 0 and rho == 0.75:
                imported_config = replace(config, output_directory=TUNING_ARTIFACT)
                validate_artifact(
                    TUNING_ARTIFACT,
                    resolved_learned_config(
                        imported_config, "shared_residual", "forward"
                    ),
                    "shared_residual",
                )
                summary = json.loads(
                    (TUNING_ARTIFACT / "summary.json").read_text(encoding="utf-8")
                )
                print(f"  reused {TUNING_ARTIFACT}", flush=True)
            else:
                summary = run(config, kind="shared_residual")
                gc.collect()
            records.append(_record(summary, world_seed, rho))

    report = {
        "scope": (
            f"development worlds {sorted(args.worlds)}, configured rho values "
            f"{sorted(args.rhos)}; descriptive control, not confirmatory"
        ),
        "escape_hatch_max_ratio": ESCAPE_HATCH_MAX_RATIO,
        "selected_configuration": {
            "residual_rank": base.shared_residual_model.residual_rank,
            "residual_learning_rate": (
                base.shared_residual_model.residual_learning_rate
            ),
            "residual_penalty": base.shared_residual_model.residual_penalty,
        },
        "records": sorted(
            records,
            key=lambda row: (float(row["configured_rho"]), int(row["world_seed"])),
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sweep.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
