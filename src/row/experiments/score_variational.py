"""Score prediction P-2026-08-18-A: the variational wake learner.

Reports all three V3 currencies (V3 spec 4.2) for the variational learner
against the frozen fixed architectures on canonical mixed worlds, plus the
rate-distortion frontier of the sparse two-part code: the pruning threshold
is swept and the largest bit saving that stays inside the H11.1
non-inferiority margin is reported, rather than quoting a saving the
behavior does not support.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments import learned_lifetime
from row.experiments.learned_lifetime import _build_model, _tensor
from row.metrics import nmse
from row.mixed_world import CANONICAL_PROFILE, MixedWorldFactory

LN2 = math.log(2.0)
BITS_PER_SCALAR = 8
BASELINES = ("shared_residual", "continuous", "dense")
# The two fixed architectures the shared-residual envelope is measured
# against; P-2026-08-18-A is about beating BOTH of these under the
# literal two-part code.
FIXED = ("continuous", "dense")


def _retained_bits(summary: dict) -> int:
    return BITS_PER_SCALAR * (
        int(summary["shared_parameter_count"]) + int(summary["task_state_scalar_count"])
    )


def _two_part_objective(loss: float, bits: int) -> float:
    return loss + LN2 * bits


def _load_world(config, world_seed: int):
    factory = MixedWorldFactory(list(CANONICAL_PROFILE))
    original = learned_lifetime.World
    learned_lifetime.World = factory
    try:
        from dataclasses import replace

        return factory.generate(replace(config.world, seed=world_seed))
    finally:
        learned_lifetime.World = original


def _rebuild(config, artifact: Path, world):
    model = _build_model(config, "variational")
    for task in world.tasks:
        model.begin_task(task.task_id)
    state = torch.load(artifact / "model.pt", weights_only=True)["model_state_dict"]
    for key in state:
        if key.startswith("task_codes.") or key.startswith("task_residuals."):
            task_id = key.split(".", 1)[1]
            if task_id not in model.task_codes:
                model.begin_task(task_id)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _task_nmse(model, world) -> np.ndarray:
    return np.array(
        [
            nmse(model(_tensor(task.eval_x), task.task_id).cpu().numpy(), task.eval_y)
            for task in world.tasks
        ]
    )


def _frontier(config, artifact: Path, world, thresholds, margin: float):
    model = _rebuild(config, artifact, world)
    base = _task_nmse(model, world)
    shared_bits = BITS_PER_SCALAR * int(model.shared_parameter_count)
    total_task_scalars = int(model.task_state_scalar_count)
    points = []
    for threshold in thresholds:
        pruned = copy.deepcopy(model)
        report = pruned.apply_information_prune(threshold)
        scores = _task_nmse(pruned, world)
        retained = int(report["retained_task_scalars"])
        points.append(
            {
                "threshold_bits": threshold,
                "retained_task_scalars": retained,
                # Sparse code: 8 bits per retained scalar plus a one-bit
                # presence bitmap over all coordinates.
                "task_bits": BITS_PER_SCALAR * retained + total_task_scalars,
                "total_bits": BITS_PER_SCALAR * retained
                + total_task_scalars
                + shared_bits,
                "mean_nmse": float(np.mean(scores)),
                "mean_nmse_increase": float(np.mean(scores - base)),
                "maximum_task_nmse_increase": float(np.max(scores - base)),
                "within_margin": bool(float(np.mean(scores - base)) <= margin),
            }
        )
    dense_total = BITS_PER_SCALAR * total_task_scalars + shared_bits
    within = [point for point in points if point["within_margin"]]
    best = min(within, key=lambda point: point["total_bits"]) if within else None
    return {
        "dense_two_part_total_bits": dense_total,
        "mean_float_nmse": float(np.mean(base)),
        "non_inferiority_margin": margin,
        "points": points,
        "best_within_margin": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--variational-root", type=Path, default=Path("artifacts/v3_variational/canonical")
    )
    parser.add_argument(
        "--baseline-root", type=Path, default=Path("artifacts/v2_mixed/canonical")
    )
    parser.add_argument("--margin", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("reports/v3_variational.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    thresholds = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
    per_world = []
    for world_seed in args.worlds:
        artifact = args.variational_root / f"world_{world_seed}" / "variational"
        summary = json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
        baselines = {
            name: json.loads(
                (args.baseline_root / f"world_{world_seed}" / name / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            for name in BASELINES
        }
        world = _load_world(config, world_seed)
        frontier = _frontier(config, artifact, world, thresholds, args.margin)

        losses = {
            "variational": float(summary["cumulative_prequential_gaussian_log_loss"]),
            **{
                name: float(value["cumulative_prequential_gaussian_log_loss"])
                for name, value in baselines.items()
            },
        }
        bits = {
            "variational_dense_code": frontier["dense_two_part_total_bits"],
            **{name: _retained_bits(value) for name, value in baselines.items()},
        }
        if frontier["best_within_margin"] is not None:
            bits["variational_sparse_code"] = frontier["best_within_margin"]["total_bits"]
        objectives = {
            "variational_dense_code": _two_part_objective(
                losses["variational"], bits["variational_dense_code"]
            ),
            **{
                name: _two_part_objective(losses[name], bits[name])
                for name in BASELINES
            },
        }
        if "variational_sparse_code" in bits:
            objectives["variational_sparse_code"] = _two_part_objective(
                losses["variational"], bits["variational_sparse_code"]
            )
        best_variational = min(
            value
            for key, value in objectives.items()
            if key.startswith("variational")
        )
        # Envelope gain: how much of shared-residual's raw prequential
        # advantage over the better fixed architecture the variational
        # learner retains.
        better_fixed = min(losses[name] for name in FIXED)
        envelope = better_fixed - losses["shared_residual"]
        retained_share = (
            (better_fixed - losses["variational"]) / envelope if envelope else float("nan")
        )
        per_world.append(
            {
                "world_seed": world_seed,
                "losses": losses,
                "retained_bits": bits,
                "two_part_objective": objectives,
                "beats_both_fixed_under_two_part": bool(
                    all(best_variational < objectives[name] for name in FIXED)
                ),
                "beats_shared_residual_under_two_part": bool(
                    best_variational < objectives["shared_residual"]
                ),
                "envelope_gain_nats": envelope,
                "retained_envelope_share": retained_share,
                "variational_diagnostics": {
                    key: value
                    for key, value in summary["variational"].items()
                    if key not in {"per_task_kl_bits", "sparse_two_part"}
                },
                "rate_distortion": frontier,
            }
        )

    verdict = {
        "prediction": "P-2026-08-18-A",
        "claim": (
            "variational wake learner beats BOTH fixed architectures under the "
            "literal two-part code at lambda = ln 2, retaining at least half of "
            "the raw prequential envelope gain"
        ),
        "worlds_beating_both_fixed": sum(
            row["beats_both_fixed_under_two_part"] for row in per_world
        ),
        "worlds_total": len(per_world),
        "mean_retained_envelope_share": float(
            np.mean([row["retained_envelope_share"] for row in per_world])
        ),
        "per_world": per_world,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in verdict.items() if k != "per_world"}, indent=2))
    for row in per_world:
        best = row["rate_distortion"]["best_within_margin"]
        print(
            f"world {row['world_seed']}: beats_both={row['beats_both_fixed_under_two_part']} "
            f"envelope_share={row['retained_envelope_share']:.2f} "
            f"best_within_margin_bits={best['total_bits'] if best else None} "
            f"dense_bits={row['rate_distortion']['dense_two_part_total_bits']}"
        )


if __name__ == "__main__":
    main()
