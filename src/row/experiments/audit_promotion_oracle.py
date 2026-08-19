"""Oracle gate: is there anything for PROMOTE to win in this testbed?

Promotion is two operations, not one: ADD shared capacity and REMOVE the
task-local capacity the new shared object makes redundant. An oracle that
only grows the library pays for the new operators and saves nothing, which
is exactly what the first measurement of this gate showed (grown library
predicted better by 3,077 nats but cost 7,312 more bits, losing J by 1,991
in 3/3 worlds).

This computes the full frontier for both conditions — saturated library at
K, and grown library at K + 2 — sweeping the retained residual rank down to
zero and reporting J = L + lambda*D against the behavioral cost. The gate
passes if the grown library reaches a lower J than anything the saturated
library can reach at comparable distortion; only then does an explicit
promotion operator have something to win.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments import learned_lifetime
from row.experiments.learned_lifetime import _build_model, _tensor
from row.metrics import nmse
from row.mixed_world import CANONICAL_PROFILE
from row.task_group_world import TaskGroupSpec, TaskGroupWorldFactory

LN2 = math.log(2.0)
BITS = 8


def _load(config, path: Path, world_seed: int, spec: TaskGroupSpec, slots: int):
    factory = TaskGroupWorldFactory(list(CANONICAL_PROFILE), spec)
    original = learned_lifetime.World
    learned_lifetime.World = factory
    try:
        world = factory.generate(replace(config.world, seed=world_seed, reuse_rho=1.0))
    finally:
        learned_lifetime.World = original
    local = replace(
        config,
        shared_residual_model=replace(
            config.shared_residual_model, operator_slots=slots
        ),
    )
    model = _build_model(local, "shared_residual")
    state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    model.load_state_dict(state)
    model.eval()
    return model, world


@torch.no_grad()
def _scores(model, world) -> np.ndarray:
    return np.array(
        [
            nmse(model(_tensor(task.eval_x), task.task_id).cpu().numpy(), task.eval_y)
            for task in world.tasks
        ]
    )


@torch.no_grad()
def _reduce_rank(model, target_rank: int) -> int:
    """Keep the highest-energy `target_rank` components per task step."""

    retained = 0
    for task_id in model.task_codes:
        _, residual_u, residual_v, residual_b = model._unpack(task_id)
        u, v, b = residual_u.clone(), residual_v.clone(), residual_b.clone()
        for step in range(model.task_steps):
            energy = torch.tensor(
                [
                    float(torch.norm(u[step, :, k]) * torch.norm(v[step, k, :]))
                    for k in range(model.residual_rank)
                ]
            )
            for position, component in enumerate(torch.argsort(energy, descending=True)):
                if position >= target_rank:
                    u[step, :, component] = 0.0
                    v[step, component, :] = 0.0
                    b[step, component] = 0.0
        model.task_residuals[task_id].copy_(
            torch.cat((u.reshape(-1), v.reshape(-1), b.reshape(-1)))
        )
        retained += model.task_steps * min(target_rank, model.residual_rank) * (
            2 * model.d + 1
        )
    return retained


def frontier(config, path: Path, world_seed: int, spec: TaskGroupSpec, slots: int):
    model, world = _load(config, path, world_seed, spec, slots)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    loss = float(summary["cumulative_prequential_gaussian_log_loss"])
    shared_bits = BITS * int(model.shared_parameter_count)
    route_bits = BITS * model.route_size * len(model.task_codes)
    base = _scores(model, world)
    points = []
    for rank in (2, 1, 0):
        clone = copy.deepcopy(model)
        retained = _reduce_rank(clone, rank)
        clone.eval()
        scores = _scores(clone, world)
        task_bits = route_bits + BITS * retained
        total = shared_bits + task_bits
        points.append(
            {
                "residual_rank": rank,
                "shared_bits": shared_bits,
                "task_bits": task_bits,
                "total_bits": total,
                "two_part_objective": loss + LN2 * total,
                "mean_nmse": float(np.mean(scores)),
                "mean_nmse_increase": float(np.mean(scores - base)),
            }
        )
    return {
        "world_seed": world_seed,
        "slots": slots,
        "lifetime_loss": loss,
        "mean_float_nmse": float(np.mean(base)),
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--saturated", type=Path, default=Path("artifacts/v3_taskgroup/eta0.9_k6_onset16_frozen16")
    )
    parser.add_argument(
        "--grown", type=Path, default=Path("artifacts/v3_taskgroup/eta0.9_k8_onset16_frozen16_fs6")
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--onset", type=int, default=16)
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v3_promotion_oracle.json")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    spec = TaskGroupSpec(groups=2, eta=args.eta, future_tasks=8, family_onset=args.onset)
    rows = []
    for world_seed in args.worlds:
        saturated = frontier(
            config, args.saturated / f"world_{world_seed}" / "shared_residual",
            world_seed, spec, 6,
        )
        grown = frontier(
            config, args.grown / f"world_{world_seed}" / "shared_residual",
            world_seed, spec, 8,
        )
        rows.append({"saturated": saturated, "grown": grown})

    print("PROMOTION ORACLE FRONTIER (J = L + ln2 * bits; lower is better)")
    for row in rows:
        world = row["saturated"]["world_seed"]
        print(f"  world {world}:")
        for label in ("saturated", "grown"):
            for point in row[label]["points"]:
                print(
                    f"    {label:>9} K={row[label]['slots']} rank{point['residual_rank']}: "
                    f"J={point['two_part_objective']:>10.0f} bits={point['total_bits']:>7} "
                    f"NMSE={point['mean_nmse']:.5f} (+{point['mean_nmse_increase']:.5f})"
                )
    best_sat = [
        min(row["saturated"]["points"], key=lambda p: p["two_part_objective"])
        for row in rows
    ]
    best_grown = [
        min(row["grown"]["points"], key=lambda p: p["two_part_objective"]) for row in rows
    ]
    gains = [
        s["two_part_objective"] - g["two_part_objective"]
        for s, g in zip(best_sat, best_grown, strict=True)
    ]
    print(
        f"\n  best-J gain from growing the library: mean {np.mean(gains):+.0f} nats, "
        f"positive in {sum(value > 0 for value in gains)}/{len(gains)} worlds"
    )
    print("  GATE PASSES if positive: an expanded library wins the two-part cell,")
    print("  so there is something for an explicit promoter to discover.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
