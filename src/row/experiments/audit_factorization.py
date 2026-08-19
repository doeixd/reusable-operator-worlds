"""V4.2 gate: can distinct abstractions share a parameterized parent?

V4.1's two gates both failed on this testbed, and the failures were
informative rather than merely negative. There is no REDUNDANCY to
eliminate (no abstraction substitutes for another at a
contribution-relative tolerance, and re-homing is net negative), and no
OBSOLESCENCE to collect (the dormancy pair's refusal control does not
refuse). What survives is that the library holds several behaviorally
DISTINCT abstractions where the teacher has two hidden families.

V4.2 asks whether those distinct objects are irreducibly distinct or are
instantiations of a smaller basis:

    A_i(z)  ~=  A(z ; alpha_i)  =  C(z) + sum_k alpha_ik B_k(z)

This is anti-unification, not deletion. Nothing is forgotten; the
vocabulary changes from a set of atoms to one operator with arguments.

THE COMPARISON LADDER. A rank-K fit that merely approximates everything
badly is not a discovery, so the audit reports four representations, and
only the contrast between them supports a claim:

    original A_i          behavioral ceiling
    C + alpha_i B         the factorization hypothesis
    C alone (rank 0)      generic collapse -- the null that a centroid
                          explains the library
    no abstraction        deletion, the floor

The gate passes only if `C + alpha_i B` is close to the ceiling AND
materially better than rank-0 collapse. Rank 0 beating the floor by
itself would only show that the abstractions have a nonzero mean.

BOTH CURRENCIES, ALWAYS. Bits saved is reported next to held-out
Gaussian loss paid. The retracted V4.1 H14 result was produced by
counting a structural gain without pricing its behavioral cost, so a
positive bit count alone is never a pass here.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments import learned_lifetime
from row.experiments.learned_lifetime import _build_model, _tensor
from row.metrics import gaussian_nll
from row.mixed_world import CANONICAL_PROFILE
from row.task_group_world import TaskGroupSpec, TaskGroupWorldFactory

LN2 = math.log(2.0)


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
        shared_residual_model=replace(config.shared_residual_model, operator_slots=slots),
    )
    model = _build_model(local, "lifecycle")
    state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
    count = sum(1 for key in state if key.startswith("abstractions."))
    for index in range(count):
        model.abstractions.append(
            torch.nn.Parameter(state[f"abstractions.{index}"].clone(), requires_grad=False)
        )
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    model.load_state_dict(state)
    model.eval()
    return model, world


@torch.no_grad()
def _loss(model, tasks, references) -> float:
    total = 0.0
    for task_id in references:
        task = tasks.get(task_id)
        if task is None:
            continue
        total += gaussian_nll(
            model(_tensor(task.eval_x), task_id).cpu().numpy(), task.eval_y, 0.1
        )
    return total


def audit(config, path: Path, world_seed: int, spec: TaskGroupSpec, slots: int,
          ranks=(0, 1, 2, 4)) -> dict:
    model, world = _load(config, path, world_seed, spec, slots)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    lifecycle = summary.get("lifecycle")
    if lifecycle is None or "task_reference" not in lifecycle:
        raise ValueError(f"{path} has no persisted reference table")
    references = {k: int(v) for k, v in lifecycle["task_reference"].items()}
    for task_id, reference in references.items():
        model.task_reference[task_id] = reference
        model.retired.add(task_id)
    tasks = {task.task_id: task for task in world.tasks}
    probe = _tensor(
        np.random.default_rng(world_seed + 4242).normal(size=(256, config.world.state_dim))
    )

    original = [p.detach().clone() for p in model.abstractions]
    ceiling = _loss(model, tasks, references)

    # The floor: every dependent loses its abstraction entirely.
    kept = dict(model.task_reference)
    for task_id in list(model.task_reference):
        model.task_reference.pop(task_id)
    floor = _loss(model, tasks, references)
    model.task_reference.update(kept)

    width = model.residual_u_size + model.residual_v_size + model.residual_b_size
    rows = []
    for rank in ranks:
        if rank == 0:
            with torch.no_grad():
                centre = torch.stack(original).mean(dim=0)
                rebuilt = torch.stack([centre] * len(original))
            bits_after = 8 * width
            distortion = None
        else:
            report = model.factorize(probe, rank=rank)
            rebuilt = report["rebuilt"]
            bits_after = report["bits_after"]
            distortion = report["relative_distortion"]
        with torch.no_grad():
            for index in range(len(model.abstractions)):
                model.abstractions[index].copy_(rebuilt[index])
            value = _loss(model, tasks, references)
            for index in range(len(model.abstractions)):
                model.abstractions[index].copy_(original[index])
        bits_before = 8 * width * len(original)
        rows.append(
            {
                "rank": rank,
                "loss": value,
                "loss_cost_vs_ceiling": value - ceiling,
                "relative_distortion": distortion,
                "bits_before": bits_before,
                "bits_after": bits_after,
                "bits_saved_nats": LN2 * (bits_before - bits_after),
                "net_nats": LN2 * (bits_before - bits_after) - (value - ceiling),
            }
        )
    return {
        "world_seed": world_seed,
        "abstractions": len(original),
        "ceiling_loss": ceiling,
        "floor_loss": floor,
        "contribution_nats": floor - ceiling,
        "ranks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v4_dev/structured"))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arm", default="lifecycle")
    parser.add_argument("--onset", type=int, default=16)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--output", type=Path, default=Path("reports/v4_factorization.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    spec = TaskGroupSpec(
        groups=2, eta=0.9, future_tasks=8, family_onset=args.onset,
        new_primitive_families=True,
    )
    results = []
    print("V4.2 FACTORIZATION GATE   A_i(z) ~= C(z) + sum_k alpha_ik B_k(z)")
    print("  (both currencies: bits saved MINUS held-out nats paid)")
    for world in args.worlds:
        row = audit(config, args.root / f"world_{world}" / args.arm, world, spec,
                    args.slots, tuple(args.ranks))
        results.append(row)
        print(
            "\n  world %d: %d abstractions, worth %,.0f nats over the floor"
            .replace(",.0f", ".0f")
            % (world, row["abstractions"], row["contribution_nats"])
        )
        print("    %-6s %12s %12s %10s" % ("rank", "nats paid", "bits saved", "net"))
        for entry in row["ranks"]:
            print(
                "    %-6d %12.0f %12.0f %10.0f"
                % (entry["rank"], entry["loss_cost_vs_ceiling"],
                   entry["bits_saved_nats"], entry["net_nats"])
            )

    def best(row, rank):
        return next(e for e in row["ranks"] if e["rank"] == rank)

    wins = [
        r for r in results
        if any(best(r, k)["net_nats"] > 0 for k in args.ranks if k > 0)
        and max(best(r, k)["net_nats"] for k in args.ranks if k > 0)
        > best(r, 0)["net_nats"]
    ]
    print(
        "\n  GATE %s: a rank>0 family beats BOTH the ceiling cost and the\n"
        "  rank-0 collapse null in %d/%d worlds."
        % ("PASSES" if len(wins) == len(results) else "FAILS", len(wins), len(results))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
