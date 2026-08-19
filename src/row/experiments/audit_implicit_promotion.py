"""Is the family structure CAUSALLY carried by shared slots plus references?

The correlational finding is that route codes separate by hidden family far
more strongly than residuals do, and that per-group route mass concentrates
on different basis slots. Correlation in a route vector is weak evidence on
its own, so this runs three interventions on trained artifacts:

  1. EQUALIZE  - replace every task's route with the pooled mean route.
  2. SWAP      - give each task the mean route of the OTHER family, with
                 own-family mean substitution as the control for "any
                 change hurts".
  3. ABLATE    - remove one basis slot at a time (renormalizing the mixture)
                 and look for differential damage by family.

If swapping costs materially more than own-family substitution, and slot
ablation damages the families differentially in the direction their route
mass predicts, then the learner holds shared functional objects addressed by
family-specific references — implicit promotion performed during wake.

Interpretive caution kept in the code because it is easy to overstate: the
shared slots here were PREALLOCATED and already paid for. Nothing charged
the optimizer for using them, so this is not evidence of MDL-driven
promotion. It is evidence that unused shared capacity gets used.
"""

from __future__ import annotations

import argparse
import copy
import json
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


def _load(config, artifact: Path, world_seed: int, spec: TaskGroupSpec, profile):
    factory = TaskGroupWorldFactory(profile, spec)
    original = learned_lifetime.World
    learned_lifetime.World = factory
    try:
        world = factory.generate(replace(config.world, seed=world_seed, reuse_rho=1.0))
    finally:
        learned_lifetime.World = original
    model = _build_model(config, "shared_residual")
    state = torch.load(artifact / "model.pt", weights_only=True)["model_state_dict"]
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    model.load_state_dict(state)
    model.eval()
    return model, world


@torch.no_grad()
def _per_task_nmse(model, world) -> np.ndarray:
    return np.array(
        [
            nmse(model(_tensor(task.eval_x), task.task_id).cpu().numpy(), task.eval_y)
            for task in world.tasks
        ]
    )


@torch.no_grad()
def _group_means(model, world, assignment: np.ndarray) -> dict[int, torch.Tensor]:
    means = {}
    for group in sorted(set(assignment.tolist())):
        codes = [
            model.task_codes[task.task_id].detach()
            for index, task in enumerate(world.tasks)
            if assignment[index] == group
        ]
        means[group] = torch.stack(codes).mean(dim=0)
    return means


def audit(config, artifact: Path, world_seed: int, spec: TaskGroupSpec, profile):
    model, world = _load(config, artifact, world_seed, spec, profile)
    assignment = np.array(world.group_assignment[: len(world.tasks)])
    base = _per_task_nmse(model, world)
    pooled = torch.stack(
        [model.task_codes[task.task_id].detach() for task in world.tasks]
    ).mean(dim=0)
    group_means = _group_means(model, world, assignment)

    def _substitute(codes_for) -> np.ndarray:
        clone = copy.deepcopy(model)
        with torch.no_grad():
            for index, task in enumerate(world.tasks):
                clone.task_codes[task.task_id].copy_(codes_for(index))
        clone.eval()
        return _per_task_nmse(clone, world)

    equalized = _substitute(lambda index: pooled)
    own_group = _substitute(lambda index: group_means[assignment[index]])
    other_group = _substitute(
        lambda index: group_means[1 - assignment[index]]
        if len(group_means) == 2
        else pooled
    )

    ablations = []
    for slot in range(model.operator_slots):
        clone = copy.deepcopy(model)
        with torch.no_grad():
            for task in world.tasks:
                # Route codes are stored flat and reshaped in _unpack.
                view = clone.task_codes[task.task_id].view(
                    clone.task_steps, clone.operator_slots
                )
                view[:, slot] = -1e9
        clone.eval()
        scores = _per_task_nmse(clone, world)
        damage = scores - base
        ablations.append(
            {
                "slot": slot,
                "mean_damage_group0": float(np.mean(damage[assignment == 0])),
                "mean_damage_group1": float(np.mean(damage[assignment == 1])),
                "differential": float(
                    np.mean(damage[assignment == 0]) - np.mean(damage[assignment == 1])
                ),
            }
        )

    return {
        "world_seed": world_seed,
        "eta": spec.eta,
        "groups": spec.groups,
        "baseline_mean_nmse": float(np.mean(base)),
        "equalize_mean_nmse_increase": float(np.mean(equalized - base)),
        "own_group_mean_nmse_increase": float(np.mean(own_group - base)),
        "other_group_mean_nmse_increase": float(np.mean(other_group - base)),
        # The decisive contrast: substituting the WRONG family's reference
        # against substituting the right family's, so the comparison is not
        # confounded by "any substitution degrades".
        "swap_penalty_over_own_group": float(
            np.mean(other_group - base) - np.mean(own_group - base)
        ),
        "slot_ablation": ablations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v3_taskgroup/eta0.9"))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--uniform-rho", type=float, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v3_implicit_promotion.json")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    profile = (
        [args.uniform_rho] * 6 if args.uniform_rho is not None else list(CANONICAL_PROFILE)
    )
    rows = []
    for world_seed in args.worlds:
        artifact = args.root / f"world_{world_seed}" / "shared_residual"
        if not (artifact / "summary.json").exists():
            continue
        spec = TaskGroupSpec(groups=args.groups, eta=args.eta, future_tasks=8)
        rows.append(audit(config, artifact, world_seed, spec, profile))

    print("causal interventions on the reference channel (NMSE increase):")
    for row in rows:
        print(
            f"  world {row['world_seed']}: equalize {row['equalize_mean_nmse_increase']:+.5f} | "
            f"own-family mean {row['own_group_mean_nmse_increase']:+.5f} | "
            f"other-family mean {row['other_group_mean_nmse_increase']:+.5f} | "
            f"swap penalty {row['swap_penalty_over_own_group']:+.5f}"
        )
    swaps = [row["swap_penalty_over_own_group"] for row in rows]
    print(
        f"  mean swap penalty {np.mean(swaps):+.5f}, positive in "
        f"{sum(value > 0 for value in swaps)}/{len(swaps)} worlds"
    )
    for row in rows:
        best = max(row["slot_ablation"], key=lambda entry: abs(entry["differential"]))
        print(
            f"  world {row['world_seed']} most family-differential slot: {best['slot']} "
            f"(group0 {best['mean_damage_group0']:+.5f} vs group1 {best['mean_damage_group1']:+.5f})"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
