"""V4.1 world-validity gate: is there room for TIMED deletion?

The spec (§4.3, §10 step 1) requires this before any operator is tuned. It
takes the online learner's BIRTHS as given and optimises each
abstraction's deletion time independently, scanning

    t_d in {consolidation points} union {infinity}

which is O(|L| * T) and affordable. The earlier "retain forever versus
delete at birth" formulation is rejected because it is blind to the case
V4 exists to study — useful early, obsolete later — and it was to be the
gate, so it could have declared a world unable to test DELETE precisely
where timed deletion is the whole opportunity.

Accounting follows §0.2's extended objective:

    J = L + lambda*D_T + kappa*SUM_t D_live(t)

Deleting abstraction A at t_d:
  * removes A's bits from the final description;
  * ends A's occupancy at t_d instead of at T;
  * forces every dependent to fall back, which restores a private residual
    (bits back) and costs whatever prediction the abstraction was buying.

The fallback is priced PESSIMISTICALLY (dependents lose the abstraction's
contribution entirely rather than re-adapting from replay). That makes the
gate conservative in the correct direction: it understates deletion's
value, so a gate that passes anyway really passes.
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
from row.metrics import gaussian_nll
from row.mixed_world import CANONICAL_PROFILE
from row.task_group_world import TaskGroupSpec, TaskGroupWorldFactory

LN2 = math.log(2.0)
BITS = 8
SLEEPS = (24, 32, 48, 64)


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
def _task_loss(model, task, sigma: float) -> float:
    """Held-out Gaussian log loss for one task, in nats."""

    prediction = model(_tensor(task.eval_x), task.task_id).cpu().numpy()
    return gaussian_nll(prediction, task.eval_y, sigma)


def audit(config, path: Path, world_seed: int, spec: TaskGroupSpec, slots: int,
          kappa: float) -> dict[str, object]:
    model, world = _load(config, path, world_seed, spec, slots)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    promotion = summary["promotion"]
    sigma = config.evaluation.gaussian_sigma
    tasks = len(world.tasks)
    residual_bits = BITS * (
        model.residual_u_size + model.residual_v_size + model.residual_b_size
    )

    # Birth time per abstraction, in order of promotion.
    births = [
        record["lifetime_index"]
        for record in promotion["ledger"]
        if record["decision"] == "promote"
    ]
    index_of = {task.task_id: i for i, task in enumerate(world.tasks)}

    # Dependencies come from the ARTIFACT, never from the reloaded model.
    # `task_reference` is a plain dict and is absent from `state_dict`, so a
    # reloaded model reports an empty reference table — which silently turns
    # this audit into an analysis of an unused library where deletion is
    # trivially free. A first run of this gate "passed" that way, reporting
    # 0 dependents for every abstraction. Refusing to run is the only safe
    # behavior, because the failure mode looks exactly like success.
    lifecycle = summary.get("lifecycle")
    if lifecycle is None or "task_reference" not in lifecycle:
        raise ValueError(
            f"{path} has no persisted reference table: run the `lifecycle` "
            "model, which records lineage, rather than the V3 `promoting` model"
        )
    dependents: dict[int, list[str]] = {}
    for task_id, reference in lifecycle["task_reference"].items():
        dependents.setdefault(int(reference), []).append(task_id)
    if not dependents:
        raise ValueError(f"{path} records a library with no dependents at all")

    rows = []
    for abstraction in range(len(model.abstractions)):
        members = dependents.get(abstraction, [])
        born = births[abstraction] if abstraction < len(births) else 0
        # Prediction cost of losing this abstraction, measured directly.
        clone = copy.deepcopy(model)
        with torch.no_grad():
            for task_id in members:
                clone.task_reference.pop(task_id, None)
                clone.retired.discard(task_id)
        clone.eval()
        loss_delta = 0.0
        for task_id in members:
            task = world.tasks[index_of[task_id]]
            loss_delta += _task_loss(clone, task, sigma) - _task_loss(model, task, sigma)

        best = None
        for deletion_time in [*SLEEPS, tasks]:
            if deletion_time < born:
                continue
            keeps = deletion_time >= tasks
            # Final description: the abstraction's own bits, and the private
            # residuals its dependents must carry once it is gone.
            delta_final = 0.0 if keeps else (-residual_bits + residual_bits * len(members))
            # Occupancy: bit-time the abstraction holds.
            occupancy_kept = residual_bits * (tasks - born)
            occupancy_now = residual_bits * ((tasks if keeps else deletion_time) - born)
            delta_occupancy = occupancy_now - occupancy_kept
            delta_j = (
                (0.0 if keeps else loss_delta)
                + LN2 * delta_final
                + kappa * delta_occupancy
            )
            if best is None or delta_j < best["delta_j"]:
                best = {
                    "deletion_time": None if keeps else deletion_time,
                    "delta_j": delta_j,
                }
        rows.append(
            {
                "abstraction": abstraction,
                "born_at": born,
                "dependents": len(members),
                "loss_cost_of_losing_it": loss_delta,
                "best_deletion_time": best["deletion_time"],
                "best_delta_j": best["delta_j"],
            }
        )

    oracle_gain = -sum(min(0.0, row["best_delta_j"]) for row in rows)
    return {
        "world_seed": world_seed,
        "kappa": kappa,
        "library_size": len(model.abstractions),
        "abstractions": rows,
        "deletions_that_pay": sum(1 for row in rows if row["best_delta_j"] < 0),
        "oracle_gain_nats": oracle_gain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/v4_dev/structured"),
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--onset", type=int, default=16)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument(
        "--kappas", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v4_lifecycle_oracle.json")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    spec = TaskGroupSpec(
        groups=2, eta=0.9, future_tasks=8, family_onset=args.onset,
        new_primitive_families=True,
    )
    results = []
    print("V4.1 WORLD-VALIDITY GATE: is there room for timed deletion?")
    for kappa in args.kappas:
        rows = [
            audit(config, args.root / f"world_{w}" / "lifecycle", w, spec, args.slots, kappa)
            for w in args.worlds
        ]
        results.extend(rows)
        gains = [row["oracle_gain_nats"] for row in rows]
        pays = [row["deletions_that_pay"] for row in rows]
        libs = [row["library_size"] for row in rows]
        print(
            f"  kappa={kappa:<7g} oracle gain {np.mean(gains):>9.0f} nats  "
            f"deletions that pay {np.mean(pays):.1f} of {np.mean(libs):.1f} abstractions"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n  GATE PASSES if the oracle finds material room for timed deletion.")


if __name__ == "__main__":
    main()
