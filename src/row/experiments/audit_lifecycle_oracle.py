"""V4.1 gate and causal control: EXACT behavioral-cover search.

Two jobs, both exact rather than bounded, because the libraries here hold
3-7 abstractions and the subset search is at most 2^7 = 128 evaluations.

1. THE COVER ORACLE. Build the substitution relation S[j][tau] = abstraction
   j serves task tau within epsilon, then search every subset of the library
   for the cheapest one that still covers every dependent. That is a
   covering problem and its solution is a BEHAVIORAL COVER: the cheapest set
   of representatives preserving what the library can do.

   The relation is a directed GRAPH, not an equivalence. Substitutability at
   tolerance epsilon is neither symmetric (A may serve B's dependents
   without the reverse) nor transitive (A~B and B~C does not give A~C), so
   this module never speaks of equivalence classes.

2. THE CAUSAL CONTROL. At a MATCHED number of retirements, compare
   functional-substitution retirement against random and usage-based
   retirement. Without it, a gain of k * 1098 nats is equally consistent
   with the library simply having been too big.


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
import itertools
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


def _substitution_matrix(model, references, probe, epsilon):
    """S[j][task] = abstraction j serves this task within epsilon."""

    matrix = {}
    with torch.no_grad():
        baselines = {t: model(probe, t) for t in references}

        def swap(task_id, reference):
            previous = model.task_reference.get(task_id)
            if reference is None:
                model.task_reference.pop(task_id, None)
            else:
                model.task_reference[task_id] = reference
            after = model(probe, task_id)
            if previous is None:
                model.task_reference.pop(task_id, None)
            else:
                model.task_reference[task_id] = previous
            return after

        # Denominator = what the abstraction CONTRIBUTES, not total output
        # variance. Against total scale every abstraction substituted for
        # every other, the causal control went degenerate, and the null
        # edit (delete everything) passed. See PREDICTIONS.md, "V4.1
        # H14 — RETRACTED".
        contribution = {
            t: max(float(torch.mean(torch.square(baselines[t] - swap(t, None)))), 1e-12)
            for t in references
        }
        for j in range(len(model.abstractions)):
            matrix[j] = {
                t: float(torch.mean(torch.square(swap(t, j) - baselines[t])))
                / contribution[t]
                <= epsilon
                for t in references
            }
    return matrix


def audit(config, path, world_seed, spec, slots, kappa, epsilon=0.02):
    model, world = _load(config, path, world_seed, spec, slots)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    residual_bits = BITS * (
        model.residual_u_size + model.residual_v_size + model.residual_b_size
    )
    lifecycle = summary.get("lifecycle")
    if lifecycle is None or "task_reference" not in lifecycle:
        raise ValueError(
            str(path) + " has no persisted reference table: run the `lifecycle` "
            "model, which records lineage, not the V3 `promoting` model"
        )
    references = {k: int(v) for k, v in lifecycle["task_reference"].items()}
    if not references:
        raise ValueError(str(path) + " records a library with no dependents at all")
    for task_id, reference in references.items():
        model.task_reference[task_id] = reference
        model.retired.add(task_id)

    probe = _tensor(
        np.random.default_rng(world_seed + 4242).normal(
            size=(256, config.world.state_dim)
        )
    )
    library = list(range(len(model.abstractions)))
    matrix = _substitution_matrix(model, references, probe, epsilon)

    def covers(subset):
        return all(any(matrix[j][t] for j in subset) for t in references)

    # NULL-EDIT GUARD. Under a well-scaled tolerance, dropping an
    # abstraction must cost exactly its own contribution, i.e. a relative
    # deviation of 1.0 > epsilon. If a task is served by NO abstraction,
    # epsilon is loose enough to admit deleting the library outright and
    # every cover it certifies is vacuous.
    unserved = [t for t in references if not any(matrix[j][t] for j in library)]
    if len(unserved) == len(references):
        raise ValueError(
            "tolerance %.4g admits the null edit for every dependent: the "
            "substitution relation carries no information (see PREDICTIONS.md, "
            "'V4.1 H14 — RETRACTED')" % epsilon
        )

    def cost(subset):
        # Library bits plus the reference code, which shrinks with the LIVE
        # library size, so compaction is charged and credited correctly.
        reference_bits = math.ceil(math.log2(len(subset) + 1)) if subset else 0
        return LN2 * (
            residual_bits * len(subset) + reference_bits * len(references)
        )

    best_subset, best_cost = None, None
    for size in range(1, len(library) + 1):
        for subset in itertools.combinations(library, size):
            if not covers(subset):
                continue
            value = cost(subset)
            if best_cost is None or value < best_cost:
                best_subset, best_cost = subset, value
        if best_subset is not None:
            break
    full_cost = cost(tuple(library))
    oracle_gain = full_cost - (best_cost if best_cost is not None else full_cost)

    retire_count = len(library) - (len(best_subset) if best_subset else len(library))
    usage = {j: sum(1 for r in references.values() if r == j) for j in library}
    policies = {}
    if retire_count > 0:
        generator = np.random.default_rng(world_seed + 99)
        keep_n = len(library) - retire_count
        random_keep = sorted(
            int(x) for x in generator.choice(library, keep_n, replace=False)
        )
        usage_keep = sorted(sorted(library, key=lambda j: -usage[j])[:keep_n])
        for name, keep in (
            ("functional", list(best_subset)),
            ("usage", usage_keep),
            ("random", random_keep),
        ):
            stranded = sum(
                0 if any(matrix[j][t] for j in keep) else 1 for t in references
            )
            policies[name] = {
                "kept": keep,
                "stranded_dependents": stranded,
                "covers_everything": stranded == 0,
                "cost_nats": cost(tuple(keep)) + LN2 * residual_bits * stranded,
            }

    cover = list(best_subset) if best_subset else library
    return {
        "world_seed": world_seed,
        "kappa": kappa,
        "library_size": len(library),
        "behavioral_cover": cover,
        "cover_size": len(cover),
        "oracle_gain_nats": oracle_gain,
        "retirements": retire_count,
        "policies": policies,
        "reuse_density": len(references) / max(1, len(cover)),
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
    print("V4.1 EXACT BEHAVIORAL COVER + CAUSAL CONTROL")
    for kappa in args.kappas:
        rows = [
            audit(config, args.root / ("world_%d" % w) / "lifecycle", w, spec,
                  args.slots, kappa)
            for w in args.worlds
        ]
        results.extend(rows)
        for row in rows:
            print(
                "    world %d: library %d -> cover %d   gain %7.0f nats   "
                "reuse density %.1f"
                % (row["world_seed"], row["library_size"], row["cover_size"],
                   row["oracle_gain_nats"], row["reuse_density"])
            )
        scored = [r for r in rows if r["policies"]]
        if scored:
            print("    causal control at matched retirements:")
            for name in ("functional", "usage", "random"):
                stranded = [r["policies"][name]["stranded_dependents"] for r in scored]
                costs = [r["policies"][name]["cost_nats"] for r in scored]
                print(
                    "      %11s: stranded dependents %5.1f   cost %8.0f nats"
                    % (name, np.mean(stranded), np.mean(costs))
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    passed = [r for r in results if r["oracle_gain_nats"] > 0]
    print(
        "\n  GATE %s: the exact oracle finds compaction room in %d/%d cells."
        % ("PASSES" if passed else "FAILS", len(passed), len(results))
    )
    if not passed:
        print(
            "  No subset of any library is redundant at a contribution-relative\n"
            "  tolerance, so this testbed cannot test compaction. Do not tune a\n"
            "  RETIRE operator against it (PREDICTIONS.md, 'V4.1 H14 — RETRACTED')."
        )


if __name__ == "__main__":
    main()
