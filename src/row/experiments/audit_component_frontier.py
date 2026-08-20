"""V5.0: the rate-distortion frontier of each representational component.

Everything downstream should be priced in `D*(R)` -- the bits a component
actually needs -- not in the arbitrary 8-bit serialization this project
has used as a retention proxy. The V4R coding audit already showed that
proxy overstates description length several-fold; this module produces
the replacement currency, per component and per tolerance.

    R_c(epsilon) = min bits/scalar such that Delta L_c <= epsilon

with `Delta L_c` evaluated ONLY on the computations that depend on
component `c`. That restriction is not cosmetic. Scoring a component
over tasks it does not participate in returns 0.0 at every bit depth,
which reads as "infinitely compressible" and actually means "never
measured"; that vacuous result was produced once during the V4R audit
and is the reason each component here carries an explicit participant
set and a non-vacuity assertion.

Components, matching the scorer's retained-description scope:

    private    per-task residuals of tasks NOT using an abstraction
    shared     promoted abstractions
    basis      the shared operator basis
    routes     per-task route/code state (retained by every task)

Reported as bits/scalar and as total nats at lambda = ln 2, so the four
can be summed into a component-resolved D* for any tolerance.
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
DEPTHS = (1, 2, 3, 4, 5, 6, 8)


def quantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    levels = max(2, 2 ** bits)
    scale = float(tensor.abs().max().clamp_min(1e-12))
    step = 2 * scale / (levels - 1)
    return torch.round(tensor / step) * step


def _load(config, kind: str, path: Path, seed: int, spec: TaskGroupSpec, slots: int):
    factory = TaskGroupWorldFactory(list(CANONICAL_PROFILE), spec)
    original = learned_lifetime.World
    learned_lifetime.World = factory
    try:
        world = factory.generate(replace(config.world, seed=seed, reuse_rho=1.0))
    finally:
        learned_lifetime.World = original
    local = replace(
        config, shared_residual_model=replace(config.shared_residual_model, operator_slots=slots)
    )
    model = _build_model(local, kind)
    state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
    count = sum(1 for k in state if k.startswith("abstractions."))
    if count and hasattr(model, "abstractions"):
        for index in range(count):
            model.abstractions.append(
                torch.nn.Parameter(state[f"abstractions.{index}"].clone(), requires_grad=False)
            )
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    model.load_state_dict(state)
    model.eval()
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    table = summary.get("lifecycle", {}).get("task_reference") or \
        summary.get("reference_table", {}).get("task_reference") or {}
    for task_id, reference in table.items():
        model.task_reference[task_id] = int(reference)
        model.retired.add(task_id)
    return model, world


@torch.no_grad()
def _loss(model, tasks, ids) -> float:
    return sum(
        gaussian_nll(model(_tensor(tasks[t].eval_x), t).cpu().numpy(), tasks[t].eval_y, 0.1)
        for t in ids
    )


def frontier(model, tasks, participants, get, put, tolerances) -> dict:
    """Minimum depth per tolerance, scored only over `participants`."""

    if not participants:
        return {"participants": 0, "vacuous": True}
    base = _loss(model, tasks, participants)
    curve = {}
    for bits in DEPTHS:
        original = get()
        with torch.no_grad():
            put([quantize(x, bits) for x in original])
        curve[bits] = _loss(model, tasks, participants) - base
        with torch.no_grad():
            put(original)
    # NON-VACUITY: a component that costs nothing even at 1 bit was not
    # actually exercised by its participant set.
    vacuous = abs(curve[1]) < 1e-6
    out = {"participants": len(participants), "vacuous": vacuous,
           "curve_nats": {str(k): v for k, v in curve.items()}}
    for tol in tolerances:
        budget = tol * len(participants)
        out[f"bits@{tol:g}"] = next((b for b in DEPTHS if curve[b] <= budget), 8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v4r_sealed/structured"))
    parser.add_argument("--kind", default="lifecycle")
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(400, 410)))
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[2.0, 10.0, 30.0])
    parser.add_argument("--output", type=Path, default=Path("reports/v5_component_frontier.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    spec = TaskGroupSpec(groups=2, eta=0.9, future_tasks=8, family_onset=16,
                         new_primitive_families=True)
    rows = []
    for world in args.worlds:
        path = args.root / f"world_{world}" / args.kind
        if not (path / "summary.json").exists():
            continue
        model, generated = _load(config, args.kind, path, world, spec, args.slots)
        tasks = {t.task_id: t for t in generated.tasks}
        retired = getattr(model, "retired", set())
        private = [t for t in model.task_residuals if t not in retired and t in tasks]
        promoted = [t for t in tasks if t in retired]
        allt = list(tasks)

        cell = {"world": world}
        cell["private"] = frontier(
            model, tasks, private,
            lambda: [model.task_residuals[t].detach().clone() for t in private],
            lambda vs: [model.task_residuals[t].copy_(v) for t, v in zip(private, vs)],
            args.tolerances)
        absts = list(getattr(model, "abstractions", []))
        cell["shared"] = frontier(
            model, tasks, promoted,
            lambda: [q.detach().clone() for q in absts],
            lambda vs: [q.copy_(v) for q, v in zip(absts, vs)],
            args.tolerances) if absts else {"participants": 0, "vacuous": True}
        basis = list(model.basis.parameters())
        cell["basis"] = frontier(
            model, tasks, allt,
            lambda: [q.detach().clone() for q in basis],
            lambda vs: [q.copy_(v) for q, v in zip(basis, vs)],
            args.tolerances)
        codes = list(model.task_codes.values())
        cell["routes"] = frontier(
            model, tasks, allt,
            lambda: [q.detach().clone() for q in codes],
            lambda vs: [q.copy_(v) for q, v in zip(codes, vs)],
            args.tolerances)
        rows.append(cell)

    print("V5.0 COMPONENT RATE-DISTORTION FRONTIER  D*(R)")
    print("  minimum bits/scalar, each component scored only on its participants\n")
    header = "  %-9s %7s " % ("component", "n") + " ".join(
        "%9s" % ("eps=%g" % t) for t in args.tolerances)
    print(header)
    for name in ("private", "shared", "basis", "routes"):
        cells = [r[name] for r in rows if not r[name].get("vacuous")]
        if not cells:
            print("  %-9s %7s   (no non-vacuous cells)" % (name, "-"))
            continue
        n = float(np.mean([c["participants"] for c in cells]))
        vals = " ".join("%9.1f" % np.mean([c[f"bits@{t:g}"] for c in cells])
                        for t in args.tolerances)
        print("  %-9s %7.1f %s" % (name, n, vals))
    vac = [(r["world"], k) for r in rows for k in ("private", "shared", "basis", "routes")
           if r[k].get("vacuous")]
    print(f"\n  vacuous cells (component not exercised by its participants): {len(vac)}")
    print(f"  worlds scored: {len(rows)}")
    print("\n  Against the 8-bit proxy, these are the honest per-component costs.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
