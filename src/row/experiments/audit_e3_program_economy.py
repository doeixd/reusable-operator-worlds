"""E3: are the solutions economical PROGRAMS over the learned vocabulary?

`E3_PROGRAM_ECONOMY_PLAN.md`. A coding and economics audit on existing artifacts.

    E3a  syntax sufficiency  store nothing but (z_1..z_D), verified BITWISE
    E3b  two-part economy    D*(library) + program bits, against a continuous-route
                             representation and a fully PRIVATE one
    E3c  semantic controls   wrong route / shuffled library / wrong depth collapse,
                             while the GAUGE permutation (route and library permuted
                             CONSISTENTLY) preserves behaviour exactly

`D*` follows the project's established instrument
(`audit_meta_recurrence.rate_distortion_bits`): quantize at depths 1..8, measure
functional error against the operator's OWN CONTRIBUTION, and interpolate in
`(bits, log error)` to a real-valued rate. Ported here to the learned operator
class, which stores `U`, `V`, `b` and a learnable `alpha` rather than a teacher
`Primitive`.

The private alternative is compressed against EACH TASK'S OWN distribution, so a
private operator only has to be accurate where that task uses it. That is what
makes E3b a real test rather than an operator count: V4R found local private
compression beating shared structure, and the same mechanism is available here.

Correctness is FUNCTIONAL throughout; no claim reads route agreement against
teacher IDs. Fails closed; atomic report.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from row.config import load_config
from row.experiments.audit_e0_export import git_commit, load_model
from row.world import World, WorldConfig

WORLDS = (0, 1, 2)
BUDGETS = (0.01, 0.05, 0.25)          # contribution-relative distortion budgets
DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)   # Amendment 1: headroom
D_MAX = 8                              # length code: log2(D_MAX) bits
BEHAVIOUR_TOLERANCE = 0.10             # >10% NMSE degradation = not behaviour-preserving
COLLAPSE_MARGIN = 1.0                  # E3c: wrong codes must degrade by >= 1 log unit


def operator_scalars(operator) -> int:
    total = operator.U.numel() + operator.V.numel() + operator.b.numel()
    return total + (operator.alpha.numel() if torch.is_tensor(operator.alpha) else 0)


@torch.no_grad()
def quantize_operator(operator, bits: int):
    """Symmetric per-tensor quantization of every STORED scalar."""
    local = copy.deepcopy(operator)
    levels = max(2, 2 ** bits)
    for name in ("U", "V", "b"):
        tensor = getattr(local, name)
        scale = float(tensor.abs().max())
        if scale == 0.0:
            continue
        step = 2 * scale / (levels - 1)
        tensor.copy_(torch.round(tensor / step) * step)
    if torch.is_tensor(local.alpha):
        scale = float(local.alpha.abs().max())
        if scale > 0:
            step = 2 * scale / (levels - 1)
            local.alpha.copy_(torch.round(local.alpha / step) * step)
    return local


@torch.no_grad()
def contribution(operator, probe: torch.Tensor) -> torch.Tensor:
    """What the operator DOES: its effect on the state, not the state itself."""
    return operator(probe) - probe


@torch.no_grad()
def rate_distortion_bits(operator, probe: torch.Tensor, budget_fraction: float) -> float:
    """Bits per scalar meeting a CONTRIBUTION-relative budget, interpolated.

    Mirrors `audit_meta_recurrence.rate_distortion_bits`: a real-valued rate,
    because integer depths make the measure jump by whole bits.
    """
    reference = contribution(operator, probe)
    energy = float(torch.mean(reference ** 2))
    budget = budget_fraction * energy
    depths, errors = [], []
    for bits in DEPTHS:
        error = float(torch.mean((contribution(quantize_operator(operator, bits), probe)
                                  - reference) ** 2))
        depths.append(float(bits))
        errors.append(max(error, 1e-30))
        if error <= budget:
            if len(depths) == 1:
                return float(bits)
            x0, x1 = depths[-2], depths[-1]
            y0, y1 = math.log(errors[-2]), math.log(errors[-1])
            target = math.log(max(budget, 1e-30))
            if y0 == y1:
                return float(bits)
            return x0 + (x1 - x0) * (y0 - target) / (y0 - y1)
    return float(DEPTHS[-1])


def _interpolate(depths, excesses, target):
    """Smallest real-valued rate meeting a target, interpolated in (bits, log excess)."""
    x0, x1 = depths[-2], depths[-1]
    y0, y1 = math.log(max(excesses[-2], 1e-30)), math.log(max(excesses[-1], 1e-30))
    goal = math.log(max(target, 1e-30))
    if y0 == y1:
        return float(x1)
    return x0 + (x1 - x0) * (y0 - goal) / (y0 - y1)


def behavioural_rate(evaluate, tolerance: float = BEHAVIOUR_TOLERANCE) -> dict:
    """Smallest bits/scalar whose COMPOSED NMSE ratio is <= 1 + tolerance.

    Amendment 1: a per-operator contribution budget under-controls end-to-end
    error because error compounds through composition (E8 measured that
    compounding directly), so the rate that matters is defined on the composed
    prediction rather than on one operator's own effect.

    `evaluate(bits) -> nmse_ratio` is supplied by the caller, so one definition
    serves both the shared library (ratio over all tasks) and a private per-task
    stack (ratio for that one task).
    """

    depths, excesses = [], []
    for bits in DEPTHS:
        ratio = evaluate(bits)
        excesses.append(max(ratio - 1.0, 1e-12))
        depths.append(float(bits))
        if ratio <= 1.0 + tolerance:
            rate = float(bits) if len(depths) == 1 else _interpolate(depths, excesses, tolerance)
            return {"bits_per_scalar": rate, "nmse_ratio": ratio, "saturated": False}
    return {"bits_per_scalar": float(DEPTHS[-1]), "nmse_ratio": evaluate(DEPTHS[-1]),
            "saturated": True}


def nmse(pred, y) -> float:
    return float(torch.mean((pred - y) ** 2) / (torch.var(y, unbiased=False) + 1e-12))


@torch.no_grad()
def route_indices(model, task_id: str) -> list[int]:
    return [int(v) for v in torch.argmax(model.task_codes[task_id], dim=-1)]


@torch.no_grad()
def predict_with_route(model, x: torch.Tensor, indices, library=None) -> torch.Tensor:
    """Execute an explicit program: exactly the eval-mode forward, made literal."""
    ops = library if library is not None else model.library
    z = x
    for index in indices:
        z = ops[index](z)
    return z


@torch.no_grad()
def state_probes(model, tasks, per_task: int = 64) -> dict:
    """For each task, the states its own program actually visits (for private D*)."""
    out = {}
    for task in tasks:
        x = torch.tensor(task.eval_x[:per_task], dtype=torch.float32)
        indices = route_indices(model, task.task_id)
        states, z = [], x
        for index in indices:
            states.append(z)
            z = model.library[index](z)
        out[task.task_id] = {"indices": indices, "inputs": states}
    return out


def continuous_economy(world: int, config_path, scalars_hint=None) -> dict:
    """D*(library) + per-task route bits for a learner that actually uses a mixture.

    Amendment 2: available for the worlds with a compatible exact-reuse
    continuous artifact; reported as absent elsewhere rather than substituted.
    """

    path = Path("artifacts/rho_development/rho_1") / f"world_{world}" / "continuous"
    if not (path / "model.pt").exists():
        return {"available": False, "reason": f"no continuous artifact at {path}"}
    raw = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    config = load_config(config_path)
    config = replace(config, world=replace(config.world, **raw["world"]))
    fields = set(config.continuous_model.__dataclass_fields__)
    config = replace(config, continuous_model=replace(
        config.continuous_model,
        **{k: v for k, v in raw["continuous_model"].items() if k in fields}))
    model, _, _ = load_model(config, path, "continuous")
    world_obj = World.generate(WorldConfig(**raw["world"]))
    tasks = [t for t in world_obj.tasks if t.task_id in model.task_codes]
    scalars = operator_scalars(model.basis[0])
    base = {}
    for task in tasks:
        x = torch.tensor(task.eval_x, dtype=torch.float32)
        y = torch.tensor(task.eval_y, dtype=torch.float32)
        with torch.no_grad():
            base[task.task_id] = nmse(model(x, task.task_id), y)
    base_mean = float(np.mean(list(base.values())))

    def library_ratio(bits):
        quantized = copy.deepcopy(model)
        for index, op in enumerate(quantized.basis):
            quantized.basis[index] = quantize_operator(op, bits)
        vals = []
        for task in tasks:
            x = torch.tensor(task.eval_x, dtype=torch.float32)
            y = torch.tensor(task.eval_y, dtype=torch.float32)
            with torch.no_grad():
                vals.append(nmse(quantized(x, task.task_id), y))
        return float(np.mean(vals) / max(base_mean, 1e-12))

    library_rate = behavioural_rate(library_ratio)
    d_library = library_rate["bits_per_scalar"] * scalars * len(model.basis)

    route_rates = []
    for task in tasks:
        x = torch.tensor(task.eval_x, dtype=torch.float32)
        y = torch.tensor(task.eval_y, dtype=torch.float32)
        reference = base[task.task_id]
        code = model.task_codes[task.task_id]

        def route_ratio(bits, code=code, x=x, y=y, reference=reference, task=task):
            local = copy.deepcopy(model)
            target = local.task_codes[task.task_id]
            scale = float(code.abs().max())
            with torch.no_grad():
                if scale > 0:
                    step = 2 * scale / (max(2, 2 ** bits) - 1)
                    target.copy_(torch.round(code / step) * step)
                value = nmse(local(x, task.task_id), y)
            return float(value / max(reference, 1e-12))

        route_rates.append(behavioural_rate(route_ratio)["bits_per_scalar"])
    route_scalars = int(model.task_codes[tasks[0].task_id].numel())
    d_routes = float(sum(r * route_scalars for r in route_rates))
    return {"available": True, "artifact": str(path), "tasks": len(tasks),
            "library_rate": library_rate, "scalars_per_operator": scalars,
            "library_operators": len(model.basis), "D_library_bits": d_library,
            "route_scalars_per_task": route_scalars,
            "mean_route_bits_per_scalar": float(np.mean(route_rates)),
            "D_routes_bits": d_routes, "D_continuous_bits": d_library + d_routes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e3_program_economy.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    out = {"frozen_plan": "E3_PROGRAM_ECONOMY_PLAN.md", "git_commit": git_commit(),
           "protocol": {"budgets": list(BUDGETS), "depths": list(DEPTHS), "D_max": D_MAX,
                        "behaviour_tolerance": BEHAVIOUR_TOLERANCE,
                        "collapse_margin": COLLAPSE_MARGIN,
                        "note": "correctness is functional; no claim reads route agreement"},
           "worlds": {}}
    for world in WORLDS:
        path = Path("artifacts/e1_disc") / f"world_{world}"
        raw = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
        config = load_config(args.config)
        config = replace(config, world=replace(config.world, **raw["world"]))
        fields = set(config.discrete_model.__dataclass_fields__)
        config = replace(config, discrete_model=replace(
            config.discrete_model,
            **{k: v for k, v in raw["discrete_model"].items() if k in fields}))
        model, _, _ = load_model(config, path, "discrete")
        world_obj = World.generate(WorldConfig(**raw["world"]))
        tasks = [t for t in world_obj.tasks if t.task_id in model.task_codes]
        slots = model.operator_slots
        depth = config.world.program_length
        probe = torch.tensor(np.random.default_rng(np.random.SeedSequence(
            [767, world, 2])).normal(size=(256, config.world.state_dim)), dtype=torch.float32)

        # ---- E3a: is the literal discrete sequence the whole program? --------
        deviations, exact = [], 0
        for task in tasks:
            x = torch.tensor(task.eval_x, dtype=torch.float32)
            y = torch.tensor(task.eval_y, dtype=torch.float32)
            with torch.no_grad():
                original = model(x, task.task_id)
            literal = predict_with_route(model, x, route_indices(model, task.task_id))
            deviations.append(abs(nmse(literal, y) - nmse(original, y)))
            exact += int(bool(torch.equal(original, literal)))
        e3a = {"tasks": len(tasks), "bitwise_identical": exact,
               "max_nmse_deviation": float(max(deviations)),
               "passes": bool(max(deviations) < 1e-6)}

        # ---- E3b: the two-part accounting ------------------------------------
        probes = state_probes(model, tasks)
        scalars = operator_scalars(model.library[0])
        length_code = math.log2(D_MAX)
        per_task_program = length_code + depth * math.log2(slots)
        program_bits = len(tasks) * per_task_program

        base_nmse = {}
        for task in tasks:
            x = torch.tensor(task.eval_x, dtype=torch.float32)
            y = torch.tensor(task.eval_y, dtype=torch.float32)
            with torch.no_grad():
                base_nmse[task.task_id] = nmse(model(x, task.task_id), y)
        base_mean = float(np.mean(list(base_nmse.values())))

        def shared_ratio(bits):
            quantized = copy.deepcopy(model)
            for index, op in enumerate(quantized.library):
                quantized.library[index] = quantize_operator(op, bits)
            vals = []
            for task in tasks:
                x = torch.tensor(task.eval_x, dtype=torch.float32)
                y = torch.tensor(task.eval_y, dtype=torch.float32)
                with torch.no_grad():
                    vals.append(nmse(quantized(x, task.task_id), y))
            return float(np.mean(vals) / max(base_mean, 1e-12))

        shared_rate = behavioural_rate(shared_ratio)
        d_library = shared_rate["bits_per_scalar"] * scalars * len(model.library)
        d_program = d_library + program_bits

        # PRIVATE: each task keeps its OWN copies of the operators it uses, and
        # each may be compressed until THAT TASK degrades by the tolerance. A
        # private operator serves one distribution; a shared one serves all of
        # them. That asymmetry is the substance of the comparison (V4R found
        # local private compression beating shared structure).
        private_total, private_rates = 0.0, []
        for task in tasks:
            indices = probes[task.task_id]["indices"]
            x = torch.tensor(task.eval_x, dtype=torch.float32)
            y = torch.tensor(task.eval_y, dtype=torch.float32)
            reference = base_nmse[task.task_id]

            def task_ratio(bits, indices=indices, x=x, y=y, reference=reference):
                ops = [quantize_operator(model.library[i], bits) for i in indices]
                with torch.no_grad():
                    z = x
                    for op in ops:
                        z = op(z)
                return float(nmse(z, y) / max(reference, 1e-12))

            rate = behavioural_rate(task_ratio)
            private_rates.append(rate["bits_per_scalar"])
            private_total += rate["bits_per_scalar"] * scalars * len(indices)

        per_task_private = private_total / max(len(tasks), 1)
        amortization = (d_library / (per_task_private - per_task_program)
                        if per_task_private > per_task_program else None)
        economy = {
            "behavioural_rate_shared": shared_rate,
            "mean_private_bits_per_scalar": float(np.mean(private_rates)),
            "scalars_per_operator": scalars,
            "library_operators": len(model.library),
            "D_library_bits": d_library,
            "bits_per_task_program": per_task_program,
            "program_bits": program_bits,
            "D_program_bits": d_program,
            "D_private_bits": private_total,
            "program_beats_private": bool(d_program < private_total),
            "amortization_point_tasks": amortization,
        }
        continuous = continuous_economy(world, args.config)
        economy["continuous"] = continuous
        if continuous.get("available"):
            economy["program_beats_continuous"] = bool(
                d_program < continuous["D_continuous_bits"])
        else:
            economy["program_beats_continuous"] = None

        contribution_budgets = {}
        for budget in BUDGETS:
            bits = [rate_distortion_bits(op, probe, budget) for op in model.library]
            contribution_budgets[str(budget)] = {
                "mean_bits_per_scalar": float(np.mean(bits)),
                "D_library_bits": float(sum(b * scalars for b in bits)),
                "composed_nmse_ratio": shared_ratio(max(1, int(math.ceil(float(np.mean(bits)))))),
                "note": "secondary currency: a per-operator budget under-controls composed error",
            }

        # ---- E3c: semantic controls ------------------------------------------
        rng = np.random.default_rng(np.random.SeedSequence([767, world, 8]))
        permutation = list(rng.permutation(slots))
        controls = {k: [] for k in ("true", "wrong_route", "shuffled_library",
                                    "wrong_depth", "gauge")}
        gauge_exact = 0
        for task in tasks:
            x = torch.tensor(task.eval_x, dtype=torch.float32)
            y = torch.tensor(task.eval_y, dtype=torch.float32)
            indices = route_indices(model, task.task_id)
            controls["true"].append(nmse(predict_with_route(model, x, indices), y))
            wrong = [int(v) for v in rng.integers(0, slots, len(indices))]
            controls["wrong_route"].append(nmse(predict_with_route(model, x, wrong), y))
            shuffled = [permutation[i] for i in indices]        # route kept, meanings moved
            controls["shuffled_library"].append(nmse(predict_with_route(model, x, shuffled), y))
            controls["wrong_depth"].append(nmse(predict_with_route(model, x, indices[:-1]), y))
            # GAUGE: relabel the library AND the route consistently
            inverse = [0] * slots
            for new, old in enumerate(permutation):
                inverse[old] = new
            gauge_library = [model.library[permutation[i]] for i in range(slots)]
            gauge_route = [inverse[i] for i in indices]
            gauge_pred = predict_with_route(model, x, gauge_route, library=gauge_library)
            controls["gauge"].append(nmse(gauge_pred, y))
            gauge_exact += int(bool(torch.equal(
                gauge_pred, predict_with_route(model, x, indices))))
        geo = {k: float(np.exp(np.mean(np.log(np.maximum(v, 1e-12)))))
               for k, v in controls.items()}
        e3c = {"geomean_nmse": geo, "gauge_bitwise_tasks": gauge_exact, "tasks": len(tasks),
               "gauge_preserves": bool(gauge_exact == len(tasks)),
               "collapse": {k: float(np.log(geo[k]) - np.log(geo["true"]))
                            for k in ("wrong_route", "shuffled_library", "wrong_depth")}}

        out["worlds"][str(world)] = {"E3a": e3a,
                                     "E3b": {"economy": economy,
                                             "contribution_budgets": contribution_budgets},
                                     "E3c": e3c, "slots": slots, "depth": depth,
                                     "tasks": len(tasks)}
        print(f"[w{world}] E3a bitwise {e3a['bitwise_identical']}/{e3a['tasks']} "
              f"maxdev {e3a['max_nmse_deviation']:.2e}", flush=True)
        econ = economy
        print("    behavioural rate {0:.2f} b/scalar (private mean {1:.2f}) | library {2:.0f}"
              " + programs {3:.0f} = {4:.0f} vs private {5:.0f} -> wins={6} amortize@{7}".format(
                  econ["behavioural_rate_shared"]["bits_per_scalar"],
                  econ["mean_private_bits_per_scalar"], econ["D_library_bits"],
                  econ["program_bits"], econ["D_program_bits"], econ["D_private_bits"],
                  econ["program_beats_private"],
                  econ["amortization_point_tasks"] and round(econ["amortization_point_tasks"], 1)),
              flush=True)
        print(f"    E3c true {geo['true']:.5f} wrong_route {geo['wrong_route']:.5f} "
              f"shuffled {geo['shuffled_library']:.5f} wrong_depth {geo['wrong_depth']:.5f} "
              f"gauge {geo['gauge']:.5f} (bitwise {gauge_exact}/{len(tasks)})", flush=True)

    # ---- decisions ---------------------------------------------------------
    worlds = [out["worlds"][str(w)] for w in WORLDS]
    e3a_pass = sum(w["E3a"]["passes"] for w in worlds) == 3
    e3b_pass = (sum(w["E3b"]["economy"]["program_beats_private"] for w in worlds) >= 2
                and not any(w["E3b"]["economy"]["behavioural_rate_shared"]["saturated"]
                            for w in worlds))
    continuous_clause = [w["E3b"]["economy"]["program_beats_continuous"] for w in worlds
                         if w["E3b"]["economy"]["program_beats_continuous"] is not None]
    e3c_pass = (sum(w["E3c"]["gauge_preserves"] for w in worlds) == 3
                and all(sum(w["E3c"]["collapse"][k] >= COLLAPSE_MARGIN for w in worlds) >= 2
                        for k in ("wrong_route", "shuffled_library", "wrong_depth")))
    out["decisions"] = {"E3a_syntax_sufficient": bool(e3a_pass),
                        "E3b_program_economy": bool(e3b_pass),
                        "E3b_secondary_beats_continuous": (
                            None if not continuous_clause
                            else bool(sum(continuous_clause) >= max(1, len(continuous_clause) - 1))),
                        "E3b_continuous_worlds": len(continuous_clause),
                        "E3c_semantics_causal": bool(e3c_pass)}
    out["outcome"] = ("PROGRAM REPRESENTATION: sufficient, economical and causal"
                      if e3a_pass and e3b_pass and e3c_pass
                      else "PARTIAL: " + ", ".join(
                          k for k, v in out["decisions"].items()
                          if isinstance(v, bool) and not v) + " did not pass")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(json.dumps(out["decisions"], indent=2))
    print(f"OUTCOME {out['outcome']}")


if __name__ == "__main__":
    main()
