"""Sealed export confirmation, seeds 800-829 (EXPORT_CONFIRMATION_PLAN.md).

Five independently preregistered components on one support-split lifetime per
world, with a HIERARCHICAL verdict so a mechanistic miss cannot erase a direct
behavioural result:

    C1  program representation   syntax sufficiency / two-part economy / causal semantics
    C2  frozen export            G_export, with the oracle and inference legs separated
    C3  systematic composition   H1, H2 and H3 kept separate; H3 is the flagship
    C4  length closure           depth 4 load-bearing, depth 2 secondary
    C5  compositional drift      slope b and continuation ratio q, scored separately

Structural assertions are FATAL. Verdicts are read once, from the frozen table.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
from row.experiments.audit_e0_export import assign, distance_matrix, git_commit, load_model
from row.experiments.audit_e1_export import ADAPT_LR, ADAPT_STEPS, geo, nmse, scratch_model
from row.experiments.audit_e3_program_economy import (behavioural_rate, operator_scalars,
                                                      predict_with_route, quantize_operator,
                                                      route_indices, state_probes)
from row.experiments.audit_e8_length import VariableDepthDiscrete, adapt_cell, oracle_cell
from row.support_split_world import (SupportSplitSpec, _build_tasks,
                                     generate_support_split_world)
from row.world import World, WorldConfig

SEEDS = tuple(range(800, 830))
STRATA = ("H1", "H2", "H3")
DEPTHS = {"depth2": 2, "depth4": 4}
PER_CELL = 8
D_MAX = 8
CACHE = Path("reports/export_confirmation_cache")
ART = Path("artifacts/export_confirmation")

# ---- registered thresholds (EXPORT_CONFIRMATION_PLAN.md) --------------------
WORLD_FRACTION = 28                    # "in >= 28 of 30 worlds"
C1B_QD_MEAN, C1B_QD_WORLD = 2.0, 1.5
C1B_NSTAR = (3.0, 6.0)
C1C_COLLAPSE = 1.0
C2_GEXPORT = (0.75, 1.15)
C2_ORACLE_LEG = 1.5
C3_MEAN, C3_WORLD = 1.5, 1.0
C4_D4_MEAN, C4_D4_WORLD = 1.25, 0.75
C4_D2_MEAN = 1.5
C5_B = (0.3, 0.9)
C5_Q = (0.45, 0.95)
C5_Q_DISCONTINUITY = 1.5


def fingerprint() -> str:
    payload = {"seeds": list(SEEDS), "strata": list(STRATA), "depths": DEPTHS,
               "per_cell": PER_CELL, "steps": ADAPT_STEPS, "lr": ADAPT_LR}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cached(key: str, compute):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.json"
    stamp = fingerprint()
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("protocol") != stamp:
            raise SystemExit(f"cached cell {key} under a different protocol; refuse to mix")
        return stored["value"]
    value = compute()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"protocol": stamp, "value": value}), encoding="utf-8")
    os.replace(tmp, path)
    return value


def fatal(condition: bool, message: str) -> None:
    """Structural assertions fail the run; they do not warn."""
    if not condition:
        raise SystemExit(f"FATAL STRUCTURAL ASSERTION: {message}")


def pairs_of(program) -> set:
    return {(i, program[i], program[i + 1]) for i in range(len(program) - 1)}


def load_world(world: int, config_path: Path, art_root: Path) -> dict:
    path = art_root / f"world_{world}"
    fatal((path / "summary.json").exists(), f"missing artifact {path}")
    raw = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    config = load_config(config_path)
    config = replace(config, world=replace(config.world, **raw["world"]))
    fields = set(config.discrete_model.__dataclass_fields__)
    config = replace(config, discrete_model=replace(
        config.discrete_model, **{k: v for k, v in raw["discrete_model"].items() if k in fields}))
    shipped, _, _ = load_model(config, path, "discrete")
    variable = copy.deepcopy(shipped)
    variable.__class__ = VariableDepthDiscrete
    split = json.loads((path / "support_split.json").read_text(encoding="utf-8"))
    # The world MUST be the support-split one. `World.generate` produces the same
    # 64 opaque task IDs from the same seed but DIFFERENT programs (63/64 differ),
    # so using it would silently pair this model's routes with another world's
    # targets — a mismatched-target bug that identical IDs hide perfectly.
    world_obj = generate_support_split_world(WorldConfig(**raw["world"]), SupportSplitSpec())
    fatal([tuple(t.program.primitive_ids) for t in world_obj.tasks]
          == [tuple(p) for p in split["train_programs"]],
          f"reconstructed world does not match the recorded training split at {path}")
    probe = torch.tensor(np.random.default_rng(np.random.SeedSequence(
        [800, world, 1])).normal(size=(256, config.world.state_dim)), dtype=torch.float32)
    d = distance_matrix(shipped, world_obj, config.world.program_length, probe, "library",
                        config.world.seed)
    best, cost, margin = assign(d["matrix"])
    return {"path": path, "config": config, "shipped": shipped, "model": variable,
            "split": split, "world": world_obj, "assignment": best,
            "assignment_margin": margin, "probe": probe}


def structural_checks(cell: dict) -> None:
    split = cell["split"]
    held = [tuple(p) for p in split["withheld_placements"]]
    train = [tuple(p) for p in split["train_programs"]]
    train_set = set(train)
    seen_pairs = set()
    for program in train:
        seen_pairs |= pairs_of(program)
    for program in train:
        fatal(not any(program[i] == p for p, i in held),
              f"training program {program} touches a withheld placement")
    for name in STRATA:
        for program in (tuple(p) for p in split["strata"][name]):
            fatal(program not in train_set, f"{name} program {program} appears in training")
            if name == "H2":
                fatal(not (pairs_of(program) <= seen_pairs),
                      f"H2 program {program} has no unseen adjacent pair")
            if name == "H3":
                fatal(any(program[i] == p for p, i in held),
                      f"H3 program {program} has no withheld placement")
    diagnostics = split["diagnostics"]
    fatal(diagnostics["balance_ratio"] <= 2.0, "training balance exceeds 2.0")
    fatal(diagnostics["context_min"] >= 3, "a primitive has fewer than 3 contexts")
    fatal(cell["config"].world.program_length == 3, "trained depth is not 3")
    # the variable-depth executor must BE the shipped one at depth 3
    x = torch.tensor(np.random.default_rng(np.random.SeedSequence(
        [800, cell["config"].world.seed, 7])).normal(
        size=(32, cell["config"].world.state_dim)), dtype=torch.float32)
    with torch.no_grad():
        for task_id in list(cell["shipped"].task_codes)[:8]:
            fatal(bool(torch.equal(cell["shipped"](x, task_id), cell["model"](x, task_id))),
                  "variable-depth executor differs from the shipped one at depth 3")


def stable_tag(tag: str) -> int:
    """A PROCESS-INDEPENDENT integer for a seed component.

    `hash()` is randomized per process in Python, so a sealed sample drawn from
    it would differ between runs of the same code — the project's oldest
    recorded rule (AGENTS.md: NumPy SeedSequence with explicit integer
    components, never the built-in hash).
    """

    return int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)


def sample(pool, world: int, tag: str, count: int) -> list:
    rng = np.random.default_rng(np.random.SeedSequence([800, world, stable_tag(tag)]))
    order = rng.permutation(len(pool))
    return [tuple(pool[int(i)]) for i in order[:count]]


def score_world(world: int, cell: dict, donor: dict) -> dict:
    structural_checks(cell)
    config, model, split = cell["config"], cell["model"], cell["split"]
    library = cell["world"].library
    slots = model.operator_slots
    depth = config.world.program_length
    tasks = [t for t in cell["world"].tasks if t.task_id in model.task_codes]
    row = {"assignment_margin": cell["assignment_margin"],
           "split_diagnostics": split["diagnostics"]}

    # ---- C1a syntax sufficiency ------------------------------------------
    exact = 0
    for task in tasks:
        x = torch.tensor(task.eval_x, dtype=torch.float32)
        with torch.no_grad():
            original = cell["shipped"](x, task.task_id)
        literal = predict_with_route(cell["shipped"], x, route_indices(cell["shipped"], task.task_id))
        exact += int(bool(torch.equal(original, literal)))
    row["C1a"] = {"tasks": len(tasks), "bitwise": exact, "passes": exact == len(tasks)}

    # ---- C1b two-part economy --------------------------------------------
    def economy():
        scalars = operator_scalars(cell["shipped"].library[0])
        per_task_program = math.log2(D_MAX) + depth * math.log2(slots)
        base = {}
        for task in tasks:
            x = torch.tensor(task.eval_x, dtype=torch.float32)
            y = torch.tensor(task.eval_y, dtype=torch.float32)
            with torch.no_grad():
                base[task.task_id] = nmse(cell["shipped"](x, task.task_id), y)
        base_mean = float(np.mean(list(base.values())))

        def shared_ratio(bits):
            q = copy.deepcopy(cell["shipped"])
            for index, op in enumerate(q.library):
                q.library[index] = quantize_operator(op, bits)
            vals = []
            for task in tasks:
                x = torch.tensor(task.eval_x, dtype=torch.float32)
                y = torch.tensor(task.eval_y, dtype=torch.float32)
                with torch.no_grad():
                    vals.append(nmse(q(x, task.task_id), y))
            return float(np.mean(vals) / max(base_mean, 1e-12))

        rate = behavioural_rate(shared_ratio)
        fatal(not rate["saturated"], "shared behavioural rate saturated the bit ceiling")
        d_library = rate["bits_per_scalar"] * scalars * len(cell["shipped"].library)
        probes = state_probes(cell["shipped"], tasks)
        private = 0.0
        for task in tasks:
            indices = probes[task.task_id]["indices"]
            x = torch.tensor(task.eval_x, dtype=torch.float32)
            y = torch.tensor(task.eval_y, dtype=torch.float32)
            reference = base[task.task_id]

            def task_ratio(bits, indices=indices, x=x, y=y, reference=reference):
                ops = [quantize_operator(cell["shipped"].library[i], bits) for i in indices]
                with torch.no_grad():
                    z = x
                    for op in ops:
                        z = op(z)
                return float(nmse(z, y) / max(reference, 1e-12))

            r = behavioural_rate(task_ratio)
            fatal(not r["saturated"], f"private behavioural rate saturated for {task.task_id}")
            private += r["bits_per_scalar"] * scalars * len(indices)
        d_program = d_library + len(tasks) * per_task_program
        per_task_private = private / len(tasks)
        n_star = (d_library / (per_task_private - per_task_program)
                  if per_task_private > per_task_program else None)
        return {"bits_per_scalar": rate["bits_per_scalar"], "D_library": d_library,
                "D_program": d_program, "D_private": private,
                "Q_D": float(np.log(private / d_program)), "N_star": n_star}

    row["C1b"] = cached(f"w{world}_C1b", economy)

    # ---- C1c causal semantics --------------------------------------------
    def semantics():
        rng = np.random.default_rng(np.random.SeedSequence([800, world, 5]))
        permutation = [int(v) for v in rng.permutation(slots)]
        inverse = [0] * slots
        for new, old in enumerate(permutation):
            inverse[old] = new
        gauge_library = [cell["shipped"].library[permutation[i]] for i in range(slots)]
        values = {k: [] for k in ("true", "wrong_route", "shuffled_library", "wrong_depth")}
        gauge_exact = 0
        for task in tasks:
            x = torch.tensor(task.eval_x, dtype=torch.float32)
            y = torch.tensor(task.eval_y, dtype=torch.float32)
            indices = route_indices(cell["shipped"], task.task_id)
            truth = predict_with_route(cell["shipped"], x, indices)
            values["true"].append(nmse(truth, y))
            values["wrong_route"].append(nmse(predict_with_route(
                cell["shipped"], x, [int(v) for v in rng.integers(0, slots, len(indices))]), y))
            values["shuffled_library"].append(nmse(predict_with_route(
                cell["shipped"], x, [permutation[i] for i in indices]), y))
            values["wrong_depth"].append(nmse(predict_with_route(
                cell["shipped"], x, indices[:-1]), y))
            gauge = predict_with_route(cell["shipped"], x, [inverse[i] for i in indices],
                                       library=gauge_library)
            gauge_exact += int(bool(torch.equal(gauge, truth)))
        g = {k: geo(v) for k, v in values.items()}
        return {"geomean": g, "gauge_bitwise": gauge_exact, "tasks": len(tasks),
                "collapse": {k: float(np.log(g[k]) - np.log(g["true"]))
                             for k in ("wrong_route", "shuffled_library", "wrong_depth")}}

    row["C1c"] = cached(f"w{world}_C1c", semantics)

    # ---- C2 / C3: held-out strata ----------------------------------------
    row["strata"] = {}
    for stratum in STRATA:
        programs = sample(split["strata"][stratum], world, stratum, PER_CELL)
        fatal(len(programs) == PER_CELL, f"{stratum} has fewer than {PER_CELL} programs")

        def cellwise(programs=programs, stratum=stratum):
            arms = {a: [] for a in ("O", "O-W", "R", "R-W", "S")}
            weak = 0
            for index, program in enumerate(programs):
                task = _build_tasks(config.world, library, [program],
                                    [f"task_conf_{stratum}_{index}"], index_offset=3000 + index)[0]
                tag = f"w{world}_{stratum}_{index}"
                res = {
                    "O": oracle_cell(model, task, program, cell["assignment"],
                                     f"cO_{tag}", library),
                    "O-W": oracle_cell(donor["model"], task, program, donor["assignment"],
                                       f"cOW_{tag}", library),
                    "R": adapt_cell(model, task, f"cR_{tag}", len(program), False, library, program),
                    "R-W": adapt_cell(donor["model"], task, f"cRW_{tag}", len(program), False,
                                      library, program),
                    "S": adapt_cell(scratch_model(config, "discrete", 7717), task, f"cS_{tag}",
                                    len(program), True, library, program),
                }
                for arm in arms:
                    arms[arm].append(res[arm]["query_nmse"])
                if any(res[a].get("support_reduction_objective", 1.0) <= 0.01 for a in ("R", "S")):
                    weak += 1
            g = {a: geo(v) for a, v in arms.items()}
            denominator = g["S"] - g["O"]
            return {"geomean": g, "weak": weak,
                    "delta_comp": float(np.log(g["S"]) - np.log(g["R"])),
                    "oracle_leg": float(np.log(g["S"]) - np.log(g["O"])),
                    "G_export": (float((g["S"] - g["R"]) / denominator)
                                 if denominator > 0 else None)}

        row["strata"][stratum] = cached(f"w{world}_{stratum}", cellwise)
        fatal(row["strata"][stratum]["weak"] == 0,
              f"{stratum} has arms that did not adapt in world {world}")

    # ---- C4 / C5: length ---------------------------------------------------
    row["depths"] = {}
    for name, dep in DEPTHS.items():
        longer = World.generate(replace(config.world, program_length=dep,
                                        tasks=min(config.world.teacher_primitives ** dep, 64)))
        pool = [tuple(t.program.primitive_ids) for t in longer.tasks]
        programs = sample(pool, world, name, PER_CELL)
        fatal(all(len(p) == dep for p in programs), f"{name} programs are not depth {dep}")

        def cellwise(programs=programs, dep=dep, name=name):
            arms = {a: [] for a in ("O", "O-W", "R", "S")}
            per_step, weak = [], 0
            for index, program in enumerate(programs):
                task = _build_tasks(config.world, library, [program],
                                    [f"task_conf_{name}_{index}"], index_offset=4000 + index)[0]
                tag = f"w{world}_{name}_{index}"
                res = {
                    "O": oracle_cell(model, task, program, cell["assignment"], f"dO_{tag}", library),
                    "O-W": oracle_cell(donor["model"], task, program, donor["assignment"],
                                       f"dOW_{tag}", library),
                    "R": adapt_cell(model, task, f"dR_{tag}", dep, False, library, program),
                    "S": adapt_cell(scratch_model(config, "discrete", 7717), task, f"dS_{tag}",
                                    dep, True, library, program),
                }
                for arm in arms:
                    arms[arm].append(res[arm]["query_nmse"])
                per_step.append(res["O"]["per_step_nmse"])
                if any(res[a].get("support_reduction_objective", 1.0) <= 0.01 for a in ("R", "S")):
                    weak += 1
            g = {a: geo(v) for a, v in arms.items()}
            steps = [geo([r[t] for r in per_step]) for t in range(dep)]
            return {"geomean": g, "weak": weak, "per_step": steps,
                    "delta": float(np.log(g["S"]) - np.log(g["R"])),
                    "oracle_leg": float(np.log(g["S"]) - np.log(g["O"]))}

        row["depths"][name] = cached(f"w{world}_{name}", cellwise)
        fatal(row["depths"][name]["weak"] == 0, f"{name} has arms that did not adapt")

    steps = row["depths"]["depth4"]["per_step"]
    t = np.arange(1, len(steps) + 1)
    slope = float(np.polyfit(t, np.log(np.maximum(steps, 1e-30)), 1)[0])
    ratios = [steps[i + 1] / max(steps[i], 1e-30) for i in range(len(steps) - 1)]
    row["C5"] = {"per_step": steps, "b": slope,
                 "ratios": ratios,
                 "q": float(ratios[-1] / max(np.mean(ratios[:-1]), 1e-12))}
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/export_confirmation.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--artifacts", type=Path, default=ART)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    seeds = list(args.seeds)
    cells = {w: load_world(w, args.config, args.artifacts) for w in seeds}
    worlds = {}
    for index, w in enumerate(seeds):
        donor = cells[seeds[(index + 1) % len(seeds)]]
        worlds[str(w)] = score_world(w, cells[w], donor)
        r = worlds[str(w)]
        print(f"[w{w}] C1a {r['C1a']['bitwise']}/{r['C1a']['tasks']} "
              f"Q_D {r['C1b']['Q_D']:.2f} N* {r['C1b']['N_star'] and round(r['C1b']['N_star'],1)} "
              f"gauge {r['C1c']['gauge_bitwise']}/{r['C1c']['tasks']} | "
              f"H1 {r['strata']['H1']['delta_comp']:+.2f} H2 {r['strata']['H2']['delta_comp']:+.2f} "
              f"H3 {r['strata']['H3']['delta_comp']:+.2f} | D2 {r['depths']['depth2']['delta']:+.2f} "
              f"D4 {r['depths']['depth4']['delta']:+.2f} | b {r['C5']['b']:.2f} q {r['C5']['q']:.2f}",
              flush=True)
    n = len(seeds)
    need = WORLD_FRACTION if n == len(SEEDS) else max(1, int(round(WORLD_FRACTION / len(SEEDS) * n)))

    def frac(predicate) -> int:
        return sum(1 for w in worlds.values() if predicate(w))

    qd = [w["C1b"]["Q_D"] for w in worlds.values()]
    ns = [w["C1b"]["N_star"] for w in worlds.values() if w["C1b"]["N_star"] is not None]
    gexp = [w["strata"]["H1"]["G_export"] for w in worlds.values()
            if w["strata"]["H1"]["G_export"] is not None]
    decisions = {
        "C1a": frac(lambda w: w["C1a"]["passes"]) >= need,
        "C1b": (float(np.mean(qd)) >= C1B_QD_MEAN
                and frac(lambda w: w["C1b"]["Q_D"] >= C1B_QD_WORLD) >= need
                and C1B_NSTAR[0] <= float(np.mean(ns)) <= C1B_NSTAR[1]),
        "C1c": (frac(lambda w: w["C1c"]["gauge_bitwise"] == w["C1c"]["tasks"]) >= need
                and all(frac(lambda w, k=k: w["C1c"]["collapse"][k] >= C1C_COLLAPSE) >= need
                        for k in ("wrong_route", "shuffled_library", "wrong_depth"))),
        "C2": (C2_GEXPORT[0] <= float(np.mean(gexp)) <= C2_GEXPORT[1]
               and frac(lambda w: w["strata"]["H1"]["oracle_leg"] >= C2_ORACLE_LEG) >= need),
        "C3": {s: (float(np.mean([w["strata"][s]["delta_comp"] for w in worlds.values()])) >= C3_MEAN
                   and frac(lambda w, s=s: w["strata"][s]["delta_comp"] > C3_WORLD) >= need)
               for s in STRATA},
        "C4": (float(np.mean([w["depths"]["depth4"]["delta"] for w in worlds.values()])) >= C4_D4_MEAN
               and frac(lambda w: w["depths"]["depth4"]["delta"] > C4_D4_WORLD) >= need),
        "C4_depth2_secondary": float(np.mean(
            [w["depths"]["depth2"]["delta"] for w in worlds.values()])) >= C4_D2_MEAN,
        "C5": (C5_B[0] <= float(np.mean([w["C5"]["b"] for w in worlds.values()])) <= C5_B[1]
               and C5_Q[0] <= float(np.mean([w["C5"]["q"] for w in worlds.values()])) <= C5_Q[1]
               and frac(lambda w: w["C5"]["q"] < C5_Q_DISCONTINUITY) >= need),
    }
    c1 = decisions["C1a"] and decisions["C1b"] and decisions["C1c"]
    c3 = all(decisions["C3"].values())
    verdict = {
        "PROGRAM_CONFIRMED": bool(c1),
        "EXPORT_CONFIRMED": bool(decisions["C2"]),
        "COMPOSITION_CONFIRMED": bool(decisions["C2"] and c3),
        "LENGTH_CLOSED_COMPOSITION_CONFIRMED": bool(decisions["C2"] and c3 and decisions["C4"]),
        "FULL_PROGRAM_LANGUAGE_BLOCK_CONFIRMED": bool(c1 and decisions["C2"] and c3
                                                      and decisions["C4"]),
        "DRIFT_MECHANISM_CONFIRMED": bool(decisions["C5"]),
    }
    out = {"frozen_plan": "EXPORT_CONFIRMATION_PLAN.md", "git_commit": git_commit(),
           "seeds": seeds, "worlds_required": need,
           "summary": {"Q_D_mean": float(np.mean(qd)), "N_star_mean": float(np.mean(ns)),
                       "G_export_mean": float(np.mean(gexp)),
                       "delta_comp_mean": {s: float(np.mean(
                           [w["strata"][s]["delta_comp"] for w in worlds.values()])) for s in STRATA},
                       "delta_depth4_mean": float(np.mean(
                           [w["depths"]["depth4"]["delta"] for w in worlds.values()])),
                       "delta_depth2_mean": float(np.mean(
                           [w["depths"]["depth2"]["delta"] for w in worlds.values()])),
                       "b_mean": float(np.mean([w["C5"]["b"] for w in worlds.values()])),
                       "q_mean": float(np.mean([w["C5"]["q"] for w in worlds.values()]))},
           "decisions": decisions, "verdict": verdict, "worlds": worlds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(json.dumps(out["summary"], indent=2))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
