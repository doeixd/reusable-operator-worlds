"""E5: can the learner efficiently WRITE the programs we know exist?

`E5_SYNTHESIZER_PLAN.md` (Amendment 1). Two settings:

    D = 3   1,728 programs   ENUM is cheap; the recognizer's real opponent
    D = 6   ~2.99M programs  ENUM infeasible; C_find can actually differ

Arms: O (oracle program), ENUM (exhaustive support-scored search), OPT (the
sealed route optimization), REC (the amortized writer, top-k re-ranked), S
(scratch). Costs are reported in EXECUTIONS and DEVICE-SECONDS, with the
recognizer charged for its own training via `C_amortize`.

Correctness is FUNCTIONAL. Route agreement is reported and gates nothing.
Fails closed; per-cell cache; atomic report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from row.config import load_config
from row.experiments.audit_e0_export import assign, distance_matrix, git_commit, load_model
from row.experiments.audit_e1_export import ADAPT_LR, ADAPT_STEPS, geo, nmse
from row.experiments.audit_e8_length import VariableDepthDiscrete, adapt_cell
from row.models.program_recognizer import ProgramRecognizer
from row.support_split_world import _build_tasks
from row.world import World, WorldConfig

WORLDS = (0, 1, 2)
SETTINGS = {"D3": 3, "D6": 6}
PER_SETTING = 12
TOP_K = (1, 5, 25)
ORACLE_GAP = 0.15
COST_RATIO = 0.1
D6_GATE = 0.75
REC_EPOCHS = 400
REC_LR = 1e-3
CACHE = Path("reports/e5_cache")


def fingerprint() -> str:
    payload = {"settings": SETTINGS, "per_setting": PER_SETTING, "k": list(TOP_K),
               "steps": ADAPT_STEPS, "lr": ADAPT_LR, "rec_epochs": REC_EPOCHS, "rec_lr": REC_LR}
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
    if not condition:
        raise SystemExit(f"FATAL: {message}")


def stable_tag(tag: str) -> int:
    return int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)


@torch.no_grad()
def execute(library, x: torch.Tensor, program) -> torch.Tensor:
    z = x
    for index in program:
        z = library[index](z)
    return z


def load_cell(world: int, config_path: Path) -> dict:
    path = Path("artifacts/e1_disc") / f"world_{world}"
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
    world_obj = World.generate(WorldConfig(**raw["world"]))
    probe = torch.tensor(np.random.default_rng(np.random.SeedSequence(
        [900, world, 1])).normal(size=(256, config.world.state_dim)), dtype=torch.float32)
    d = distance_matrix(shipped, world_obj, config.world.program_length, probe, "library",
                        config.world.seed)
    best, _, _ = assign(d["matrix"])
    # the depth-3 equivalence control, re-verified per world
    x = torch.tensor(np.random.default_rng(np.random.SeedSequence(
        [900, world, 7])).normal(size=(32, config.world.state_dim)), dtype=torch.float32)
    with torch.no_grad():
        for task_id in list(shipped.task_codes)[:8]:
            fatal(bool(torch.equal(shipped(x, task_id), variable(x, task_id))),
                  "variable-depth executor differs from the shipped one at depth 3")
    return {"config": config, "shipped": shipped, "model": variable, "world": world_obj,
            "assignment": best, "path": path}


def train_recognizer(cell: dict, world: int, depth: int) -> dict:
    """Train the writer on this world's TRAINED tasks only, library frozen."""
    config, shipped = cell["config"], cell["shipped"]
    slots = shipped.operator_slots
    tasks = [t for t in cell["world"].tasks if t.task_id in shipped.task_codes]
    xs, ys, targets = [], [], []
    for task in tasks:
        with torch.no_grad():
            route = [int(v) for v in torch.argmax(shipped.task_codes[task.task_id], dim=-1)]
        xs.append(torch.tensor(task.train_x, dtype=torch.float32))
        ys.append(torch.tensor(task.train_y, dtype=torch.float32))
        targets.append(route)
    trained_depth = config.world.program_length
    # A recognizer for depth `depth` is trained on depth-3 supervision when the
    # setting is deeper: the heads beyond the trained depth have no supervised
    # signal, so they are trained on RE-EXECUTED depth-`depth` programs drawn
    # from the library itself (self-supervision through the frozen executor).
    synthetic = []
    if depth != trained_depth:
        rng = np.random.default_rng(np.random.SeedSequence([900, world, depth]))
        for index in range(256):
            program = [int(v) for v in rng.integers(0, slots, depth)]
            x = torch.tensor(rng.normal(size=(config.world.examples_per_task,
                                              config.world.state_dim)), dtype=torch.float32)
            synthetic.append((x, execute(shipped.library, x, program), program))
    model = ProgramRecognizer(config.world.state_dim, slots, depth,
                              seed=5000 + world * 17 + depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=REC_LR)
    start = time.process_time()
    first_loss, last_loss = None, None
    for epoch in range(REC_EPOCHS):
        total = 0.0
        batch = synthetic if depth != trained_depth else list(zip(xs, ys, targets))
        for x, y, program in batch:
            optimizer.zero_grad()
            logits = model(x, y)
            loss = sum(torch.nn.functional.cross_entropy(
                logits[step].unsqueeze(0), torch.tensor([program[step]])) for step in range(depth))
            loss.backward()
            optimizer.step()
            total += float(loss)
        if epoch == 0:
            first_loss = total / max(len(batch), 1)
        last_loss = total / max(len(batch), 1)
    seconds = time.process_time() - start
    return {"model": model, "train_seconds": seconds, "first_loss": first_loss,
            "last_loss": last_loss, "examples": len(synthetic) if synthetic else len(tasks),
            "uniform_loss": depth * math.log(slots)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e5_synthesizer.json"))
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    out = {"frozen_plan": "E5_SYNTHESIZER_PLAN.md (Amendment 1)", "git_commit": git_commit(),
           "protocol": {"settings": SETTINGS, "per_setting": PER_SETTING, "k": list(TOP_K),
                        "oracle_gap": ORACLE_GAP, "cost_ratio": COST_RATIO,
                        "note": "correctness is functional; route agreement gates nothing"},
           "settings": {}}
    cells = {w: load_cell(w, args.config) for w in WORLDS}
    for name in args.settings:
        depth = SETTINGS[name]
        space = cells[0]["shipped"].operator_slots ** depth
        enum_feasible = space <= 20_000
        per_world = {}
        for world in WORLDS:
            cell = cells[world]
            config, model = cell["config"], cell["model"]
            # TWO index spaces, deliberately named apart: tasks are built from
            # the TEACHER library (6 primitives), while every arm writes and
            # executes programs over the LEARNER's slots (12 operators).
            teacher_library = cell["world"].library
            library = cell["shipped"].library
            slots = model.operator_slots
            rng = np.random.default_rng(np.random.SeedSequence([900, world, stable_tag(name)]))
            trained = {tuple(t.program.primitive_ids) for t in cell["world"].tasks}
            # Test TASKS are teacher programs (indices over the teacher's
            # primitives); the ARMS write programs over the learner's slots. The
            # two index spaces are different sizes and must not be conflated.
            programs = []
            while len(programs) < PER_SETTING:
                candidate = tuple(int(v) for v in
                                  rng.integers(0, config.world.teacher_primitives, depth))
                if depth == config.world.program_length and candidate in trained:
                    continue
                if candidate not in programs:
                    programs.append(candidate)
            rec = train_recognizer(cell, world, depth)
            fatal(rec["last_loss"] < rec["first_loss"], "recognizer training loss did not fall")
            fatal(rec["last_loss"] < rec["uniform_loss"],
                  "recognizer no better than a uniform prior")
            arms = {a: [] for a in ("O", "ENUM", "OPT", "S")}
            arms.update({f"REC{k}": [] for k in TOP_K})
            costs = {a: [] for a in arms}
            agree = {f"REC{k}": [] for k in TOP_K}
            for index, program in enumerate(programs):
                task = _build_tasks(config.world, teacher_library, [program],
                                    [f"task_e5_{name}_{index}"], index_offset=5000 + index)[0]
                sx = torch.tensor(task.train_x, dtype=torch.float32)
                sy = torch.tensor(task.train_y, dtype=torch.float32)
                qx = torch.tensor(task.eval_x, dtype=torch.float32)
                qy = torch.tensor(task.eval_y, dtype=torch.float32)
                tag = f"w{world}_{name}_{index}"

                def cellwise(task=task, program=program, tag=tag, sx=sx, sy=sy, qx=qx, qy=qy):
                    res = {}
                    t0 = time.process_time()
                    res["O"] = {"nmse": nmse(execute(
                        library, qx, [cell["assignment"][int(p)] for p in program]), qy),
                        "executions": 0, "seconds": time.process_time() - t0}
                    if enum_feasible:
                        t0 = time.process_time()
                        best, best_loss = None, float("inf")
                        for candidate in itertools.product(range(slots), repeat=depth):
                            loss = float(torch.mean((execute(library, sx, candidate) - sy) ** 2))
                            if loss < best_loss:
                                best, best_loss = candidate, loss
                        res["ENUM"] = {"nmse": nmse(execute(library, qx, best), qy),
                                       "executions": space, "program": list(best),
                                       "seconds": time.process_time() - t0}
                    t0 = time.process_time()
                    opt = adapt_cell(model, task, f"e5OPT_{tag}", depth, False,
                                     teacher_library, program)
                    res["OPT"] = {"nmse": opt["query_nmse"], "executions": 2 * ADAPT_STEPS,
                                  "seconds": time.process_time() - t0,
                                  "support_reduction": opt["support_reduction_objective"]}
                    t0 = time.process_time()
                    scratch_lib = copy.deepcopy(model)
                    s = adapt_cell(scratch_lib, task, f"e5S_{tag}", depth, True,
                                   teacher_library, program)
                    res["S"] = {"nmse": s["query_nmse"], "executions": 2 * ADAPT_STEPS,
                                "seconds": time.process_time() - t0,
                                "support_reduction": s["support_reduction_objective"]}
                    for k in TOP_K:
                        t0 = time.process_time()
                        candidates = rec["model"].top_k_programs(sx, sy, k)
                        best, best_loss = candidates[0], float("inf")
                        for candidate in candidates:
                            loss = float(torch.mean((execute(library, sx, candidate) - sy) ** 2))
                            if loss < best_loss:
                                best, best_loss = candidate, loss
                        res[f"REC{k}"] = {"nmse": nmse(execute(library, qx, best), qy),
                                          "executions": k, "program": list(best),
                                          "seconds": time.process_time() - t0,
                                          "agrees_with_oracle": bool(
                                              list(best) == [cell["assignment"][int(p)]
                                                             for p in program])}
                    return res

                res = cached(tag, cellwise)
                for arm in arms:
                    if arm in res:
                        arms[arm].append(res[arm]["nmse"])
                        costs[arm].append((res[arm]["executions"], res[arm]["seconds"]))
                for k in TOP_K:
                    agree[f"REC{k}"].append(res[f"REC{k}"]["agrees_with_oracle"])
            summary = {a: geo(v) for a, v in arms.items() if v}
            cost_summary = {a: {"executions": float(np.mean([c[0] for c in v])),
                                "seconds": float(np.mean([c[1] for c in v]))}
                            for a, v in costs.items() if v}
            per_world[str(world)] = {
                "programs": [list(p) for p in programs],
                "geomean_nmse": summary, "cost": cost_summary,
                "oracle_gap": {a: float(np.log(summary[a]) - np.log(summary["O"]))
                               for a in summary if a != "O"},
                "recognizer": {k: v for k, v in rec.items() if k != "model"},
                "C_amortize_seconds_per_task": rec["train_seconds"] / PER_SETTING,
                "route_agreement": {k: float(np.mean(v)) for k, v in agree.items()},
                "enum_feasible": enum_feasible, "program_space": space,
            }
            r = per_world[str(world)]
            print(f"[{name} w{world}] O {summary['O']:.5f} "
                  + (f"ENUM {summary['ENUM']:.5f} " if enum_feasible else "ENUM infeasible ")
                  + f"OPT {summary['OPT']:.5f} S {summary['S']:.5f} | "
                  + " ".join(f"REC{k} {summary[f'REC{k}']:.5f}" for k in TOP_K)
                  + f" | gaps " + " ".join(f"{k}:{r['oracle_gap'][f'REC{k}']:+.2f}" for k in TOP_K),
                  flush=True)
        out["settings"][name] = {"depth": depth, "program_space": space,
                                 "enum_feasible": enum_feasible, "worlds": per_world}
    # ---- decisions ---------------------------------------------------------
    decisions = {}
    for name, block in out["settings"].items():
        worlds = block["worlds"]
        gate_ok = True
        if block["depth"] != 3:
            gate = [float(np.log(w["geomean_nmse"]["S"]) - np.log(w["geomean_nmse"]["O"]))
                    for w in worlds.values()]
            gate_ok = sum(g >= D6_GATE for g in gate) >= 2
        best_cost = {}
        for w in worlds.values():
            reference = min([w["cost"][a]["executions"] for a in ("ENUM", "OPT") if a in w["cost"]])
            ref_sec = min([w["cost"][a]["seconds"] for a in ("ENUM", "OPT") if a in w["cost"]])
            best_cost[id(w)] = (reference, ref_sec)
        per_k = {}
        for k in TOP_K:
            key = f"REC{k}"
            quality = sum(w["oracle_gap"][key] <= ORACLE_GAP for w in worlds.values()) >= 2
            cost = sum(
                w["cost"][key]["executions"] <= COST_RATIO * best_cost[id(w)][0]
                and w["cost"][key]["seconds"] <= COST_RATIO * best_cost[id(w)][1]
                for w in worlds.values()) >= 2
            per_k[key] = {"quality": bool(quality), "cost": bool(cost),
                          "synthesis": bool(quality and cost)}
        outcome = ("UNINTERPRETABLE (oracle gate failed)" if not gate_ok
                   else "SYNTHESIS DEMONSTRATED" if any(v["synthesis"] for v in per_k.values())
                   else "QUALITY WITHOUT AMORTIZATION" if any(v["quality"] for v in per_k.values())
                   else "AMORTIZATION WITHOUT QUALITY" if any(v["cost"] for v in per_k.values())
                   else "NO SYNTHESIS")
        decisions[name] = {"gate_ok": bool(gate_ok), "per_k": per_k, "outcome": outcome}
    out["decisions"] = decisions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    for name, d in decisions.items():
        print(f"{name}: {d['outcome']}  {json.dumps(d['per_k'])}")


if __name__ == "__main__":
    main()
