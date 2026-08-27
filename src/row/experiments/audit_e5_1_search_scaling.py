"""E5.1: which horizon binds first, execution or search?

`E5_1_SEARCH_SCALING_PLAN.md`, frozen. E5 measured findability at exactly two
depths and saw no degradation across a 1,728x growth in program space. This
module sweeps depth and search budget together to separate two horizons the
project has always conflated:

    D_execute  the depth at which the frozen operators stop composing well
               enough for ANY program to solve the task (eligibility fails)
    D_search   the depth at which finding a good program stops working, given
               that good programs still exist

Arms: O (oracle program through the E0.1 assignment), OPT (support-only route
optimization at K steps), ENUM (exhaustive, only where the space is small), S
(scratch: a FRESH library and route, per `scratch_model`, matching E1 and E8).

Correctness is FUNCTIONAL. Route agreement is reported and gates nothing.
Fails closed; protocol-fingerprinted per-cell cache; atomic report.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_e1_export import ADAPT_LR, ADAPT_STEPS, geo, nmse, scratch_model
from row.experiments.audit_e5_synthesizer import execute, fatal, load_cell
from row.experiments.audit_e8_length import adapt_cell
from row.support_split_world import _build_tasks

WORLDS = (0, 1, 2)
DEPTHS = (3, 4, 5, 6, 7, 8, 9, 10)
BUDGETS = (250, 500, 1000, 2000)
ANCHOR_K = 2000
PER_DEPTH = 8
ENUM_MAX = 250_000
ELIGIBLE_MARGIN = 0.75      # oracle over scratch, log units
GAP = 0.15                  # oracle gap defining "search still works"
CACHE = Path("reports/e5_1_cache")


def fingerprint() -> str:
    payload = {"depths": list(DEPTHS), "budgets": list(BUDGETS), "per_depth": PER_DEPTH,
               "lr": ADAPT_LR, "anchor": ANCHOR_K, "enum_max": ENUM_MAX,
               "margin": ELIGIBLE_MARGIN, "gap": GAP}
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


def two_of_three(flags) -> bool:
    return sum(1 for f in flags if f) >= 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e5_1_search_scaling.json"))
    parser.add_argument("--depths", nargs="+", type=int, default=list(DEPTHS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")

    out = {"frozen_plan": "E5_1_SEARCH_SCALING_PLAN.md", "git_commit": git_commit(),
           "protocol": {"depths": list(args.depths), "budgets": list(BUDGETS),
                        "per_depth": PER_DEPTH, "lr": ADAPT_LR,
                        "anchor_k": ANCHOR_K, "enum_max": ENUM_MAX,
                        "eligible_margin": ELIGIBLE_MARGIN, "oracle_gap": GAP,
                        "scratch": "scratch_model(config, 'discrete', 7717), as in E1/E8",
                        "note": "correctness is functional; route agreement gates nothing"},
           "depths": {}}
    # `load_cell` re-verifies the depth-3 executor equivalence control per world.
    cells = {w: load_cell(w, args.config) for w in WORLDS}
    slots = cells[0]["shipped"].operator_slots

    for depth in args.depths:
        space = slots ** depth
        enum_feasible = space <= ENUM_MAX
        per_world = {}
        for world in WORLDS:
            cell = cells[world]
            config = cell["config"]
            teacher_library = cell["world"].library      # 6 teacher primitives
            library = cell["shipped"].library            # 12 learner slots
            # Programs depend on (world, depth) ONLY, so every budget and every
            # arm at this depth sees exactly the same tasks.
            rng = np.random.default_rng(np.random.SeedSequence([951, world, depth]))
            trained = {tuple(t.program.primitive_ids) for t in cell["world"].tasks}
            programs: list[tuple[int, ...]] = []
            while len(programs) < PER_DEPTH:
                candidate = tuple(int(v) for v in
                                  rng.integers(0, config.world.teacher_primitives, depth))
                if depth == config.world.program_length and candidate in trained:
                    continue
                if candidate not in programs:
                    programs.append(candidate)

            arms: dict[str, list[float]] = {"O": [], "ENUM": [], "S": []}
            arms.update({f"OPT{k}": [] for k in BUDGETS})
            costs: dict[str, list[tuple[float, float]]] = {a: [] for a in arms}
            reductions: dict[str, list[float]] = {f"OPT{k}": [] for k in BUDGETS}

            for index, program in enumerate(programs):
                task = _build_tasks(config.world, teacher_library, [program],
                                    [f"task_e51_d{depth}_{index}"],
                                    index_offset=51000 + depth * 100 + index)[0]
                sx = torch.tensor(task.train_x, dtype=torch.float32)
                sy = torch.tensor(task.train_y, dtype=torch.float32)
                qx = torch.tensor(task.eval_x, dtype=torch.float32)
                qy = torch.tensor(task.eval_y, dtype=torch.float32)
                tag = f"w{world}_d{depth}_{index}"

                def cellwise(program=program, task=task, tag=tag,
                             sx=sx, sy=sy, qx=qx, qy=qy):
                    res: dict[str, dict] = {}
                    t0 = time.process_time()
                    oracle_route = [cell["assignment"][int(p)] for p in program]
                    res["O"] = {"nmse": nmse(execute(library, qx, oracle_route), qy),
                                "executions": 0, "seconds": time.process_time() - t0,
                                "program": oracle_route}
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
                    for k in BUDGETS:
                        t0 = time.process_time()
                        opt = adapt_cell(cell["model"], task, f"e51O{k}_{tag}", depth,
                                         False, teacher_library, program, steps=k)
                        res[f"OPT{k}"] = {
                            "nmse": opt["query_nmse"], "executions": 2 * k,
                            "seconds": time.process_time() - t0,
                            "program": opt.get("route"),
                            "support_reduction": opt["support_reduction_objective"]}
                    t0 = time.process_time()
                    # A FRESH library, as in E1 and E8 -- never a copy of the
                    # trained one, which would make this a fine-tuning arm.
                    s = adapt_cell(scratch_model(config, "discrete", 7717), task,
                                   f"e51S_{tag}", depth, True, teacher_library, program,
                                   steps=ANCHOR_K)
                    res["S"] = {"nmse": s["query_nmse"], "executions": 2 * ANCHOR_K,
                                "seconds": time.process_time() - t0,
                                "support_reduction": s["support_reduction_objective"]}
                    return res

                res = cached(tag, cellwise)
                for arm in arms:
                    if arm in res:
                        arms[arm].append(res[arm]["nmse"])
                        costs[arm].append((res[arm]["executions"], res[arm]["seconds"]))
                for k in BUDGETS:
                    reductions[f"OPT{k}"].append(res[f"OPT{k}"]["support_reduction"])

            summary = {a: geo(v) for a, v in arms.items() if v}
            gaps = {a: float(np.log(summary[a]) - np.log(summary["O"]))
                    for a in summary if a != "O"}
            # Non-vacuity 2: OPT must actually have optimized in every cell.
            bad = [a for a, v in reductions.items() if min(v) <= 0.0]
            per_world[str(world)] = {
                "programs": [list(p) for p in programs],
                "geomean_nmse": summary, "oracle_gap": gaps,
                "cost": {a: {"executions": float(np.mean([c[0] for c in v])),
                             "seconds": float(np.mean([c[1] for c in v]))}
                         for a, v in costs.items() if v},
                "support_reduction_min": {a: float(min(v)) for a, v in reductions.items()},
                "cells_without_optimization": bad,
                "eligible": bool(math.log(summary["S"]) - math.log(summary["O"])
                                 >= ELIGIBLE_MARGIN),
                "scratch_over_oracle": float(np.log(summary["S"]) - np.log(summary["O"])),
                "enum_feasible": enum_feasible, "program_space": space,
            }
            r = per_world[str(world)]
            print(f"[d{depth} w{world}] O {summary['O']:.5f} "
                  + (f"ENUM {summary['ENUM']:.5f} " if enum_feasible
                     else f"ENUM {space:,} infeasible ")
                  + f"S {summary['S']:.5f} (S/O {r['scratch_over_oracle']:+.2f}, "
                  + ("ELIGIBLE" if r["eligible"] else "INELIGIBLE") + ") | "
                  + " ".join(f"K{k} {summary[f'OPT{k}']:.5f}({gaps[f'OPT{k}']:+.2f})"
                             for k in BUDGETS), flush=True)

        eligible = two_of_three([per_world[str(w)]["eligible"] for w in WORLDS])
        kstar = None
        for k in BUDGETS:
            if two_of_three([per_world[str(w)]["oracle_gap"][f"OPT{k}"] <= GAP for w in WORLDS]):
                kstar = k
                break
        out["depths"][str(depth)] = {
            "worlds": per_world, "eligible": eligible, "program_space": space,
            "enum_feasible": enum_feasible,
            "k_star": kstar if kstar is not None else f">{max(BUDGETS)}",
            "anchor_search_ok": two_of_three(
                [per_world[str(w)]["oracle_gap"][f"OPT{ANCHOR_K}"] <= GAP for w in WORLDS]),
        }
        print(f"  depth {depth}: eligible={eligible} k*={out['depths'][str(depth)]['k_star']} "
              f"anchor_search_ok={out['depths'][str(depth)]['anchor_search_ok']}", flush=True)
        write(out, args.output)
        if not eligible:
            print(f"  D_execute reached at depth {depth}; stopping the sweep.", flush=True)
            break

    swept = sorted(int(d) for d in out["depths"])
    d_execute = next((d for d in swept if not out["depths"][str(d)]["eligible"]), None)
    d_search = next((d for d in swept if out["depths"][str(d)]["eligible"]
                     and not out["depths"][str(d)]["anchor_search_ok"]), None)
    if d_execute is not None and (d_search is None or d_execute < d_search):
        outcome = "EXECUTION BINDS FIRST"
    elif d_search is not None and (d_execute is None or d_search < d_execute):
        outcome = "SEARCH BINDS FIRST"
    else:
        outcome = "NEITHER BINDS"
    # Non-vacuity 3 and 4, evaluated over what was actually swept.
    ks = {out["depths"][str(d)]["k_star"] for d in swept if out["depths"][str(d)]["eligible"]}
    oracles = [geo([out["depths"][str(d)]["worlds"][str(w)]["geomean_nmse"]["O"]
                    for w in WORLDS]) for d in swept]
    out["estimands"] = {
        "D_execute": d_execute, "D_search": d_search, "outcome": outcome,
        "k_star_by_depth": {str(d): out["depths"][str(d)]["k_star"] for d in swept},
        "budget_axis_vacuous": len(ks) <= 1,
        "oracle_error_by_depth": {str(d): oracles[i] for i, d in enumerate(swept)},
        "depth_axis_monotone": all(oracles[i] <= oracles[i + 1] for i in range(len(oracles) - 1)),
    }
    write(out, args.output)
    print(f"\nD_execute={d_execute} D_search={d_search} -> {outcome}")
    print(f"k* by depth: {out['estimands']['k_star_by_depth']}")
    print(f"budget axis vacuous: {out['estimands']['budget_axis_vacuous']} | "
          f"oracle error monotone in depth: {out['estimands']['depth_axis_monotone']}")


def write(payload: dict, path: Path) -> None:
    """Atomic: a completed report reaches disk before anything is printed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
