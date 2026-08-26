"""E1: does a frozen library execute a program it never trained on?

`E1_FROZEN_EXPORT_PLAN.md` (Amendment 1), under `EXPORT_BRANCH_PROGRAM.md`
(Amendments 1-2). Interface is E1-P throughout: the library is frozen and only
the route is inferred. DISC has no private residual channel, so E1-PR and E1-R
are undefined here and are recorded as such rather than run.

Arms per held-out program:

    O    teacher program through this world's functional assignment (no fitting)
    O-W  teacher program through ANOTHER world's library and assignment
    R    route inferred from support, this world's frozen library
    R-W  route inferred from support, another world's frozen library
    S    scratch: fresh library AND route trained on support
    F    full finetune: this library and route trained on support

`H1` (triple-novel) and `H2` (pair-novel) are reported separately.
Fails closed; per-cell cache with a protocol fingerprint; atomic report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from row.config import load_config
from row.experiments.audit_e0_export import (SUBSTRATES, assign, distance_matrix, git_commit,
                                             load_model, world_of)
from row.experiments.learned_lifetime import _build_model
from row.world import Program

WORLDS = (0, 1, 2)
PER_STRATUM = 12
ADAPT_STEPS = 2000
ADAPT_LR = 0.01
MARGIN = 0.15
CACHE = Path("reports/e1_cache")


def protocol_fingerprint() -> str:
    payload = {"per_stratum": PER_STRATUM, "steps": ADAPT_STEPS, "lr": ADAPT_LR,
               "margin": MARGIN, "arms": ["O", "O-W", "R", "R-W", "S", "F"],
               "diagnostics": "amendment2-mode-consistent"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cached(key: str, compute):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.json"
    fingerprint = protocol_fingerprint()
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("protocol") != fingerprint:
            raise SystemExit(f"cached cell {key} under protocol {stored.get('protocol')}, "
                             f"not {fingerprint}; delete reports/e1_cache to rescore")
        return stored["value"]
    value = compute()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"protocol": fingerprint, "value": value}), encoding="utf-8")
    os.replace(tmp, path)
    return value


def pairs_of(program) -> set:
    return {(i, program[i], program[i + 1]) for i in range(len(program) - 1)}


def held_out_programs(world, seed: int) -> dict:
    """Unseen programs, split into triple-novel (H1) and pair-novel (H2)."""
    depth = world.config.program_length
    k = world.config.teacher_primitives
    trained = [tuple(task.program.primitive_ids) for task in world.tasks]
    trained_set = set(trained)
    seen_pairs = set()
    seen_positions = {(i, p) for program in trained for i, p in enumerate(program)}
    for program in trained:
        seen_pairs |= pairs_of(program)
    h1, h2 = [], []
    for program in itertools.product(range(k), repeat=depth):
        if program in trained_set:
            continue
        if any((i, p) not in seen_positions for i, p in enumerate(program)):
            continue                      # a primitive never trained in that position
        (h1 if pairs_of(program) <= seen_pairs else h2).append(program)
    rng = np.random.default_rng(np.random.SeedSequence([761, seed]))
    out = {}
    for name, pool in (("H1", h1), ("H2", h2)):
        order = rng.permutation(len(pool))
        out[name] = [pool[int(i)] for i in order[:PER_STRATUM]]
    return {"programs": out, "available": {"H1": len(h1), "H2": len(h2)},
            "trained": len(trained_set)}


def synth_task(world, program, index: int, seed: int):
    """Support and query examples for an unseen program, from disjoint streams."""
    rng_support = np.random.default_rng(np.random.SeedSequence([762, seed, index, 0]))
    rng_query = np.random.default_rng(np.random.SeedSequence([762, seed, index, 1]))
    d = world.config.state_dim
    support_x = rng_support.normal(size=(world.config.examples_per_task, d))
    query_x = rng_query.normal(size=(world.config.evaluation_examples, d))
    executor = Program(tuple(program))
    return {
        "support_x": torch.tensor(support_x, dtype=torch.float32),
        "support_y": torch.tensor(executor.execute(world.library, support_x), dtype=torch.float32),
        "query_x": torch.tensor(query_x, dtype=torch.float32),
        "query_y": torch.tensor(executor.execute(world.library, query_x), dtype=torch.float32),
    }


def nmse(pred, y) -> float:
    return float(torch.mean((pred - y) ** 2) / (torch.var(y, unbiased=False) + 1e-12))


def oracle_route(model, task, program, assignment, probe_id: str) -> dict:
    local = copy.deepcopy(model)
    local.begin_task(probe_id)
    with torch.no_grad():
        logits = torch.full_like(local.task_codes[probe_id], -50.0)
        for step, primitive in enumerate(program):
            logits[step, assignment[int(primitive)]] = 50.0
        local.task_codes[probe_id].copy_(logits)
    local.eval()
    with torch.no_grad():
        return {"query_nmse": nmse(local(task["query_x"], probe_id), task["query_y"]),
                "support_nmse": nmse(local(task["support_x"], probe_id), task["support_y"])}


def adapt(model, task, probe_id: str, train_library: bool) -> dict:
    """Fit the route (and optionally the library) on SUPPORT only."""
    local = copy.deepcopy(model)
    local.begin_task(probe_id)
    for parameter in local.parameters():
        parameter.requires_grad_(False)
    code = local.task_codes[probe_id]
    code.requires_grad_(True)
    params = [code]
    if train_library:
        for operator in local.library:
            for parameter in operator.parameters():
                parameter.requires_grad_(True)
                params.append(parameter)
    optimizer = torch.optim.Adam(params, lr=ADAPT_LR)
    # Amendment 2: both endpoints of each reduction are measured in the SAME
    # mode. This learner is relaxed-in-training and hard-at-evaluation, so a
    # train-mode initial against an eval-mode final measures nothing.
    local.train()
    with torch.no_grad():
        initial_objective = nmse(local(task["support_x"], probe_id), task["support_y"])
    local.eval()
    with torch.no_grad():
        initial_eval = nmse(local(task["support_x"], probe_id), task["support_y"])
    local.train()
    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        loss = torch.mean((local(task["support_x"], probe_id) - task["support_y"]) ** 2)
        if not bool(torch.isfinite(loss)):
            raise SystemExit(f"non-finite adaptation loss for {probe_id}")
        loss.backward(inputs=params)
        optimizer.step()
    with torch.no_grad():
        final_objective = nmse(local(task["support_x"], probe_id), task["support_y"])
    local.eval()
    with torch.no_grad():
        final_support = nmse(local(task["support_x"], probe_id), task["support_y"])
        query = nmse(local(task["query_x"], probe_id), task["query_y"])
    return {"query_nmse": query, "support_nmse": final_support,
            "support_reduction_objective":
                (initial_objective - final_objective) / max(initial_objective, 1e-12),
            "support_reduction_eval":
                (initial_eval - final_support) / max(initial_eval, 1e-12)}


def scratch_model(config, kind: str, seed_offset: int):
    local = replace(config, discrete_model=replace(config.discrete_model,
                                                   seed=config.discrete_model.seed + seed_offset))
    model = _build_model(local, kind)
    model.eval()
    return model


def geo(values) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(values, 1e-12)))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e1_export.json"))
    parser.add_argument("--worlds", nargs="+", type=int, default=list(WORLDS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")

    models, worlds, assignments, configs = {}, {}, {}, {}
    # The wrong-library arm needs its donor loaded even when only a subset of
    # worlds is scored (Amendment 1: world w receives world (w+1) mod 3).
    needed = sorted(set(args.worlds) | {(w + 1) % 3 for w in args.worlds})
    for w in needed:
        path = Path(SUBSTRATES[f"DISC_w{w}"]["path"])
        raw = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
        config = load_config(args.config)
        config = replace(config, world=replace(config.world, **raw["world"]))
        fields = set(config.discrete_model.__dataclass_fields__)
        config = replace(config, discrete_model=replace(
            config.discrete_model, **{k: v for k, v in raw["discrete_model"].items() if k in fields}))
        model, _, _ = load_model(config, path, "discrete")
        world = world_of(path)
        probe = torch.tensor(np.random.default_rng(
            np.random.SeedSequence([760, world.config.seed, 99])).normal(
            size=(256, world.config.state_dim)), dtype=torch.float32)
        d = distance_matrix(model, world, world.config.program_length, probe, "library",
                            world.config.seed)
        best, cost, margin = assign(d["matrix"])
        models[w], worlds[w], assignments[w], configs[w] = model, world, best, config
        print(f"[w{w}] assignment mean distance {cost:.3f}, margin {margin:.3f}", flush=True)

    out = {"frozen_plan": "E1_FROZEN_EXPORT_PLAN.md (Amendment 1)", "git_commit": git_commit(),
           "protocol": {"interface": "E1-P (frozen library, route only); E1-PR/E1-R undefined "
                                     "on DISC, which has no private residual channel",
                        "steps": ADAPT_STEPS, "lr": ADAPT_LR, "per_stratum": PER_STRATUM,
                        "margin": MARGIN, "donor": "world w receives world (w+1) mod 3's library"},
           "worlds": {}}
    for w in args.worlds:
        world, model, assignment = worlds[w], models[w], assignments[w]
        donor = (w + 1) % 3
        selection = held_out_programs(world, world.config.seed)
        rows = {"available": selection["available"], "trained_programs": selection["trained"],
                "donor_world": donor, "strata": {}}
        for stratum, programs in selection["programs"].items():
            arms = {a: [] for a in ("O", "O-W", "R", "R-W", "S", "F")}
            checks, exempt = [], []
            for index, program in enumerate(programs):
                task = synth_task(world, program, index, world.config.seed)
                tag = f"w{w}_{stratum}_{index}"
                cell = cached(f"{tag}", lambda task=task, program=program, tag=tag, donor=donor: {
                    "O": oracle_route(model, task, program, assignment, f"e1O_{tag}"),
                    "O-W": oracle_route(models[donor], task, program, assignments[donor],
                                        f"e1OW_{tag}"),
                    "R": adapt(model, task, f"e1R_{tag}", train_library=False),
                    "R-W": adapt(models[donor], task, f"e1RW_{tag}", train_library=False),
                    "S": adapt(scratch_model(configs[w], "discrete", 7717), task, f"e1S_{tag}",
                               train_library=True),
                    "F": adapt(model, task, f"e1F_{tag}", train_library=True),
                })
                for arm in arms:
                    arms[arm].append(cell[arm]["query_nmse"])
                checks.append({a: cell[a].get("support_reduction_objective")
                               for a in ("R", "S")})     # Amendment 2: claim-bearing arms only
                exempt.append({a: {k: cell[a].get(k) for k in
                                   ("support_reduction_objective", "support_reduction_eval")}
                               for a in ("R-W", "F")})
                print(f"[w{w} {stratum} {index}] O {cell['O']['query_nmse']:.4f} "
                      f"O-W {cell['O-W']['query_nmse']:.4f} R {cell['R']['query_nmse']:.4f} "
                      f"S {cell['S']['query_nmse']:.4f} F {cell['F']['query_nmse']:.4f}", flush=True)
            summary = {arm: geo(values) for arm, values in arms.items()}
            weak = [c for c in checks if any(v is not None and v <= 0.01 for v in c.values())]
            rows["strata"][stratum] = {
                "programs": [list(p) for p in programs], "n": len(programs),
                "geomean_query_nmse": summary,
                "margin_O_vs_S": float(np.log(summary["S"]) - np.log(summary["O"])),
                "margin_O_vs_OW": float(np.log(summary["O-W"]) - np.log(summary["O"])),
                "margin_R_vs_S": float(np.log(summary["S"]) - np.log(summary["R"])),
                "R_minus_O": float(np.log(summary["R"]) - np.log(summary["O"])),
                "C_repair": float(summary["R"] - summary["F"]),
                "G_export": (float((summary["S"] - summary["R"]) / (summary["S"] - summary["O"]))
                             if summary["S"] - summary["O"] > 0 and summary["S"] > 2 * summary["O"]
                             else None),
                "weak_adaptation_cells_claim_bearing": len(weak),
                "exempt_arms_note": ("R-W and F are exempt from the non-vacuity clause "
                                     "(Amendment 2): a wrong library and a destructive "
                                     "finetune budget are DESIGNED not to improve"),
                "exempt_arm_reductions": {
                    a: {k: float(np.mean([e[a][k] for e in exempt])) for k in
                        ("support_reduction_objective", "support_reduction_eval")}
                    for a in ("R-W", "F")},
            }
        out["worlds"][str(w)] = rows
    # ---- decisions -------------------------------------------------------
    decisions = {}
    for stratum in ("H1", "H2"):
        per_world = {str(w): out["worlds"][str(w)]["strata"][stratum] for w in args.worlds}
        e1a = (sum(r["margin_O_vs_S"] >= MARGIN for r in per_world.values()) >= 2
               and sum(r["margin_O_vs_OW"] >= MARGIN for r in per_world.values()) >= 2)
        e1b = (sum(r["margin_R_vs_S"] >= MARGIN for r in per_world.values()) >= 2
               and sum(r["R_minus_O"] <= MARGIN for r in per_world.values()) >= 2)
        decisions[stratum] = {"E1a_vocabulary_exports": bool(e1a),
                              "E1b_route_findable": bool(e1b),
                              "per_world": {k: {m: r[m] for m in
                                                ("margin_O_vs_S", "margin_O_vs_OW",
                                                 "margin_R_vs_S", "R_minus_O", "G_export")}
                                            for k, r in per_world.items()}}
    out["decisions"] = decisions
    e1a_any = any(d["E1a_vocabulary_exports"] for d in decisions.values())
    e1b_any = any(d["E1b_route_findable"] for d in decisions.values())
    out["outcome"] = ("E1a+E1b: VOCABULARY EXPORTS AND ROUTE IS FINDABLE" if e1a_any and e1b_any
                      else "E1a ONLY: VOCABULARY EXPORTS, WRITER/SEARCH MISSING" if e1a_any
                      else "E1a FAILS: STOP - objects are not exportable program primitives")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    for stratum, d in decisions.items():
        print(f"{stratum}: E1a={d['E1a_vocabulary_exports']} E1b={d['E1b_route_findable']}")
    print(f"OUTCOME {out['outcome']}")


if __name__ == "__main__":
    main()
