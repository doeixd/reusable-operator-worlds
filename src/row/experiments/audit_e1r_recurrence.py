"""E1-R: does export depend on the world having recurrent structure?

`E1R_RECURRENCE_CONTROL_PLAN.md`, a control under `EXPORT_BRANCH_PROGRAM.md`.

E1 passed only at exact reuse. Its controls varied the LIBRARY (scratch, wrong
world); this one varies RECURRENCE, the property V1 established as causal.

The held-out object is a held-out TASK, not a held-out program: at `rho < 1` the
teacher primitives are task-specific, so the same program index denotes a
different operator for a different task and programs are not comparable across
`rho`. Generating the world with `tasks = 76` reproduces the trained world's
first 64 tasks bitwise and supplies 12 the lifetime never saw; at `rho = 1` the
protocol reduces exactly to E1's, so those cells double as a reproduction check.

Arms: O (oracle route through a per-task functional assignment; a CEILING, and
it uses teacher information), R (support-only route inference over the frozen
library; the arm the verdict reads), S (scratch). Fails closed; atomic report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from row.config import load_config
from row.experiments.audit_e0_export import (assign, distance_matrix, git_commit, load_model,
                                             teacher_fn)
from row.experiments.audit_e1_export import adapt, geo, nmse, scratch_model
from row.world import World, WorldConfig

RHOS = ("0.0", "0.9", "1.0")
WORLDS = (0, 1, 2)
HELD_OUT = 12
CACHE = Path("reports/e1r_cache")


def artifact_for(rho: str, world: int) -> Path:
    if rho == "1.0":
        return Path("artifacts/e1_disc") / f"world_{world}"
    return Path("artifacts/e1r_disc") / f"rho{rho.replace('.', 'p')}_world_{world}"


def protocol_fingerprint() -> str:
    payload = {"rhos": list(RHOS), "held_out": HELD_OUT, "arms": ["O", "R", "S"],
               "protocol": "held-out TASK, tasks=76 prefix"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cached(key: str, compute):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.json"
    fingerprint = protocol_fingerprint()
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("protocol") != fingerprint:
            raise SystemExit(f"cached cell {key} under protocol {stored.get('protocol')}, "
                             f"not {fingerprint}; delete reports/e1r_cache to rescore")
        return stored["value"]
    value = compute()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"protocol": fingerprint, "value": value}), encoding="utf-8")
    os.replace(tmp, path)
    return value


def extended_world(path: Path, extra: int) -> tuple[World, World]:
    """The trained world, and the same world generated longer."""
    raw = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))["world"]
    trained = World.generate(WorldConfig(**raw))
    longer = World.generate(WorldConfig(**{**raw, "tasks": raw["tasks"] + extra}))
    for index in range(raw["tasks"]):
        a, b = trained.tasks[index], longer.tasks[index]
        if a.task_id != b.task_id or a.program.primitive_ids != b.program.primitive_ids \
                or not np.array_equal(a.train_x, b.train_x):
            raise SystemExit(f"prefix property violated at task {index} for {path}")
    return trained, longer


def measured_recurrence(path: Path) -> dict:
    """The project's OWN registered measurement, read from the artifact.

    `World.functional_reuse_diagnostics` writes `world_functional_reuse.json`
    for every run; it reports mean pairwise residual-function correlation, which
    is exactly 1.0 at `rho = 1` and ~0 at `rho = 0`. Recomputing it here would
    risk a second, subtly different definition of the project's central
    coordinate — an early version of this scorer did exactly that and read 0.983
    at `rho = 0`.
    """

    data = json.loads((path / "world_functional_reuse.json").read_text(encoding="utf-8"))
    return {"mean_pairwise_residual_correlation":
                float(data["mean_pairwise_residual_correlation"]),
            "configured_rho": float(data["configured_rho"]),
            "source": "artifact world_functional_reuse.json"}


def per_task_assignment(model, task, world_config, probe: torch.Tensor) -> dict:
    """Match the frozen library to THIS task's teacher primitives, functionally."""
    class _Shim:                      # distance_matrix() reads `.library` and `.config`
        pass
    shim = _Shim()
    shim.library = task.teacher_library
    shim.config = world_config
    d = distance_matrix(model, shim, world_config.program_length, probe, "library",
                        world_config.seed)
    best, cost, margin = assign(d["matrix"])
    return {"assignment": best, "mean_distance": cost, "margin": margin}


def oracle_route(model, task, program, assignment, probe_id: str) -> dict:
    local = copy.deepcopy(model)
    local.begin_task(probe_id)
    with torch.no_grad():
        logits = torch.full_like(local.task_codes[probe_id], -50.0)
        for step, primitive in enumerate(program):
            logits[step, assignment[int(primitive)]] = 50.0
        local.task_codes[probe_id].copy_(logits)
    local.eval()
    x = torch.tensor(task.eval_x, dtype=torch.float32)
    y = torch.tensor(task.eval_y, dtype=torch.float32)
    with torch.no_grad():
        return {"query_nmse": nmse(local(x, probe_id), y)}


def as_tensors(task) -> dict:
    return {"support_x": torch.tensor(task.train_x, dtype=torch.float32),
            "support_y": torch.tensor(task.train_y, dtype=torch.float32),
            "query_x": torch.tensor(task.eval_x, dtype=torch.float32),
            "query_y": torch.tensor(task.eval_y, dtype=torch.float32)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e1r_recurrence.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    out = {"frozen_plan": "E1R_RECURRENCE_CONTROL_PLAN.md", "git_commit": git_commit(),
           "protocol": {"held_out_tasks": HELD_OUT, "rhos": list(RHOS),
                        "note": "held-out TASK protocol; reduces to E1's at rho = 1"},
           "cells": {}}
    for rho in RHOS:
        for world in WORLDS:
            path = artifact_for(rho, world)
            if not (path / "summary.json").exists():
                raise SystemExit(f"missing artifact {path}")
            raw = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
            config = load_config(args.config)
            config = replace(config, world=replace(config.world, **raw["world"]))
            fields = set(config.discrete_model.__dataclass_fields__)
            config = replace(config, discrete_model=replace(
                config.discrete_model,
                **{k: v for k, v in raw["discrete_model"].items() if k in fields}))
            model, _, _ = load_model(config, path, "discrete")
            trained, longer = extended_world(path, HELD_OUT)
            held = longer.tasks[len(trained.tasks):]
            probe = torch.tensor(np.random.default_rng(np.random.SeedSequence(
                [763, world, 5])).normal(size=(256, config.world.state_dim)),
                dtype=torch.float32)
            recurrence = measured_recurrence(path)
            arms = {"O": [], "R": [], "S": []}
            weak, margins = 0, []
            for index, task in enumerate(held):
                tag = f"rho{rho.replace('.', 'p')}_w{world}_t{index}"
                cell = cached(tag, lambda task=task, index=index, tag=tag: (lambda fit: {
                    "O": oracle_route(model, task, task.program.primitive_ids,
                                      fit["assignment"], f"e1rO_{tag}"),
                    "assignment_distance": fit["mean_distance"],
                    "assignment_margin": fit["margin"],
                    "R": adapt(model, as_tensors(task), f"e1rR_{tag}", train_library=False),
                    "S": adapt(scratch_model(config, "discrete", 7717), as_tensors(task),
                               f"e1rS_{tag}", train_library=True),
                })(per_task_assignment(model, task, config.world, probe)))
                for arm in arms:
                    arms[arm].append(cell[arm]["query_nmse"])
                margins.append(cell["assignment_margin"])
                if any(cell[a].get("support_reduction_objective", 1.0) <= 0.01 for a in ("R", "S")):
                    weak += 1
            summary = {arm: geo(values) for arm, values in arms.items()}
            key = f"rho{rho}_w{world}"
            out["cells"][key] = {
                "artifact": str(path), "measured_recurrence": recurrence,
                "geomean_query_nmse": summary,
                "M_R": float(np.log(summary["S"]) - np.log(summary["R"])),
                "M_O": float(np.log(summary["S"]) - np.log(summary["O"])),
                "assignment_margin_mean": float(np.mean(margins)),
                "weak_adaptation_cells": weak, "n": len(held),
            }
            print(f"[rho {rho} w{world}] r={recurrence['mean_pairwise_residual_correlation']:+.3f} O {summary['O']:.5f} "
                  f"R {summary['R']:.5f} S {summary['S']:.5f} | M_R {out['cells'][key]['M_R']:+.3f} "
                  f"M_O {out['cells'][key]['M_O']:+.3f} weak={weak}", flush=True)
    # ---- decision --------------------------------------------------------
    def mr(rho):
        return [out["cells"][f"rho{rho}_w{w}"]["M_R"] for w in WORLDS
                if f"rho{rho}_w{w}" in out["cells"]]
    high, low, mid = mr("1.0"), mr("0.0"), mr("0.9")
    if not (high and low and mid):
        out["decision"] = {"outcome": "INCOMPLETE",
                           "M_R_rho1": high, "M_R_rho09": mid, "M_R_rho0": low}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("INCOMPLETE: not every rho x world cell is present")
        return
    dependent = sum(v >= 1.0 for v in high) >= 2 and sum(v <= 0.3 for v in low) >= 2
    independent = sum(v >= 1.0 for v in low) >= 2
    out["decision"] = {
        "M_R_rho1": high, "M_R_rho09": mid, "M_R_rho0": low,
        "outcome": ("RECURRENCE-DEPENDENT" if dependent
                    else "RECURRENCE-INDEPENDENT" if independent else "INTERMEDIATE"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(f"M_R by rho: 1.0 {[round(v,2) for v in high]}  0.9 {[round(v,2) for v in mid]}  "
          f"0.0 {[round(v,2) for v in low]}")
    print(f"OUTCOME {out['decision']['outcome']}")


if __name__ == "__main__":
    main()
