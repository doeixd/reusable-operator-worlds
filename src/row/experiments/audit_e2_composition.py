"""E2: systematic composition, and whether an operator survives a new position.

`E2_COMPOSITION_PLAN.md` (Amendment 1). Three strata over programs the lifetime
never trained on:

    H1  triple-novel   unseen program, every adjacent pair seen
    H2  pair-novel     unseen program, some adjacent pair never seen
    H3  position-novel places a primitive in a position it NEVER occupied

`H3` is the rung's point and the question E1 structurally could not ask.

Arms are E1's, so the two are directly comparable: O (teacher program through the
functional assignment), O-W and R-W (another world's library), R (route inferred
from support), S (scratch). Interface E1-P; the discrete substrate has no private
residual channel. Fails closed; per-cell cache; atomic report.
"""
from __future__ import annotations

import argparse
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
from row.experiments.audit_e0_export import assign, distance_matrix, git_commit, load_model
from row.experiments.audit_e1_export import ADAPT_LR, ADAPT_STEPS, adapt, geo, nmse, scratch_model
from row.experiments.audit_e1r_recurrence import as_tensors, oracle_route
from row.world import World, WorldConfig

WORLDS = (0, 1, 2)
STRATA = ("H1", "H2", "H3")
PER_STRATUM = 12
MARGIN = 0.15
CACHE = Path("reports/e2_cache")
ART = Path("artifacts/e2_support_split")


def protocol_fingerprint() -> str:
    payload = {"strata": list(STRATA), "per_stratum": PER_STRATUM, "steps": ADAPT_STEPS,
               "lr": ADAPT_LR, "margin": MARGIN, "arms": ["O", "O-W", "R", "R-W", "S"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cached(key: str, compute):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.json"
    fingerprint = protocol_fingerprint()
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("protocol") != fingerprint:
            raise SystemExit(f"cached cell {key} under protocol {stored.get('protocol')}, "
                             f"not {fingerprint}; delete reports/e2_cache to rescore")
        return stored["value"]
    value = compute()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"protocol": fingerprint, "value": value}), encoding="utf-8")
    os.replace(tmp, path)
    return value


def load_cell(world: int, config_path: Path):
    path = ART / f"world_{world}"
    raw = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    config = load_config(config_path)
    config = replace(config, world=replace(config.world, **raw["world"]))
    fields = set(config.discrete_model.__dataclass_fields__)
    config = replace(config, discrete_model=replace(
        config.discrete_model, **{k: v for k, v in raw["discrete_model"].items() if k in fields}))
    model, _, _ = load_model(config, path, "discrete")
    split = json.loads((path / "support_split.json").read_text(encoding="utf-8"))
    world_obj = World.generate(WorldConfig(**raw["world"]))     # for the base teacher library
    probe = torch.tensor(np.random.default_rng(np.random.SeedSequence(
        [765, world, 9])).normal(size=(256, config.world.state_dim)), dtype=torch.float32)
    d = distance_matrix(model, world_obj, config.world.program_length, probe, "library",
                        config.world.seed)
    best, cost, margin = assign(d["matrix"])
    return {"model": model, "config": config, "split": split, "world": world_obj,
            "assignment": best, "assignment_distance": cost, "assignment_margin": margin}


def build_task(config, library, program, index: int):
    """A held-out task, built with the world's own conventions at an unused index."""
    from row.support_split_world import _build_tasks
    return _build_tasks(config.world, library, [tuple(program)],
                        [f"task_e2_{index}"], index_offset=1000 + index)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e2_composition.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    cells = {w: load_cell(w, args.config) for w in WORLDS}
    out = {"frozen_plan": "E2_COMPOSITION_PLAN.md (Amendment 1)", "git_commit": git_commit(),
           "protocol": {"strata": list(STRATA), "per_stratum": PER_STRATUM,
                        "steps": ADAPT_STEPS, "lr": ADAPT_LR, "margin": MARGIN,
                        "interface": "E1-P; the discrete substrate has no residual channel",
                        "donor": "world w receives world (w+1) mod 3's library"},
           "worlds": {}}
    for w in WORLDS:
        cell = cells[w]
        donor = cells[(w + 1) % 3]
        split = cell["split"]
        held = split["withheld_placements"]
        trained = {tuple(p) for p in split["train_programs"]}
        rows = {"withheld_placements": held, "split_diagnostics": split["diagnostics"],
                "assignment_distance": cell["assignment_distance"],
                "assignment_margin": cell["assignment_margin"],
                "donor_world": (w + 1) % 3, "strata": {}}
        rng = np.random.default_rng(np.random.SeedSequence([765, w, 3]))
        for stratum in STRATA:
            pool = [tuple(p) for p in split["strata"][stratum]]
            order = rng.permutation(len(pool))
            programs = [pool[int(i)] for i in order[:PER_STRATUM]]
            leaked = [p for p in programs if p in trained]
            if leaked:
                raise SystemExit(f"held-out program present in training: {leaked[:3]}")
            if stratum == "H3":
                ok = [p for p in programs if any(p[i] == pr for pr, i in held)]
                if len(ok) != len(programs):
                    raise SystemExit("H3 program without a withheld placement")
            arms = {a: [] for a in ("O", "O-W", "R", "R-W", "S")}
            weak = 0
            for index, program in enumerate(programs):
                task = build_task(cell["config"], cell["world"].library, program, index)
                tag = f"w{w}_{stratum}_{index}"
                res = cached(tag, lambda task=task, program=program, tag=tag: {
                    "O": oracle_route(cell["model"], task, program, cell["assignment"],
                                      f"e2O_{tag}"),
                    "O-W": oracle_route(donor["model"], task, program, donor["assignment"],
                                        f"e2OW_{tag}"),
                    "R": adapt(cell["model"], as_tensors(task), f"e2R_{tag}", train_library=False),
                    "R-W": adapt(donor["model"], as_tensors(task), f"e2RW_{tag}",
                                 train_library=False),
                    "S": adapt(scratch_model(cell["config"], "discrete", 7717), as_tensors(task),
                               f"e2S_{tag}", train_library=True),
                })
                for arm in arms:
                    arms[arm].append(res[arm]["query_nmse"])
                if any(res[a].get("support_reduction_objective", 1.0) <= 0.01 for a in ("R", "S")):
                    weak += 1
            g = {arm: geo(values) for arm, values in arms.items()}
            rows["strata"][stratum] = {
                "n": len(programs), "programs": [list(p) for p in programs],
                "geomean_query_nmse": g, "weak_adaptation_cells": weak,
                "margin_O_vs_S": float(np.log(g["S"]) - np.log(g["O"])),
                "margin_O_vs_OW": float(np.log(g["O-W"]) - np.log(g["O"])),
                "margin_R_vs_S": float(np.log(g["S"]) - np.log(g["R"])),
                "R_minus_O": float(np.log(g["R"]) - np.log(g["O"])),
            }
            r = rows["strata"][stratum]
            print(f"[w{w} {stratum}] O {g['O']:.5f} O-W {g['O-W']:.5f} R {g['R']:.5f} "
                  f"S {g['S']:.5f} | O-S {r['margin_O_vs_S']:+.3f} O-OW {r['margin_O_vs_OW']:+.3f} "
                  f"R-S {r['margin_R_vs_S']:+.3f} weak={weak}", flush=True)
        out["worlds"][str(w)] = rows
    decisions = {}
    for stratum in STRATA:
        per = {str(w): out["worlds"][str(w)]["strata"][stratum] for w in WORLDS}
        holds = (sum(r["margin_O_vs_S"] >= MARGIN for r in per.values()) >= 2
                 and sum(r["margin_O_vs_OW"] >= MARGIN for r in per.values()) >= 2
                 and sum(r["margin_R_vs_S"] >= MARGIN for r in per.values()) >= 2)
        decisions[stratum] = {
            "composition_holds": bool(holds),
            "margins": {k: {m: r[m] for m in ("margin_O_vs_S", "margin_O_vs_OW",
                                              "margin_R_vs_S", "R_minus_O")}
                        for k, r in per.items()}}
    out["decisions"] = decisions
    h3 = decisions["H3"]["composition_holds"]
    base = decisions["H1"]["composition_holds"] and decisions["H2"]["composition_holds"]
    out["outcome"] = ("COMPOSITION HOLDS INCLUDING POSITION-NOVEL PLACEMENTS" if base and h3
                      else "COMPOSITION HOLDS WITHIN THE TRAINED POSITIONAL ENVELOPE; H3 FAILS"
                      if base else "COMPOSITION DOES NOT HOLD")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    for stratum, d in decisions.items():
        print(f"{stratum}: composition_holds={d['composition_holds']}")
    print(f"OUTCOME {out['outcome']}")


if __name__ == "__main__":
    main()
