"""E8: is the operator interface closed under variable-length composition?

`E8_LENGTH_PLAN.md`, under `EXPORT_BRANCH_PROGRAM.md`. E2 established same-depth
systematicity; E8 changes the LENGTH, which separates a learned length-3 algebra
from operators that iterate as a DSL.

    E8a  depth 2, familiar positions
    E8b  depth 4, whose fourth operation occupies a position that never existed

The executor is the only thing that changes: `DiscreteLibraryLearner.forward`
loops `for step in range(self.task_steps)`, so depth is a model constant. Here a
task's route carries its own length and the executor loops over that. The library
is frozen and no trained artifact is modified. Registered control: at depth 3 the
variable-depth executor must reproduce the shipped one BITWISE, on the real
artifacts, before any cell is scored.

Per-step errors `e_1..e_D` are recorded against the teacher's intermediate states
so that a depth-4 failure can be read as fourth-step breakage versus compounding.
Fails closed; per-cell cache; atomic report.
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
from row.experiments.audit_e0_export import assign, distance_matrix, git_commit, load_model
from row.experiments.audit_e1_export import ADAPT_LR, ADAPT_STEPS, geo, nmse, scratch_model
from row.models.learned_models import DiscreteLibraryLearner
from row.support_split_world import _build_tasks
from row.world import Program, World, WorldConfig

WORLDS = (0, 1, 2)
CONDITIONS = {"E8a_depth2": 2, "E8b_depth4": 4}
PER_CONDITION = 24
MARGIN = 0.15
CACHE = Path("reports/e8_cache")


class VariableDepthDiscrete(DiscreteLibraryLearner):
    """The shipped executor with depth taken from the TASK's route, not the model.

    `forward` is otherwise identical, so a depth-`task_steps` route reproduces the
    shipped executor exactly; that equivalence is asserted on the real artifacts
    before E8 scores anything.
    """

    def begin_task_depth(self, task_id: str, depth: int) -> torch.nn.Parameter:
        if task_id in self.task_codes:
            del self.task_codes[task_id]
        self.task_codes[task_id] = torch.nn.Parameter(torch.zeros(depth, self.operator_slots))
        return self.task_codes[task_id]

    def _coefficients(self, task_id: str) -> torch.Tensor:
        logits = self.task_codes[task_id]
        if self.training:
            return torch.softmax(logits / self.temperature, dim=-1)
        indices = torch.argmax(logits, dim=-1)
        return torch.nn.functional.one_hot(indices, self.operator_slots).to(logits.dtype)

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        z = x
        coefficients = self._coefficients(task_id)
        for step in range(coefficients.shape[0]):        # <- the only change
            candidates = torch.stack([operator(z) for operator in self.library], dim=0)
            weights = coefficients[step].view(self.operator_slots, 1, 1)
            z = torch.sum(weights * candidates, dim=0)
        return z

    @torch.no_grad()
    def trace(self, x: torch.Tensor, task_id: str) -> list[torch.Tensor]:
        """The state after every step, for the per-step diagnostic."""
        z, out = x, []
        coefficients = self._coefficients(task_id)
        for step in range(coefficients.shape[0]):
            candidates = torch.stack([operator(z) for operator in self.library], dim=0)
            z = torch.sum(coefficients[step].view(self.operator_slots, 1, 1) * candidates, dim=0)
            out.append(z)
        return out


def protocol_fingerprint() -> str:
    payload = {"conditions": CONDITIONS, "per_condition": PER_CONDITION,
               "steps": ADAPT_STEPS, "lr": ADAPT_LR, "margin": MARGIN,
               "arms": ["O", "O-W", "R", "R-W", "S"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cached(key: str, compute):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.json"
    fingerprint = protocol_fingerprint()
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("protocol") != fingerprint:
            raise SystemExit(f"cached cell {key} under protocol {stored.get('protocol')}, "
                             f"not {fingerprint}; delete reports/e8_cache to rescore")
        return stored["value"]
    value = compute()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"protocol": fingerprint, "value": value}), encoding="utf-8")
    os.replace(tmp, path)
    return value


def teacher_trace(program, library, x: np.ndarray) -> list[np.ndarray]:
    z, out = np.asarray(x, dtype=np.float64), []
    for index in program:
        z = library[index](z)
        out.append(z.copy())
    return out


def as_tensors(task) -> dict:
    return {"support_x": torch.tensor(task.train_x, dtype=torch.float32),
            "support_y": torch.tensor(task.train_y, dtype=torch.float32),
            "query_x": torch.tensor(task.eval_x, dtype=torch.float32),
            "query_y": torch.tensor(task.eval_y, dtype=torch.float32)}


def oracle_cell(model, task, program, assignment, probe_id: str, library) -> dict:
    local = copy.deepcopy(model)
    if not isinstance(local, VariableDepthDiscrete):
        local.__class__ = VariableDepthDiscrete
    local.begin_task_depth(probe_id, len(program))
    with torch.no_grad():
        logits = torch.full_like(local.task_codes[probe_id], -50.0)
        for step, primitive in enumerate(program):
            logits[step, assignment[int(primitive)]] = 50.0
        local.task_codes[probe_id].copy_(logits)
    local.eval()
    t = as_tensors(task)
    with torch.no_grad():
        query = nmse(local(t["query_x"], probe_id), t["query_y"])
        states = local.trace(t["query_x"], probe_id)
    reference = teacher_trace(program, library, task.eval_x)
    per_step = []
    for step, state in enumerate(states):
        target = torch.tensor(reference[step], dtype=torch.float32)
        per_step.append(nmse(state, target))
    return {"query_nmse": query, "per_step_nmse": per_step}


def adapt_cell(model, task, probe_id: str, depth: int, train_library: bool, library,
               program, steps: int | None = None) -> dict:
    """`steps` defaults to ADAPT_STEPS, so every pre-E5.1 caller is unchanged."""
    local = copy.deepcopy(model)
    if not isinstance(local, VariableDepthDiscrete):
        local.__class__ = VariableDepthDiscrete    # the scratch arm gets the same executor
    local.begin_task_depth(probe_id, depth)
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
    t = as_tensors(task)
    optimizer = torch.optim.Adam(params, lr=ADAPT_LR)
    local.train()
    with torch.no_grad():
        initial = nmse(local(t["support_x"], probe_id), t["support_y"])
    for _ in range(ADAPT_STEPS if steps is None else int(steps)):
        optimizer.zero_grad()
        loss = torch.mean((local(t["support_x"], probe_id) - t["support_y"]) ** 2)
        if not bool(torch.isfinite(loss)):
            raise SystemExit(f"non-finite adaptation loss for {probe_id}")
        loss.backward(inputs=params)
        optimizer.step()
    with torch.no_grad():
        final_objective = nmse(local(t["support_x"], probe_id), t["support_y"])
    local.eval()
    with torch.no_grad():
        query = nmse(local(t["query_x"], probe_id), t["query_y"])
        states = local.trace(t["query_x"], probe_id)
    reference = teacher_trace(program, library, task.eval_x)
    per_step = [nmse(state, torch.tensor(reference[i], dtype=torch.float32))
                for i, state in enumerate(states)]
    with torch.no_grad():
        route = [int(v) for v in torch.argmax(local.task_codes[probe_id], dim=-1)]
    return {"query_nmse": query, "per_step_nmse": per_step, "route": route,
            "support_reduction_objective": (initial - final_objective) / max(initial, 1e-12)}


def load_cell(world: int, config_path: Path):
    path = Path("artifacts/e1_disc") / f"world_{world}"
    raw = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    config = load_config(config_path)
    config = replace(config, world=replace(config.world, **raw["world"]))
    fields = set(config.discrete_model.__dataclass_fields__)
    config = replace(config, discrete_model=replace(
        config.discrete_model, **{k: v for k, v in raw["discrete_model"].items() if k in fields}))
    shipped, _, _ = load_model(config, path, "discrete")
    world_obj = World.generate(WorldConfig(**raw["world"]))
    variable = copy.deepcopy(shipped)
    variable.__class__ = VariableDepthDiscrete
    probe = torch.tensor(np.random.default_rng(np.random.SeedSequence(
        [766, world, 4])).normal(size=(256, config.world.state_dim)), dtype=torch.float32)
    d = distance_matrix(shipped, world_obj, config.world.program_length, probe, "library",
                        config.world.seed)
    best, cost, margin = assign(d["matrix"])
    return {"shipped": shipped, "model": variable, "config": config, "world": world_obj,
            "assignment": best, "assignment_margin": margin, "path": path}


def equivalence_control(cell) -> dict:
    """Registered gate: at depth 3 the variable-depth executor IS the shipped one."""
    shipped, variable = cell["shipped"], cell["model"]
    x = torch.tensor(np.random.default_rng(np.random.SeedSequence(
        [766, cell["config"].world.seed, 77])).normal(
        size=(64, cell["config"].world.state_dim)), dtype=torch.float32)
    checked, equal = 0, True
    with torch.no_grad():
        for task_id in list(shipped.task_codes)[:16]:
            a = shipped(x, task_id)
            b = variable(x, task_id)
            equal = equal and bool(torch.equal(a, b))
            checked += 1
    return {"tasks_checked": checked, "bitwise_equal": bool(equal)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e8_length.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    cells = {w: load_cell(w, args.config) for w in WORLDS}
    controls = {str(w): equivalence_control(cells[w]) for w in WORLDS}
    if not all(c["bitwise_equal"] for c in controls.values()):
        raise SystemExit(f"variable-depth executor is not bitwise equal at depth 3: {controls}")
    print(f"equivalence control: bitwise equal at depth 3 in all worlds "
          f"({sum(c['tasks_checked'] for c in controls.values())} tasks)", flush=True)

    out = {"frozen_plan": "E8_LENGTH_PLAN.md", "git_commit": git_commit(),
           "protocol": {"conditions": CONDITIONS, "per_condition": PER_CONDITION,
                        "steps": ADAPT_STEPS, "lr": ADAPT_LR, "margin": MARGIN,
                        "oracle_first": True},
           "equivalence_control": controls, "worlds": {}}
    for w in WORLDS:
        cell = cells[w]
        donor = cells[(w + 1) % 3]
        trained = {tuple(t.program.primitive_ids) for t in cell["world"].tasks}
        rows = {"donor_world": (w + 1) % 3, "assignment_margin": cell["assignment_margin"],
                "conditions": {}}
        for name, depth in CONDITIONS.items():
            raw = cell["config"].world
            longer = World.generate(replace(raw, program_length=depth,
                                            tasks=min(raw.teacher_primitives ** depth, 64)))
            rng = np.random.default_rng(np.random.SeedSequence([766, w, depth]))
            pool = [tuple(t.program.primitive_ids) for t in longer.tasks]
            order = rng.permutation(len(pool))
            programs = [pool[int(i)] for i in order[:PER_CONDITION]]
            if any(p in trained for p in programs):
                raise SystemExit("a test program appears in the training set")
            if any(len(p) != depth for p in programs):
                raise SystemExit("test program of the wrong depth")
            arms = {a: [] for a in ("O", "O-W", "R", "R-W", "S")}
            per_step = {a: [] for a in ("O", "R")}
            weak = 0
            for index, program in enumerate(programs):
                task = _build_tasks(raw, cell["world"].library, [program],
                                    [f"task_e8_{name}_{index}"], index_offset=2000 + index)[0]
                tag = f"w{w}_{name}_{index}"
                res = cached(tag, lambda task=task, program=program, tag=tag, depth=depth: {
                    "O": oracle_cell(cell["model"], task, program, cell["assignment"],
                                     f"e8O_{tag}", cell["world"].library),
                    "O-W": oracle_cell(donor["model"], task, program, donor["assignment"],
                                       f"e8OW_{tag}", cell["world"].library),
                    "R": adapt_cell(cell["model"], task, f"e8R_{tag}", depth, False,
                                    cell["world"].library, program),
                    "R-W": adapt_cell(donor["model"], task, f"e8RW_{tag}", depth, False,
                                      cell["world"].library, program),
                    "S": adapt_cell(scratch_model(cell["config"], "discrete", 7717), task,
                                    f"e8S_{tag}", depth, True, cell["world"].library, program),
                })
                for arm in arms:
                    arms[arm].append(res[arm]["query_nmse"])
                for arm in per_step:
                    per_step[arm].append(res[arm]["per_step_nmse"])
                if any(res[a].get("support_reduction_objective", 1.0) <= 0.01 for a in ("R", "S")):
                    weak += 1
            g = {arm: geo(values) for arm, values in arms.items()}
            steps = {arm: [float(np.exp(np.mean(np.log(np.maximum(
                [row[t] for row in values], 1e-12))))) for t in range(depth)]
                for arm, values in per_step.items()}
            rows["conditions"][name] = {
                "depth": depth, "n": len(programs), "geomean_query_nmse": g,
                "per_step_geomean_nmse": steps, "weak_adaptation_cells": weak,
                "margin_O_vs_S": float(np.log(g["S"]) - np.log(g["O"])),
                "margin_O_vs_OW": float(np.log(g["O-W"]) - np.log(g["O"])),
                "margin_R_vs_S": float(np.log(g["S"]) - np.log(g["R"])),
                "R_minus_O": float(np.log(g["R"]) - np.log(g["O"])),
            }
            r = rows["conditions"][name]
            print(f"[w{w} {name}] O {g['O']:.5f} O-W {g['O-W']:.5f} R {g['R']:.5f} S {g['S']:.5f}"
                  f" | O-S {r['margin_O_vs_S']:+.3f} O-OW {r['margin_O_vs_OW']:+.3f}"
                  f" R-S {r['margin_R_vs_S']:+.3f} | e_O {[round(v, 4) for v in steps['O']]}",
                  flush=True)
        out["worlds"][str(w)] = rows
    decisions = {}
    for name in CONDITIONS:
        per = {str(w): out["worlds"][str(w)]["conditions"][name] for w in WORLDS}
        oracle = (sum(r["margin_O_vs_S"] >= MARGIN for r in per.values()) >= 2
                  and sum(r["margin_O_vs_OW"] >= MARGIN for r in per.values()) >= 2)
        inference = sum(r["margin_R_vs_S"] >= MARGIN for r in per.values()) >= 2
        decisions[name] = {"oracle_closed": bool(oracle), "inference_closed": bool(inference),
                           "length_closed": bool(oracle and inference)}
    out["decisions"] = decisions
    d2, d4 = decisions["E8a_depth2"], decisions["E8b_depth4"]
    out["outcome"] = (
        "LENGTH-CLOSED AT BOTH DEPTHS" if d2["length_closed"] and d4["length_closed"]
        else "DEPTH-4 ORACLE EXTRAPOLATES; THE WRITER DOES NOT"
        if d4["oracle_closed"] and not d4["inference_closed"]
        else "COMPOSITION BOUNDED BY THE EXECUTOR: DEPTH 2 CLOSED, DEPTH 4 ORACLE NOT"
        if d2["length_closed"] and not d4["oracle_closed"]
        else "DEPTH 2 NOT CLOSED: RE-EXAMINE THE EQUIVALENCE CONTROL")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    for name, d in decisions.items():
        print(f"{name}: oracle={d['oracle_closed']} inference={d['inference_closed']}")
    print(f"OUTCOME {out['outcome']}")


if __name__ == "__main__":
    main()
