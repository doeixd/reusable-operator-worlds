"""E9: partial coverage -- does `p + alpha + eps` close the gap?

`E9_PARTIAL_COVERAGE_PLAN.md`, frozen at Amendment 1.

E8D and its coverage reframe both died on the same measured fact: route-only
inference is within +/-0.07 log units of the best available arm in 8 of 9 cells
across E1 and E1-R. `p` is sufficient wherever anything is sufficient, because
this generator makes tasks that are in-language or out-of-language with nothing
between. E9 builds the missing middle by PERTURBING one teacher primitive:

    delta_U   perturb `U` within span{U_k}   -- alpha can express it exactly
    delta_V   perturb `V`                     -- alpha cannot express it at all

At `delta = 0` the construction reproduces the existing tasks BITWISE; that
reduction is asserted per world before anything is scored.

Arms, all fitted on support only over the FROZEN learner library:

    P         route only
    P+A       route + alpha (K = 16 on slot 11)
    A-RAND    route + alpha, U_k frozen at random init   (matched budget)
    P+A+E     route + alpha + a rank-2 output patch
    CEILING   route + slot 11 FULLY free (U, V, b)       (Amendment 1)

`GENERATOR` -- the true perturbed teacher -- is reported as a zero-error
reference and is NEVER a denominator: targets are noiseless, so using it would
make every share identically zero.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from row.arm_provenance import describe_arm
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_e1_export import ADAPT_LR, ADAPT_STEPS, nmse
from row.experiments.audit_e5_synthesizer import fatal, load_cell
from row.experiments.audit_e8_length import VariableDepthDiscrete
from row.support_split_world import _build_tasks

WORLDS = (0, 1, 2)
DEPTH = 3                       # the world's native program length
PROGRAMS = 12
K_ARGS = 16                     # H39's confirmed argument dimension
# Amendment 4: the parameterized slot is chosen PER WORLD as the learner
# slot functionally matched to the perturbed teacher primitive. A fixed
# index put the argument channel on an unrelated operator in 2 of 3 worlds.
PATCH_RANK = 2
DELTAS = (0.0, 0.5, 1.0, 2.0)   # Amendment 2: sized so the gate can fire
DIRECTIONS = ("U", "V")
CACHE = Path("reports/e9_cache")


class PSlot(nn.Module):
    """`P(alpha)(z) = tanh(z + a . (U_0 + sum_k alpha_k U_k) tanh(V z + b))`.

    At `alpha = 0` this is exactly the base slot, so the whole learner reproduces
    the frozen artifact bitwise. `dL/dalpha_k = <dL/dU, U_k>` is nonzero at
    `alpha = 0`, so the stationary point that bit two earlier residual schemas
    does not exist here.
    """

    def __init__(self, base, matrices: torch.Tensor, free_operator: bool = False) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("matrices", matrices)
        self.alpha = nn.Parameter(torch.zeros(matrices.shape[0]))
        self.free_operator = free_operator
        if free_operator:                      # CEILING: the whole operator is free
            self.U = nn.Parameter(base.U.detach().clone())
            self.V = nn.Parameter(base.V.detach().clone())
            self.b = nn.Parameter(base.b.detach().clone())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.free_operator:
            hidden = torch.tanh(torch.nn.functional.linear(z, self.V, self.b))
            return torch.tanh(z + self.base.alpha * torch.nn.functional.linear(hidden, self.U))
        hidden = torch.tanh(torch.nn.functional.linear(z, self.base.V, self.base.b))
        U = self.base.U + torch.einsum("k,kdr->dr", self.alpha, self.matrices)
        return torch.tanh(z + self.base.alpha * torch.nn.functional.linear(hidden, U))


class Patch(nn.Module):
    """A rank-2 output residual, initialized AWAY from zero.

    A zero-initialized rank-2 residual never moves (`d/du = tanh(0)`,
    `d/dv ~ u`), which froze training once and an adaptation fit once.
    """

    def __init__(self, d: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.V = nn.Parameter(torch.randn(PATCH_RANK, d, generator=g) * 0.05)
        self.b = nn.Parameter(torch.zeros(PATCH_RANK))
        self.U = nn.Parameter(torch.randn(d, PATCH_RANK, generator=g) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            torch.tanh(torch.nn.functional.linear(x, self.V, self.b)), self.U)


def fingerprint() -> str:
    payload = {"depth": DEPTH, "programs": PROGRAMS, "K": K_ARGS,
               "patch_rank": PATCH_RANK, "deltas": list(DELTAS),
               "steps": ADAPT_STEPS, "lr": ADAPT_LR}
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


def perturb(library, index: int, direction: str, delta: float,
            matrices: np.ndarray, rng) -> tuple:
    """A copy of the teacher library with ONE primitive perturbed.

    At `delta = 0` the copy is the original, so the whole construction reduces
    exactly to the existing generator.
    """
    if delta == 0.0:
        return tuple(library)
    out = list(library)
    p = out[index]
    if direction == "U":
        coeffs = rng.normal(size=matrices.shape[0])
        step = np.einsum("k,kdr->dr", coeffs, matrices)
        step = step / max(np.linalg.norm(step, ord=2), 1e-12)
        out[index] = dc_replace(p, U=p.U + delta * step)
    elif direction == "V":
        step = rng.normal(size=p.V.shape)
        step = step / max(np.linalg.norm(step, ord=2), 1e-12)
        out[index] = dc_replace(p, V=p.V + delta * step)
    else:
        raise ValueError(direction)
    return tuple(out)


def fit(model, task, probe_id: str, arm: str, matrices: torch.Tensor,
        rand_matrices: torch.Tensor, d: int, seed: int, pslot_index: int) -> dict:
    """Fit the arm's channels on SUPPORT only and score the query."""
    local = copy.deepcopy(model)
    local.__class__ = VariableDepthDiscrete
    local.begin_task_depth(probe_id, DEPTH)
    for p in local.parameters():
        p.requires_grad_(False)
    params = [local.task_codes[probe_id]]
    local.task_codes[probe_id].requires_grad_(True)

    pslot = patch = None
    if arm in ("P+A", "A-RAND", "P+A+E", "CEILING"):
        basis = rand_matrices if arm == "A-RAND" else matrices
        pslot = PSlot(local.library[pslot_index], basis,
                      free_operator=(arm == "CEILING"))
        local.library = nn.ModuleList(
            [pslot if i == pslot_index else op for i, op in enumerate(local.library)])
        if arm == "CEILING":
            params += [pslot.U, pslot.V, pslot.b]
        else:
            params.append(pslot.alpha)
    if arm == "P+A+E":
        patch = Patch(d, seed)
        params += list(patch.parameters())

    for p in params:
        p.requires_grad_(True)
    t = {k: torch.tensor(getattr(task, k), dtype=torch.float32)
         for k in ("train_x", "train_y", "eval_x", "eval_y")}
    optimizer = torch.optim.Adam(params, lr=ADAPT_LR)
    local.train()

    def predict(x):
        out = local(x, probe_id)
        return out + patch(x) if patch is not None else out

    with torch.no_grad():
        initial = float(torch.mean((predict(t["train_x"]) - t["train_y"]) ** 2))
    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        loss = torch.mean((predict(t["train_x"]) - t["train_y"]) ** 2)
        if not bool(torch.isfinite(loss)):
            raise SystemExit(f"non-finite loss for {probe_id} arm {arm}")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = float(torch.mean((predict(t["train_x"]) - t["train_y"]) ** 2))
    local.eval()
    with torch.no_grad():
        query = nmse(predict(t["eval_x"]), t["eval_y"])
    return {"query_nmse": query,
            "support_reduction": (initial - final) / max(initial, 1e-12),
            "alpha_norm": (float(torch.norm(pslot.alpha)) if pslot is not None
                           and not pslot.free_operator else None),
            "patch_norm": (float(torch.norm(patch.U)) if patch is not None else None)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e9_partial_coverage.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")

    out = {"frozen_plan": "E9_PARTIAL_COVERAGE_PLAN.md (Amendment 1)",
           "git_commit": git_commit(),
           "protocol": {"depth": DEPTH, "programs": PROGRAMS, "K": K_ARGS,
                        "pslot": "per-world matched slot", "deltas": list(DELTAS),
                        "directions": list(DIRECTIONS), "steps": ADAPT_STEPS,
                        "ceiling": "route + slot 11 fully free (Amendment 1)",
                        "note": "GENERATOR is a zero-error reference, never a denominator"},
           "worlds": {}}

    for world in WORLDS:
        cell = load_cell(world, args.config)
        config = cell["config"]
        d = config.world.state_dim
        rank = config.discrete_model.operator_rank
        base_library = cell["world"].library

        # One argument basis per world, used BOTH for the alpha channel and for
        # the delta_U perturbation -- that is what makes delta_U lie in alpha's
        # span rather than merely resemble it.
        arng = np.random.default_rng(np.random.SeedSequence([1200, world]))
        mats = arng.normal(size=(K_ARGS, d, rank))
        mats = np.stack([m / max(np.linalg.norm(m, ord=2), 1e-12) for m in mats])
        matrices = torch.tensor(mats, dtype=torch.float32)
        rrng = np.random.default_rng(np.random.SeedSequence([1201, world]))
        rmats = rrng.normal(size=(K_ARGS, d, rank))
        rmats = np.stack([m / max(np.linalg.norm(m, ord=2), 1e-12) for m in rmats])
        rand_matrices = torch.tensor(rmats, dtype=torch.float32)

        # Amendment 2: the perturbed primitive is chosen FIRST and every held-out
        # program must contain it. Drawing programs freely and then perturbing
        # the most-used primitive left most programs untouched and halved the
        # realized manipulation.
        prng = np.random.default_rng(np.random.SeedSequence([1202, world]))
        trained = {tuple(t.program.primitive_ids) for t in cell["world"].tasks}
        target_primitive = int(prng.integers(0, config.world.teacher_primitives))
        programs = []
        guard = 0
        while len(programs) < PROGRAMS and guard < 20000:
            guard += 1
            cand = tuple(int(v) for v in prng.integers(0, config.world.teacher_primitives, DEPTH))
            if target_primitive not in cand or cand in trained or cand in programs:
                continue
            programs.append(cand)
        fatal(len(programs) == PROGRAMS,
              f"world {world}: only {len(programs)} programs contain primitive "
              f"{target_primitive}")
        # Amendment 4: attach the argument channel to the learner slot that the
        # perturbed TEACHER primitive is functionally matched to. Teacher
        # primitive indices and learner slot indices are different spaces.
        pslot_index = int(cell["assignment"][target_primitive])

        entry = {"programs": [list(p) for p in programs],
                 "perturbed_primitive": target_primitive,
                 "parameterized_slot": pslot_index,
                 "slot_note": "matched to the perturbed primitive via the E0.1 assignment",
                 "cells": {}}

        for direction in DIRECTIONS:
            for delta in sorted(DELTAS):      # delta = 0 first: it is the reference
                # `ParameterDict` keys cannot contain ".", and the probe id is
                # derived from this key, so the delta is written as `0p5`.
                key = f"{direction}_{str(delta).replace('.', 'p')}"
                lrng = np.random.default_rng(np.random.SeedSequence([1203, world,
                                                                     int(delta * 1000)]))
                lib = perturb(base_library, target_primitive, direction, delta, mats, lrng)
                if delta == 0.0:
                    fatal(all(a is b for a, b in zip(lib, base_library)),
                          "delta = 0 did not reduce to the base library")

                arms = {a: [] for a in ("P", "P+A", "A-RAND", "P+A+E", "CEILING")}
                gen, shift, checks = [], [], []
                for index, program in enumerate(programs):
                    task = _build_tasks(config.world, lib, [program],
                                        [f"task_e9_{key}_{index}"],
                                        index_offset=90000 + index)[0]
                    base_task = _build_tasks(config.world, base_library, [program],
                                             [f"task_e9_base_{index}"],
                                             index_offset=90000 + index)[0]
                    shift.append(float(np.mean((task.eval_y - base_task.eval_y) ** 2) /
                                       max(np.mean(base_task.eval_y ** 2), 1e-12)))
                    gen.append(0.0)   # the true perturbed teacher reproduces targets exactly
                    tag = f"w{world}_{key}_{index}"

                    def cellwise(task=task, tag=tag):
                        res = {}
                        for arm in arms:
                            got = fit(cell["model"], task, f"e9{arm}_{tag}", arm,
                                      matrices, rand_matrices, d, seed=7700 + index,
                                      pslot_index=pslot_index)
                            res[arm] = got
                        return res

                    res = cached(tag, cellwise)
                    for arm in arms:
                        fatal(res[arm]["support_reduction"] > 0.0,
                              f"arm {arm} did not optimize at {tag}")
                        arms[arm].append(res[arm]["query_nmse"])
                    checks.append(res)

                geo = {a: float(np.exp(np.mean(np.log(np.maximum(v, 1e-12)))))
                       for a, v in arms.items()}
                # Amendment 3: the denominator is a MEASURED reference -- the same
                # arm at delta = 0 -- not a fitted ceiling. Three fitted ceilings
                # failed here; in a noiseless small-support regime capacity buys
                # overfitting before it buys reach.
                baseline = entry["cells"].get(f"{direction}_0p0", {}).get(
                    "geomean_nmse", {}).get("P")
                degradation = (math.log(geo["P"]) - math.log(baseline)
                               if baseline else None)
                recovery = {a: ((math.log(geo["P"]) - math.log(geo[a])) / degradation
                                if degradation and abs(degradation) > 1e-9 else None)
                            for a in ("P+A", "A-RAND", "P+A+E", "CEILING")}
                entry["cells"][key] = {
                    "direction": direction, "delta": delta,
                    "geomean_nmse": geo,
                    "degradation": degradation,
                    "gap_P_to_CEILING_diagnostic": math.log(geo["P"]) - math.log(geo["CEILING"]),
                    "realized_perturbation": float(np.mean(shift)),
                    "recovery": recovery,
                    "alpha_norm": float(np.mean([c["P+A"]["alpha_norm"] for c in checks])),
                    "patch_norm": float(np.mean([c["P+A+E"]["patch_norm"] for c in checks])),
                    "generator_nmse": 0.0}
                r = entry["cells"][key]
                rec = lambda a: ("n/a" if r["recovery"][a] is None
                                 else f"{r['recovery'][a]:+.2f}")
                deg = "n/a" if r["degradation"] is None else f"{r['degradation']:+.2f}"
                print(f"[w{world} d{direction}={delta}] shift {r['realized_perturbation']:.4f} "
                      f"| P {geo['P']:.5f} | degradation {deg} | recovery "
                      f"P+A {rec('P+A')} A-RAND {rec('A-RAND')} P+A+E {rec('P+A+E')} "
                      f"CEIL {rec('CEILING')} | |a| {r['alpha_norm']:.3f} "
                      f"|e| {r['patch_norm']:.3f}", flush=True)
                write(out, args.output)
        out["worlds"][str(world)] = entry
        write(out, args.output)

    write(out, args.output)
    print("\nE9 complete. Gate: the gap must OPEN with delta, or the rung is unscoreable.")


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
