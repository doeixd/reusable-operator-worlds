"""E0.1 + E1.0: do these objects look like program primitives? (E0_PHASE0_AUDIT_PLAN.md, Am. 1-2)

Two audits on EXISTING artifacts, no new lifetimes.

E0.1 CONTEXTUAL FUNCTIONAL SUBSTITUTABILITY. A learned object is compared with a
teacher primitive by what it DOES when placed in the same program context, not
by its parameters (the slot-index-versus-primitive-index lesson):

    d(A_i, P_j) = E || c[A_i](x) - c[P_j](x) ||^2 / E || c[P_j](x) - c[id](x) ||^2

normalised by the teacher operation's CONTRIBUTION in that context, not by total
output scale (Amendment 2: the frozen denominator repeated the V4.1 error). A
distance of 1.0 means "as wrong as omitting the operation entirely". Hungarian
assignment; random-permutation and shuffled-library controls; a null-edit guard
and a degenerate-context guard.

E1.0 ORACLE-ROUTE GATE. Freeze the library, supply the TEACHER program through
the functionally matched slots, and evaluate on programs the lifetime actually
trained on. A substrate is eligible for E1 iff its oracle-route NMSE is within
2.0x its intact NMSE. This is the gate that decides whether a later export STOP
would be about export or merely about mixture routing.

Fails closed; atomic report.
"""
from __future__ import annotations

import argparse
import json
import pickle
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment

from row.config import load_config
from row.experiments.learned_lifetime import _build_model
from row.world import World, WorldConfig

CONTEXTS_PER_POSITION = 8
PROBE = 256
GATE_RATIO = 2.0
DEGENERATE_FRACTION = 0.01

SUBSTRATES = {
    "DISC": {"path": "artifacts/discrete/seed_0", "kind": "discrete", "route": "library"},
    # E1: three fresh discrete lifetimes from current committed code. The
    # `DISC` row above predates learnable operator scales and is kept as an
    # era-labelled reference, never pooled with these.
    "DISC_w0": {"path": "artifacts/e1_disc/world_0", "kind": "discrete", "route": "library"},
    "DISC_w1": {"path": "artifacts/e1_disc/world_1", "kind": "discrete", "route": "library"},
    "DISC_w2": {"path": "artifacts/e1_disc/world_2", "kind": "discrete", "route": "library"},
    "MIX": {"path": "artifacts/rho_development/rho_1/world_1/continuous", "kind": "continuous",
            "route": "basis"},
    "MIX_w2": {"path": "artifacts/rho_development/rho_1/world_2/continuous", "kind": "continuous",
               "route": "basis"},
}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def world_of(path: Path) -> World:
    raw = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    return World.generate(WorldConfig(**raw["world"]))


def teacher_fn(primitive):
    """The teacher operation as a torch callable on float32 state."""
    U = torch.tensor(primitive.U, dtype=torch.float32)
    V = torch.tensor(primitive.V, dtype=torch.float32)
    b = torch.tensor(primitive.b, dtype=torch.float32)
    alpha = float(primitive.alpha)

    def apply(z: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(z @ V.T + b)
        return torch.tanh(z + alpha * (hidden @ U.T))

    return apply


def load_model(config, path: Path, kind: str):
    """Build and load, detecting the artifact's alpha convention from the file.

    Artifacts predating learnable operator scales have no `*.alpha` entries. The
    convention is read from the checkpoint rather than assumed, so a legacy
    artifact is reconstructed as the model that actually trained rather than
    silently given twelve fresh trainable scalars.
    """

    try:
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Legacy ROW checkpoints stored NumPy-valued summaries beside the tensor
        # state, which the restricted loader refuses. The artifact's own
        # `config.yaml` has already been read and used to build this model, and
        # current checkpoints are tensor-only (AGENTS.md).
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        legacy_pickle = True
    else:
        legacy_pickle = False
    state = state["model_state_dict"] if "model_state_dict" in state else state
    learnable = any(key.endswith(".alpha") for key in state)
    section = "discrete_model" if kind == "discrete" else "continuous_model"
    current = getattr(config, section)
    if bool(current.learnable_alpha) != learnable:
        config = replace(config, **{section: replace(current, learnable_alpha=learnable)})
    model = _build_model(config, kind)
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(f"state mismatch at {path}: {result}")
    model.eval()
    return model, learnable, legacy_pickle


def contexts_for(world: World, position: int, depth: int, seed_world: int) -> list[tuple]:
    rng = np.random.default_rng(np.random.SeedSequence([760, seed_world, position]))
    k = len(world.library)
    out = []
    for _ in range(CONTEXTS_PER_POSITION):
        pre = tuple(int(v) for v in rng.integers(0, k, position))
        post = tuple(int(v) for v in rng.integers(0, k, depth - 1 - position))
        out.append((pre, post))
    return out


def run_context(x, pre, post, middle, teachers):
    z = x
    for index in pre:
        z = teachers[index](z)
    z = middle(z)
    for index in post:
        z = teachers[index](z)
    return z


def distance_matrix(model, world, depth, probe, route_attr, seed_world) -> dict:
    teachers = [teacher_fn(p) for p in world.library]
    objects = list(getattr(model, route_attr))
    n_obj, n_teach = len(objects), len(teachers)
    numerator = np.zeros((n_obj, n_teach))
    denominator = np.zeros(n_teach)
    null_edit = np.zeros(n_teach)
    per_position = np.zeros((depth, n_obj, n_teach))
    dropped = 0
    with torch.no_grad():
        for position in range(depth):
            for pre, post in contexts_for(world, position, depth, seed_world):
                target = {j: run_context(probe, pre, post, teachers[j], teachers) for j in range(n_teach)}
                skipped = run_context(probe, pre, post, lambda z: z, teachers)
                contribution = {j: float(torch.mean((target[j] - skipped) ** 2)) for j in range(n_teach)}
                mean_contribution = float(np.mean(list(contribution.values())))
                for j in range(n_teach):
                    if contribution[j] < DEGENERATE_FRACTION * mean_contribution:
                        dropped += 1
                        continue
                    denominator[j] += contribution[j]
                    null_edit[j] += float(torch.mean((skipped - target[j]) ** 2))
                    for i, obj in enumerate(objects):
                        value = float(torch.mean((run_context(probe, pre, post, obj, teachers)
                                                  - target[j]) ** 2))
                        numerator[i, j] += value
                        per_position[position, i, j] += value
    matrix = numerator / np.maximum(denominator, 1e-12)
    return {"matrix": matrix, "null_edit": null_edit / np.maximum(denominator, 1e-12),
            "per_position": per_position / np.maximum(denominator, 1e-12),
            "dropped_contexts": dropped}


def assign(matrix: np.ndarray) -> tuple[dict, float, float]:
    rows, cols = linear_sum_assignment(matrix.T)          # one learned object per teacher
    best = {int(t): int(o) for t, o in zip(rows, cols)}
    cost = float(matrix[cols, rows].mean())
    margin = float("inf")
    for teacher, obj in best.items():
        blocked = matrix.copy()
        blocked[obj, teacher] = 1e9
        r2, c2 = linear_sum_assignment(blocked.T)
        alt = float(blocked[c2, r2].mean())
        if alt < 1e8:
            margin = min(margin, alt - cost)
    return best, cost, margin


def nmse(model, task, task_id) -> float:
    x = torch.tensor(task.eval_x, dtype=torch.float32)
    y = torch.tensor(task.eval_y, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x, task_id)
    return float(torch.mean((pred - y) ** 2) / (torch.var(y, unbiased=False) + 1e-12))


def oracle_route_nmse(model, world, assignment, slots: int, permute=None) -> dict:
    """Force each task's route to the teacher program through matched slots."""
    intact, oracle = [], []
    saved = {tid: model.task_codes[tid].detach().clone() for tid in model.task_codes}
    try:
        for task in world.tasks:
            tid = task.task_id
            if tid not in model.task_codes:
                continue
            intact.append(nmse(model, task, tid))
            logits = torch.full_like(model.task_codes[tid], -50.0)
            for step, primitive in enumerate(task.program.primitive_ids):
                slot = assignment[int(primitive)]
                if permute is not None:
                    slot = permute[slot]
                logits[step, slot] = 50.0
            with torch.no_grad():
                model.task_codes[tid].copy_(logits)
            oracle.append(nmse(model, task, tid))
    finally:
        with torch.no_grad():
            for tid, value in saved.items():
                model.task_codes[tid].copy_(value)
    return {"intact_nmse": float(np.mean(intact)), "oracle_route_nmse": float(np.mean(oracle)),
            "ratio": float(np.mean(oracle) / max(np.mean(intact), 1e-12)), "tasks": len(intact)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e0_export_audit.json"))
    parser.add_argument("--substrates", nargs="+", default=list(SUBSTRATES))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    out = {"frozen_plan": "E0_PHASE0_AUDIT_PLAN.md (Amendments 1-2)", "git_commit": git_commit(),
           "protocol": {"contexts_per_position": CONTEXTS_PER_POSITION, "probe": PROBE,
                        "gate_ratio": GATE_RATIO,
                        "distance": "contribution-normalised; 1.0 == as wrong as omitting the operation"},
           "substrates": {}}
    for name in args.substrates:
        spec = SUBSTRATES[name]
        path = Path(spec["path"])
        if not (path / "model.pt").exists():
            raise SystemExit(f"missing artifact {path}")
        raw = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
        world = world_of(path)
        config = load_config(args.config)
        config = replace(config, world=replace(config.world, **raw["world"]))
        key = "discrete_model" if spec["kind"] == "discrete" else "continuous_model"
        if key in raw:
            section = getattr(config, key)
            fields = set(section.__dataclass_fields__)
            config = replace(config, **{key: replace(
                section, **{k: v for k, v in raw[key].items() if k in fields})})
        model, learnable_alpha, legacy_pickle = load_model(config, path, spec["kind"])
        depth = world.config.program_length
        probe = torch.tensor(
            np.random.default_rng(np.random.SeedSequence([760, world.config.seed, 99])).normal(
                size=(PROBE, world.config.state_dim)), dtype=torch.float32)
        d = distance_matrix(model, world, depth, probe, spec["route"], world.config.seed)
        best, cost, margin = assign(d["matrix"])
        rng = np.random.default_rng(np.random.SeedSequence([760, world.config.seed, 5]))
        slots = d["matrix"].shape[0]
        random_assignment = {j: int(v) for j, v in enumerate(rng.permutation(slots)[:len(world.library)])}
        random_cost = float(np.mean([d["matrix"][o, t] for t, o in random_assignment.items()]))
        shuffle = list(rng.permutation(slots))
        gate = oracle_route_nmse(model, world, best, slots)
        gate_random = oracle_route_nmse(model, world, random_assignment, slots)
        gate_shuffled = oracle_route_nmse(model, world, best, slots, permute=shuffle)
        null_edit = float(np.mean(d["null_edit"]))
        matched = [d["matrix"][o, t] for t, o in best.items()]
        above_null = [t for t, o in best.items() if d["matrix"][o, t] > d["null_edit"][t]]
        position_means = [float(np.mean([d["per_position"][p][o, t] for t, o in best.items()]))
                          for p in range(depth)]
        spread = (max(position_means) - min(position_means)) / max(np.mean(position_means), 1e-12)
        out["substrates"][name] = {
            "artifact": str(path), "kind": spec["kind"],
            "learnable_alpha": bool(learnable_alpha),
            "legacy_pickle_checkpoint": bool(legacy_pickle),
            "E0_1": {
                "assignment": {str(t): o for t, o in best.items()},
                "mean_distance": cost, "assignment_margin": margin,
                "random_assignment_distance": random_cost,
                "null_edit_distance": null_edit,
                "matched_distances": [float(v) for v in matched],
                "per_position_distance": position_means,
                "position_spread_fraction": float(spread),
                "dropped_degenerate_contexts": d["dropped_contexts"],
                "objects_worse_than_null_edit": above_null,
                "beats_random_by_2x": bool(random_cost >= 2 * cost),
                "position_stable_within_50pct": bool(spread <= 0.5),
                "substitutable": bool(random_cost >= 2 * cost and spread <= 0.5 and not above_null),
            },
            "E1_0": {
                "teacher_route": gate, "random_assignment_route": gate_random,
                "shuffled_library_route": gate_shuffled,
                "gate_ratio": GATE_RATIO,
                "controls_ok": bool(gate["ratio"] < gate_random["ratio"]
                                    and gate["ratio"] < gate_shuffled["ratio"]),
                "eligible_for_E1": bool(gate["ratio"] <= GATE_RATIO
                                        and gate["ratio"] < gate_random["ratio"]
                                        and gate["ratio"] < gate_shuffled["ratio"]),
            },
        }
        e = out["substrates"][name]
        print(f"[{name}] E0.1 matched {cost:.3f} (random {random_cost:.3f}, null-edit {null_edit:.3f}, "
              f"margin {margin:.3f}, position spread {spread:.2f}) -> substitutable="
              f"{e['E0_1']['substitutable']}", flush=True)
        print(f"[{name}] E1.0 intact {gate['intact_nmse']:.5f} oracle-route {gate['oracle_route_nmse']:.5f} "
              f"ratio {gate['ratio']:.2f} (random {gate_random['ratio']:.2f}, "
              f"shuffled {gate_shuffled['ratio']:.2f}) -> eligible={e['E1_0']['eligible_for_E1']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    eligible = [n for n, v in out["substrates"].items() if v["E1_0"]["eligible_for_E1"]]
    print(f"eligible for E1: {eligible or 'NONE'}")


if __name__ == "__main__":
    main()
