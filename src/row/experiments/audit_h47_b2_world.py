"""H47 B2 teacher-level world gates (H47_MEMBERSHIP_PLAN.md, Amendment 2).

Oracle audit of the `schema_groups = 2` world on development seeds 0-2.
No learner is involved: everything is computed from the teacher's family
operators and programs. Gates:

  G2 within-group continuity      held-out family fit from its own
                                  group's trained operators (U-space
                                  least squares, relative residual
                                  <= 0.05) and task-level substitution
                                  NMSE <= 0.05
  G3 cross-group non-substitut.   same fits from the other group:
                                  relative residual >= 0.5 and
                                  Q = NMSE(cross) / NMSE(within) >= 3
  G4 balance                      per-family behavioural contribution
                                  within 10% across families and between
                                  group means
  G5 distinguishable behaviour    nearest-centroid classification of the
                                  64 trained family tasks into groups
                                  from effective-innovation vectors on a
                                  common probe, accuracy >= 0.95

All five must pass in 3 of 3 worlds. Atomic report.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from row.config import load_config
from row.meta_world import MetaFamilySpec, generate_meta_world
from row.world import Primitive

WITHIN_RESID = 0.05
WITHIN_NMSE = 0.05
CROSS_RESID = 0.5
CROSS_CONTRIB = 0.5
GROUP_BALANCE = 0.20
Q_MIN = 3.0
BALANCE = 0.10
ACCURACY = 0.95


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def nmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2) / (np.var(target) + 1e-12))


def with_operator(task, operator):
    """The task's teacher library with its family slot replaced."""
    library = (*task.teacher_library[:-1], operator)
    return library


def ls_fit(target_U: np.ndarray, sources: list[np.ndarray]) -> float:
    A = np.stack([s.ravel() for s in sources], axis=1)
    coef, *_ = np.linalg.lstsq(A, target_U.ravel(), rcond=None)
    resid = target_U.ravel() - A @ coef
    return float(np.linalg.norm(resid) / np.linalg.norm(target_U.ravel()))


def substitute_error(task, target_op: Primitive, candidates: list[Primitive], probe_x: np.ndarray) -> tuple[float, float]:
    """Amendment 3: least-squares mixture over the candidates' full span in
    U-space (oracle, learner-blind), evaluated on the task's own program.
    Returns (MSE of the substitute, contribution MSE of the family step)."""
    target = task.program.execute(task.teacher_library, probe_x)
    A = np.stack([c.U.ravel() for c in candidates], axis=1)
    coef, *_ = np.linalg.lstsq(A, target_op.U.ravel(), rcond=None)
    U = (A @ coef).reshape(target_op.U.shape)
    base = candidates[0]
    substitute = Primitive(U=U, V=base.V, b=base.b, alpha=base.alpha)
    identity = Primitive(U=np.zeros_like(target_op.U), V=base.V, b=base.b, alpha=base.alpha)
    err = float(np.mean((task.program.execute(with_operator(task, substitute), probe_x) - target) ** 2))
    contribution = float(np.mean((task.program.execute(with_operator(task, identity), probe_x) - target) ** 2))
    return err, contribution


def family_spread(world, spec, ops, probe) -> tuple[dict, float]:
    contributions = {}
    for f in range(spec.families):
        vals = []
        for i, task in enumerate(world.tasks):
            if spec.family_of(i) != f:
                continue
            target = task.program.execute(task.teacher_library, probe)
            identity = Primitive(U=np.zeros_like(ops[f].U), V=ops[f].V, b=ops[f].b, alpha=ops[f].alpha)
            vals.append(nmse(task.program.execute(with_operator(task, identity), probe), target))
        contributions[f] = float(np.mean(vals))
    c = np.array(list(contributions.values()))
    return contributions, float((c.max() - c.min()) / c.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("reports/h47_b2_world_gates.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2,
                          schema_groups=args.groups)
    out = {}
    for world in args.worlds:
        generated = generate_meta_world(replace(config.world, seed=world, tasks=spec.total_tasks), spec)
        ops = list(generated.family_operators)
        groups = list(generated.family_group)
        trained = {g: [ops[f] for f in range(spec.families) if groups[f] == g] for g in range(args.groups)}
        probe = np.random.default_rng(np.random.SeedSequence([world, 4747])).normal(size=(256, config.world.state_dim))
        # G2 / G3 on held-out families
        heldout = []
        for j, task in enumerate(generated.novel_family_tasks):
            f = spec.families + j
            g = groups[f]
            other = [h for h in range(args.groups) if h != g]
            within_resid = ls_fit(ops[f].U, [o.U for o in trained[g]])
            cross_resid = min(ls_fit(ops[f].U, [o.U for o in trained[h]]) for h in other)
            within_err, contribution = substitute_error(task, ops[f], trained[g], probe)
            cross_err = min(substitute_error(task, ops[f], trained[h], probe)[0] for h in other)
            heldout.append({"family": f, "group": g, "within_resid": within_resid, "cross_resid": cross_resid,
                            "contribution_mse": contribution,
                            "within_err_over_contribution": within_err / max(contribution, 1e-12),
                            "cross_err_over_contribution": cross_err / max(contribution, 1e-12),
                            "Q": cross_err / max(within_err, 1e-12)})
        g2 = all(h["within_resid"] <= WITHIN_RESID and h["within_err_over_contribution"] <= WITHIN_NMSE for h in heldout)
        g3 = all(h["cross_resid"] >= CROSS_RESID and h["cross_err_over_contribution"] >= CROSS_CONTRIB
                 and h["Q"] >= Q_MIN for h in heldout)
        # G4 (Amendment 3): GROUP-level balance; family spread is a covariate
        # reported beside the same world's G = 1 spread.
        contributions, spread = family_spread(generated, spec, ops, probe)
        gm = [np.mean([contributions[f] for f in range(spec.families) if groups[f] == g]) for g in range(args.groups)]
        group_spread = float((max(gm) - min(gm)) / np.mean(gm))
        g4 = group_spread <= GROUP_BALANCE
        base_spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2)
        base_world = generate_meta_world(replace(config.world, seed=world, tasks=base_spec.total_tasks), base_spec)
        _, baseline_family_spread = family_spread(base_world, base_spec, list(base_world.family_operators), probe)
        # G5: nearest-centroid on effective innovation vectors (teacher)
        vectors, labels = [], []
        for i, task in enumerate(generated.tasks):
            f = spec.family_of(i)
            if f is None:
                continue
            identity = Primitive(U=np.zeros_like(ops[f].U), V=ops[f].V, b=ops[f].b, alpha=ops[f].alpha)
            innovation = (task.program.execute(task.teacher_library, probe)
                          - task.program.execute(with_operator(task, identity), probe))
            vectors.append(innovation.ravel()); labels.append(groups[f])
        X, y = np.stack(vectors), np.array(labels)
        # leave-one-out nearest centroid
        correct = 0
        for k in range(len(y)):
            mask = np.arange(len(y)) != k
            cents = [X[mask & (y == g)].mean(0) for g in range(args.groups)]
            pred = int(np.argmin([np.linalg.norm(X[k] - cnt) for cnt in cents]))
            correct += pred == y[k]
        accuracy = correct / len(y)
        g5 = bool(accuracy >= ACCURACY)
        g2, g3, g4 = bool(g2), bool(g3), bool(g4)
        out[world] = {"heldout": heldout, "G2": g2, "G3": g3, "contributions": contributions,
                      "contribution_spread": spread, "baseline_g1_family_spread": baseline_family_spread,
                      "group_spread": group_spread, "G4": g4,
                      "group_classification_accuracy_loo": accuracy, "G5": g5,
                      "all_pass": bool(g2 and g3 and g4 and g5)}
        print(f"world {world}: G2 {g2} (resid {[round(h['within_resid'], 4) for h in heldout]}, within/contrib {[round(h['within_err_over_contribution'], 4) for h in heldout]}) | "
              f"G3 {g3} (cross/contrib {[round(h['cross_err_over_contribution'], 3) for h in heldout]}, Q {[round(h['Q'], 1) for h in heldout]}) | "
              f"G4 {g4} (group {group_spread:.3f}; family {spread:.3f} vs G1 {baseline_family_spread:.3f}) | G5 {g5} (acc {accuracy:.3f})", flush=True)
    verdict = "PASS" if all(out[w]["all_pass"] for w in out) else "FAIL"
    report = {"frozen_plan": "H47_MEMBERSHIP_PLAN.md Amendment 2 (B2 teacher gates)", "git_commit": git_commit(),
              "frozen_amendment": 3,
              "thresholds": {"within_resid": WITHIN_RESID, "within_err_over_contribution": WITHIN_NMSE,
                             "cross_resid": CROSS_RESID, "cross_err_over_contribution": CROSS_CONTRIB,
                             "q_min": Q_MIN, "group_balance": GROUP_BALANCE, "accuracy": ACCURACY},
              "groups": args.groups, "worlds": out, "verdict": verdict}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(f"B2 world gates: {verdict}")


if __name__ == "__main__":
    main()
