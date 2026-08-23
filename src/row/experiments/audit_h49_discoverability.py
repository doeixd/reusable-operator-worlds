"""H49 structural-discoverability census (H49_DISCOVERABILITY_PLAN.md).

On the frozen K = 4 artifacts of the schema_groups = 2 world (M_4 label-
free, L_4 told the group), for six candidate partitions applied as
RE-FIT routing policies: discard each trained family task's local state,
re-fit it alpha-only (route code + alphas; residual frozen at its
task-free init) on its own 128 examples under the policy (B1, 2,000
updates), and read evaluation NMSE. From these: C_LOO (geometric mean
NMSE), the two-part proxy D* (nats for the parameters the policy makes a
task pay for), and substitutability (own-group versus other-group slot).
No unseen task, no future label, no new lifetime. Fails closed on non-
vacuity; atomic report.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.score_h39b_pslot import load_pslot, read_json
from row.experiments.score_h39c_ksweep import validate_cell
from row.meta_world import MetaFamilySpec, generate_meta_world

K = 4
STEPS = 2000
LR = 0.01
PRECISION = 1.0 / 256
SIGMA = 0.1
P_LOO_MIN = 0.15
MARGIN_MIN = 0.10
WORLDS = (0, 1, 2)
BASE = {"model": "pslot", "snapshot_history": True, "schema_groups": 2, "slot_args": K,
        "freeze_args": False, "freeze_matrices": False, "pslot_index": 11, "pslot_count": 2}
ARTIFACTS = {"M4": ("artifacts/h39c/w_m4", dict(BASE)),
             "L4": ("artifacts/h39c/w_l4", {**BASE, "route_policy": {"kind": "mask_group"}})}
PAIRINGS = {"TRUE": ((0, 1), (2, 3)), "WRONG-A": ((0, 2), (1, 3)), "WRONG-B": ((0, 3), (1, 2))}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def candidates(spec, family_of_task: dict[str, int], world: int) -> dict[str, dict[str, int | None]]:
    """task_id -> slot POSITION (0 = slot 11, 1 = slot 10) or None (distributed)."""
    out = {}
    for name, (a, b) in PAIRINGS.items():
        out[name] = {t: (0 if f in a else 1) for t, f in family_of_task.items()}
    ids = sorted(family_of_task)
    for r in (1, 2):
        rng = np.random.default_rng(np.random.SeedSequence([49, world, r]))
        perm = rng.permutation(len(ids))
        out[f"RANDOM-{r}"] = {ids[i]: (0 if rank < len(ids) // 2 else 1) for rank, i in enumerate(perm)}
    out["DISTRIBUTED"] = {t: None for t in family_of_task}
    return out


@torch.no_grad()
def nmse(model, tid, x, y) -> float:
    var = float(torch.var(y, unbiased=False)) or 1.0
    return float(torch.mean((model(x, tid) - y) ** 2)) / var


def description_nats(code: torch.Tensor, alpha: torch.Tensor, position: int | None) -> float:
    """Unit-Gaussian two-part proxy at precision 1/256 for the parameters
    the policy makes the task pay for: the route code and the alphas of
    the slot(s) it routes through."""
    params = [code.detach().flatten()]
    params.append(alpha.detach().flatten() if position is None else alpha.detach()[position].flatten())
    theta = torch.cat(params)
    return float((0.5 * theta ** 2 + math.log(1.0 / PRECISION)).sum())


def refit(base_model, task, position: int | None, label: str) -> dict:
    model = copy.deepcopy(base_model)
    tid = task.task_id
    for p in model.parameters():
        p.requires_grad_(False)
    # Discard the task's local state: fresh code, fresh alphas, residual at
    # the task-free init (frozen), retirement lifted, reference kept.
    model.retired.discard(tid)
    model.task_mask.pop(tid, None)
    with torch.no_grad():
        model.task_codes[tid].zero_()
        model.task_alphas[tid].zero_()
        model.task_residuals[tid].copy_(model.initial_residual_state)
    if position is not None:
        model.task_mask[tid] = int(model.pslot_indices[position])
    code, alpha = model.task_codes[tid], model.task_alphas[tid]
    code.requires_grad_(True)
    alpha.requires_grad_(True)
    x = torch.tensor(task.train_x, dtype=torch.float32)
    y = torch.tensor(task.train_y, dtype=torch.float32)
    ex = torch.tensor(task.eval_x, dtype=torch.float32)
    ey = torch.tensor(task.eval_y, dtype=torch.float32)
    initial_support = float(torch.mean((model(x, tid) - y) ** 2))
    optimizer = torch.optim.Adam([code, alpha], lr=LR)
    finite = True
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = torch.mean((model(x, tid) - y) ** 2)
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        loss.backward(inputs=[code, alpha])
        optimizer.step()
    final_support = float(torch.mean((model(x, tid) - y) ** 2))
    value = nmse(model, tid, ex, ey)
    return {"task_id": tid, "position": position, "nmse": value,
            "support_reduction": (initial_support - final_support) / initial_support if initial_support > 0 else 0.0,
            "alpha_norm": float(torch.linalg.norm(alpha.detach())),
            "description_nats": description_nats(code, alpha, position),
            "finite": bool(finite and math.isfinite(value))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/h49_discoverability.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2, schema_groups=2)
    sources, worlds = {}, {}
    for world in WORLDS:
        for name, (root, record) in ARTIFACTS.items():
            sources[f"{name}/world_{world}"] = validate_cell(Path(root) / f"world_{world}" / "lifecycle", record, world)

    for world in WORLDS:
        generated = generate_meta_world(replace(config.world, seed=world, tasks=spec.total_tasks), spec)
        family_tasks = [(t, spec.family_of(i)) for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None]
        family_of_task = {t.task_id: f for t, f in family_tasks}
        cands = candidates(spec, family_of_task, world)
        assert cands["RANDOM-1"] != cands["RANDOM-2"] and all(cands["RANDOM-1"] != cands[p] for p in PAIRINGS)
        per_artifact = {}
        for name, (root, record) in ARTIFACTS.items():
            model = load_pslot(config, Path(root) / f"world_{world}" / "lifecycle", record)
            rows = {}
            for cand, assignment in cands.items():
                fits = [refit(model, t, assignment[t.task_id], f"{name}_w{world}_{cand}") for t, _ in family_tasks]
                if not all(f["finite"] and f["alpha_norm"] > 0 and f["support_reduction"] > 0.01 for f in fits):
                    bad = [f for f in fits if not (f["finite"] and f["alpha_norm"] > 0 and f["support_reduction"] > 0.01)]
                    raise SystemExit(f"non-vacuity failed: {name} world {world} {cand}: {bad[:2]}")
                rows[cand] = {"fits": fits,
                              "C_LOO": float(np.exp(np.mean(np.log([f["nmse"] for f in fits])))),
                              "D_star_nats": float(np.mean([f["description_nats"] for f in fits]))}
                print(f"world {world} {name} {cand:11s}: C_LOO {rows[cand]['C_LOO']:.5f} D* {rows[cand]['D_star_nats']:.1f}", flush=True)
            # substitutability: own-group (TRUE) vs other-group (complement of TRUE)
            complement = {t: 1 - p for t, p in cands["TRUE"].items()}
            other = [refit(model, t, complement[t.task_id], f"{name}_w{world}_COMPL") for t, _ in family_tasks]
            own = rows["TRUE"]["fits"]
            s_subst = float(np.mean([np.log(o["nmse"]) - np.log(w["nmse"]) for o, w in zip(other, own)]))
            rows["COMPLEMENT"] = {"fits": other, "C_LOO": float(np.exp(np.mean(np.log([f["nmse"] for f in other])))),
                                  "D_star_nats": float(np.mean([f["description_nats"] for f in other]))}
            log_true = np.log(rows["TRUE"]["C_LOO"])
            best_wrong = min(rows[c]["C_LOO"] for c in ("WRONG-A", "WRONG-B", "RANDOM-1", "RANDOM-2"))
            scale = 1.0 / (2 * SIGMA * SIGMA)  # nats per unit MSE at sigma 0.1... reported on scaled currency
            def two_part(c):
                # C_LOO in nats-equivalent over 128 examples at sigma: 128 * mse/(2 sigma^2); nmse -> mse via mean target var ~ use nmse*var; keep currency consistent by using nmse*128*scale*var_mean
                return rows[c]["D_star_nats"] + 128 * scale * rows[c]["C_LOO"] * mean_var
            mean_var = float(np.mean([np.var(t.eval_y) for t, _ in family_tasks]))
            per_artifact[name] = {
                "candidates": {c: {k: v for k, v in r.items() if k != "fits"} for c, r in rows.items()},
                "rows": {c: r["fits"] for c, r in rows.items()},
                "P_LOO": float(np.log(rows["DISTRIBUTED"]["C_LOO"]) - log_true),
                "margin_vs_best_wrong": float(np.log(best_wrong) - log_true),
                "P_D": rows["DISTRIBUTED"]["D_star_nats"] - rows["TRUE"]["D_star_nats"],
                "two_part_TRUE": two_part("TRUE"), "two_part_DISTRIBUTED": two_part("DISTRIBUTED"),
                "S_subst": s_subst,
            }
            a = per_artifact[name]
            print(f"world {world} {name}: P_LOO {a['P_LOO']:+.3f} margin {a['margin_vs_best_wrong']:+.3f} "
                  f"P_D {a['P_D']:+.1f} two-part TRUE {a['two_part_TRUE']:.1f} vs DIST {a['two_part_DISTRIBUTED']:.1f} "
                  f"S_subst {s_subst:+.3f}", flush=True)
        worlds[world] = per_artifact

    def signal_loo(name):
        return sum(worlds[w][name]["P_LOO"] >= P_LOO_MIN and worlds[w][name]["margin_vs_best_wrong"] >= MARGIN_MIN for w in WORLDS) >= 2

    def signal_d(name):
        return sum(worlds[w][name]["P_D"] > 0 and worlds[w][name]["two_part_TRUE"] < worlds[w][name]["two_part_DISTRIBUTED"] for w in WORLDS) >= 2
    instrument_ok = sum(worlds[w]["L4"]["margin_vs_best_wrong"] >= 0.15 or
                        (np.log(worlds[w]["L4"]["candidates"]["WRONG-A"]["C_LOO"]) - np.log(worlds[w]["L4"]["candidates"]["TRUE"]["C_LOO"]) >= 0.15
                         and np.log(worlds[w]["L4"]["candidates"]["WRONG-B"]["C_LOO"]) - np.log(worlds[w]["L4"]["candidates"]["TRUE"]["C_LOO"]) >= 0.15)
                        for w in WORLDS) >= 2
    m_signal = signal_loo("M4") or signal_d("M4")
    l_signal = signal_loo("L4") or signal_d("L4")
    if not instrument_ok:
        outcome = "NOT READ: instrument cannot see the partition on L4"
    elif m_signal:
        outcome = "A — DISCOVERABLE"
    elif l_signal:
        outcome = "C — SIGNAL NEEDS ORGANIZATION"
    else:
        outcome = "B — UNDERDETERMINED"
    report = {"frozen_plan": "H49_DISCOVERABILITY_PLAN.md", "git_commit": git_commit(),
              "protocol": {"K": K, "steps": STEPS, "lr": LR, "precision": PRECISION, "sigma": SIGMA,
                           "p_loo_min": P_LOO_MIN, "margin_min": MARGIN_MIN},
              "sources": sources, "worlds": worlds,
              "signals": {"M4_loo": signal_loo("M4"), "M4_dstar": signal_d("M4"),
                          "L4_loo": signal_loo("L4"), "L4_dstar": signal_d("L4"), "instrument_ok": instrument_ok},
              "outcome": outcome}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    for name in ("M4", "L4"):
        print(f"{name}: P_LOO {[round(worlds[w][name]['P_LOO'], 3) for w in WORLDS]} margin "
              f"{[round(worlds[w][name]['margin_vs_best_wrong'], 3) for w in WORLDS]} P_D {[round(worlds[w][name]['P_D'], 1) for w in WORLDS]} "
              f"S_subst {[round(worlds[w][name]['S_subst'], 3) for w in WORLDS]}")
    print(f"OUTCOME {outcome}")


if __name__ == "__main__":
    main()
