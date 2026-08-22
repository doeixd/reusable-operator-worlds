"""Stage A audits of the sealed H39 result (H39_NEXT_STEPS_PLAN.md, Amendment 1).

A1  "made, not mined", apples-to-apples: the census construction (fit a
    coordinate in the affine span of the finished learner's REALIZED task
    objects -- private residuals plus promoted abstractions, NOT the
    learned argument matrices) on M's own sealed artifacts and on O's,
    against M's registered online alpha-only endpoint R_M.
A2  compensation: zero both slots' alphas on every trained family task,
    freeze everything else shared, re-fit only the task's route code and
    private residual on its own 128 training examples (B1), and compare
    evaluation NMSE with the intact model. If the world-mean lands in
    [1.10, 1.50) a second optimizer (Adam 0.05) is run on every cell and
    both must agree above 1.25.

Reads only completed sealed cells and the sealed report; writes
`reports/h39_confirmation_followups.json` atomically before printing.
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
from row.experiments.audit_effective_operator import load_learner
from row.experiments.audit_v6r_adaptation_geometry import CHECKPOINTS, SCALE
from row.experiments.census_h39_schema import alpha_fit as census_alpha_fit
from row.experiments.score_h39b_pslot import load_pslot, read_json
from row.experiments.score_h39_confirmation import RECORDS, SEEDS, validate_cell
from row.meta_world import MetaFamilySpec, generate_meta_world

A1_PASS_MARGIN = 0.5
A1_FAIL_MARGIN = 0.2
A2_PASS = 1.25
A2_FAIL = 1.10
A2_ROBUST_BAND = (1.10, 1.50)
STEPS = 2000


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def realized_population(model, family_ids: list[str]):
    vectors = [model.task_residuals[t].detach().clone() for t in family_ids
               if t not in model.retired]
    vectors.extend(a.detach().clone() for a in getattr(model, "abstractions", []))
    stack = torch.stack(vectors)
    mean = stack.mean(dim=0)
    _, singular, vh = torch.linalg.svd(stack - mean, full_matrices=False)
    return stack, mean, singular, vh


def census_fit(model, futures, family_ids, rank: int, label: str, freeze_alphas: bool):
    stack, mean, singular, vh = realized_population(model, family_ids)
    rank = min(rank if rank else stack.shape[0] - 1, vh.shape[0])
    basis = vh[:rank].T.contiguous()
    base = copy.deepcopy(model)
    if freeze_alphas:
        # The census must see only the realized population: M's argument
        # channel is closed for the probe (alphas zero, frozen). The
        # residual offset `mean` keeps the fit off the stationary point.
        for p in base.task_alphas.values():
            p.requires_grad_(False)
    fits = []
    for index, task in enumerate(futures):
        fit = census_alpha_fit(base, task, mean, basis, 128, f"{label}_r{rank}_t{index}")
        fit.update({"task_index": index, "rank": rank})
        fits.append(fit)
    return {"rank": rank, "population_size": int(stack.shape[0]),
            "variance_explained": float((singular[:rank] ** 2).sum() / (singular ** 2).sum()),
            "fits": fits,
            "endpoint_scaled": float(np.mean([f["final_query_scaled"] for f in fits]))}


@torch.no_grad()
def nmse(model, task_id: str, x, y) -> float:
    var = float(torch.var(y, unbiased=False)) or 1.0
    return float(torch.mean((model(x, task_id) - y) ** 2)) / var


def _refit(model, tid, x, y, ex, ey, params, learning_rate):
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    curve, finite = {}, True
    for update in range(1, STEPS + 1):
        optimizer.zero_grad()
        loss = torch.mean((model(x, tid) - y) ** 2)
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        loss.backward(inputs=params)
        optimizer.step()
        if update in CHECKPOINTS:
            curve[str(update)] = nmse(model, tid, ex, ey)
    return nmse(model, tid, ex, ey), curve, finite


def compensation_fit(base_model, task, learning_rate: float, label: str) -> dict:
    """Amendment 2: both arms get the same re-fit budget; only alpha differs."""
    tid = task.task_id
    x = torch.tensor(task.train_x, dtype=torch.float32)
    y = torch.tensor(task.train_y, dtype=torch.float32)
    ex = torch.tensor(task.eval_x, dtype=torch.float32)
    ey = torch.tensor(task.eval_y, dtype=torch.float32)
    intact = nmse(base_model, tid, ex, ey)
    retired = tid in base_model.retired
    results = {}
    for arm in ("alpha_free", "alpha_zeroed"):
        model = copy.deepcopy(base_model)
        for p in model.parameters():
            p.requires_grad_(False)
        # Retirement froze the private residual out of the computation; the
        # re-fit lifts it for BOTH arms so they differ only in alpha.
        model.retired.discard(tid)
        code, residual, alpha = model.task_codes[tid], model.task_residuals[tid], model.task_alphas[tid]
        params = [code, residual]
        if arm == "alpha_zeroed":
            with torch.no_grad():
                alpha.zero_()
        else:
            alpha.requires_grad_(True)
            params.append(alpha)
        code.requires_grad_(True)
        residual.requires_grad_(True)
        before = nmse(model, tid, ex, ey)
        final, curve, finite = _refit(model, tid, x, y, ex, ey, params, learning_rate)
        results[arm] = {"nmse_before_refit": before, "nmse_after_refit": final, "curve": curve,
                        "finite": finite,
                        "plateau_change_1000_to_2000": curve.get("1000", final) - final}
    free, zeroed = results["alpha_free"]["nmse_after_refit"], results["alpha_zeroed"]["nmse_after_refit"]
    return {"task_id": tid, "retired": retired, "learning_rate": learning_rate,
            "nmse_intact": intact, "arms": results,
            "uncompensated_ratio": (results["alpha_zeroed"]["nmse_before_refit"] / intact
                                    if intact > 0 else float("nan")),
            "compensated_ratio": zeroed / free if free > 0 else float("nan"),
            "plateau_change_1000_to_2000": max(abs(r["plateau_change_1000_to_2000"]) for r in results.values()),
            "finite": bool(all(r["finite"] for r in results.values()) and math.isfinite(zeroed / free))}


def geomean(values) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=float)))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/h39_confirmation"))
    parser.add_argument("--sealed", type=Path, default=Path("reports/h39_confirmation.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/h39_confirmation_followups.json"))
    parser.add_argument("--a2-sample", type=int, default=0,
                        help="EXPLORATORY only: limit A2 to the first N family tasks per world")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.a2_sample and args.output == Path("reports/h39_confirmation_followups.json"):
        raise SystemExit("a sampled A2 requires a non-registered --output")
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")

    sealed = read_json(args.sealed)
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2)
    worlds = {}
    for seed in SEEDS:
        paths = {arm: args.root / arm / f"world_{seed}" / "lifecycle" for arm in ("ordinary", "m2k32")}
        for arm, path in paths.items():
            validate_cell(path, arm, seed)
        generated = generate_meta_world(replace(config.world, seed=seed, tasks=spec.total_tasks), spec)
        futures = list(generated.novel_family_tasks)
        family_tasks = [t for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None]
        family_ids = [t.task_id for t in family_tasks]
        anchor = sealed["worlds"][str(seed)]["anchor_k128"]
        r_m = sealed["worlds"][str(seed)]["arms"]["m2k32"]["ratio"]

        m = load_pslot(config, paths["m2k32"], RECORDS["m2k32"])
        o = load_learner(config, paths["ordinary"], 12, kind="prospective")
        a1 = {"anchor_k128": anchor, "R_M_online": r_m}
        for name, model, freeze in (("M", m, True), ("O", o, False)):
            for rank in (8, 0):
                result = census_fit(model, futures, family_ids, rank, f"census_{name}_w{seed}", freeze)
                key = f"{name}_rank_{'max' if rank == 0 else rank}"
                a1[key] = {**result, "ratio": result["endpoint_scaled"] / anchor}
        print(f"world {seed} A1: R_M {r_m:.3f} | census(M) max {a1['M_rank_max']['ratio']:.3f} "
              f"r8 {a1['M_rank_8']['ratio']:.3f} | census(O) max {a1['O_rank_max']['ratio']:.3f}", flush=True)

        tasks_for_a2 = family_tasks[:args.a2_sample] if args.a2_sample else family_tasks
        a2_rows = [compensation_fit(m, task, 0.01, f"comp_w{seed}") for task in tasks_for_a2]
        comp = np.array([r["compensated_ratio"] for r in a2_rows])
        a2 = {"rows": a2_rows, "tasks": len(a2_rows),
              "median_compensated_ratio": float(np.median(comp)),
              "mean_compensated_ratio": float(comp.mean()),
              "median_uncompensated_ratio": float(np.median([r["uncompensated_ratio"] for r in a2_rows])),
              "max_plateau_change": float(max(abs(r["plateau_change_1000_to_2000"]) for r in a2_rows)),
              "finite": all(r["finite"] for r in a2_rows)}
        print(f"world {seed} A2: uncompensated median {a2['median_uncompensated_ratio']:.2f} -> "
              f"compensated median {a2['median_compensated_ratio']:.3f}", flush=True)
        worlds[seed] = {"A1": a1, "A2": a2}

    # ---- A1 decision ------------------------------------------------------
    g_rm = geomean([worlds[s]["A1"]["R_M_online"] for s in SEEDS])
    g_cm = geomean([worlds[s]["A1"]["M_rank_max"]["ratio"] for s in SEEDS])
    g_cm8 = geomean([worlds[s]["A1"]["M_rank_8"]["ratio"] for s in SEEDS])
    g_co = geomean([worlds[s]["A1"]["O_rank_max"]["ratio"] for s in SEEDS])
    if g_cm - g_rm >= A1_PASS_MARGIN and g_co - g_rm >= A1_PASS_MARGIN:
        a1_verdict = "PASS"
    elif g_cm - g_rm <= A1_FAIL_MARGIN:
        a1_verdict = "FAIL"
    else:
        a1_verdict = "PARTIAL"
    # ---- A2 decision ------------------------------------------------------
    medians = np.array([worlds[s]["A2"]["median_compensated_ratio"] for s in SEEDS])
    a2_mean = float(medians.mean())
    robustness = None
    if A2_ROBUST_BAND[0] <= a2_mean < A2_ROBUST_BAND[1] and not args.a2_sample:
        print("A2 in the robustness band; running Adam 0.05 on every cell", flush=True)
        robust_medians = []
        for seed in SEEDS:
            m = load_pslot(config, args.root / "m2k32" / f"world_{seed}" / "lifecycle", RECORDS["m2k32"])
            generated = generate_meta_world(replace(config.world, seed=seed, tasks=spec.total_tasks), spec)
            family_tasks = [t for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None]
            rows = [compensation_fit(m, task, 0.05, f"comp05_w{seed}") for task in family_tasks]
            worlds[seed]["A2"]["rows_adam_005"] = rows
            robust_medians.append(float(np.median([r["compensated_ratio"] for r in rows])))
        robustness = {"optimizer": "adam", "lr": 0.05, "mean_of_medians": float(np.mean(robust_medians)),
                      "per_world": robust_medians}
    if a2_mean >= A2_ROBUST_BAND[1]:
        a2_verdict = "PASS"
    elif a2_mean >= A2_PASS and robustness is not None and robustness["mean_of_medians"] >= A2_PASS:
        a2_verdict = "PASS"
    elif a2_mean >= A2_PASS:
        a2_verdict = "PARTIAL"
    elif a2_mean < A2_FAIL:
        a2_verdict = "FAIL"
    else:
        a2_verdict = "PARTIAL"
    report = {
        "frozen_plan": "H39_NEXT_STEPS_PLAN.md Stage A (Amendments 1-2)", "git_commit": git_commit(),
        "status": "EXPLORATORY (sampled A2)" if args.a2_sample else "REGISTERED",
        "protocol": {"steps": STEPS, "b1_lr": 0.01, "a1_pass_margin": A1_PASS_MARGIN,
                     "a1_fail_margin": A1_FAIL_MARGIN, "a2_pass": A2_PASS, "a2_fail": A2_FAIL,
                     "a2_robust_band": list(A2_ROBUST_BAND), "sigma": 0.1, "scale": SCALE},
        "worlds": worlds,
        "A1": {"geomean_R_M_online": g_rm, "geomean_census_M_max": g_cm, "geomean_census_M_rank8": g_cm8,
               "geomean_census_O_max": g_co, "verdict": a1_verdict,
               "registered_sentence_if_pass": ("useful coordinates were not recoverable from the final "
                                               "extensional task-object population; they had to be maintained "
                                               "in an explicit intensional channel during learning")},
        "A2": {"mean_of_world_median_compensated_ratio": a2_mean, "per_world_medians": medians.tolist(),
               "robustness": robustness, "verdict": a2_verdict},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(f"A1: online R_M {g_rm:.3f} | census(M) max-rank {g_cm:.3f} (rank 8 {g_cm8:.3f}) | "
          f"census(O) max-rank {g_co:.3f} -> {a1_verdict}")
    print(f"A2: mean of world-median compensated ratio {a2_mean:.3f} "
          f"(robustness {robustness}) -> {a2_verdict}")


if __name__ == "__main__":
    main()
