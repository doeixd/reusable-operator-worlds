"""Scorer for H47 B1 (H47_MEMBERSHIP_PLAN.md, Amendment 1): the cost of
imposing discrete commitment on a continuous computational family.

Arms on worlds 0-2: M (H39d cap_m2k32, reused), L_arb (arbitrary hard
mask), H_early (anneal 8->24), H_late (anneal 40->56). Endpoints by the
confirmation fits (alpha-only k=128 B1 with robustness, full k=128, k=1),
channel use, and the route entropy diagnostics. Tolerances are relative
to M per world, as frozen: J cost if (J_X - J_M) / |J_M - J_O| >= 0.25;
R cost if log R_X - log R_M >= +0.15 (alpha) / +0.08 (full); gain at the
negatives; otherwise NEUTRAL; a rule holds in >= 2 of 3 worlds. Result
matrix CONTINUOUS / COMPILE-AFTER-FORMATION / WRONG-ONTOLOGY / REDUNDANT
/ mixed. Fails closed; atomic report.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_v6r_adaptation_geometry import CHECKPOINTS
from row.experiments.score_h39b_pslot import channel_use, factorized_fit, load_pslot, read_json
from row.experiments.score_h39c_ksweep import PROTOCOLS, WORLDS, validate_cell
from row.meta_world import MetaFamilySpec, generate_meta_world

J_COST = 0.25
R_ALPHA_TOL = 0.15
R_FULL_TOL = 0.08
ENTROPY_TARGET = 0.20
ENTROPY_PAIR = 0.05
BASE = {"model": "pslot", "snapshot_history": True, "slot_args": 32, "freeze_args": False,
        "freeze_matrices": False, "pslot_index": 11, "pslot_count": 2}
ARMS = {
    "m": ("artifacts/h39c/cap_m2k32", dict(BASE)),
    "larb": ("artifacts/h39c/b1_larb", {**BASE, "route_policy": {"kind": "mask_arbitrary"}}),
    "hearly": ("artifacts/h39c/b1_hearly", {**BASE, "route_policy": {"kind": "anneal", "start": 8, "commit": 24, "final": 0.1}}),
    "hlate": ("artifacts/h39c/b1_hlate", {**BASE, "route_policy": {"kind": "anneal", "start": 40, "commit": 56, "final": 0.1}}),
}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def classify(delta: float, tol: float) -> str:
    return "COST" if delta >= tol else "GAIN" if delta <= -tol else "NEUTRAL"


def majority(labels: list[str], label: str) -> bool:
    return sum(l == label for l in labels) >= 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--h39d", type=Path, default=Path("reports/h39d_capacity.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/h47_b1.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2)
    h39d = read_json(args.h39d)
    sources, worlds = {}, {}
    for world in WORLDS:
        for arm, (root, record) in ARMS.items():
            path = Path(root) / f"world_{world}" / "lifecycle"
            sources[f"{arm}/world_{world}"] = validate_cell(path, record, world)
    print("all 12 cells validated", flush=True)

    for world in WORLDS:
        generated = generate_meta_world(replace(config.world, seed=world, tasks=spec.total_tasks), spec)
        futures = list(generated.novel_family_tasks)
        family_tasks = [t for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None]
        family_ids = {t.task_id for t in family_tasks}
        anchor = h39d["worlds"][str(world)]["anchor_k128"]
        j_o = h39d["worlds"][str(world)]["ordinary_loss"]
        out = {}
        for arm, (root, record) in ARMS.items():
            path = Path(root) / f"world_{world}" / "lifecycle"
            model = load_pslot(config, path, record)
            fits = []
            for index, task in enumerate(futures):
                for support, mode, protocols in ((128, "alpha_only", PROTOCOLS), (128, "full", PROTOCOLS[:1]),
                                                 (1, "alpha_only", PROTOCOLS[:1]), (1, "full", PROTOCOLS[:1])):
                    for opt, lr, steps, tag in protocols:
                        fit = factorized_fit(model, task, support, mode, opt, lr, steps,
                                             f"{arm}_w{world}_{tag}_{mode}_k{support}_t{index}")
                        fit.update({"task_index": index, "protocol": tag})
                        fits.append(fit)

            def endpoint(mode, support, tag):
                return float(np.mean([f["final_query_scaled"] for f in fits
                                      if f["mode"] == mode and f["support"] == support and f["protocol"] == tag]))
            ent = np.array([min(model.conditional_entropy_bits(t)) for t in family_ids if t in model.task_codes])
            masked = sum(1 for t in family_ids if t in model.task_mask)
            use = channel_use(model, family_tasks)
            diag = read_json(path / "pslot.json")["diagnostics"]
            nonvac = {
                "argument_matrices_moved": all(v > 1e-3 for v in diag["argument_matrices_relative_movement_by_slot"]),
                "alpha_moves_in_every_fit": all(f["alpha_norm"] > 0 for f in fits if f["mode"] == "alpha_only"),
                "finite": all(f["finite"] for f in fits if f["protocol"] == "B1"),
            }
            if arm == "larb":
                nonvac["mask_fraction_1"] = masked == len(family_ids)
            if arm in ("hearly", "hlate"):
                nonvac["final_entropy_at_target"] = float(np.median(ent)) <= ENTROPY_TARGET
                nonvac["temperature_final"] = abs(float(model.route_temperature) - 0.1) < 1e-6
            if not all(nonvac.values()):
                raise SystemExit(f"non-vacuity failed: {arm} world {world}: {nonvac}")
            out[arm] = {
                "fits": fits, "J": read_json(path / "summary.json")["cumulative_prequential_gaussian_log_loss"],
                "R_alpha": endpoint("alpha_only", 128, "B1") / anchor,
                "R_alpha_robust": {t: endpoint("alpha_only", 128, t) / anchor for t in ("B2_adam", "B2_lbfgs")},
                "R_full": endpoint("full", 128, "B1") / anchor,
                "alpha_only_k1": endpoint("alpha_only", 1, "B1"), "full_k1": endpoint("full", 1, "B1"),
                "median_conditional_entropy_bits": float(np.median(ent)), "mean_conditional_entropy_bits": float(ent.mean()),
                "masked_family_tasks": masked, "route_temperature": float(model.route_temperature),
                "channel_use": {k: v for k, v in use.items() if k != "rows"}, "non_vacuity": nonvac,
            }
        m = out["m"]
        scale = abs(m["J"] - j_o)
        deltas = {}
        for arm in ("larb", "hearly", "hlate"):
            x = out[arm]
            deltas[arm] = {
                "J_rel": (x["J"] - m["J"]) / scale, "dlog_R_alpha": float(np.log(x["R_alpha"]) - np.log(m["R_alpha"])),
                "dlog_R_full": float(np.log(x["R_full"]) - np.log(m["R_full"])),
            }
            deltas[arm]["J_class"] = "COST" if deltas[arm]["J_rel"] >= J_COST else "GAIN" if deltas[arm]["J_rel"] <= -J_COST else "NEUTRAL"
            deltas[arm]["R_alpha_class"] = classify(deltas[arm]["dlog_R_alpha"], R_ALPHA_TOL)
            deltas[arm]["R_full_class"] = classify(deltas[arm]["dlog_R_full"], R_FULL_TOL)
        worlds[world] = {"anchor_k128": anchor, "J_O": j_o, "arms": out, "deltas": deltas}
        print(f"world {world}: " + " | ".join(
            f"{a}: J {out[a]['J'] - m['J']:+.0f} R_a {out[a]['R_alpha']:.3f} R_f {out[a]['R_full']:.3f} H {out[a]['median_conditional_entropy_bits']:.3f}"
            for a in ARMS) + f" | M R_a {m['R_alpha']:.3f}", flush=True)

    # ---- rules ----------------------------------------------------------
    cls = {arm: {k: [worlds[w]["deltas"][arm][k] for w in WORLDS] for k in ("J_class", "R_alpha_class", "R_full_class")}
           for arm in ("larb", "hearly", "hlate")}
    verdict_arm = {arm: {k: ("COST" if majority(v, "COST") else "GAIN" if majority(v, "GAIN") else "NEUTRAL" if majority(v, "NEUTRAL") else "MIXED")
                         for k, v in cls[arm].items()} for arm in cls}
    pair_ok = all(abs(worlds[w]["arms"]["hearly"]["median_conditional_entropy_bits"]
                      - worlds[w]["arms"]["hlate"]["median_conditional_entropy_bits"]) <= ENTROPY_PAIR for w in WORLDS)
    larb_cost_alpha = verdict_arm["larb"]["R_alpha_class"] == "COST"
    hearly_cost = verdict_arm["hearly"]["R_alpha_class"] == "COST" or verdict_arm["hearly"]["J_class"] == "COST"
    hlate_neutral = verdict_arm["hlate"]["R_alpha_class"] == "NEUTRAL" and verdict_arm["hlate"]["J_class"] == "NEUTRAL"
    hlate_cost_alpha = verdict_arm["hlate"]["R_alpha_class"] == "COST"
    all_neutral = all(v == "NEUTRAL" for arm in verdict_arm for v in verdict_arm[arm].values())
    if not pair_ok:
        label = "NOT COMPARABLE (H_early/H_late final entropies differ by > 0.05 bits)"
    elif larb_cost_alpha and hearly_cost and hlate_cost_alpha:
        label = "CONTINUOUS"
    elif hearly_cost and hlate_neutral:
        label = "COMPILE-AFTER-FORMATION"
    elif larb_cost_alpha and verdict_arm["hearly"]["R_alpha_class"] == "NEUTRAL" and verdict_arm["hlate"]["R_alpha_class"] == "NEUTRAL":
        label = "WRONG-ONTOLOGY"
    elif all_neutral:
        label = "REDUNDANT"
    else:
        label = "MIXED (reported cell by cell)"
    report = {"frozen_plan": "H47_MEMBERSHIP_PLAN.md Amendment 1 (B1)", "git_commit": git_commit(),
              "protocol": {"j_cost": J_COST, "r_alpha_tol": R_ALPHA_TOL, "r_full_tol": R_FULL_TOL,
                           "entropy_target": ENTROPY_TARGET, "entropy_pair": ENTROPY_PAIR,
                           "protocols": [{"tag": t, "optimizer": o, "lr": lr, "steps": s} for o, lr, s, t in PROTOCOLS],
                           "query_checkpoints": list(CHECKPOINTS), "sigma": 0.1},
              "sources": sources, "worlds": worlds, "per_arm_verdicts": verdict_arm,
              "h_pair_comparable": pair_ok, "label": label}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    for arm in verdict_arm:
        print(f"{arm}: {verdict_arm[arm]}  per-world dlogR_alpha {[round(worlds[w]['deltas'][arm]['dlog_R_alpha'], 3) for w in WORLDS]}")
    print(f"LABEL {label}")


if __name__ == "__main__":
    main()
