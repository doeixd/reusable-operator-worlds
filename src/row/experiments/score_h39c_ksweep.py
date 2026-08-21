"""Scorer for the H39c K-sweep (H39C_KSWEEP_PLAN.md).

Arms P_K (K in 2,4,8,16) and the matched-budget control G_8 (U_k frozen at
init) on worlds 0-2, against the ordinary V6 artifacts and V6R anchors.
Fits, channel-use, and loaders are the H39b functions, unchanged. Decision
rules are the plan's: parity_K, fertile_K (baseline-relative channel-use),
learned_directions (P_8 vs G_8), trend, and the fixed verdict table.
Fails closed; report written atomically before any console summary.
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
from row.experiments.audit_effective_operator import load_learner
from row.experiments.audit_v6r_adaptation_geometry import (
    CHECKPOINTS, SCALE, adam_fit, artifact_path, validate_artifact,
)
from row.experiments.score_h39b_pslot import (
    channel_use, dstar_proxy, factorized_fit, load_pslot, read_json,
)
from row.meta_world import MetaFamilySpec, generate_meta_world

KS = (2, 4, 8, 16)
WORLDS = (0, 1, 2)
PASS_RATIO = 1.5
PARITY_NATS = 2000.0
ALPHA_ZEROED_MIN = 1.25
MASS_MULTIPLE = 2.0
LEARNED_MARGIN = 0.2
REQUIRED = ("model.pt", "summary.json", "rho_profile.json", "fingerprint.json",
            "config.yaml", "history.pt", "pslot.json")
PROTOCOLS = [("adam", 0.01, 2000, "B1"), ("adam", 0.05, 2000, "B2_adam"),
             ("lbfgs", 1.0, 500, "B2_lbfgs")]


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def record_for(arm: str) -> dict:
    k = 8 if arm == "g8" else int(arm[1:])
    return {"model": "pslot", "snapshot_history": True, "slot_args": k,
            "freeze_args": False, "freeze_matrices": arm == "g8", "pslot_index": 11}


def validate_cell(path: Path, expected: dict, world: int) -> dict:
    missing = [f for f in REQUIRED if not (path / f).exists()]
    if missing:
        raise SystemExit(f"incomplete cell {path}: missing {missing}")
    provenance = read_json(path / "rho_profile.json")
    if provenance.get("h39_pilot") != expected:
        raise SystemExit(f"record mismatch at {path}: {provenance.get('h39_pilot')} != {expected}")
    arm = provenance.get("v6_arm") or {}
    if arm.get("arm") != "ordinary" or arm.get("operator_slots") != 12 \
            or arm.get("sleeps") != [16, 24, 32, 48, 64] or not arm.get("lifecycle"):
        raise SystemExit(f"protocol mismatch at {path}: {arm}")
    fingerprint = read_json(path / "fingerprint.json")
    if int(fingerprint.get("world_seed", -1)) != world:
        raise SystemExit(f"world mismatch at {path}")
    return {"path": str(path), "git_commit": fingerprint.get("git_commit"),
            "resolved_config_sha256": fingerprint.get("resolved_config_sha256")}


@torch.no_grad()
def ordinary_slot_mass(path: Path) -> float:
    state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
    codes = torch.stack([torch.softmax(v.reshape(3, 12), -1)
                         for k, v in state.items() if k.startswith("task_codes.")])
    return float(codes[:, :, 11].mean(0).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/h39c"))
    parser.add_argument("--ordinary-root", type=Path, default=Path("artifacts/v6_clean"))
    parser.add_argument("--anchor", type=Path, default=Path("reports/v6r_adaptation_geometry.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/h39c_ksweep.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    arms = [f"p{k}" for k in KS] + ["g8"]
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2)
    anchor_report = read_json(args.anchor)

    sources, worlds_out = {}, {}
    for world in WORLDS:
        ordinary_path = artifact_path(args.ordinary_root, "ordinary", world)
        sources[f"ordinary/world_{world}"] = validate_artifact(ordinary_path, "ordinary", world)
        for arm in arms:
            path = args.root / f"ksweep_{arm}" / f"world_{world}" / "lifecycle"
            sources[f"{arm}/world_{world}"] = validate_cell(path, record_for(arm), world)
    print("all 15 cells and 3 ordinary artifacts validated", flush=True)

    for world in WORLDS:
        ordinary_path = artifact_path(args.ordinary_root, "ordinary", world)
        generated = generate_meta_world(replace(config.world, seed=world, tasks=spec.total_tasks), spec)
        futures = list(generated.novel_family_tasks)
        family_tasks = [t for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None]
        ordinary = load_learner(config, ordinary_path, 12, kind="prospective")
        rows = [r for r in anchor_report["rows"] if r["arm"] == "ordinary" and r["world"] == world
                and r["method"] == "adam_001" and r["support"] == 128]
        reproduced = []
        for index, task in enumerate(futures):
            result, _, _ = adam_fit(ordinary, task, support=128, learning_rate=0.01, steps=2000,
                                    label=f"anchor_w{world}_t{index}", checkpoints=CHECKPOINTS)
            expected = next(r["final_query_scaled"] for r in rows if r["task_index"] == index)
            observed = result["final_query_mse"] / SCALE
            if abs(expected - observed) > 1e-12:
                raise SystemExit(f"anchor mismatch world {world} task {index}")
            reproduced.append({"task_index": index, "expected": expected, "observed": observed})
        anchor = float(np.mean([r["expected"] for r in reproduced]))
        ordinary_loss = read_json(ordinary_path / "summary.json")["cumulative_prequential_gaussian_log_loss"]
        base_mass = ordinary_slot_mass(ordinary_path)
        print(f"world {world}: anchor {anchor:.5f} reproduced; ordinary slot-12 mass {base_mass:.4f}", flush=True)

        arm_out = {}
        for arm in arms:
            path = args.root / f"ksweep_{arm}" / f"world_{world}" / "lifecycle"
            model = load_pslot(config, path, record_for(arm))
            fits = []
            for index, task in enumerate(futures):
                for support, mode, protocols in ((128, "alpha_only", PROTOCOLS),
                                                 (128, "full", PROTOCOLS[:1]),
                                                 (1, "alpha_only", PROTOCOLS[:1]),
                                                 (1, "full", PROTOCOLS[:1])):
                    for opt, lr, steps, tag in protocols:
                        fit = factorized_fit(model, task, support, mode, opt, lr, steps,
                                             f"{arm}_w{world}_{tag}_{mode}_k{support}_t{index}")
                        fit.update({"task_index": index, "task_id": task.task_id, "protocol": tag})
                        fits.append(fit)

            def endpoint(mode, support, tag):
                vals = [f["final_query_scaled"] for f in fits
                        if f["mode"] == mode and f["support"] == support and f["protocol"] == tag]
                return float(np.mean(vals))
            primary = endpoint("alpha_only", 128, "B1")
            robust = {t: endpoint("alpha_only", 128, t) / anchor for t in ("B2_adam", "B2_lbfgs")}
            diagnostics = read_json(path / "pslot.json")["diagnostics"]
            use = channel_use(model, family_tasks)
            loss = read_json(path / "summary.json")["cumulative_prequential_gaussian_log_loss"]
            matrices_initial = bool(torch.equal(model.argument_matrices.detach(),
                                                model.initial_argument_matrices))
            nonvac = {
                "argument_matrices": (matrices_initial if arm == "g8"
                                      else diagnostics["argument_matrices_relative_movement"] > 1e-3),
                "family_alpha_nonzero": use["alpha_norm_mean"] > 0.0,
                "alpha_moves_in_every_fit": all(f["alpha_norm"] > 0 for f in fits if f["mode"] == "alpha_only"),
                "support_falls_over_1pct": all(f["support_reduction"] > 0.01 for f in fits if f["protocol"] == "B1"),
                "k0_differs_from_final": all(f["query_curve_mse"]["0"] != f["final_query_mse"] for f in fits),
                "primary_finite": all(f["finite"] for f in fits if f["protocol"] == "B1"),
            }
            if not all(nonvac.values()):
                raise SystemExit(f"non-vacuity failed for {arm} world {world}: {nonvac}")
            ratio = primary / anchor
            used = (use["alpha_zeroed_ratio"] >= ALPHA_ZEROED_MIN
                    and use["route_mass_P_max_step"] >= MASS_MULTIPLE * base_mass)
            arm_out[arm] = {
                "fits": fits, "loss": loss, "loss_gap_nats": loss - ordinary_loss,
                "alpha_only_k128_ratio": ratio, "robustness_ratios": robust,
                "robust_pass": any(v <= PASS_RATIO for v in robust.values()),
                "full_k128_ratio": endpoint("full", 128, "B1") / anchor,
                "alpha_only_k1": endpoint("alpha_only", 1, "B1"),
                "full_k1": endpoint("full", 1, "B1"),
                "channel_use": {k: v for k, v in use.items() if k != "rows"},
                "channel_rows": use["rows"],
                "used": bool(used), "mass_threshold": MASS_MULTIPLE * base_mass,
                "pslot_diagnostics": diagnostics, "dstar_proxy_bits": dstar_proxy(model),
                "non_vacuity": nonvac,
            }
            print(f"  {arm}: loss gap {loss - ordinary_loss:+.1f}, alpha-only ratio {ratio:.3f} "
                  f"(robust {min(robust.values()):.3f}), full {arm_out[arm]['full_k128_ratio']:.3f}, "
                  f"mass {use['route_mass_P_max_step']:.3f}/{MASS_MULTIPLE * base_mass:.3f}, "
                  f"alpha-zeroed {use['alpha_zeroed_ratio']:.2f}, used={used}", flush=True)
        worlds_out[world] = {"anchor_k128": anchor, "anchor_rows": reproduced,
                             "ordinary_loss": ordinary_loss, "ordinary_slot12_mass": base_mass,
                             "arms": arm_out}

    # ---- decision rules --------------------------------------------------
    per_k = {}
    for k in KS:
        arm = f"p{k}"
        parity = all(worlds_out[w]["arms"][arm]["loss_gap_nats"] <= PARITY_NATS for w in WORLDS)
        fertile_worlds = [w for w in WORLDS if worlds_out[w]["arms"][arm]["alpha_only_k128_ratio"] <= PASS_RATIO
                          and worlds_out[w]["arms"][arm]["robust_pass"] and worlds_out[w]["arms"][arm]["used"]]
        per_k[k] = {"parity": parity, "fertile_worlds": fertile_worlds,
                    "fertile": len(fertile_worlds) >= 2,
                    "ratios": [worlds_out[w]["arms"][arm]["alpha_only_k128_ratio"] for w in WORLDS],
                    "loss_gaps": [worlds_out[w]["arms"][arm]["loss_gap_nats"] for w in WORLDS],
                    "full_ratios": [worlds_out[w]["arms"][arm]["full_k128_ratio"] for w in WORLDS],
                    "used": [worlds_out[w]["arms"][arm]["used"] for w in WORLDS]}
    g_vs_p = [worlds_out[w]["arms"]["g8"]["alpha_only_k128_ratio"]
              - worlds_out[w]["arms"]["p8"]["alpha_only_k128_ratio"] for w in WORLDS]
    learned_directions = all(d > 0 for d in g_vs_p) and float(np.mean(g_vs_p)) > LEARNED_MARGIN
    trend_worlds = [w for w in WORLDS if all(
        worlds_out[w]["arms"][f"p{KS[i]}"]["alpha_only_k128_ratio"]
        >= worlds_out[w]["arms"][f"p{KS[i + 1]}"]["alpha_only_k128_ratio"] for i in range(len(KS) - 1))]
    trend = len(trend_worlds) >= 2
    any_parity = any(per_k[k]["parity"] for k in KS)
    fertile_ks = [k for k in KS if per_k[k]["parity"] and per_k[k]["fertile"]]
    if not any_parity:
        verdict = "NOT COMPARABLE"
    elif fertile_ks and learned_directions:
        verdict = "A"
    elif fertile_ks:
        verdict = "A-capacity"
    elif learned_directions and trend:
        verdict = "P"
    else:
        verdict = "B"
    descriptions = {
        "A": "in-basis argument is fertile and its learned directions matter; licenses writing a frozen confirmation plan (seeds 700-729, not opened)",
        "A-capacity": "some K passes but learned directions do not beat the frozen-direction control: a capacity result, not sharing",
        "P": "no K reaches 1.5x, but learned directions beat the control 3/3 and the gap shrinks with K: real, capacity-limited at K <= 16",
        "B": "not fertile",
        "NOT COMPARABLE": "parity failed for every K",
    }
    report = {
        "frozen_plan": "H39C_KSWEEP_PLAN.md", "status": "DEVELOPMENT worlds 0-2",
        "git_commit": git_commit(),
        "protocol": {"ks": list(KS), "worlds": list(WORLDS), "pass_ratio": PASS_RATIO,
                     "parity_nats": PARITY_NATS, "alpha_zeroed_min": ALPHA_ZEROED_MIN,
                     "mass_multiple_of_baseline": MASS_MULTIPLE, "learned_margin": LEARNED_MARGIN,
                     "protocols": [{"tag": t, "optimizer": o, "lr": lr, "steps": s} for o, lr, s, t in PROTOCOLS],
                     "sigma": 0.1, "query_checkpoints": list(CHECKPOINTS)},
        "sources": sources, "worlds": worlds_out, "per_k": per_k,
        "g8_minus_p8_ratio_by_world": g_vs_p, "learned_directions": learned_directions,
        "trend_worlds": trend_worlds, "trend": trend, "fertile_ks": fertile_ks,
        "verdict": verdict, "description": descriptions[verdict],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    for k in KS:
        print(f"K={k}: ratios {[round(r, 3) for r in per_k[k]['ratios']]} loss gaps "
              f"{[round(g) for g in per_k[k]['loss_gaps']]} used {per_k[k]['used']} fertile={per_k[k]['fertile']}")
    print(f"G8 - P8 ratio by world: {[round(d, 3) for d in g_vs_p]}; learned_directions={learned_directions}; "
          f"trend worlds {trend_worlds}")
    print(f"VERDICT {verdict}: {descriptions[verdict]}")


if __name__ == "__main__":
    main()
