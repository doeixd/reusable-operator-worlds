"""Scorer for H39d: larger K and multi-slot arguments (H39D_CAPACITY_PLAN.md).

Arms P32, P64 (one parameterized slot) and M2K16, M2K32 (two slots) on
worlds 0-2, against the ordinary V6 artifacts, the V6R anchors, and the
H39c K=16 points (read from reports/h39c_ksweep.json, not rerun). Fits,
channel-use, and loaders are the H39b functions. Decision rules are the
plan's: parity, fertile (functional usage criterion), trend over K = 16 ->
32 -> 64, slot_structure, and the fixed verdict table A / P+ / S.
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

ARMS = ("p32", "p64", "m2k16", "m2k32")
SINGLE = {"p32": 32, "p64": 64}
MULTI = {"m2k16": 16, "m2k32": 32}
IMPROVEMENT_MIN = 0.1
H39C_BEST_MEAN = None  # read from the H39c report
WORLDS = (0, 1, 2)
PASS_RATIO = 1.5
PARITY_NATS = 2000.0
ALPHA_ZEROED_MIN = 1.25
REQUIRED = ("model.pt", "summary.json", "rho_profile.json", "fingerprint.json",
            "config.yaml", "history.pt", "pslot.json")
PROTOCOLS = [("adam", 0.01, 2000, "B1"), ("adam", 0.05, 2000, "B2_adam"),
             ("lbfgs", 1.0, 500, "B2_lbfgs")]


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def record_for(arm: str) -> dict:
    record = {"model": "pslot", "snapshot_history": True,
              "slot_args": SINGLE.get(arm, MULTI.get(arm)),
              "freeze_args": False, "freeze_matrices": False, "pslot_index": 11}
    if arm in MULTI:
        record["pslot_count"] = 2
    return record


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
    parser.add_argument("--h39c", type=Path, default=Path("reports/h39c_ksweep.json"))
    parser.add_argument("--ordinary-root", type=Path, default=Path("artifacts/v6_clean"))
    parser.add_argument("--anchor", type=Path, default=Path("reports/v6r_adaptation_geometry.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/h39d_capacity.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    arms = list(ARMS)
    h39c = read_json(args.h39c)
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2)
    anchor_report = read_json(args.anchor)

    sources, worlds_out = {}, {}
    for world in WORLDS:
        ordinary_path = artifact_path(args.ordinary_root, "ordinary", world)
        sources[f"ordinary/world_{world}"] = validate_artifact(ordinary_path, "ordinary", world)
        for arm in arms:
            path = args.root / f"cap_{arm}" / f"world_{world}" / "lifecycle"
            sources[f"{arm}/world_{world}"] = validate_cell(path, record_for(arm), world)
    print("all 12 cells and 3 ordinary artifacts validated", flush=True)

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
            path = args.root / f"cap_{arm}" / f"world_{world}" / "lifecycle"
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
            nonvac = {
                "argument_matrices": all(m > 1e-3 for m in diagnostics["argument_matrices_relative_movement_by_slot"]),
                "family_alpha_nonzero": all(a > 0.0 for a in diagnostics["alpha_norm_mean_by_slot"]),
                "alpha_moves_in_every_fit": all(f["alpha_norm"] > 0 for f in fits if f["mode"] == "alpha_only"),
                "support_falls_over_1pct": all(f["support_reduction"] > 0.01 for f in fits if f["protocol"] == "B1"),
                "k0_differs_from_final": all(f["query_curve_mse"]["0"] != f["final_query_mse"] for f in fits),
                "primary_finite": all(f["finite"] for f in fits if f["protocol"] == "B1"),
            }
            if not all(nonvac.values()):
                raise SystemExit(f"non-vacuity failed for {arm} world {world}: {nonvac}")
            ratio = primary / anchor
            used = use["alpha_zeroed_ratio"] >= ALPHA_ZEROED_MIN
            arm_out[arm] = {
                "fits": fits, "loss": loss, "loss_gap_nats": loss - ordinary_loss,
                "alpha_only_k128_ratio": ratio, "robustness_ratios": robust,
                "robust_pass": any(v <= PASS_RATIO for v in robust.values()),
                "full_k128_ratio": endpoint("full", 128, "B1") / anchor,
                "alpha_only_k1": endpoint("alpha_only", 1, "B1"),
                "full_k1": endpoint("full", 1, "B1"),
                "channel_use": {k: v for k, v in use.items() if k != "rows"},
                "channel_rows": use["rows"],
                "used": bool(used),
                "pslot_diagnostics": diagnostics, "dstar_proxy_bits": dstar_proxy(model),
                "non_vacuity": nonvac,
            }
            print(f"  {arm}: loss gap {loss - ordinary_loss:+.1f}, alpha-only ratio {ratio:.3f} "
                  f"(robust {min(robust.values()):.3f}), full {arm_out[arm]['full_k128_ratio']:.3f}, "
                  f"mass {use['route_mass_P_max_step']:.3f} (ordinary {base_mass:.3f}), "
                  f"alpha-zeroed {use['alpha_zeroed_ratio']:.2f}, used={used}", flush=True)
        worlds_out[world] = {"anchor_k128": anchor, "anchor_rows": reproduced,
                             "ordinary_loss": ordinary_loss, "ordinary_slot12_mass": base_mass,
                             "arms": arm_out}

    # ---- decision rules --------------------------------------------------
    per_arm = {}
    for arm in arms:
        cells = [worlds_out[w]["arms"][arm] for w in WORLDS]
        parity = all(c["loss_gap_nats"] <= PARITY_NATS for c in cells)
        fertile_worlds = [w for w, c in zip(WORLDS, cells)
                          if c["alpha_only_k128_ratio"] <= PASS_RATIO and c["robust_pass"] and c["used"]]
        ratios = [c["alpha_only_k128_ratio"] for c in cells]
        per_arm[arm] = {"parity": parity, "fertile_worlds": fertile_worlds,
                        "fertile": len(fertile_worlds) >= 2, "ratios": ratios,
                        "mean_ratio": float(np.mean(ratios)),
                        "loss_gaps": [c["loss_gap_nats"] for c in cells],
                        "full_ratios": [c["full_k128_ratio"] for c in cells],
                        "used": [c["used"] for c in cells]}
    k16 = {w: h39c["worlds"][str(w)]["arms"]["p16"]["alpha_only_k128_ratio"] for w in WORLDS}
    trend_worlds = [w for w in WORLDS
                    if k16[w] >= per_arm["p32"]["ratios"][w] >= per_arm["p64"]["ratios"][w]]
    trend = len(trend_worlds) >= 2
    slot_worlds_32 = [w for w in WORLDS if per_arm["m2k16"]["ratios"][w] < per_arm["p32"]["ratios"][w]]
    slot_worlds_64 = [w for w in WORLDS if per_arm["m2k32"]["ratios"][w] < per_arm["p64"]["ratios"][w]]
    slot_structure = len(slot_worlds_32) >= 2 and len(slot_worlds_64) >= 2
    h39c_best_mean = float(np.mean(list(k16.values())))
    best_arm = min(arms, key=lambda a: per_arm[a]["mean_ratio"])
    improvement = h39c_best_mean - per_arm[best_arm]["mean_ratio"]
    fertile_arms = [a for a in arms if per_arm[a]["parity"] and per_arm[a]["fertile"]]
    if not any(per_arm[a]["parity"] for a in arms):
        verdict = "NOT COMPARABLE"
    elif fertile_arms:
        verdict = "A"
    elif trend and improvement >= IMPROVEMENT_MIN:
        verdict = "P+"
    else:
        verdict = "S"
    descriptions = {
        "A": "some arm is fertile on >= 2/3 worlds: licenses WRITING a confirmation plan (seeds 700-729, not opened)",
        "P+": "no fertile arm; trend holds and the best mean ratio improves on H39c K=16 by >= 0.1: still capacity-limited; one more development rung may be written",
        "S": "saturated: no fertile arm and the single-slot linear-in-U argument has stopped improving; the next design is not more K",
        "NOT COMPARABLE": "parity failed for every arm",
    }
    report = {
        "frozen_plan": "H39D_CAPACITY_PLAN.md", "status": "DEVELOPMENT worlds 0-2",
        "git_commit": git_commit(),
        "protocol": {"arms": list(ARMS), "worlds": list(WORLDS), "pass_ratio": PASS_RATIO,
                     "parity_nats": PARITY_NATS, "alpha_zeroed_min": ALPHA_ZEROED_MIN,
                     "improvement_min": IMPROVEMENT_MIN,
                     "protocols": [{"tag": t, "optimizer": o, "lr": lr, "steps": s} for o, lr, s, t in PROTOCOLS],
                     "sigma": 0.1, "query_checkpoints": list(CHECKPOINTS)},
        "sources": sources, "worlds": worlds_out, "per_arm": per_arm,
        "h39c_k16_ratios": k16, "h39c_k16_mean": h39c_best_mean,
        "trend_worlds": trend_worlds, "trend": trend,
        "slot_structure_worlds": {"m2k16_lt_p32": slot_worlds_32, "m2k32_lt_p64": slot_worlds_64},
        "slot_structure": slot_structure, "best_arm": best_arm, "improvement_over_h39c": improvement,
        "fertile_arms": fertile_arms, "verdict": verdict, "description": descriptions[verdict],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    for arm in arms:
        print(f"{arm}: ratios {[round(r, 3) for r in per_arm[arm]['ratios']]} mean {per_arm[arm]['mean_ratio']:.3f} "
              f"loss gaps {[round(g) for g in per_arm[arm]['loss_gaps']]} used {per_arm[arm]['used']} "
              f"fertile={per_arm[arm]['fertile']}")
    print(f"H39c K=16 mean {h39c_best_mean:.3f}; best {best_arm} improvement {improvement:+.3f}; "
          f"trend worlds {trend_worlds}; slot_structure={slot_structure} {report['slot_structure_worlds']}")
    print(f"VERDICT {verdict}: {descriptions[verdict]}")


if __name__ == "__main__":
    main()
