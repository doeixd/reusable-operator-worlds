"""Sealed-block scorer for H39_CONFIRMATION_PLAN.md (seeds 700-729).

Arms O (ordinary), M (two slots, K=32, learned directions), G (same with
U_k frozen at init). Per world: O's full-fit k=128 endpoint is computed
fresh (the V6R anchor construction); M and G get alpha-only and full fits.
Estimands E1-E5 with 95% paired bootstrap intervals over worlds and the
plan's predicted intervals; verdict CONFIRMED / CONFIRMED-RELATIVE / FAILED
printed from the fixed table. Fails closed on an incomplete block, a
mismatched record, or a non-vacuity failure; every cell is reported; the
report is written atomically before any console summary.
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
from row.experiments.audit_v6r_adaptation_geometry import CHECKPOINTS, SCALE, adam_fit
from row.experiments.score_h39b_pslot import (
    channel_use, dstar_proxy, factorized_fit, load_pslot, read_json,
)
from row.meta_world import MetaFamilySpec, generate_meta_world

SEEDS = list(range(700, 730))
PASS_RATIO = 1.5
ALPHA_ZEROED_MIN = 1.25
BOOT = 10_000
BOOT_SEED = (700, 39)
PREDICTED = {"E1_mean_D": (0.8, 2.2), "E2_mean_R": (1.2, 1.8), "E2_fraction": (0.55, 0.75),
             "E3_mean_L": (-2000.0, -500.0), "E4_mean_F": (0.65, 0.90)}
E2_FRACTION_MIN = 0.5
E5_MIN_WORLDS = 27
REQUIRED = ("model.pt", "summary.json", "rho_profile.json", "fingerprint.json",
            "config.yaml", "history.pt")
PROTOCOLS = [("adam", 0.01, 2000, "B1"), ("adam", 0.05, 2000, "B2_adam"),
             ("lbfgs", 1.0, 500, "B2_lbfgs")]
RECORDS = {
    "ordinary": {"model": "prospective", "snapshot_history": True},
    "m2k32": {"model": "pslot", "snapshot_history": True, "slot_args": 32, "freeze_args": False,
              "freeze_matrices": False, "pslot_index": 11, "pslot_count": 2},
    "g2k32": {"model": "pslot", "snapshot_history": True, "slot_args": 32, "freeze_args": False,
              "freeze_matrices": True, "pslot_index": 11, "pslot_count": 2},
}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def validate_cell(path: Path, arm: str, seed: int) -> dict:
    missing = [f for f in REQUIRED if not (path / f).exists()]
    if missing:
        raise SystemExit(f"sealed block incomplete: {path} missing {missing}")
    provenance = read_json(path / "rho_profile.json")
    if provenance.get("h39_pilot") != RECORDS[arm]:
        raise SystemExit(f"record mismatch at {path}: {provenance.get('h39_pilot')}")
    protocol = provenance.get("v6_arm") or {}
    expected = {"arm": "ordinary", "operator_slots": 12, "sleeps": [16, 24, 32, 48, 64],
                "lifecycle": True, "freeze_basis_at": None, "freeze_slots": None}
    bad = {k: protocol.get(k) for k, v in expected.items() if protocol.get(k) != v}
    if bad:
        raise SystemExit(f"protocol mismatch at {path}: {bad}")
    fingerprint = read_json(path / "fingerprint.json")
    if int(fingerprint.get("world_seed", -1)) != seed:
        raise SystemExit(f"world mismatch at {path}")
    return {"path": str(path), "git_commit": fingerprint.get("git_commit"),
            "resolved_config_sha256": fingerprint.get("resolved_config_sha256")}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/h39_confirmation"))
    parser.add_argument("--output", type=Path, default=Path("reports/h39_confirmation.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    prereg = subprocess.run(["python", "tools/check_prereg.py"])
    if prereg.returncode != 0:
        raise SystemExit("prereg check failed")

    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2)
    sources = {}
    for seed in SEEDS:
        for arm in RECORDS:
            path = args.root / arm / f"world_{seed}" / "lifecycle"
            sources[f"{arm}/world_{seed}"] = validate_cell(path, arm, seed)
    print("sealed block complete: 90/90 cells validated", flush=True)

    worlds = {}
    for seed in SEEDS:
        generated = generate_meta_world(replace(config.world, seed=seed, tasks=spec.total_tasks), spec)
        futures = list(generated.novel_family_tasks)
        if len(futures) != 2:
            raise SystemExit(f"world {seed}: expected 2 futures")
        family_tasks = [t for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None]
        paths = {arm: args.root / arm / f"world_{seed}" / "lifecycle" for arm in RECORDS}

        ordinary = load_learner(config, paths["ordinary"], 12, kind="prospective")
        o_fits = []
        for index, task in enumerate(futures):
            result, _, _ = adam_fit(ordinary, task, support=128, learning_rate=0.01, steps=2000,
                                    label=f"o_w{seed}_t{index}", checkpoints=CHECKPOINTS)
            if not result["finite"]:
                raise SystemExit(f"non-finite ordinary fit world {seed} task {index}")
            o_fits.append({"task_index": index, **result,
                           "final_query_scaled": result["final_query_mse"] / SCALE})
        anchor = float(np.mean([f["final_query_scaled"] for f in o_fits]))
        o_loss = read_json(paths["ordinary"] / "summary.json")["cumulative_prequential_gaussian_log_loss"]

        arms_out = {}
        for arm in ("m2k32", "g2k32"):
            model = load_pslot(config, paths[arm], RECORDS[arm])
            fits = []
            for index, task in enumerate(futures):
                for support, mode, protocols in ((128, "alpha_only", PROTOCOLS),
                                                 (128, "full", PROTOCOLS[:1]),
                                                 (1, "alpha_only", PROTOCOLS[:1]),
                                                 (1, "full", PROTOCOLS[:1])):
                    for opt, lr, steps, tag in protocols:
                        fit = factorized_fit(model, task, support, mode, opt, lr, steps,
                                             f"{arm}_w{seed}_{tag}_{mode}_k{support}_t{index}")
                        fit.update({"task_index": index, "protocol": tag})
                        fits.append(fit)

            def endpoint(mode, support, tag):
                return float(np.mean([f["final_query_scaled"] for f in fits
                                      if f["mode"] == mode and f["support"] == support
                                      and f["protocol"] == tag]))
            diagnostics = read_json(paths[arm] / "pslot.json")["diagnostics"]
            use = channel_use(model, family_tasks)
            matrices_at_init = all(torch.equal(m.detach(), init) for _, m, init in model.all_argument_matrices())
            nonvac = {
                "argument_matrices": (matrices_at_init if arm == "g2k32"
                                      else all(v > 1e-3 for v in diagnostics["argument_matrices_relative_movement_by_slot"])),
                "family_alpha_nonzero": all(a > 0 for a in diagnostics["alpha_norm_mean_by_slot"]),
                "alpha_moves_in_every_fit": all(f["alpha_norm"] > 0 for f in fits if f["mode"] == "alpha_only"),
                "support_falls_over_1pct": all(f["support_reduction"] > 0.01 for f in fits if f["protocol"] == "B1"),
                "k0_differs_from_final": all(f["query_curve_mse"]["0"] != f["final_query_mse"] for f in fits),
                "finite": all(f["finite"] for f in fits if f["protocol"] == "B1"),
            }
            if not all(nonvac.values()):
                raise SystemExit(f"non-vacuity failed: {arm} world {seed}: {nonvac}")
            loss = read_json(paths[arm] / "summary.json")["cumulative_prequential_gaussian_log_loss"]
            arms_out[arm] = {
                "fits": fits, "loss": loss, "loss_gap_nats": loss - o_loss,
                "alpha_only_k128_B1": endpoint("alpha_only", 128, "B1"),
                "ratio": endpoint("alpha_only", 128, "B1") / anchor,
                "robustness_ratios": {t: endpoint("alpha_only", 128, t) / anchor for t in ("B2_adam", "B2_lbfgs")},
                "full_ratio": endpoint("full", 128, "B1") / anchor,
                "alpha_only_k1": endpoint("alpha_only", 1, "B1"), "full_k1": endpoint("full", 1, "B1"),
                "channel_use": {k: v for k, v in use.items() if k != "rows"},
                "pslot_diagnostics": diagnostics, "dstar_proxy_bits": dstar_proxy(model),
                "non_vacuity": nonvac,
            }
        worlds[seed] = {"anchor_k128": anchor, "ordinary_fits": o_fits, "ordinary_loss": o_loss,
                        "arms": arms_out}
        m, g = arms_out["m2k32"], arms_out["g2k32"]
        print(f"world {seed}: R_M {m['ratio']:.3f} R_G {g['ratio']:.3f} L {m['loss_gap_nats']:+.0f} "
              f"F {m['full_ratio']:.3f} U {m['channel_use']['alpha_zeroed_ratio']:.2f}", flush=True)

    # ---- estimands -----------------------------------------------------
    R_M = np.array([worlds[s]["arms"]["m2k32"]["ratio"] for s in SEEDS])
    R_G = np.array([worlds[s]["arms"]["g2k32"]["ratio"] for s in SEEDS])
    D = R_G - R_M
    L = np.array([worlds[s]["arms"]["m2k32"]["loss_gap_nats"] for s in SEEDS])
    F = np.array([worlds[s]["arms"]["m2k32"]["full_ratio"] for s in SEEDS])
    U = np.array([worlds[s]["arms"]["m2k32"]["channel_use"]["alpha_zeroed_ratio"] for s in SEEDS])
    rng = np.random.default_rng(np.random.SeedSequence(list(BOOT_SEED)))

    def estimand(name, values, sign_ok, predicted_key, ci_excludes=None):
        mean = float(values.mean())
        lo, hi = bootstrap_ci(values, rng)
        sign = bool(sign_ok(mean))
        ci_pass = True if ci_excludes is None else not (lo <= ci_excludes <= hi)
        plo, phi = PREDICTED[predicted_key]
        interval = plo <= mean <= phi
        return {"mean": mean, "ci95": [lo, hi], "sign_pass": sign, "ci_excludes_null": ci_pass,
                "predicted_interval": [plo, phi], "in_predicted_interval": bool(interval),
                "pass": bool(sign and ci_pass and interval),
                "partial": bool(sign and ci_pass and not interval),
                "per_world": values.tolist()}
    E1 = estimand("E1", D, lambda m: m > 0, "E1_mean_D", ci_excludes=0.0)
    E3 = estimand("E3", L, lambda m: m < 0, "E3_mean_L", ci_excludes=0.0)
    E4 = estimand("E4", F, lambda m: m < 1, "E4_mean_F", ci_excludes=1.0)
    fraction = float(np.mean(R_M <= PASS_RATIO))
    mean_R = float(R_M.mean())
    E2 = {"fraction_le_1_5": fraction, "fraction_rule_pass": fraction >= E2_FRACTION_MIN,
          "predicted_fraction": PREDICTED["E2_fraction"],
          "fraction_in_predicted": PREDICTED["E2_fraction"][0] <= fraction <= PREDICTED["E2_fraction"][1],
          "mean_R": mean_R, "mean_R_ci95": list(bootstrap_ci(R_M, rng)),
          "predicted_mean": PREDICTED["E2_mean_R"],
          "mean_in_predicted": PREDICTED["E2_mean_R"][0] <= mean_R <= PREDICTED["E2_mean_R"][1],
          "per_world": R_M.tolist()}
    E2["pass"] = bool(E2["fraction_rule_pass"] and E2["mean_in_predicted"])
    E2["partial"] = bool(E2["fraction_rule_pass"] and not E2["mean_in_predicted"])
    used_worlds = int(np.sum(U >= ALPHA_ZEROED_MIN))
    E5 = {"worlds_used": used_worlds, "required": E5_MIN_WORLDS, "pass": used_worlds >= E5_MIN_WORLDS,
          "per_world": U.tolist()}
    core = E1["pass"] and E3["pass"] and E4["pass"] and E5["pass"]
    core_partial = (E1["sign_pass"] and E1["ci_excludes_null"] and E3["sign_pass"] and E3["ci_excludes_null"]
                    and E4["sign_pass"] and E4["ci_excludes_null"] and E5["pass"])
    if core_partial and E2["fraction_rule_pass"]:
        verdict = "CONFIRMED" if core and E2["pass"] else "CONFIRMED (with PARTIAL estimands)"
    elif core_partial:
        verdict = "CONFIRMED-RELATIVE" if core else "CONFIRMED-RELATIVE (with PARTIAL estimands)"
    else:
        verdict = "FAILED"
    partials = [n for n, e in (("E1", E1), ("E2", E2), ("E3", E3), ("E4", E4)) if e.get("partial")]
    report = {
        "frozen_plan": "H39_CONFIRMATION_PLAN.md", "status": "SEALED CONFIRMATION seeds 700-729",
        "git_commit": git_commit(),
        "protocol": {"seeds": SEEDS, "pass_ratio": PASS_RATIO, "alpha_zeroed_min": ALPHA_ZEROED_MIN,
                     "bootstrap": {"resamples": BOOT, "seed": list(BOOT_SEED)},
                     "protocols": [{"tag": t, "optimizer": o, "lr": lr, "steps": s} for o, lr, s, t in PROTOCOLS],
                     "sigma": 0.1, "query_checkpoints": list(CHECKPOINTS), "predicted": PREDICTED},
        "sources": sources, "worlds": worlds,
        "estimands": {"E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5},
        "partial_estimands": partials, "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(f"E1 mean D {E1['mean']:.3f} CI {E1['ci95']} pass={E1['pass']} partial={E1['partial']}")
    print(f"E2 fraction {fraction:.3f} mean R {mean_R:.3f} CI {E2['mean_R_ci95']} pass={E2['pass']} partial={E2['partial']}")
    print(f"E3 mean L {E3['mean']:.1f} CI {E3['ci95']} pass={E3['pass']} partial={E3['partial']}")
    print(f"E4 mean F {E4['mean']:.3f} CI {E4['ci95']} pass={E4['pass']} partial={E4['partial']}")
    print(f"E5 used {used_worlds}/30 pass={E5['pass']}")
    print(f"VERDICT {verdict}")


if __name__ == "__main__":
    main()
