"""Scorer for the H48b schema-width opportunity sweep (H48B_WIDTH_SWEEP_PLAN.md).

G = 2 world, worlds 0-2, K in {2, 4, 8, 16, 32}; per K, M_K (ignores
identity) versus L_K (told the group). Deltas are oracle-minus-learner in
log endpoint units (positive = told-identity is better) and in nats for
present cost. Rules and the four outcomes are the plan's. Fails closed;
atomic report.
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
from row.experiments.audit_h47_baselines import ari_nmi, two_slot_stats
from row.experiments.audit_v6r_adaptation_geometry import CHECKPOINTS
from row.experiments.score_h39b_pslot import channel_use, dstar_proxy, factorized_fit, load_pslot, read_json
from row.experiments.score_h39c_ksweep import PROTOCOLS, WORLDS, validate_cell
from row.meta_world import MetaFamilySpec, generate_meta_world

KS = (2, 4, 8, 16, 32)
ALPHA_PAYS = 0.15
FULL_PAYS = 0.08
USAGE_MIN = 1.25


def record(k: int, told: bool) -> dict:
    r = {"model": "pslot", "snapshot_history": True, "schema_groups": 2, "slot_args": k,
         "freeze_args": False, "freeze_matrices": False, "pslot_index": 11, "pslot_count": 2}
    if told:
        r["route_policy"] = {"kind": "mask_group"}
    return r


def cell_path(k: int, told: bool, world: int) -> Path:
    if k == 32:
        root = "artifacts/h39c/b2_ltrue" if told else "artifacts/h39c/b2_m"
    else:
        root = f"artifacts/h39c/w_{'l' if told else 'm'}{k}"
    return Path(root) / f"world_{world}" / "lifecycle"


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def majority(flags) -> bool:
    return sum(bool(f) for f in flags) >= 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/h48b_width.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2, schema_groups=2)
    sources = {}
    for k in KS:
        for told in (False, True):
            for world in WORLDS:
                path = cell_path(k, told, world)
                sources[f"K{k}_{'L' if told else 'M'}/world_{world}"] = validate_cell(path, record(k, told), world)
                if (read_json(path / "rho_profile.json").get("meta_family_spec") or {}).get("schema_groups") != 2:
                    raise SystemExit(f"{path} is not a G=2 world")
    print("30 cells validated", flush=True)

    results = {}
    for world in WORLDS:
        generated = generate_meta_world(replace(config.world, seed=world, tasks=spec.total_tasks), spec)
        futures = list(generated.novel_family_tasks)
        family_tasks = [t for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None]
        group_of = {t.task_id: spec.group_of_family(spec.family_of(i))
                    for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None}
        for k in KS:
            pair = {}
            for told in (False, True):
                path = cell_path(k, told, world)
                model = load_pslot(config, path, record(k, told))
                fits = []
                for index, task in enumerate(futures):
                    for support, mode, protocols in ((128, "alpha_only", PROTOCOLS), (128, "full", PROTOCOLS[:1]),
                                                     (1, "alpha_only", PROTOCOLS[:1]), (1, "full", PROTOCOLS[:1])):
                        for opt, lr, steps, tag in protocols:
                            fit = factorized_fit(model, task, support, mode, opt, lr, steps,
                                                 f"w_k{k}_{'L' if told else 'M'}_w{world}_{tag}_{mode}_k{support}_t{index}")
                            fit.update({"task_index": index, "protocol": tag})
                            fits.append(fit)

                def endpoint(mode, support, tag):
                    return float(np.mean([f["final_query_scaled"] for f in fits
                                          if f["mode"] == mode and f["support"] == support and f["protocol"] == tag]))
                use = channel_use(model, family_tasks)
                diag = read_json(path / "pslot.json")["diagnostics"]
                stats = {t: two_slot_stats(model.task_codes[t].detach(), model.task_steps, model.operator_slots)
                         for t in group_of if t in model.task_codes}
                ari, nmi = ari_nmi([group_of[t] for t in stats], [stats[t]["dominant_slot"] for t in stats])
                ent = float(np.median([min(model.conditional_entropy_bits(t)) for t in stats]))
                nonvac = {"argument_matrices_moved": all(v > 1e-3 for v in diag["argument_matrices_relative_movement_by_slot"]),
                          "alpha_moves_in_every_fit": all(f["alpha_norm"] > 0 for f in fits if f["mode"] == "alpha_only"),
                          "finite": all(f["finite"] for f in fits if f["protocol"] == "B1")}
                if told:
                    nonvac["mask_fraction_1"] = sum(1 for t in group_of if t in model.task_mask) == len(group_of)
                elif k <= 4:
                    nonvac["usage_over_1_25"] = use["alpha_zeroed_ratio"] >= USAGE_MIN
                if not all(nonvac.values()):
                    raise SystemExit(f"non-vacuity failed: K={k} {'L' if told else 'M'} world {world}: {nonvac}")
                pair["L" if told else "M"] = {
                    "fits": fits, "J": read_json(path / "summary.json")["cumulative_prequential_gaussian_log_loss"],
                    "E_alpha": endpoint("alpha_only", 128, "B1"),
                    "E_alpha_robust": {t: endpoint("alpha_only", 128, t) for t in ("B2_adam", "B2_lbfgs")},
                    "E_full": endpoint("full", 128, "B1"),
                    "E_alpha_k1": endpoint("alpha_only", 1, "B1"), "E_full_k1": endpoint("full", 1, "B1"),
                    "channel_use": {kk: v for kk, v in use.items() if kk != "rows"},
                    "dstar_proxy_bits": dstar_proxy(model),
                    "route": {"median_entropy_bits": ent, "ari_vs_group": ari, "nmi_vs_group": nmi},
                    "non_vacuity": nonvac,
                }
            m, l = pair["M"], pair["L"]
            d_alpha = float(np.log(m["E_alpha"]) - np.log(l["E_alpha"]))
            d_alpha_robust = {t: float(np.log(m["E_alpha_robust"][t]) - np.log(l["E_alpha_robust"][t]))
                              for t in ("B2_adam", "B2_lbfgs")}
            results[(k, world)] = {"arms": pair, "delta_alpha": d_alpha, "delta_alpha_robust": d_alpha_robust,
                                   "robust_sign_agrees": all(np.sign(v) == np.sign(d_alpha) for v in d_alpha_robust.values()),
                                   "delta_full": float(np.log(m["E_full"]) - np.log(l["E_full"])),
                                   "delta_J": m["J"] - l["J"]}
            r = results[(k, world)]
            print(f"K={k:2d} world {world}: d_alpha {d_alpha:+.3f} (robust {min(d_alpha_robust.values()):+.3f}) "
                  f"d_full {r['delta_full']:+.3f} d_J {r['delta_J']:+.0f} | M entropy {m['route']['median_entropy_bits']:.3f} "
                  f"ARI {m['route']['ari_vs_group']:.2f} usage {m['channel_use']['alpha_zeroed_ratio']:.2f}", flush=True)

    per_k = {}
    for k in KS:
        rows = [results[(k, w)] for w in WORLDS]
        alpha_pays = majority(r["delta_alpha"] >= ALPHA_PAYS and r["robust_sign_agrees"] for r in rows)
        present_pays = majority(r["delta_J"] >= 0 for r in rows)
        full_pays = majority(r["delta_full"] >= FULL_PAYS for r in rows)
        per_k[k] = {"alpha_pays": alpha_pays, "present_pays": present_pays, "full_pays": full_pays,
                    "delta_alpha": [r["delta_alpha"] for r in rows], "delta_full": [r["delta_full"] for r in rows],
                    "delta_J": [r["delta_J"] for r in rows],
                    "m_entropy": [r["arms"]["M"]["route"]["median_entropy_bits"] for r in rows],
                    "m_ari": [r["arms"]["M"]["route"]["ari_vs_group"] for r in rows]}
    paying = [k for k in KS if per_k[k]["alpha_pays"] and per_k[k]["present_pays"]]
    k_star = max(paying) if paying and not per_k[32]["alpha_pays"] else None
    fully = [k for k in KS if per_k[k]["alpha_pays"] and per_k[k]["present_pays"] and per_k[k]["full_pays"]]
    alpha_any = [k for k in KS if per_k[k]["alpha_pays"]]
    if fully:
        outcome = "FULLY LICENSED"
    elif k_star is not None and all(per_k[k]["alpha_pays"] and per_k[k]["present_pays"] for k in KS if k < k_star):
        outcome = "CROSSOVER"
    elif not alpha_any:
        outcome = "CAPACITY NOT BINDING"
    elif alpha_any and not any(per_k[k]["full_pays"] or per_k[k]["present_pays"] for k in alpha_any):
        outcome = "INNOVATION BUFFERS"
    else:
        outcome = "MIXED (reported per K)"
    report = {"frozen_plan": "H48B_WIDTH_SWEEP_PLAN.md", "git_commit": git_commit(),
              "protocol": {"ks": list(KS), "alpha_pays": ALPHA_PAYS, "full_pays": FULL_PAYS, "usage_min": USAGE_MIN,
                           "protocols": [{"tag": t, "optimizer": o, "lr": lr, "steps": s} for o, lr, s, t in PROTOCOLS],
                           "query_checkpoints": list(CHECKPOINTS), "sigma": 0.1},
              "sources": sources,
              "cells": {f"K{k}/world_{w}": v for (k, w), v in results.items()},
              "per_k": per_k, "k_star": k_star, "fully_licensed_ks": fully, "outcome": outcome}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    for k in KS:
        p = per_k[k]
        print(f"K={k:2d}: d_alpha {[round(x, 3) for x in p['delta_alpha']]} d_J {[round(x) for x in p['delta_J']]} "
              f"d_full {[round(x, 3) for x in p['delta_full']]} -> alpha {p['alpha_pays']} present {p['present_pays']} full {p['full_pays']} "
              f"| M entropy {[round(e, 2) for e in p['m_entropy']]} ARI {[round(a, 2) for a in p['m_ari']]}")
    print(f"K* = {k_star}; OUTCOME {outcome}")


if __name__ == "__main__":
    main()
