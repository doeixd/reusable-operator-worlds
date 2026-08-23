"""H47 B2 opportunity gate (H47_MEMBERSHIP_PLAN.md, Amendments 2-3).

On the schema_groups = 2 world, worlds 0-2: M_G2 (pooled soft routing,
two slots, K = 32) versus L_true (exact mask of each trained family task's
parameterized-slot mass onto its GROUP's slot). The gate IS the membership
tax, read once:

    MEMBERSHIP HAS VALUE iff log E_alpha(M) - log E_alpha(L_true) >= +0.15
                             in at least 2 of 3 worlds

where E_alpha is the alpha-only k=128 B1 scaled endpoint on the two
unseen-family futures (one per group). The ordinary anchor cancels in the
difference of logs; R values are also reported against the G = 1 ordinary
anchors for continuity only. Also: full-interface and k=1 endpoints, J
(raw nats; relative band deferred until an ordinary G = 2 run exists),
channel use, and M_G2's route baselines against GROUP labels (entropy,
margin, group consistency, ARI / NMI — diagnostics that set the H-stage
bands). Fails closed; atomic report.
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
from row.experiments.score_h39b_pslot import channel_use, factorized_fit, load_pslot, read_json
from row.experiments.score_h39c_ksweep import PROTOCOLS, WORLDS, validate_cell
from row.meta_world import MetaFamilySpec, generate_meta_world

GATE = 0.15
BASE = {"model": "pslot", "snapshot_history": True, "schema_groups": 2, "slot_args": 32,
        "freeze_args": False, "freeze_matrices": False, "pslot_index": 11, "pslot_count": 2}
ARMS = {"m": ("artifacts/h39c/b2_m", dict(BASE)),
        "ltrue": ("artifacts/h39c/b2_ltrue", {**BASE, "route_policy": {"kind": "mask_group"}})}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--h39d", type=Path, default=Path("reports/h39d_capacity.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/h47_b2_gate.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2, schema_groups=2)
    h39d = read_json(args.h39d)
    sources, worlds = {}, {}
    for world in WORLDS:
        for arm, (root, record) in ARMS.items():
            path = Path(root) / f"world_{world}" / "lifecycle"
            sources[f"{arm}/world_{world}"] = validate_cell(path, record, world)
            meta = read_json(path / "rho_profile.json").get("meta_family_spec") or {}
            if meta.get("schema_groups") != 2:
                raise SystemExit(f"{path} is not a schema_groups=2 world: {meta}")
    print("6 cells validated on the G=2 world", flush=True)

    for world in WORLDS:
        generated = generate_meta_world(replace(config.world, seed=world, tasks=spec.total_tasks), spec)
        futures = list(generated.novel_family_tasks)
        family_tasks = [t for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None]
        group_of_task = {t.task_id: spec.group_of_family(spec.family_of(i))
                         for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None}
        g1_anchor = h39d["worlds"][str(world)]["anchor_k128"]
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
                                             f"b2_{arm}_w{world}_{tag}_{mode}_k{support}_t{index}")
                        fit.update({"task_index": index, "future_group": spec.group_of_family(spec.families + index),
                                    "protocol": tag})
                        fits.append(fit)

            def endpoint(mode, support, tag):
                return float(np.mean([f["final_query_scaled"] for f in fits
                                      if f["mode"] == mode and f["support"] == support and f["protocol"] == tag]))
            # route diagnostics against GROUP labels
            state = {t: model.task_codes[t].detach() for t in group_of_task if t in model.task_codes}
            stats = {t: two_slot_stats(code, model.task_steps, model.operator_slots) for t, code in state.items()}
            labels = [group_of_task[t] for t in stats]
            dominant = [stats[t]["dominant_slot"] for t in stats]
            ent = np.array([min(model.conditional_entropy_bits(t)) for t in stats])
            margin = np.array([stats[t]["margin_at_dominant_step"] for t in stats])
            consistency = {}
            for g in (0, 1):
                members = [stats[t]["dominant_slot"] for t in stats if group_of_task[t] == g]
                maj = max(set(members), key=members.count)
                consistency[g] = {"majority_slot": maj, "fraction": members.count(maj) / len(members)}
            ari, nmi = ari_nmi(labels, dominant)
            use = channel_use(model, family_tasks)
            diag = read_json(path / "pslot.json")["diagnostics"]
            nonvac = {
                "argument_matrices_moved": all(v > 1e-3 for v in diag["argument_matrices_relative_movement_by_slot"]),
                "alpha_moves_in_every_fit": all(f["alpha_norm"] > 0 for f in fits if f["mode"] == "alpha_only"),
                "finite": all(f["finite"] for f in fits if f["protocol"] == "B1"),
            }
            if arm == "ltrue":
                nonvac["mask_fraction_1"] = sum(1 for t in group_of_task if t in model.task_mask) == len(group_of_task)
                nonvac["mask_matches_group"] = all(
                    model.task_mask[t] == model.pslot_indices[group_of_task[t]] for t in group_of_task if t in model.task_mask)
            if not all(nonvac.values()):
                raise SystemExit(f"non-vacuity failed: {arm} world {world}: {nonvac}")
            out[arm] = {
                "fits": fits, "J": read_json(path / "summary.json")["cumulative_prequential_gaussian_log_loss"],
                "E_alpha_k128_B1": endpoint("alpha_only", 128, "B1"),
                "E_alpha_k128_robust": {t: endpoint("alpha_only", 128, t) for t in ("B2_adam", "B2_lbfgs")},
                "E_full_k128_B1": endpoint("full", 128, "B1"),
                "E_alpha_k1": endpoint("alpha_only", 1, "B1"), "E_full_k1": endpoint("full", 1, "B1"),
                "R_alpha_vs_G1_anchor": endpoint("alpha_only", 128, "B1") / g1_anchor,
                "route": {"median_entropy_bits": float(np.median(ent)), "mean_entropy_bits": float(ent.mean()),
                          "median_margin": float(np.median(margin)), "group_consistency": consistency,
                          "ari_nmi_vs_group": [ari, nmi], "masked_tasks": len(model.task_mask)},
                "channel_use": {k: v for k, v in use.items() if k != "rows"}, "non_vacuity": nonvac,
            }
        m, l = out["m"], out["ltrue"]
        gap = float(np.log(m["E_alpha_k128_B1"]) - np.log(l["E_alpha_k128_B1"]))
        gap_robust = {t: float(np.log(m["E_alpha_k128_robust"][t]) - np.log(l["E_alpha_k128_robust"][t]))
                      for t in ("B2_adam", "B2_lbfgs")}
        worlds[world] = {"g1_anchor_k128": g1_anchor, "arms": out, "tax_log_alpha": gap,
                         "tax_log_alpha_robust": gap_robust,
                         "tax_log_full": float(np.log(m["E_full_k128_B1"]) - np.log(l["E_full_k128_B1"])),
                         "tax_J_nats": m["J"] - l["J"], "gate_world": gap >= GATE}
        print(f"world {world}: E_alpha M {m['E_alpha_k128_B1']:.5f} L {l['E_alpha_k128_B1']:.5f} tax {gap:+.3f} "
              f"(robust {min(gap_robust.values()):+.3f}) | full tax {worlds[world]['tax_log_full']:+.3f} | J M-L {m['J'] - l['J']:+.0f} | "
              f"M entropy {m['route']['median_entropy_bits']:.3f} ARI {m['route']['ari_nmi_vs_group'][0]:.2f} | "
              f"L entropy {l['route']['median_entropy_bits']:.3f}", flush=True)

    passes = sum(worlds[w]["gate_world"] for w in WORLDS)
    verdict = "MEMBERSHIP HAS VALUE" if passes >= 2 else "NO-OPPORTUNITY"
    m_ent = [worlds[w]["arms"]["m"]["route"]["median_entropy_bits"] for w in WORLDS]
    report = {"frozen_plan": "H47_MEMBERSHIP_PLAN.md Amendments 2-3 (B2 gate)", "git_commit": git_commit(),
              "protocol": {"gate_log_units": GATE, "worlds": list(WORLDS),
                           "protocols": [{"tag": t, "optimizer": o, "lr": lr, "steps": s} for o, lr, s, t in PROTOCOLS],
                           "query_checkpoints": list(CHECKPOINTS), "sigma": 0.1},
              "sources": sources, "worlds": worlds, "worlds_passing": passes, "verdict": verdict,
              "m_g2_baselines_for_h_stage": {
                  "median_entropy_bits_by_world": m_ent,
                  "ari_by_world": [worlds[w]["arms"]["m"]["route"]["ari_nmi_vs_group"][0] for w in WORLDS],
                  "log_E_alpha_sd": float(np.std([np.log(worlds[w]["arms"]["m"]["E_alpha_k128_B1"]) for w in WORLDS])),
                  "log_E_full_sd": float(np.std([np.log(worlds[w]["arms"]["m"]["E_full_k128_B1"]) for w in WORLDS]))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(f"tax log alpha by world: {[round(worlds[w]['tax_log_alpha'], 3) for w in WORLDS]}; "
          f"M_G2 entropy {[round(e, 3) for e in m_ent]}; VERDICT {verdict}")


if __name__ == "__main__":
    main()
