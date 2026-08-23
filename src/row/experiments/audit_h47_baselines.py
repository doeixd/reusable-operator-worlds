"""H47 baselines: how decisive is M's label-free routing, on worlds 0-2?

Read-only, on the H39d two-slot K=32 artifacts (`artifacts/h39c/cap_m2k32`)
and `reports/h39d_capacity.json`. Produces the numbers the H47 plan must
set its tolerances and annealing targets against BEFORE it is frozen
(review 64): route mass and entropy over the two parameterized slots,
early (from `history.pt`, the route at task completion) and late (final
artifact); per-task margins; dominant-slot consistency within each teacher
family; ARI / NMI as diagnostics; and the J / R quantities with their
cross-world spread. No decision is made here.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.score_h39b_pslot import read_json
from row.meta_world import MetaFamilySpec, generate_meta_world

RECORD = {"model": "pslot", "snapshot_history": True, "slot_args": 32, "freeze_args": False,
          "freeze_matrices": False, "pslot_index": 11, "pslot_count": 2}
SLOTS = (11, 10)


def two_slot_stats(code: torch.Tensor, steps: int, slots: int) -> dict:
    soft = torch.softmax(code.reshape(steps, slots), dim=-1)
    mass = soft[:, list(SLOTS)]                      # (steps, 2)
    total = mass.sum(-1)                              # mass on parameterized slots
    cond = mass / total.clamp_min(1e-12).unsqueeze(-1)
    entropy = -(cond * cond.clamp_min(1e-12).log()).sum(-1) / np.log(2)  # bits, in [0,1]
    margin = (cond[:, 0] - cond[:, 1]).abs()
    step = int(total.argmax())                        # the step where P fires most
    return {"mass_total_by_step": total.tolist(), "cond_p11_by_step": cond[:, 0].tolist(),
            "entropy_bits_by_step": entropy.tolist(), "margin_by_step": margin.tolist(),
            "dominant_step": step, "dominant_slot": int(SLOTS[int(cond[step].argmax())]),
            "entropy_at_dominant_step": float(entropy[step]),
            "margin_at_dominant_step": float(margin[step]),
            "mass_total_at_dominant_step": float(total[step])}


def ari_nmi(labels_a: list[int], labels_b: list[int]) -> tuple[float, float]:
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    ua, ub = np.unique(a), np.unique(b)
    n = len(a)
    table = np.array([[np.sum((a == x) & (b == y)) for y in ub] for x in ua], dtype=float)
    comb = lambda v: v * (v - 1) / 2
    sum_ij = comb(table).sum(); sum_a = comb(table.sum(1)).sum(); sum_b = comb(table.sum(0)).sum()
    expected = sum_a * sum_b / comb(n) if n > 1 else 0.0
    max_index = (sum_a + sum_b) / 2
    ari = (sum_ij - expected) / (max_index - expected) if max_index != expected else 1.0
    pa, pb, pab = table.sum(1) / n, table.sum(0) / n, table / n
    h = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))
    mi = np.sum(pab[pab > 0] * np.log(pab[pab > 0] / np.outer(pa, pb)[pab > 0]))
    nmi = mi / np.sqrt(h(pa) * h(pb)) if h(pa) > 0 and h(pb) > 0 else 0.0
    return float(ari), float(nmi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/h39c/cap_m2k32"))
    parser.add_argument("--h39d", type=Path, default=Path("reports/h39d_capacity.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/h47_baselines.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2)
    h39d = read_json(args.h39d)
    out = {}
    for world in (0, 1, 2):
        path = args.root / f"world_{world}" / "lifecycle"
        if read_json(path / "rho_profile.json").get("h39_pilot") != RECORD:
            raise SystemExit(f"record mismatch at {path}")
        state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
        history = torch.load(path / "history.pt", weights_only=True)
        generated = generate_meta_world(replace(config.world, seed=world, tasks=spec.total_tasks), spec)
        fam = {t.task_id: spec.family_of(i) for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None}
        late, early = {}, {}
        for tid, family in fam.items():
            late[tid] = two_slot_stats(state[f"task_codes.{tid}"], 3, 12)
            early[tid] = two_slot_stats(history["codes"][tid], 3, 12)
        families = [fam[t] for t in fam]
        dom_late = [late[t]["dominant_slot"] for t in fam]
        dom_early = [early[t]["dominant_slot"] for t in fam]
        consistency = {}
        for f in range(4):
            members = [late[t]["dominant_slot"] for t in fam if fam[t] == f]
            majority = max(set(members), key=members.count)
            consistency[f] = {"majority_slot": majority,
                              "fraction_on_majority": members.count(majority) / len(members)}
        ent_late = np.array([late[t]["entropy_at_dominant_step"] for t in fam])
        ent_early = np.array([early[t]["entropy_at_dominant_step"] for t in fam])
        mar_late = np.array([late[t]["margin_at_dominant_step"] for t in fam])
        mar_early = np.array([early[t]["margin_at_dominant_step"] for t in fam])
        mass_late = np.array([late[t]["mass_total_at_dominant_step"] for t in fam])
        pct = lambda a: {f"p{q}": float(np.percentile(a, q)) for q in (10, 25, 50, 75, 90)}
        arm = h39d["worlds"][str(world)]["arms"]["m2k32"]
        ordinary_loss = h39d["worlds"][str(world)]["ordinary_loss"]
        out[world] = {
            "route_entropy_bits_late": {"mean": float(ent_late.mean()), **pct(ent_late)},
            "route_entropy_bits_early": {"mean": float(ent_early.mean()), **pct(ent_early)},
            "route_margin_late": {"mean": float(mar_late.mean()), **pct(mar_late)},
            "route_margin_early": {"mean": float(mar_early.mean()), **pct(mar_early)},
            "parameterized_mass_at_dominant_step_late": {"mean": float(mass_late.mean()), **pct(mass_late)},
            "family_consistency_late": consistency,
            "mean_family_consistency_late": float(np.mean([c["fraction_on_majority"] for c in consistency.values()])),
            "ari_nmi_late": ari_nmi(families, dom_late),
            "ari_nmi_early": ari_nmi(families, dom_early),
            "dominant_slot_changed_early_to_late": int(sum(a != b for a, b in zip(dom_early, dom_late))),
            "J_M": arm["loss"], "J_O": ordinary_loss, "J_gap_M_minus_O": arm["loss_gap_nats"],
            "R_alpha_M": arm["alpha_only_k128_ratio"], "R_full_M": arm["full_k128_ratio"],
            "alpha_zeroed_ratio_M": arm["channel_use"]["alpha_zeroed_ratio"],
        }
        print(f"world {world}: entropy late mean {ent_late.mean():.3f} (p50 {np.median(ent_late):.3f}) early {ent_early.mean():.3f}; "
              f"margin late p50 {np.median(mar_late):.3f}; consistency {out[world]['mean_family_consistency_late']:.2f}; "
              f"ARI {out[world]['ari_nmi_late'][0]:.2f} NMI {out[world]['ari_nmi_late'][1]:.2f}; "
              f"J gap {arm['loss_gap_nats']:+.0f}; R_alpha {arm['alpha_only_k128_ratio']:.3f}; R_full {arm['full_k128_ratio']:.3f}")
    cross = {k: {"mean": float(np.mean([out[w][k] for w in out])), "sd": float(np.std([out[w][k] for w in out]))}
             for k in ("J_gap_M_minus_O", "R_alpha_M", "R_full_M")}
    cross["log_R_alpha_M"] = {"mean": float(np.mean([np.log(out[w]["R_alpha_M"]) for w in out])),
                              "sd": float(np.std([np.log(out[w]["R_alpha_M"]) for w in out]))}
    cross["log_R_full_M"] = {"mean": float(np.mean([np.log(out[w]["R_full_M"]) for w in out])),
                             "sd": float(np.std([np.log(out[w]["R_full_M"]) for w in out]))}
    report = {"purpose": "H47 tolerance and annealing baselines; no decision", "worlds": out, "cross_world": cross}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print("cross-world:", json.dumps(cross))


if __name__ == "__main__":
    main()
