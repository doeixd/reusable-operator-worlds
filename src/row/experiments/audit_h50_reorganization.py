"""H50 propose -> reorganize -> score (H50_REORGANIZATION_PLAN.md, Am. 1-2).

Six external candidates (TRUE, WRONG-A/B, RANDOM-1/2 with H49's seeds,
SHAM) are each migrated from the SAME frozen M_4 checkpoint under an
identical bounded budget (unfreeze only U_k and family-task local state;
retrospective sweeps over the 64 family tasks in lifetime order, one Adam
step per task per pass, seeded batches shared across arms), then scored
with the H49 LOO instrument on past data only. m in {4, 16, 64} migrated;
{16, 64} scored (m = 4 scored only if SEPARATION holds at 16); m = 0 rows
come from reports/h49_discoverability.json. Load-bearing comparison:
TRUE > max(WRONG/RANDOM) by +0.15 log C_LOO in >= 2/3 worlds, plus
TRUE > SHAM (+0.15) as the optimization-credit control. Recovery fraction
against L_4 is primary-descriptive. Sibling endpoints are computed only
after past-data decisions are written. Fails closed; atomic report.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_h49_discoverability import refit
from row.experiments.score_h39b_pslot import load_pslot, read_json
from row.experiments.score_h39b_pslot import factorized_fit
from row.meta_world import MetaFamilySpec, generate_meta_world

WORLDS = (0, 1, 2)
BUDGETS = (4, 16, 64)
SCORED = (16, 64)
MARGIN = 0.15
SUBST_MARGIN = 0.30
BASE = {"model": "pslot", "snapshot_history": True, "schema_groups": 2, "slot_args": 4,
        "freeze_args": False, "freeze_matrices": False, "pslot_index": 11, "pslot_count": 2}
PAIRINGS = {"TRUE": ((0, 1), (2, 3)), "WRONG-A": ((0, 2), (1, 3)), "WRONG-B": ((0, 3), (1, 2))}
ARMS = ("TRUE", "WRONG-A", "WRONG-B", "RANDOM-1", "RANDOM-2", "SHAM")
H49_KEY = {"TRUE": "TRUE", "WRONG-A": "WRONG-A", "WRONG-B": "WRONG-B",
           "RANDOM-1": "RANDOM-1", "RANDOM-2": "RANDOM-2", "SHAM": "DISTRIBUTED"}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def assignments(family_of_task: dict[str, int], world: int) -> dict[str, dict[str, int | None]]:
    out = {}
    for name, (a, b) in PAIRINGS.items():
        out[name] = {t: (0 if f in a else 1) for t, f in family_of_task.items()}
    ids = sorted(family_of_task)
    for r in (1, 2):
        rng = np.random.default_rng(np.random.SeedSequence([49, world, r]))  # Amendment 2: H49's partitions
        perm = rng.permutation(len(ids))
        out[f"RANDOM-{r}"] = {ids[i]: (0 if rank < len(ids) // 2 else 1) for rank, i in enumerate(perm)}
    out["SHAM"] = {t: None for t in family_of_task}
    return out


def migrate(model, tasks_in_order, assignment, passes: int, data_rng: np.random.Generator):
    """In place: `passes` sweeps, one Adam step per task per pass."""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = [model.argument_matrices]
    if model.pslot_count > 1:
        trainable.append(model.extra_argument_matrices)
    for p in trainable:
        p.requires_grad_(True)
    groups = [{"params": trainable, "lr": 0.003, "weight_decay": 1e-4}]
    task_params = []
    for task in tasks_in_order:
        tid = task.task_id
        model.retired.discard(tid)
        model.task_mask.pop(tid, None)
        position = migrate.assignment[tid]
        if position is not None:
            model.task_mask[tid] = int(model.pslot_indices[position])
        code, alpha, residual = model.task_codes[tid], model.task_alphas[tid], model.task_residuals[tid]
        for p in (code, alpha, residual):
            p.requires_grad_(True)
        groups.append({"params": [code, alpha], "lr": 0.05, "weight_decay": 0.0})
        groups.append({"params": [residual], "lr": 0.01, "weight_decay": 0.0})
        task_params.append((tid, task))
    optimizer = torch.optim.AdamW(groups)
    steps = 0
    for _ in range(passes):
        for tid, task in task_params:
            idx = data_rng.integers(0, task.train_x.shape[0], 8)
            x = torch.tensor(task.train_x[idx], dtype=torch.float32)
            y = torch.tensor(task.train_y[idx], dtype=torch.float32)
            optimizer.zero_grad()
            loss = torch.mean((model(x, tid) - y) ** 2)
            if not bool(torch.isfinite(loss)):
                raise SystemExit(f"non-finite migration loss at step {steps}")
            loss.backward()
            optimizer.step()
            steps += 1
    return steps


@torch.no_grad()
def family_nmse(model, tasks) -> float:
    vals = []
    for task in tasks:
        x = torch.tensor(task.eval_x, dtype=torch.float32)
        y = torch.tensor(task.eval_y, dtype=torch.float32)
        vals.append(float(torch.mean((model(x, task.task_id) - y) ** 2) / (torch.var(y, unbiased=False) + 1e-12)))
    return float(np.mean(vals))


@torch.no_grad()
def drift(model, reference, tasks, probe) -> float:
    vals = []
    for task in tasks[:8]:
        vals.append(float(torch.mean((model(probe, task.task_id) - reference(probe, task.task_id)) ** 2)))
    return float(np.mean(vals))


def score_loo(model, tasks, assignment, label: str) -> dict:
    fits = [refit(model, t, assignment[t.task_id], label) for t in tasks]
    bad = [f for f in fits if not (f["finite"] and f["alpha_norm"] > 0 and f["support_reduction"] > 0.01)]
    if bad:
        raise SystemExit(f"non-vacuity failed in LOO {label}: {bad[:2]}")
    return {"C_LOO": float(np.exp(np.mean(np.log([f["nmse"] for f in fits])))),
            "D_star_nats": float(np.mean([f["description_nats"] for f in fits])), "fits": fits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--h49", type=Path, default=Path("reports/h49_discoverability.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/h50_reorganization.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2, schema_groups=2)
    h49 = read_json(args.h49)
    worlds_out = {}
    for world in WORLDS:
        generated = generate_meta_world(replace(config.world, seed=world, tasks=spec.total_tasks), spec)
        family_tasks = [t for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None]
        family_of_task = {t.task_id: spec.family_of(i) for i, t in enumerate(generated.tasks)
                          if spec.family_of(i) is not None}
        futures = list(generated.novel_family_tasks)
        cands = assignments(family_of_task, world)
        base = load_pslot(config, Path("artifacts/h39c/w_m4") / f"world_{world}" / "lifecycle", BASE)
        probe = torch.tensor(np.random.default_rng(np.random.SeedSequence([50, world, 7])).normal(
            size=(128, config.world.state_dim)), dtype=torch.float32)
        m0 = h49["worlds"][str(world)]["M4"]["candidates"]
        arms_out = {}
        for arm in ARMS:
            data_rng = np.random.default_rng(np.random.SeedSequence([50, world, 11]))  # shared across arms
            model = copy.deepcopy(base)
            migrate.assignment = cands[arm]
            snapshots = {}
            done = 0
            t0 = time.time()
            for m in BUDGETS:
                migrate(model, family_tasks, cands[arm], m - done, data_rng)
                done = m
                snapshots[m] = copy.deepcopy(model)
            wall = time.time() - t0
            row = {"m0": {"C_LOO": m0[H49_KEY[arm]]["C_LOO"], "D_star_nats": m0[H49_KEY[arm]]["D_star_nats"]},
                   "migration": {"steps_per_budget": {m: m * 64 for m in BUDGETS}, "wall_clock_s": wall}}
            for m in BUDGETS:
                row["migration"][f"family_nmse_m{m}"] = family_nmse(snapshots[m], family_tasks)
                row["migration"][f"drift_m{m}"] = drift(snapshots[m], base, family_tasks, probe)
            for m in SCORED:
                s = score_loo(snapshots[m], family_tasks, cands[arm], f"h50_{arm}_w{world}_m{m}")
                row[f"m{m}"] = {"C_LOO": s["C_LOO"], "D_star_nats": s["D_star_nats"]}
                row[f"m{m}_fits"] = s["fits"]
                print(f"world {world} {arm:9s} m={m:2d}: C_LOO {s['C_LOO']:.5f} "
                      f"(m0 {row['m0']['C_LOO']:.5f}) fam-NMSE {row['migration'][f'family_nmse_m{m}']:.4f}", flush=True)
            arms_out[arm] = row
            arms_out[arm]["snapshots"] = snapshots  # in-memory only; stripped before write
        # substitutability for TRUE, best wrong, SHAM at scored budgets
        for m in SCORED:
            wrongs = {a: arms_out[a][f"m{m}"]["C_LOO"] for a in ("WRONG-A", "WRONG-B", "RANDOM-1", "RANDOM-2")}
            best_wrong = min(wrongs, key=wrongs.get)
            for arm in ("TRUE", best_wrong):
                comp = {t: 1 - p for t, p in cands[arm].items()}
                other = [refit(arms_out[arm]["snapshots"][m], t, comp[t.task_id], f"h50sub_{arm}_w{world}_m{m}")
                         for t in family_tasks]
                own = arms_out[arm][f"m{m}_fits"]
                arms_out[arm][f"m{m}"]["S_subst"] = float(np.mean(
                    [np.log(o["nmse"]) - np.log(w["nmse"]) for o, w in zip(other, own)]))
            arms_out["SHAM"][f"m{m}"]["S_subst"] = None
            arms_out[f"best_wrong_m{m}"] = best_wrong
        # sibling endpoints (after past-data quantities exist), diagnostic
        for m in (64,):
            for arm in ("TRUE", arms_out["best_wrong_m64"], "SHAM"):
                model = arms_out[arm]["snapshots"][m]
                vals = []
                for index, task in enumerate(futures):
                    fit = factorized_fit(model, task, 128, "alpha_only", "adam", 0.01, 2000,
                                         f"h50sib_{arm}_w{world}_t{index}")
                    vals.append(fit["final_query_scaled"])
                arms_out[arm]["m64"]["sibling_alpha_k128"] = float(np.mean(vals))
        for arm in ARMS:
            arms_out[arm].pop("snapshots", None)
            for m in SCORED:
                arms_out[arm].pop(f"m{m}_fits", None)
        worlds_out[world] = {"arms": arms_out,
                             "L4_reference": {"margin": h49["worlds"][str(world)]["L4"]["margin_vs_best_wrong"],
                                              "S_subst": h49["worlds"][str(world)]["L4"]["S_subst"]}}
    # ---- decisions -------------------------------------------------------
    decisions = {}
    for m in SCORED:
        rows = {}
        for world in WORLDS:
            arms = worlds_out[world]["arms"]
            true_c = arms["TRUE"][f"m{m}"]["C_LOO"]
            wrong_min = min(arms[a][f"m{m}"]["C_LOO"] for a in ("WRONG-A", "WRONG-B", "RANDOM-1", "RANDOM-2"))
            rows[world] = {"margin_vs_best_wrong": float(np.log(wrong_min) - np.log(true_c)),
                           "margin_vs_sham": float(np.log(arms["SHAM"][f"m{m}"]["C_LOO"]) - np.log(true_c))}
        sep = (sum(rows[w]["margin_vs_best_wrong"] >= MARGIN for w in WORLDS) >= 2
               and sum(rows[w]["margin_vs_sham"] >= MARGIN for w in WORLDS) >= 2)
        subst_ok = True
        for world in WORLDS:
            arms = worlds_out[world]["arms"]
            bw = arms[f"best_wrong_m{m}"]
            s_true = arms["TRUE"][f"m{m}"].get("S_subst")
            s_bw = arms[bw][f"m{m}"].get("S_subst")
            if s_true is None or s_bw is None or (s_true - s_bw) < SUBST_MARGIN:
                subst_ok = subst_ok and False
        decisions[f"m{m}"] = {"per_world": rows, "separation_cloo": bool(sep),
                              "substitutability_corroborates": bool(subst_ok),
                              "SEPARATION": bool(sep and subst_ok)}
    m_star = next((m for m in SCORED if decisions[f"m{m}"]["SEPARATION"]), None)
    recovery = {}
    for world in WORLDS:
        arms = worlds_out[world]["arms"]
        ref = worlds_out[world]["L4_reference"]["margin"]
        base_margin = 0.059 if world == 0 else (-0.034 if world == 1 else -0.043)  # H49 M4 m=0 margins
        recovery[world] = {f"m{m}": (decisions[f"m{m}"]["per_world"][world]["margin_vs_best_wrong"] - base_margin)
                           / (ref - base_margin) for m in SCORED}
    if m_star is not None:
        outcome = "DISCOVERABLE-BY-REORGANIZATION"
    elif all(not decisions[f"m{m}"]["separation_cloo"] for m in SCORED):
        sham_gain = np.mean([np.log(worlds_out[w]["arms"]["SHAM"]["m0"]["C_LOO"])
                             - np.log(worlds_out[w]["arms"]["SHAM"]["m64"]["C_LOO"]) for w in WORLDS])
        outcome = "OPTIMIZATION-ONLY" if sham_gain > 0.3 else "UNDERDETERMINED"
    else:
        outcome = "MIXED (C_LOO separates; substitutability does not corroborate)"
    report = {"frozen_plan": "H50_REORGANIZATION_PLAN.md (Amendments 1-2)", "git_commit": git_commit(),
              "protocol": {"budgets_migrated": list(BUDGETS), "budgets_scored": list(SCORED),
                           "margin": MARGIN, "subst_margin": SUBST_MARGIN,
                           "migration": {"optimizer": "adamw", "lrs": {"task": 0.05, "residual": 0.01, "Uk": 0.003},
                                         "batch": 8, "order": "lifetime", "data_seed": [50, "world", 11]}},
              "worlds": worlds_out, "decisions": decisions, "m_star": m_star,
              "recovery_fraction_cloo_margin": recovery, "outcome": outcome}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    for m in SCORED:
        d = decisions[f"m{m}"]
        print(f"m={m}: margins vs best wrong {[round(d['per_world'][w]['margin_vs_best_wrong'], 3) for w in WORLDS]} "
              f"vs SHAM {[round(d['per_world'][w]['margin_vs_sham'], 3) for w in WORLDS]} SEPARATION={d['SEPARATION']}")
    print(f"recovery {json.dumps({str(w): {k: round(v, 2) for k, v in recovery[w].items()} for w in recovery})}")
    print(f"m* = {m_star}; OUTCOME {outcome}")


if __name__ == "__main__":
    main()
