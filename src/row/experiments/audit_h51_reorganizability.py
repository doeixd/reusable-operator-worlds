"""H51 reorganizability testbed (H51_REORGANIZABILITY_PLAN.md, Am. 1-3).

H50 showed that bounded migration of the ordinary finished representation
recovers ~0% of L_4's retrospective separation. H51 asks whether that is a
property of finished representations in general or of THIS wake, by holding the
migration operator and the scorer fixed and varying only the WAKE
REPRESENTATION:

    R_0   ordinary            (H50's result, re-used)
    R_1a  trace initialization  -- each family task's local state initialized
                                  from its task-completion provenance trace
    R_1b  trace recombination   -- residual = sum_j c_j trace_j + eps over the
                                  32 traces of its cell under the candidate
                                  (SHAM: self + 31 random others; counts equal)
    R_2   decomposable wake     -- a pooled 4-component innovation basis learned
                                  during the lifetime (`pslot_factorized`)
    R_3   oracle organized      (L_4, the recovery reference; H49's rows)

Endpoint: `C_restructure(R)` = the smallest scored migration budget at which
H50's SEPARATION rule holds. Everything else -- candidates, optimizer, learning
rates, seeded batches, LOO instrument, margins -- is H50 verbatim.

Fails closed; report written atomically before any summary is printed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from row.config import load_config
from row.experiments.audit_h49_discoverability import refit
from row.experiments.audit_h50_reorganization import (
    ARMS, H49_KEY, MARGIN, SUBST_MARGIN, WORLDS, assignments, drift, family_nmse,
    git_commit, migrate, score_loo,
)
from row.experiments.score_h39b_pslot import factorized_fit, load_pslot, read_json
from row.meta_world import MetaFamilySpec, generate_meta_world
from row.models.pslot_models import ParameterizedSlotLearner
from row.models.pslot_factorized_models import PslotFactorizedLearner

# Amendment 3: m = 64 is scored first for every arm; m = 16 is scored only for
# an arm that separates at 64. H50 found flat margins in m, so the question is
# whether ANY representation separates at all. The LOO sample is never thinned.
BUDGETS = (4, 16, 64)
PRIMARY = 64
BASE_PSLOT = {"model": "pslot", "snapshot_history": True, "schema_groups": 2, "slot_args": 4,
              "freeze_args": False, "freeze_matrices": False, "pslot_index": 11, "pslot_count": 2}
BASE_R2 = dict(BASE_PSLOT, model="pslot_factorized", schema_dim=4, schema_count=1)
# H49's M_4 m = 0 margins: the COMMON baseline every arm's recovery is measured
# from (the point ordinary wake leaves the learner at).
R0_BASE_MARGIN = {0: 0.059, 1: -0.034, 2: -0.043}
TRACE_BASIS = 32


class TraceRecombiningLearner(ParameterizedSlotLearner):
    """R_1b: a task's residual is a learned mixture of provenance traces.

    `residual_i = sum_j c_ij trace_j + eps_i` over the 32 traces in the task's
    cell under the candidate partition. The wrong and random partitions give
    each task the same number of coefficients over a differently chosen trace
    set, so they are the matched-budget control by construction.
    """

    def attach_traces(self, basis: dict[str, Tensor]) -> None:
        self.trace_basis = {t: b for t, b in basis.items()}
        self.trace_coefficients = nn.ParameterDict(
            {t: nn.Parameter(torch.zeros(b.shape[1])) for t, b in basis.items()}
        )

    def effective_residual(self, task_id: str) -> Tensor:
        own = self.task_residuals[task_id]
        if task_id not in getattr(self, "trace_basis", {}):
            return own
        return own + self.trace_basis[task_id] @ self.trace_coefficients[task_id]

    def _unpack(self, task_id: str):
        route = self.task_codes[task_id]
        u, v, b = torch.split(
            self.effective_residual(task_id),
            (self.residual_u_size, self.residual_v_size, self.residual_b_size),
        )
        return (route.reshape(self.task_steps, self.operator_slots),
                u.reshape(self.task_steps, self.d, self.residual_rank),
                v.reshape(self.task_steps, self.residual_rank, self.d),
                b.reshape(self.task_steps, self.residual_rank))


def load_traces(path: Path) -> dict[str, dict[str, Tensor]]:
    history = torch.load(path / "history.pt", weights_only=True)
    return {"residuals": history["residuals"], "codes": history["codes"], "eps": history["eps"]}


def build_r1a(base, traces, family_tasks):
    """Initialize every family task's route code and eps from its trace."""
    model = copy.deepcopy(base)
    used = 0
    with torch.no_grad():
        for task in family_tasks:
            tid = task.task_id
            if tid in traces["eps"] and tid in traces["codes"]:
                model.task_residuals[tid].copy_(traces["eps"][tid])
                model.task_codes[tid].copy_(traces["codes"][tid])
                used += 1
    if used != len(family_tasks):
        raise SystemExit(f"R_1a: {used}/{len(family_tasks)} traces found")
    return model


def build_r1b(base, traces, family_tasks, assignment, world: int):
    model = copy.deepcopy(base)
    model.__class__ = TraceRecombiningLearner
    ids = [t.task_id for t in family_tasks]
    rng = np.random.default_rng(np.random.SeedSequence([51, world, 13]))
    basis = {}
    for tid in ids:
        cell = assignment.get(tid)
        if cell is None:                                   # SHAM: self + 31 random others
            others = [o for o in ids if o != tid]
            chosen = [tid] + [others[i] for i in rng.permutation(len(others))[:TRACE_BASIS - 1]]
        else:
            chosen = [o for o in ids if assignment[o] == cell]
        if len(chosen) != TRACE_BASIS:
            raise SystemExit(f"R_1b: trace basis {len(chosen)} != {TRACE_BASIS} for {tid}")
        basis[tid] = torch.stack([traces["residuals"][o] for o in chosen], dim=1)
    model.attach_traces(basis)
    return model


def migrate_arm(model, family_tasks, assignment, passes, data_rng, extra_task_params=None):
    """H50's migrate(), plus any representation-specific task-local variables.

    The extra variables enter the SAME task learning-rate group, so the step
    count and the optimizer are unchanged; only the arm's own state is larger.
    """
    migrate.assignment = assignment
    if extra_task_params is None:
        return migrate(model, family_tasks, assignment, passes, data_rng)
    # Replicate H50's migrate with the extra parameters attached.
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = [model.argument_matrices]
    if model.pslot_count > 1:
        trainable.append(model.extra_argument_matrices)
    for p in trainable:
        p.requires_grad_(True)
    groups = [{"params": trainable, "lr": 0.003, "weight_decay": 1e-4}]
    task_params = []
    for task in family_tasks:
        tid = task.task_id
        model.retired.discard(tid)
        model.task_mask.pop(tid, None)
        position = assignment[tid]
        if position is not None:
            model.task_mask[tid] = int(model.pslot_indices[position])
        code, alpha, residual = model.task_codes[tid], model.task_alphas[tid], model.task_residuals[tid]
        fast = [code, alpha] + [p for p in extra_task_params(model, tid)]
        for p in fast + [residual]:
            p.requires_grad_(True)
        groups.append({"params": fast, "lr": 0.05, "weight_decay": 0.0})
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


def run_representation(name, world, config, spec, generated, base_dir, h49, budgets_scored):
    """One representation, all six candidates, the H50 protocol verbatim."""
    family_tasks = [t for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None]
    family_of_task = {t.task_id: spec.family_of(i) for i, t in enumerate(generated.tasks)
                      if spec.family_of(i) is not None}
    futures = list(generated.novel_family_tasks)
    cands = assignments(family_of_task, world)
    probe = torch.tensor(np.random.default_rng(np.random.SeedSequence([50, world, 7])).normal(
        size=(128, config.world.state_dim)), dtype=torch.float32)
    traces = load_traces(base_dir) if name.startswith("R_1") else None
    base = load_pslot(config, base_dir, BASE_R2 if name == "R_2" else BASE_PSLOT, world_seed=world)
    m0_rows = h49["worlds"][str(world)]["M4"]["candidates"]
    arms_out = {}
    for arm in ARMS:
        data_rng = np.random.default_rng(np.random.SeedSequence([50, world, 11]))
        extra = None
        if name == "R_1a":
            model = build_r1a(base, traces, family_tasks)
        elif name == "R_1b":
            model = build_r1b(base, traces, family_tasks, cands[arm], world)
            extra = lambda m, tid: [m.trace_coefficients[tid]]
        else:
            model = copy.deepcopy(base)
            if name == "R_2":
                extra = lambda m, tid: [m.schema_alphas[tid]]
        reference = copy.deepcopy(model)
        snapshots, done, t0 = {}, 0, time.time()
        for m in BUDGETS:
            migrate_arm(model, family_tasks, cands[arm], m - done, data_rng, extra)
            done = m
            snapshots[m] = copy.deepcopy(model)
        wall = time.time() - t0
        row = {"migration": {"steps_per_budget": {m: m * 64 for m in BUDGETS}, "wall_clock_s": wall}}
        for m in BUDGETS:
            row["migration"][f"family_nmse_m{m}"] = family_nmse(snapshots[m], family_tasks)
            row["migration"][f"drift_m{m}"] = drift(snapshots[m], reference, family_tasks, probe)
        row["migration"]["family_nmse_m0"] = family_nmse(reference, family_tasks)
        if name in {"R_0", "R_1b"}:
            # m = 0 is bitwise the ordinary representation (R_1b's coefficients
            # start at zero), so H49's measured row is the arm's own baseline.
            row["m0"] = {"C_LOO": m0_rows[H49_KEY[arm]]["C_LOO"],
                         "D_star_nats": m0_rows[H49_KEY[arm]]["D_star_nats"], "source": "h49"}
        else:
            s0 = score_loo(reference, family_tasks, cands[arm], f"h51_{name}_{arm}_w{world}_m0")
            row["m0"] = {"C_LOO": s0["C_LOO"], "D_star_nats": s0["D_star_nats"], "source": "measured"}
        for m in budgets_scored:
            s = score_loo(snapshots[m], family_tasks, cands[arm], f"h51_{name}_{arm}_w{world}_m{m}")
            row[f"m{m}"] = {"C_LOO": s["C_LOO"], "D_star_nats": s["D_star_nats"]}
            row[f"m{m}_fits"] = s["fits"]
            print(f"[{name}] world {world} {arm:9s} m={m:2d}: C_LOO {s['C_LOO']:.5f} "
                  f"(m0 {row['m0']['C_LOO']:.5f}) fam-NMSE {row['migration'][f'family_nmse_m{m}']:.4f}",
                  flush=True)
        row["snapshots"] = snapshots
        arms_out[arm] = row
    for m in budgets_scored:
        wrongs = {a: arms_out[a][f"m{m}"]["C_LOO"] for a in ("WRONG-A", "WRONG-B", "RANDOM-1", "RANDOM-2")}
        best_wrong = min(wrongs, key=wrongs.get)
        for arm in ("TRUE", best_wrong):
            comp = {t: (1 - p if p is not None else None) for t, p in cands[arm].items()}
            other = [refit(arms_out[arm]["snapshots"][m], t, comp[t.task_id], f"h51sub_{name}_{arm}_w{world}_m{m}")
                     for t in family_tasks]
            own = arms_out[arm][f"m{m}_fits"]
            arms_out[arm][f"m{m}"]["S_subst"] = float(np.mean(
                [np.log(o["nmse"]) - np.log(w["nmse"]) for o, w in zip(other, own)]))
        arms_out["SHAM"][f"m{m}"]["S_subst"] = None
        arms_out[f"best_wrong_m{m}"] = best_wrong
    for arm in ("TRUE", arms_out[f"best_wrong_m{PRIMARY}"], "SHAM"):
        vals = []
        for index, task in enumerate(futures):
            fit = factorized_fit(arms_out[arm]["snapshots"][PRIMARY], task, 128, "alpha_only", "adam",
                                 0.01, 2000, f"h51sib_{name}_{arm}_w{world}_t{index}")
            vals.append(fit["final_query_scaled"])
        arms_out[arm][f"m{PRIMARY}"]["sibling_alpha_k128"] = float(np.mean(vals))
    for arm in ARMS:
        arms_out[arm].pop("snapshots", None)
        for m in budgets_scored:
            arms_out[arm].pop(f"m{m}_fits", None)
    return arms_out


def decide(arms_by_world, m, worlds=WORLDS):
    rows = {}
    for world in worlds:
        arms = arms_by_world[world]
        true_c = arms["TRUE"][f"m{m}"]["C_LOO"]
        wrong_min = min(arms[a][f"m{m}"]["C_LOO"] for a in ("WRONG-A", "WRONG-B", "RANDOM-1", "RANDOM-2"))
        rows[world] = {"margin_vs_best_wrong": float(np.log(wrong_min) - np.log(true_c)),
                       "margin_vs_sham": float(np.log(arms["SHAM"][f"m{m}"]["C_LOO"]) - np.log(true_c))}
    sep = (sum(rows[w]["margin_vs_best_wrong"] >= MARGIN for w in worlds) >= 2
           and sum(rows[w]["margin_vs_sham"] >= MARGIN for w in worlds) >= 2)
    subst_ok = True
    for world in worlds:
        arms = arms_by_world[world]
        bw = arms[f"best_wrong_m{m}"]
        s_true, s_bw = arms["TRUE"][f"m{m}"].get("S_subst"), arms[bw][f"m{m}"].get("S_subst")
        if s_true is None or s_bw is None or (s_true - s_bw) < SUBST_MARGIN:
            subst_ok = False
    return {"per_world": rows, "separation_cloo": bool(sep),
            "substitutability_corroborates": bool(subst_ok), "SEPARATION": bool(sep and subst_ok)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--h49", type=Path, default=Path("reports/h49_discoverability.json"))
    parser.add_argument("--h50", type=Path, default=Path("reports/h50_reorganization.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/h51_reorganizability.json"))
    parser.add_argument("--representations", nargs="+", default=["R_1a", "R_1b", "R_2"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2, schema_groups=2)
    h49, h50 = read_json(args.h49), read_json(args.h50)
    out = {"frozen_plan": "H51_REORGANIZABILITY_PLAN.md (Amendments 1-3)", "git_commit": git_commit(),
           "protocol": {"budgets_migrated": list(BUDGETS), "primary_budget": PRIMARY,
                        "margin": MARGIN, "subst_margin": SUBST_MARGIN,
                        "baseline_margin_R0_m0": R0_BASE_MARGIN,
                        "instrument": "H50 migrate() and H49 refit(), unchanged"},
           "representations": {}}
    for name in args.representations:
        per_world, gates = {}, {}
        for world in WORLDS:
            generated = generate_meta_world(replace(config.world, seed=world, tasks=spec.total_tasks), spec)
            base_dir = (Path("artifacts/h51/h51_r2") / f"world_{world}" / "lifecycle" if name == "R_2"
                        else Path("artifacts/h39c/w_m4") / f"world_{world}" / "lifecycle")
            if not (base_dir / "summary.json").exists():
                raise SystemExit(f"missing artifact for {name} world {world}: {base_dir}")
            per_world[world] = run_representation(name, world, config, spec, generated, base_dir,
                                                  h49, (PRIMARY,))
            gates[world] = balance_gates(name, world, base_dir)
        decisions = {f"m{PRIMARY}": decide(per_world, PRIMARY)}
        if decisions[f"m{PRIMARY}"]["SEPARATION"]:
            for world in WORLDS:                                   # locate the transition
                generated = generate_meta_world(replace(config.world, seed=world, tasks=spec.total_tasks), spec)
                base_dir = (Path("artifacts/h51/h51_r2") / f"world_{world}" / "lifecycle" if name == "R_2"
                            else Path("artifacts/h39c/w_m4") / f"world_{world}" / "lifecycle")
                extra_rows = run_representation(name, world, config, spec, generated,
                                                base_dir, h49, (16,))
                for key, value in extra_rows.items():
                    if key in per_world[world] and isinstance(value, dict):
                        per_world[world][key].update(value)   # merge m16 into the arm row
                    else:
                        per_world[world][key] = value
            decisions["m16"] = decide(per_world, 16)
        scored = sorted(int(k[1:]) for k in decisions)
        c_restructure = next((m for m in scored if decisions[f"m{m}"]["SEPARATION"]), None)
        recovery = {}
        for world in WORLDS:
            ref = h49["worlds"][str(world)]["L4"]["margin_vs_best_wrong"]
            b = R0_BASE_MARGIN[world]
            recovery[world] = {f"m{m}": (decisions[f"m{m}"]["per_world"][world]["margin_vs_best_wrong"] - b) / (ref - b)
                               for m in scored}
        out["representations"][name] = {
            "worlds": {str(w): {"arms": per_world[w], "balance_gates": gates[w]} for w in WORLDS},
            "decisions": decisions, "C_restructure": c_restructure,
            "recovery_fraction_cloo_margin": recovery,
        }
        print(f"[{name}] C_restructure = {c_restructure}; recovery "
              f"{json.dumps({str(w): {k: round(v, 2) for k, v in recovery[w].items()} for w in recovery})}",
              flush=True)
    # R_0 row, re-used from H50 for the comparison table
    out["representations"]["R_0"] = {"source": "reports/h50_reorganization.json",
                                     "C_restructure": h50["m_star"],
                                     "recovery_fraction_cloo_margin": h50["recovery_fraction_cloo_margin"],
                                     "decisions": h50["decisions"]}
    out["outcome"] = verdict(out["representations"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(f"OUTCOME {out['outcome']}")


def balance_gates(name, world, base_dir) -> dict:
    """G1-G3 against the ordinary R_0 artifact for the same world."""
    r0 = read_json(Path("artifacts/h39c/w_m4") / f"world_{world}" / "lifecycle" / "summary.json")
    if name != "R_2":
        return {"identical_to_R0": True, "note": "R_1 arms share R_0's artifact bitwise"}
    arm = read_json(base_dir / "summary.json")

    def get(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    loss0, loss = get(r0, "gaussian_log_loss", "cumulative_gaussian_log_loss"), \
        get(arm, "gaussian_log_loss", "cumulative_gaussian_log_loss")
    shared0, shared = get(r0, "shared_parameter_count"), get(arm, "shared_parameter_count")
    task0, task = get(r0, "task_state_scalar_count"), get(arm, "task_state_scalar_count")
    rel = lambda a, b: None if (a is None or b is None or a == 0) else abs(b - a) / abs(a)
    gates = {
        "G1_loss_rel": rel(loss0, loss), "G1_pass": (rel(loss0, loss) or 1.0) <= 0.10,
        "G3_shared_rel": rel(shared0, shared), "G3_total_rel": rel(
            None if shared0 is None or task0 is None else shared0 + task0,
            None if shared is None or task is None else shared + task),
        "R0": {"loss": loss0, "shared": shared0, "task_state": task0},
        "arm": {"loss": loss, "shared": shared, "task_state": task},
    }
    gates["G3_pass"] = ((gates["G3_shared_rel"] or 1.0) <= 0.20 and (gates["G3_total_rel"] or 1.0) <= 0.20)
    return gates


def verdict(reps) -> str:
    named = [n for n in ("R_1a", "R_1b", "R_2") if n in reps and "C_restructure" in reps[n]]
    if any(reps[n]["C_restructure"] is not None for n in named):
        return "REORGANIZABILITY-IS-REAL"
    best = {}
    for n in named + (["R_0"] if "R_0" in reps else []):
        rec = reps[n]["recovery_fraction_cloo_margin"]
        vals = [rec[str(w)][f"m{PRIMARY}"] if str(w) in rec else rec[w][f"m{PRIMARY}"] for w in WORLDS]
        best[n] = vals
    order = [n for n in ("R_0", "R_1a", "R_1b", "R_2") if n in best]
    graded = False
    for i in range(len(order) - 1):
        gain = [best[order[i + 1]][k] - best[order[i]][k] for k in range(len(WORLDS))]
        if sum(g >= 0.25 for g in gain) >= 2:
            graded = True
    return "PARTIAL" if graded else "FORMATION-TIME-CONFIRMED"


if __name__ == "__main__":
    main()
