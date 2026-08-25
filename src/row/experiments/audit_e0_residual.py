"""E0.2 / E7: where does task identity live? (E0_PHASE0_AUDIT_PLAN.md)

Four conditions per trained task:

    L_full         intact
    L_no_residual  the task's private residual disabled (library and route intact)
    L_no_library   the routed library contribution replaced by the identity,
                   private residual retained
    L_refit        the private residual re-fitted under the frozen library

    R_residual = (L_no_residual - L_full) / (L_no_library - L_full)

Near 0 means task identity lives in the reusable objects; near or above 1 means
the library is a prior and the task program still lives in private state. It is
a WARNING, never a stop -- E1 is the direct test.

Two of the three substrates have NO private residual channel: their whole task
state is the route. That is reported as a structural fact rather than as a
measured zero, and it is the strongest possible form of the answer for them.

Absolute losses accompany every ratio. Fails closed; atomic report.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_e0_export import SUBSTRATES, git_commit, load_model, world_of
from row.experiments.score_h39b_pslot import load_pslot
from row.meta_world import MetaFamilySpec, generate_meta_world

REFIT_STEPS = 2000
REFIT_LR = 0.01
M4_RECORD = {"model": "pslot", "snapshot_history": True, "schema_groups": 2, "slot_args": 4,
             "freeze_args": False, "freeze_matrices": False, "pslot_index": 11, "pslot_count": 2}


def variant_forward(model, x: torch.Tensor, task_id: str, use_library=True, use_residual=True):
    """The pslot-family forward with the library and the private residual switchable.

    Mirrors `ParameterizedSlotLearner.forward` exactly; `use_library=False`
    replaces the routed parent with the identity, `use_residual=False` drops the
    task's OWN innovation while keeping any promoted reference (which is part of
    the library, not of private state).
    """

    route, own_u, own_v, own_b = model._unpack(task_id)
    coefficients = model._coefficients(route, task_id)
    reference = model.task_reference.get(task_id)
    shared = model._split_residual(model.abstractions[reference]) if reference is not None else None
    retired = task_id in model.retired
    z = x
    for step in range(model.task_steps):
        if use_library:
            candidates = model._candidates(z, task_id)
            parent = torch.sum(
                coefficients[step].view(model.operator_slots, 1, 1) * candidates, dim=0)
        else:
            parent = z
        residual = torch.zeros_like(parent)
        if shared is not None:
            residual = residual + model._innovation(z, *shared, step)
        if use_residual and not retired:
            residual = residual + model._innovation(z, own_u, own_v, own_b, step)
        z = parent + residual
    return z


def nmse_of(pred, y) -> float:
    return float(torch.mean((pred - y) ** 2) / (torch.var(y, unbiased=False) + 1e-12))


def refit_residual(model, task, task_id) -> float:
    """Re-fit the task's private residual under the frozen library and route."""
    local = copy.deepcopy(model)
    for parameter in local.parameters():
        parameter.requires_grad_(False)
    local.retired.discard(task_id)
    residual = local.task_residuals[task_id]
    with torch.no_grad():
        residual.copy_(local.initial_residual_state)
    residual.requires_grad_(True)
    x = torch.tensor(task.train_x, dtype=torch.float32)
    y = torch.tensor(task.train_y, dtype=torch.float32)
    optimizer = torch.optim.Adam([residual], lr=REFIT_LR)
    for _ in range(REFIT_STEPS):
        optimizer.zero_grad()
        loss = torch.mean((variant_forward(local, x, task_id) - y) ** 2)
        if not bool(torch.isfinite(loss)):
            raise SystemExit(f"non-finite refit loss for {task_id}")
        loss.backward(inputs=[residual])
        optimizer.step()
    with torch.no_grad():
        return nmse_of(variant_forward(local, torch.tensor(task.eval_x, dtype=torch.float32),
                                       task_id),
                       torch.tensor(task.eval_y, dtype=torch.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--meta-config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e0_residual_audit.json"))
    parser.add_argument("--worlds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--refit", action="store_true", default=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    out = {"frozen_plan": "E0_PHASE0_AUDIT_PLAN.md", "git_commit": git_commit(),
           "protocol": {"refit_steps": REFIT_STEPS, "refit_lr": REFIT_LR,
                        "definition": "R_residual = (L_no_residual - L_full)/(L_no_library - L_full)"},
           "substrates": {}}

    # --- the two E1-relevant substrates: structural, not measured ------------
    for name in ("DISC", "MIX"):
        spec = SUBSTRATES[name]
        path = Path(spec["path"])
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        prefixes = sorted({key.split(".")[0] for key in state})
        has_residual = any(p in {"task_residuals", "residuals"} for p in prefixes)
        out["substrates"][name] = {
            "artifact": str(path), "state_prefixes": prefixes,
            "has_private_residual_channel": bool(has_residual),
            "R_residual": 0.0 if not has_residual else None,
            "note": ("no private residual channel exists: the entire task state is the route, "
                     "so task identity lives wholly in the routed library. Structural, not measured."),
        }
        print(f"[{name}] private residual channel: {has_residual} "
              f"(task state = {[p for p in prefixes if p.startswith('task')]})", flush=True)

    # --- the modern substrate, which does have residuals ---------------------
    config = load_config(args.meta_config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2,
                          schema_groups=2)
    rows = {}
    for world in args.worlds:
        path = Path("artifacts/h39c/w_m4") / f"world_{world}" / "lifecycle"
        if not (path / "summary.json").exists():
            raise SystemExit(f"missing artifact {path}")
        model = load_pslot(config, path, M4_RECORD, world_seed=world)
        generated = generate_meta_world(
            replace(config.world, seed=world, tasks=spec.total_tasks), spec)
        tasks = [t for t in generated.tasks if t.task_id in model.task_codes]
        full, no_residual, no_library, refits, retired = [], [], [], [], 0
        live = {"full": [], "no_residual": [], "no_library": []}
        for task in tasks:
            tid = task.task_id
            x = torch.tensor(task.eval_x, dtype=torch.float32)
            y = torch.tensor(task.eval_y, dtype=torch.float32)
            with torch.no_grad():
                a = nmse_of(variant_forward(model, x, tid), y)
                b = nmse_of(variant_forward(model, x, tid, use_residual=False), y)
                c = nmse_of(variant_forward(model, x, tid, use_library=False), y)
            full.append(a); no_residual.append(b); no_library.append(c)
            if tid in model.retired:
                retired += 1
            else:
                # A retired task has NO private residual to disable, so it
                # contributes a guaranteed zero to the numerator. With most
                # tasks retired the aggregate would be dominated by that
                # construction, so live tasks are reported separately.
                live["full"].append(a); live["no_residual"].append(b); live["no_library"].append(c)
        if args.refit:
            for task in tasks:
                refits.append(refit_residual(model, task, task.task_id))
        L_full, L_nr, L_nl = float(np.mean(full)), float(np.mean(no_residual)), float(np.mean(no_library))
        degenerate = not (L_nl - L_full > 2 * L_full)
        live_row = None
        if live["full"]:
            lf, lnr, lnl = (float(np.mean(live[k])) for k in ("full", "no_residual", "no_library"))
            live_degenerate = not (lnl - lf > 2 * lf)
            live_row = {"tasks": len(live["full"]), "L_full": lf, "L_no_residual": lnr,
                        "L_no_library": lnl, "denominator_degenerate": bool(live_degenerate),
                        "R_residual": None if live_degenerate else float((lnr - lf) / (lnl - lf))}
        rows[str(world)] = {
            "tasks": len(tasks), "retired_tasks": retired, "live_tasks_only": live_row,
            "L_full": L_full, "L_no_residual": L_nr, "L_no_library": L_nl,
            "L_refit": float(np.mean(refits)) if refits else None,
            "denominator_degenerate": bool(degenerate),
            "R_residual": None if degenerate else float((L_nr - L_full) / (L_nl - L_full)),
        }
        r = rows[str(world)]
        print(f"[M4 w{world}] full {L_full:.5f}  no-residual {L_nr:.5f}  no-library {L_nl:.5f}  "
              f"refit {r['L_refit'] if r['L_refit'] is None else round(r['L_refit'], 5)}  "
              f"retired {retired}/{len(tasks)}  R_residual="
              f"{'degenerate' if degenerate else round(r['R_residual'], 4)}"
              + (f"  | live-only ({live_row['tasks']}): full {live_row['L_full']:.5f} "
                 f"no-residual {live_row['L_no_residual']:.5f} R_residual="
                 f"{'degenerate' if live_row['denominator_degenerate'] else round(live_row['R_residual'], 4)}"
                 if live_row else "  | no live tasks"), flush=True)
    out["substrates"]["M4"] = {"artifact": "artifacts/h39c/w_m4", "worlds": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print("written", args.output)


if __name__ == "__main__":
    main()
