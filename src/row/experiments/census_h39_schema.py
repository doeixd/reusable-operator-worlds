"""H39 census gate C0 (H39_EXISTENCE_PLAN.md, Amendment 1).

Before any factorized lifetime is spent, ask whether a rank-8 LINEAR
schema over the ordinary learner's own residual vectors can express the
held-out siblings at all. The learner is frozen; only a fresh route code
and an 8-dim argument `alpha` move, with the private residual held at
zero:

    residual_sibling = mean + W alpha

`W` is the top-8 principal directions of the 198-scalar residual vectors
of all non-retired trained family tasks plus all promoted abstractions,
within ONE artifact (common parameterization, so parameter-space PCA is
legitimate here; across learners it would not be). Siblings never
trained, so fit and score are on different objects.

Pass per world: alpha-only scaled query endpoint at k=128 under protocol
B1 (Adam 0.01, 2,000 updates) is at most 1.5x the ordinary V6R k=128
endpoint for that world. C0 licenses the lifetimes only; it predicts
nothing about their result. The report is written atomically before any
console summary.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_effective_operator import load_learner
from row.experiments.audit_v6r_adaptation_geometry import (
    SCALE, artifact_path, mse, tensors, validate_artifact,
)
from row.meta_world import MetaFamilySpec, generate_meta_world

SCHEMA_RANK = 8
PASS_RATIO = 1.5
B1_LR = 0.01
B1_STEPS = 2000
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 40, 100, 250, 500, 1000, 2000)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, check=True,
    ).stdout.strip()


def ordinary_anchor(report: dict, world: int) -> float:
    """Ordinary k=128 primary-Adam scaled endpoint, mean over the world's tasks."""
    values = [
        row["final_query_scaled"] for row in report["rows"]
        if row["arm"] == "ordinary" and row["world"] == world
        and row["method"] == "adam_001" and row["support"] == 128
    ]
    if len(values) != 2:
        raise SystemExit(f"world {world}: expected 2 ordinary k=128 anchors, got {len(values)}")
    return float(np.mean(values))


def schema_from_artifact(model, family_ids: list[str], rank: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    vectors = []
    used, retired = 0, 0
    for task_id in family_ids:
        if task_id in getattr(model, "retired", set()):
            retired += 1
            continue
        vectors.append(model.task_residuals[task_id].detach().clone())
        used += 1
    abstractions = [a.detach().clone() for a in getattr(model, "abstractions", [])]
    vectors.extend(abstractions)
    stack = torch.stack(vectors)
    if rank == 0:
        rank = int(stack.shape[0]) - 1
    mean = stack.mean(dim=0)
    centered = stack - mean
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:rank].T.contiguous()  # (198, rank)
    explained = float((singular[:rank] ** 2).sum() / (singular ** 2).sum())
    info = {
        "fit_vectors": int(stack.shape[0]),
        "trained_family_tasks_used": used,
        "retired_family_tasks_skipped": retired,
        "abstractions_included": len(abstractions),
        "schema_rank": rank,
        "variance_explained": explained,
        "singular_values": [float(s) for s in singular[: rank + 4]],
    }
    return mean, basis, info


def alpha_fit(base_model, task, mean: torch.Tensor, basis: torch.Tensor,
              support: int, label: str) -> dict:
    model = copy.deepcopy(base_model)
    probe_id = f"__h39census_{label}_{task.task_id}"
    model.begin_task(probe_id)
    code = model.task_codes[probe_id]
    for parameter in model.shared_parameters():
        parameter.requires_grad_(False)
    model.task_residuals[probe_id].requires_grad_(False)
    alpha = torch.zeros(basis.shape[1], requires_grad=True)
    original_unpack = model._unpack

    def patched_unpack(task_id: str):
        if task_id != probe_id:
            return original_unpack(task_id)
        residual = mean + basis @ alpha
        u, v, b = torch.split(
            residual, (model.residual_u_size, model.residual_v_size, model.residual_b_size)
        )
        return (
            code.reshape(model.task_steps, model.operator_slots),
            u.reshape(model.task_steps, model.d, model.residual_rank),
            v.reshape(model.task_steps, model.residual_rank, model.d),
            b.reshape(model.task_steps, model.residual_rank),
        )

    model._unpack = patched_unpack
    support_x, support_y, query_x, query_y = tensors(task, support)
    optimizer = torch.optim.Adam([code, alpha], lr=B1_LR)
    curve = {"0": mse(model, probe_id, query_x, query_y)}
    initial_support = mse(model, probe_id, support_x, support_y)
    finite, completed = True, 0
    for update in range(1, B1_STEPS + 1):
        optimizer.zero_grad()
        loss = torch.mean((model(support_x, probe_id) - support_y) ** 2)
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        loss.backward(inputs=[code, alpha])
        optimizer.step()
        completed = update
        if update in CHECKPOINTS:
            curve[str(update)] = mse(model, probe_id, query_x, query_y)
    final_support = mse(model, probe_id, support_x, support_y)
    final_query = mse(model, probe_id, query_x, query_y)
    model.forget_task(probe_id)
    return {
        "support": support,
        "optimizer": "adam", "learning_rate": B1_LR,
        "requested_updates": B1_STEPS, "completed_updates": completed,
        "initial_support_mse": initial_support,
        "final_support_mse": final_support,
        "final_query_mse": final_query,
        "final_query_scaled": final_query / SCALE,
        "query_curve_mse": curve,
        "alpha": [float(a) for a in alpha.detach()],
        "alpha_norm": float(torch.linalg.vector_norm(alpha.detach())),
        "code_norm": float(torch.linalg.vector_norm(code.detach())),
        "finite": bool(finite and math.isfinite(final_support) and math.isfinite(final_query)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v6_clean"))
    parser.add_argument("--anchor", type=Path,
                        default=Path("reports/v6r_adaptation_geometry.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/h39_census.json"))
    parser.add_argument("--schema-rank", type=int, default=SCHEMA_RANK,
                        help="EXPLORATORY only when != 8; the registered gate is rank 8. "
                             "0 means the maximum available rank (n_vectors - 1).")
    args = parser.parse_args()
    exploratory = args.schema_rank != SCHEMA_RANK
    if exploratory and args.output == Path("reports/h39_census.json"):
        raise SystemExit("exploratory rank requires an explicit non-registered --output")
    worlds = [0, 1, 2]
    torch.set_num_threads(1)

    anchor = json.loads(args.anchor.read_text(encoding="utf-8"))
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2)
    sources, world_rows = [], []
    for world in worlds:
        path = artifact_path(args.root, "ordinary", world)
        sources.append(validate_artifact(path, "ordinary", world))
        world_config = replace(config.world, seed=world, tasks=spec.total_tasks)
        generated = generate_meta_world(world_config, spec)
        siblings = list(generated.novel_family_tasks)
        if len(siblings) != 2:
            raise SystemExit(f"world {world}: expected 2 sibling tasks")
        lifetime_tasks = list(generated.tasks)
        family_ids = [
            t.task_id for i, t in enumerate(lifetime_tasks) if spec.family_of(i) is not None
        ]
        if len(family_ids) != 64:
            raise SystemExit(f"world {world}: expected 64 family tasks, got {len(family_ids)}")
        base = load_learner(config, path, 12, kind="prospective")
        missing = [t for t in family_ids if t not in base.task_residuals]
        if missing:
            raise SystemExit(f"world {world}: {len(missing)} family tasks absent from artifact")
        mean, basis, info = schema_from_artifact(base, family_ids, args.schema_rank)
        baseline = ordinary_anchor(anchor, world)
        fits = []
        for index, task in enumerate(siblings):
            fit = alpha_fit(base, task, mean, basis, 128, f"w{world}_t{index}")
            if not fit["finite"]:
                raise SystemExit(f"non-finite census fit: world {world} task {index}")
            fit.update({"task_index": index, "task_id": task.task_id})
            fits.append(fit)
            print(f"world {world} task {index}: alpha-only k=128 scaled endpoint "
                  f"{fit['final_query_scaled']:.5f} (support {fit['final_support_mse']:.3g})",
                  flush=True)
        endpoint = float(np.mean([f["final_query_scaled"] for f in fits]))
        ratio = endpoint / baseline
        world_rows.append({
            "world": world, "schema": info, "fits": fits,
            "alpha_only_endpoint_scaled": endpoint,
            "ordinary_anchor_endpoint_scaled": baseline,
            "ratio": ratio, "pass": bool(ratio <= PASS_RATIO),
        })

    passes = sum(r["pass"] for r in world_rows)
    verdict = "C0 PASS: lifetimes licensed" if passes >= 2 else "C0 FAIL: H39 not run, census negative"
    report = {
        "frozen_plan": "H39_EXISTENCE_PLAN.md (Amendment 1)",
        "status": "EXPLORATORY calibration, not the registered C0" if exploratory else "REGISTERED C0",
        "git_commit": git_commit(),
        "protocol": {
            "schema_rank_requested": args.schema_rank, "registered_rank": SCHEMA_RANK, "pass_ratio": PASS_RATIO,
            "fit": {"optimizer": "adam", "lr": B1_LR, "steps": B1_STEPS, "support": 128},
            "sigma": 0.1, "trainable": ["route_code", "alpha"], "eps": "zero, frozen",
            "query_checkpoints": list(CHECKPOINTS),
        },
        "sources": sources,
        "worlds": world_rows,
        "worlds_passing": passes,
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    for r in world_rows:
        print(f"world {r['world']}: alpha-only {r['alpha_only_endpoint_scaled']:.5f} "
              f"vs ordinary {r['ordinary_anchor_endpoint_scaled']:.5f} "
              f"ratio {r['ratio']:.3f} {'PASS' if r['pass'] else 'FAIL'}")
    print(verdict)


if __name__ == "__main__":
    main()
