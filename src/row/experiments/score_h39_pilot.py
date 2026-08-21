"""Scorer for the H39 world-0 joint-formation pilot (H39_PILOT_PLAN.md,
Amendments 1-2).

One question: after a pooled linear schema `W alpha + eps` has FORMED ONLINE
under the ordinary objective, is a member of an UNSEEN family (drawn from
the same shared functional subspace) cheaply expressible through `alpha`
alone? Primary endpoint: alpha-only (eps_new = 0, frozen) k=128 scaled query
endpoint under protocol B1, against 1.5x the ordinary world-0 V6R endpoint,
with at least one robustness optimizer agreeing.

Also scored, per the plan: ordinary bit-exactness of the history rerun, the
ordinary anchor reproduced to 1e-12 before any factorized value is read,
present-task parity, channel-use ablations (schema_share), D* proxies, the
historical-span census on the ordinary arm, non-vacuity checks, and the
fixed branch table A / B* / C / B / D. Report written atomically before any
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
from row.experiments.audit_effective_operator import _build_model, load_learner
from row.experiments.audit_v6r_adaptation_geometry import (
    CHECKPOINTS, SCALE, adam_fit, artifact_path, lbfgs_fit, mse, tensors,
    validate_artifact,
)
from row.experiments.census_h39_schema import alpha_fit as census_alpha_fit
from row.experiments import learned_lifetime
from row.meta_world import MetaFamilySpec, generate_meta_world
from row.models.factorized_models import FactorizedLifecycleLearner

PASS_RATIO = 1.5
FULL_FIT_RATIO = 1.2
PARITY_NATS = 2000.0
SCHEMA_SHARE_MIN = 0.5
BITS = 8
REQUIRED = ("model.pt", "summary.json", "rho_profile.json", "fingerprint.json",
            "config.yaml", "history.pt")


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_pilot_cell(path: Path, expected: dict) -> dict:
    missing = [f for f in REQUIRED if not (path / f).exists()]
    if missing:
        raise SystemExit(f"incomplete pilot cell {path}: missing {missing}")
    provenance = read_json(path / "rho_profile.json")
    record = provenance.get("h39_pilot")
    if record != expected:
        raise SystemExit(f"pilot record mismatch at {path}: {record} != {expected}")
    arm = provenance.get("v6_arm") or {}
    if arm.get("arm") != "ordinary" or arm.get("operator_slots") != 12 \
            or arm.get("sleeps") != [16, 24, 32, 48, 64] or not arm.get("lifecycle"):
        raise SystemExit(f"protocol mismatch at {path}: {arm}")
    fingerprint = read_json(path / "fingerprint.json")
    if int(fingerprint.get("world_seed", -1)) != 0:
        raise SystemExit(f"world mismatch at {path}")
    return {"path": str(path), "git_commit": fingerprint.get("git_commit"),
            "resolved_config_sha256": fingerprint.get("resolved_config_sha256"),
            "pilot_record": record, "protocol": arm}


def load_factorized(config, path: Path, record: dict) -> FactorizedLifecycleLearner:
    settings = {
        "schema_dim": record["schema_dim"],
        "schema_count": 1 if record["schema_grouping"] == "pooled" else 5,
        "schema_seed": record["schema_seed"],
        "schema_init_scale": record["schema_init_scale"],
        "freeze_schema": record["freeze_schema"],
    }
    saved = learned_lifetime.FACTORIZED_SETTINGS
    learned_lifetime.FACTORIZED_SETTINGS = settings
    try:
        local = replace(config, shared_residual_model=replace(
            config.shared_residual_model, operator_slots=12))
        model = _build_model(local, "factorized")
    finally:
        learned_lifetime.FACTORIZED_SETTINGS = saved
    state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
    count = sum(1 for k in state if k.startswith("abstractions."))
    for index in range(count):
        model.abstractions.append(torch.nn.Parameter(
            state[f"abstractions.{index}"].clone(), requires_grad=False))
    task_schema = read_json(path / "factorized.json")["task_schema"]
    for key in state:
        if key.startswith("task_codes."):
            task_id = key.split(".", 1)[1]
            model.begin_task(task_id, schema_index=int(task_schema[task_id]))
    model.load_state_dict(state)
    model.eval()
    summary = read_json(path / "summary.json")
    table = summary.get("reference_table") or {}
    for task_id, reference in (table.get("task_reference") or {}).items():
        model.task_reference[task_id] = int(reference)
    for task_id in table.get("retired_task_ids") or []:
        model.retired.add(task_id)
    return model


def bitwise_equal(path_a: Path, path_b: Path) -> dict:
    a = torch.load(path_a / "model.pt", weights_only=True)["model_state_dict"]
    b = torch.load(path_b / "model.pt", weights_only=True)["model_state_dict"]
    keys_equal = set(a) == set(b)
    tensors_equal = keys_equal and all(torch.equal(a[k], b[k]) for k in a)
    la = read_json(path_a / "summary.json")["cumulative_prequential_gaussian_log_loss"]
    lb = read_json(path_b / "summary.json")["cumulative_prequential_gaussian_log_loss"]
    return {"keys_equal": keys_equal, "tensors_equal": bool(tensors_equal),
            "loss_a": la, "loss_b": lb, "loss_equal": la == lb}


# ---- factorized fits ---------------------------------------------------

def factorized_fit(base_model, task, support: int, mode: str, optimizer_name: str,
                   learning_rate: float, steps: int, label: str) -> dict:
    """mode: 'alpha_only' (code + alpha; eps zero and frozen) or 'full'."""
    model = copy.deepcopy(base_model)
    probe_id = f"__h39pilot_{label}_{task.task_id}"
    code, eps, alpha = model.begin_task(probe_id, schema_index=0)
    for parameter in model.shared_parameters():
        parameter.requires_grad_(False)
    for schema in model.schemas:
        schema.requires_grad_(False)
    if mode == "alpha_only":
        with torch.no_grad():
            eps.zero_()
        eps.requires_grad_(False)
        params = [code, alpha]
    elif mode == "full":
        params = [code, alpha, eps]
    else:
        raise ValueError(mode)
    initial = torch.cat([p.detach().flatten() for p in params]).clone()
    support_x, support_y, query_x, query_y = tensors(task, support)
    initial_support = mse(model, probe_id, support_x, support_y)
    curve = {"0": mse(model, probe_id, query_x, query_y)}
    finite, completed, evaluations = True, 0, 0
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        for update in range(1, steps + 1):
            optimizer.zero_grad()
            loss = torch.mean((model(support_x, probe_id) - support_y) ** 2)
            if not bool(torch.isfinite(loss)):
                finite = False
                break
            loss.backward(inputs=params)
            optimizer.step()
            completed = update
            if update in CHECKPOINTS:
                curve[str(update)] = mse(model, probe_id, query_x, query_y)
        evaluations = completed
    elif optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=500, history_size=100,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((model(support_x, probe_id) - support_y) ** 2)
            loss.backward(inputs=params)
            return loss
        try:
            optimizer.step(closure)
        except (RuntimeError, ValueError):
            finite = False
        state = optimizer.state.get(params[0]) or {}
        completed = int(state.get("n_iter", 0))
        evaluations = int(state.get("func_evals", 0))
    else:
        raise ValueError(optimizer_name)
    final_support = mse(model, probe_id, support_x, support_y)
    final_query = mse(model, probe_id, query_x, query_y)
    curve["final"] = final_query
    final = torch.cat([p.detach().flatten() for p in params])
    result = {
        "mode": mode, "support": support, "optimizer": optimizer_name,
        "learning_rate": learning_rate, "requested_updates": steps,
        "completed_updates": completed, "optimizer_evaluations": evaluations,
        "initial_support_mse": initial_support, "final_support_mse": final_support,
        "support_reduction": (initial_support - final_support) / initial_support
        if initial_support > 0 else 0.0,
        "final_query_mse": final_query, "final_query_scaled": final_query / SCALE,
        "query_curve_mse": curve,
        "local_displacement": float(torch.linalg.vector_norm(final - initial)),
        "alpha": [float(a) for a in alpha.detach()],
        "alpha_norm": float(torch.linalg.vector_norm(alpha.detach())),
        "eps_norm": float(torch.linalg.vector_norm(eps.detach())),
        "finite": bool(finite and math.isfinite(final_support) and math.isfinite(final_query)),
    }
    model.forget_task(probe_id)
    return result


# ---- channel use ---------------------------------------------------------

@torch.no_grad()
def channel_use(model: FactorizedLifecycleLearner, tasks: list) -> dict:
    live = [t for t in tasks if t.task_id in model.task_codes and t.task_id not in model.retired]
    rows = []
    for task in live:
        x = torch.tensor(task.eval_x, dtype=torch.float32)
        y = torch.tensor(task.eval_y, dtype=torch.float32)
        var = float(torch.var(y, unbiased=False)) or 1.0
        alpha = model.task_alphas[task.task_id]
        eps = model.task_residuals[task.task_id]
        alpha0, eps0 = alpha.clone(), eps.clone()

        def nmse():
            return float(torch.mean((model(x, task.task_id) - y) ** 2)) / var
        full = nmse()
        alpha.zero_(); no_schema = nmse(); alpha.copy_(alpha0)
        eps.zero_(); no_eps = nmse(); eps.copy_(eps0)
        alpha.zero_(); eps.zero_(); neither = nmse(); alpha.copy_(alpha0); eps.copy_(eps0)
        rows.append({"task_id": task.task_id, "full": full, "schema_zeroed": no_schema,
                     "eps_zeroed": no_eps, "both_zeroed": neither,
                     "alpha_norm": float(torch.linalg.norm(alpha0)),
                     "eps_norm": float(torch.linalg.norm(eps0))})
    mean = lambda k: float(np.mean([r[k] for r in rows])) if rows else float("nan")
    denom = mean("both_zeroed") - mean("full")
    share = (mean("schema_zeroed") - mean("full")) / denom if denom > 0 else float("nan")
    eps_share = (mean("eps_zeroed") - mean("full")) / denom if denom > 0 else float("nan")
    return {"live_family_tasks": len(rows), "retired_family_tasks": len(tasks) - len(rows),
            "nmse_full": mean("full"), "nmse_schema_zeroed": mean("schema_zeroed"),
            "nmse_eps_zeroed": mean("eps_zeroed"), "nmse_both_zeroed": mean("both_zeroed"),
            "schema_share": share, "eps_share": eps_share,
            "alpha_norm_mean": mean("alpha_norm"), "eps_norm_mean": mean("eps_norm"),
            "rows": rows}


def dstar_proxy(model: FactorizedLifecycleLearner) -> dict:
    live_eps = sum(model.task_residuals[t].numel() for t in model.task_residuals
                   if t not in model.retired)
    return {"bits_per_scalar": BITS,
            "schema_bits": BITS * sum(p.numel() for p in model.schemas),
            "alpha_bits": BITS * sum(p.numel() for p in model.task_alphas.values()),
            "live_eps_bits": BITS * live_eps,
            "shared_bits": BITS * model.shared_parameter_count,
            "task_state_bits": BITS * model.task_state_scalar_count}


# ---- historical span ---------------------------------------------------

def pca_schema(vectors: torch.Tensor, rank: int):
    mean = vectors.mean(dim=0)
    _, singular, vh = torch.linalg.svd(vectors - mean, full_matrices=False)
    rank = min(rank, vh.shape[0])
    basis = vh[:rank].T.contiguous()
    explained = float((singular[:rank] ** 2).sum() / (singular ** 2).sum())
    return mean, basis, {"rank": rank, "variance_explained": explained,
                         "singular_values": [float(s) for s in singular[:rank + 4]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--pilot-root", type=Path, default=Path("artifacts/h39_pilot"))
    parser.add_argument("--ordinary-root", type=Path, default=Path("artifacts/v6_clean"))
    parser.add_argument("--anchor", type=Path, default=Path("reports/v6r_adaptation_geometry.json"))
    parser.add_argument("--census", type=Path, default=Path("reports/h39_census.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/h39_pilot.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    cells = {
        "ordinary_history": {"model": "prospective", "snapshot_history": True},
        "factorized_pooled2": {"model": "factorized", "snapshot_history": True,
                               "schema_dim": 2, "schema_grouping": "pooled",
                               "schema_seed": 39001, "schema_init_scale": 0.01,
                               "freeze_schema": False},
        "factorized_pooled": {"model": "factorized", "snapshot_history": True,
                              "schema_dim": 8, "schema_grouping": "pooled",
                              "schema_seed": 39001, "schema_init_scale": 0.01,
                              "freeze_schema": False},
    }
    paths = {name: args.pilot_root / name / "world_0" / "lifecycle" for name in cells}
    sources = {name: validate_pilot_cell(paths[name], record) for name, record in cells.items()}
    ordinary_path = artifact_path(args.ordinary_root, "ordinary", 0)
    sources["ordinary_v6_clean"] = validate_artifact(ordinary_path, "ordinary", 0)

    # 1. Ordinary bit-exactness of the history rerun.
    exact = bitwise_equal(paths["ordinary_history"], ordinary_path)
    if not (exact["tensors_equal"] and exact["loss_equal"]):
        raise SystemExit(f"ordinary history rerun is not bit-exact: {exact}")
    print("ordinary history rerun bit-exact", flush=True)

    # 2. World, futures, anchor reproduction.
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2)
    generated = generate_meta_world(replace(config.world, seed=0, tasks=spec.total_tasks), spec)
    futures = list(generated.novel_family_tasks)
    if len(futures) != 2:
        raise SystemExit("expected 2 unseen-family future tasks")
    family_tasks = [t for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None]
    anchor_report = read_json(args.anchor)
    anchor_rows = [r for r in anchor_report["rows"] if r["arm"] == "ordinary"
                   and r["world"] == 0 and r["method"] == "adam_001" and r["support"] == 128]
    ordinary = load_learner(config, ordinary_path, 12, kind="prospective")
    reproduced = []
    for index, task in enumerate(futures):
        result, _, _ = adam_fit(ordinary, task, support=128, learning_rate=0.01, steps=2000,
                                label=f"anchor_t{index}", checkpoints=CHECKPOINTS)
        expected = next(r["final_query_scaled"] for r in anchor_rows if r["task_index"] == index)
        observed = result["final_query_mse"] / SCALE
        reproduced.append({"task_index": index, "expected": expected, "observed": observed,
                           "abs_difference": abs(expected - observed)})
        if abs(expected - observed) > 1e-12:
            raise SystemExit(f"ordinary anchor mismatch task {index}: {expected} vs {observed}")
    ordinary_endpoint = float(np.mean([r["expected"] for r in reproduced]))
    ordinary_k1 = float(np.mean([r["final_query_scaled"] for r in anchor_report["rows"]
                                 if r["arm"] == "ordinary" and r["world"] == 0
                                 and r["method"] == "adam_001" and r["support"] == 1]))
    print(f"ordinary anchor reproduced: k=128 {ordinary_endpoint:.5f}", flush=True)

    # 3. Factorized fits.
    protocols = [("adam", 0.01, 2000, "B1"), ("adam", 0.05, 2000, "B2_adam"), ("lbfgs", 1.0, 500, "B2_lbfgs")]
    arms = {}
    for name in ("factorized_pooled2", "factorized_pooled"):
        model = load_factorized(config, paths[name], cells[name])
        fits = []
        for index, task in enumerate(futures):
            for support in (128, 1):
                for mode in ("alpha_only", "full"):
                    for opt, lr, steps, tag in protocols:
                        if support == 1 and tag != "B1":
                            continue
                        fit = factorized_fit(model, task, support, mode, opt, lr, steps,
                                             f"{name}_{tag}_{mode}_k{support}_t{index}")
                        fit.update({"task_index": index, "task_id": task.task_id, "protocol": tag})
                        fits.append(fit)
                        print(f"{name} t{index} k={support} {mode} {tag}: "
                              f"{fit['final_query_scaled']:.5f} (support {fit['final_support_mse']:.3g}, "
                              f"|alpha| {fit['alpha_norm']:.3f})", flush=True)

        def endpoint(mode, support, tag):
            vals = [f["final_query_scaled"] for f in fits
                    if f["mode"] == mode and f["support"] == support and f["protocol"] == tag]
            return float(np.mean(vals)) if vals else float("nan")

        def finite(mode, support, tag):
            return all(f["finite"] for f in fits
                       if f["mode"] == mode and f["support"] == support and f["protocol"] == tag)
        primary = endpoint("alpha_only", 128, "B1")
        robust = {tag: endpoint("alpha_only", 128, tag) for tag in ("B2_adam", "B2_lbfgs")}
        summary = read_json(paths[name] / "summary.json")
        diagnostics = read_json(paths[name] / "factorized.json")["diagnostics"]
        use = channel_use(model, family_tasks)
        arms[name] = {
            "fits": fits,
            "alpha_only_k128_B1": primary,
            "alpha_only_k128_ratio": primary / ordinary_endpoint,
            "alpha_only_k128_robustness": robust,
            "alpha_only_k128_robustness_ratios": {k: v / ordinary_endpoint for k, v in robust.items()},
            "full_k128_B1": endpoint("full", 128, "B1"),
            "full_k128_ratio": endpoint("full", 128, "B1") / ordinary_endpoint,
            "alpha_only_k1_B1": endpoint("alpha_only", 1, "B1"),
            "full_k1_B1": endpoint("full", 1, "B1"),
            "ordinary_k1_B1": ordinary_k1,
            "primary_finite": finite("alpha_only", 128, "B1"),
            "cumulative_prequential_gaussian_log_loss": summary["cumulative_prequential_gaussian_log_loss"],
            "schema_diagnostics": diagnostics,
            "channel_use": use,
            "dstar_proxy_bits": dstar_proxy(model),
            "non_vacuity": {
                "schema_moved": all(m > 1e-3 for m in diagnostics["schema_relative_movement"]),
                "alpha_moves_in_every_fit": all(f["alpha_norm"] > 0 for f in fits if f["mode"] == "alpha_only"),
                "support_falls_over_1pct": all(f["support_reduction"] > 0.01 for f in fits if f["protocol"] == "B1"),
                "k0_differs_from_final": all(f["query_curve_mse"]["0"] != f["final_query_mse"] for f in fits),
            },
        }

    # F-pooled2 vs F-pooled not functionally identical.
    m2 = load_factorized(config, paths["factorized_pooled2"], cells["factorized_pooled2"])
    m8 = load_factorized(config, paths["factorized_pooled"], cells["factorized_pooled"])
    with torch.no_grad():
        probe_task = family_tasks[0]
        x = torch.tensor(probe_task.eval_x, dtype=torch.float32)
        arms_differ = not torch.allclose(m2(x, probe_task.task_id), m8(x, probe_task.task_id))

    # 4. Historical-span census on the ordinary arm.
    history = torch.load(paths["ordinary_history"] / "history.pt", weights_only=True)
    family_ids = [t.task_id for t in family_tasks]
    hist_vectors = torch.stack([history["residuals"][t] for t in family_ids])
    historical = {}
    for rank in (8, 16, 0):
        mean, basis, info = pca_schema(hist_vectors, rank if rank else hist_vectors.shape[0] - 1)
        fits = []
        for index, task in enumerate(futures):
            fit = census_alpha_fit(ordinary, task, mean, basis, 128, f"hist_r{info['rank']}_t{index}")
            fit.update({"task_index": index})
            fits.append(fit)
        value = float(np.mean([f["final_query_scaled"] for f in fits]))
        historical[f"rank_{info['rank']}"] = {"schema": info, "fits": fits,
                                              "alpha_only_endpoint_scaled": value,
                                              "ratio": value / ordinary_endpoint,
                                              "pass": value / ordinary_endpoint <= PASS_RATIO}
        print(f"historical span rank {info['rank']}: ratio {value / ordinary_endpoint:.3f}", flush=True)
    census = read_json(args.census)
    final_ratio = next(w["ratio"] for w in census["worlds"] if w["world"] == 0)
    hist_pass = any(v["pass"] for v in historical.values())
    historical_reading = (
        "retirement/lifecycle discarded variation directions" if hist_pass and final_ratio > PASS_RATIO
        else "ordinary wake never formed those directions" if not hist_pass
        else "historical and final spans both express the future (unexpected)")

    # 5. Branch for the primary arm.
    ordinary_loss = read_json(paths["ordinary_history"] / "summary.json")["cumulative_prequential_gaussian_log_loss"]
    primary_arm = arms["factorized_pooled2"]
    parity_gap = primary_arm["cumulative_prequential_gaussian_log_loss"] - ordinary_loss
    parity = parity_gap <= PARITY_NATS
    alpha_pass = (primary_arm["alpha_only_k128_ratio"] <= PASS_RATIO
                  and any(v <= PASS_RATIO for v in primary_arm["alpha_only_k128_robustness_ratios"].values())
                  and primary_arm["primary_finite"])
    share = primary_arm["channel_use"]["schema_share"]
    full_ok = primary_arm["full_k128_ratio"] <= FULL_FIT_RATIO
    nonvac = all(primary_arm["non_vacuity"].values()) and arms_differ
    if not nonvac:
        branch = "NOT READ: non-vacuity failed"
    elif not parity:
        branch = "D"
    elif alpha_pass and share >= SCHEMA_SHARE_MIN:
        branch = "A"
    elif alpha_pass:
        branch = "B*"
    elif full_ok:
        branch = "C"
    else:
        branch = "B"
    descriptions = {
        "A": "joint schema works: licenses a frozen three-world H39 plan with the G control",
        "B*": "alpha-only passes but the schema carries < 0.5 of family computation: accidental; treated as B",
        "C": "future works only through private relearning (full fit within 1.2x); schema not fertile",
        "B": "schema describes the past, not the neighbourhood; do not proceed to H40-H44",
        "D": "restrictive ABI: present-task parity fails; fertility untested",
    }
    report = {
        "frozen_plan": "H39_PILOT_PLAN.md (Amendments 1-2)",
        "status": "EXPLORATORY world-0 pilot",
        "git_commit": git_commit(),
        "protocol": {"world": 0, "supports": [1, 128], "sigma": 0.1,
                     "protocols": [{"tag": t, "optimizer": o, "lr": lr, "steps": s} for o, lr, s, t in protocols],
                     "pass_ratio": PASS_RATIO, "full_fit_ratio": FULL_FIT_RATIO,
                     "parity_nats": PARITY_NATS, "schema_share_min": SCHEMA_SHARE_MIN,
                     "query_checkpoints": list(CHECKPOINTS)},
        "sources": sources,
        "ordinary_bit_exact": exact,
        "anchor": {"rows": reproduced, "ordinary_k128_endpoint": ordinary_endpoint,
                   "ordinary_k1_endpoint": ordinary_k1, "ordinary_loss": ordinary_loss},
        "arms": arms,
        "arms_functionally_differ": bool(arms_differ),
        "historical_span": historical,
        "census_final_span_ratio_world0": final_ratio,
        "historical_reading": historical_reading,
        "decision": {"parity_gap_nats": parity_gap, "parity": parity,
                     "alpha_only_pass": alpha_pass, "schema_share": share,
                     "full_fit_within_1_2x": full_ok, "non_vacuity": nonvac,
                     "branch": branch, "description": descriptions.get(branch, branch)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    for name, arm in arms.items():
        print(f"{name}: alpha-only k128 ratio {arm['alpha_only_k128_ratio']:.3f} "
              f"(robust {arm['alpha_only_k128_robustness_ratios']}), full k128 ratio "
              f"{arm['full_k128_ratio']:.3f}, schema_share {arm['channel_use']['schema_share']:.3f}, "
              f"loss gap {arm['cumulative_prequential_gaussian_log_loss'] - ordinary_loss:+.1f}")
    print(f"historical: {historical_reading}")
    print(f"BRANCH {branch}: {descriptions.get(branch, '')}")


if __name__ == "__main__":
    main()
