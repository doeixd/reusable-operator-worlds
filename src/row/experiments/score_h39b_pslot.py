"""Scorer for the H39b world-0 pilot: P(alpha) in the basis
(H39B_PSLOT_PILOT_PLAN.md).

One question: after a parameterized BASIS slot P(alpha) has formed online
under the ordinary objective, is a member of an UNSEEN family (drawn from
the same shared functional subspace) cheaply expressible through `alpha`
alone (route code + alpha; residual frozen at its task-free init)? Primary endpoint: alpha-only (eps_new = 0, frozen) k=128 scaled query
endpoint under protocol B1, against 1.5x the ordinary world-0 V6R endpoint,
with at least one robustness optimizer agreeing.

Also scored, per the plan: ordinary bit-exactness of the history rerun, the
ordinary anchor reproduced to 1e-12 before any P-slot value is read,
present-task parity, channel-use (route mass on P, alpha-zeroed ratio), D* proxies, the
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
from row.models.pslot_models import ParameterizedSlotLearner

PASS_RATIO = 1.5
FULL_FIT_RATIO = 1.2
PARITY_NATS = 2000.0
P_MASS_MIN = 0.5
ALPHA_ZEROED_MIN = 1.25
P_MASS_UNUSED = 0.2
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


def load_pslot(config, path: Path, record: dict, world_seed: int | None = None):
    settings = {"slot_args": record["slot_args"], "freeze_args": record["freeze_args"],
                "freeze_matrices": record.get("freeze_matrices", False),
                "pslot_count": record.get("pslot_count", 1)}
    kind = record.get("model", "pslot")
    saved = learned_lifetime.PSLOT_SETTINGS
    saved_f = learned_lifetime.PSLOT_FACTORIZED_SETTINGS
    learned_lifetime.PSLOT_SETTINGS = settings
    if kind == "pslot_factorized":
        # H51 arm R_2: the composed learner needs its component-basis knobs and
        # the artifact's own world seed (the basis initialization stream).
        learned_lifetime.PSLOT_FACTORIZED_SETTINGS = dict(
            settings, schema_dim=record["schema_dim"], schema_count=record.get("schema_count", 1),
            schema_seed=record.get("schema_seed", 51001),
            schema_init_scale=record.get("schema_init_scale", 1e-2),
            freeze_schema=record.get("freeze_schema", False))
    try:
        local = replace(config, shared_residual_model=replace(
            config.shared_residual_model, operator_slots=12))
        if world_seed is not None:
            local = replace(local, world=replace(local.world, seed=int(world_seed)))
        model = _build_model(local, kind)
    finally:
        learned_lifetime.PSLOT_SETTINGS = saved
        learned_lifetime.PSLOT_FACTORIZED_SETTINGS = saved_f
    state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
    count = sum(1 for k in state if k.startswith("abstractions."))
    for index in range(count):
        model.abstractions.append(torch.nn.Parameter(
            state[f"abstractions.{index}"].clone(), requires_grad=False))
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    # Artifacts written before the H47 route policies lack the
    # `route_temperature` buffer (policy off == 1.0, the buffer's init).
    # Tolerate exactly that key and nothing else.
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) - {"route_temperature"} or result.unexpected_keys:
        raise SystemExit(f"state mismatch at {path}: {result}")
    model.eval()
    extras = read_json(path / "pslot.json")
    for task_id, slot in (extras.get("task_mask") or {}).items():
        model.task_mask[task_id] = int(slot)
    summary = read_json(path / "summary.json")
    table = summary.get("reference_table") or {}
    for task_id, reference in (table.get("task_reference") or {}).items():
        model.task_reference[task_id] = int(reference)
    for task_id in table.get("retired_task_ids") or []:
        model.retired.add(task_id)
    return model


def bitwise_equal(path_a: Path, path_b: Path, subset: bool = False) -> dict:
    """subset=True: every key of b must exist in a and match (a may hold more)."""
    a = torch.load(path_a / "model.pt", weights_only=True)["model_state_dict"]
    b = torch.load(path_b / "model.pt", weights_only=True)["model_state_dict"]
    keys_equal = set(b) <= set(a) if subset else set(a) == set(b)
    tensors_equal = keys_equal and all(torch.equal(a[k], b[k]) for k in b)
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
    code, eps, alpha = model.begin_task(probe_id)
    for parameter in model.shared_parameters():
        parameter.requires_grad_(False)
    model.argument_matrices.requires_grad_(False)
    if mode == "alpha_only":
        # Amendment 3: eps frozen at the shared 1e-3 initial state (no task
        # information), not zero -- zero is stationary and pins alpha.
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
        "alpha": [float(a) for a in alpha.detach().flatten()],
        "alpha_norm": float(torch.linalg.vector_norm(alpha.detach())),
        "eps_norm": float(torch.linalg.vector_norm(eps.detach())),
        "finite": bool(finite and math.isfinite(final_support) and math.isfinite(final_query)),
    }
    model.forget_task(probe_id)
    if mode == "alpha_only" and result["alpha_norm"] == 0.0:
        raise SystemExit(f"alpha did not move in alpha-only fit {label}: stationary protocol")
    return result


# ---- channel use ---------------------------------------------------------

@torch.no_grad()
def channel_use(model: ParameterizedSlotLearner, tasks: list) -> dict:
    """Route mass on P per step and the alpha-zeroed NMSE ratio, over every
    trained family task (retired ones included: P fires through the route)."""
    rows = []
    masses = []
    for task in tasks:
        if task.task_id not in model.task_codes:
            continue
        x = torch.tensor(task.eval_x, dtype=torch.float32)
        y = torch.tensor(task.eval_y, dtype=torch.float32)
        var = float(torch.var(y, unbiased=False)) or 1.0
        alpha = model.task_alphas[task.task_id]
        alpha0 = alpha.clone()
        full = float(torch.mean((model(x, task.task_id) - y) ** 2)) / var
        alpha.zero_()
        zeroed = float(torch.mean((model(x, task.task_id) - y) ** 2)) / var
        alpha.copy_(alpha0)
        code = model.task_codes[task.task_id].reshape(model.task_steps, model.operator_slots)
        soft = torch.softmax(code, dim=-1)
        mass = soft[:, model.pslot_index].tolist()
        masses.append([soft[:, slot].tolist() for slot in model.pslot_indices])
        rows.append({"task_id": task.task_id, "retired": task.task_id in model.retired,
                     "nmse_full": full, "nmse_alpha_zeroed": zeroed,
                     "route_mass_P": mass, "alpha_norm": float(torch.linalg.norm(alpha0))})
    by_slot = np.mean(masses, axis=0).tolist() if masses else []
    mass_by_step = by_slot[0] if by_slot else []
    full = float(np.mean([r["nmse_full"] for r in rows]))
    zeroed = float(np.mean([r["nmse_alpha_zeroed"] for r in rows]))
    return {"family_tasks": len(rows), "retired_family_tasks": sum(r["retired"] for r in rows),
            "route_mass_P_by_step": mass_by_step,
            "route_mass_by_parameterized_slot": dict(zip(map(str, model.pslot_indices), by_slot)),
            "route_mass_P_max_step": max(mass_by_step) if mass_by_step else float("nan"),
            "uniform_reference": 1.0 / model.operator_slots,
            "nmse_full": full, "nmse_alpha_zeroed": zeroed,
            "alpha_zeroed_ratio": zeroed / full if full > 0 else float("nan"),
            "alpha_norm_mean": float(np.mean([r["alpha_norm"] for r in rows])),
            "rows": rows}


def dstar_proxy(model: ParameterizedSlotLearner) -> dict:
    live_eps = sum(model.task_residuals[t].numel() for t in model.task_residuals
                   if t not in model.retired)
    return {"bits_per_scalar": BITS,
            "argument_matrix_bits": BITS * model.argument_matrices.numel(),
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
    parser.add_argument("--output", type=Path, default=Path("reports/h39b_pslot_pilot.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    cells = {
        "pslot2": {"model": "pslot", "snapshot_history": True, "slot_args": 2,
                   "freeze_args": False, "pslot_index": 11},
        "pslot8": {"model": "pslot", "snapshot_history": True, "slot_args": 8,
                   "freeze_args": False, "pslot_index": 11},
        "pslot2_frozen": {"model": "pslot", "snapshot_history": True, "slot_args": 2,
                          "freeze_args": True, "pslot_index": 11},
    }
    paths = {name: args.pilot_root / name / "world_0" / "lifecycle" for name in cells}
    sources = {name: validate_pilot_cell(paths[name], record) for name, record in cells.items()}
    ordinary_path = artifact_path(args.ordinary_root, "ordinary", 0)
    sources["ordinary_v6_clean"] = validate_artifact(ordinary_path, "ordinary", 0)

    # 1. The frozen-argument control must equal the ordinary learner bitwise
    # on every ordinary tensor (its extra tensors are zero alphas and the
    # untouched argument matrices).
    exact = bitwise_equal(paths["pslot2_frozen"], ordinary_path, subset=True)
    if not (exact["tensors_equal"] and exact["loss_equal"]):
        raise SystemExit(f"frozen-argument control is not bit-exact with ordinary: {exact}")
    print("pslot2_frozen bit-exact with ordinary on all ordinary tensors", flush=True)

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
    for name in ("pslot2", "pslot8"):
        model = load_pslot(config, paths[name], cells[name])
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
        diagnostics = read_json(paths[name] / "pslot.json")["diagnostics"]
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
            "pslot_diagnostics": diagnostics,
            "channel_use": use,
            "dstar_proxy_bits": dstar_proxy(model),
            "non_vacuity": {
                "argument_matrices_moved": diagnostics["argument_matrices_relative_movement"] > 1e-3,
                "family_alpha_nonzero": use["alpha_norm_mean"] > 0.0,
                "alpha_moves_in_every_fit": all(f["alpha_norm"] > 0 for f in fits if f["mode"] == "alpha_only"),
                "support_falls_over_1pct": all(f["support_reduction"] > 0.01 for f in fits if f["protocol"] == "B1"),
                "k0_differs_from_final": all(f["query_curve_mse"]["0"] != f["final_query_mse"] for f in fits),
            },
        }

    # P2 vs P8 not functionally identical.
    m2 = load_pslot(config, paths["pslot2"], cells["pslot2"])
    m8 = load_pslot(config, paths["pslot8"], cells["pslot8"])
    with torch.no_grad():
        probe_task = family_tasks[0]
        x = torch.tensor(probe_task.eval_x, dtype=torch.float32)
        arms_differ = not torch.allclose(m2(x, probe_task.task_id), m8(x, probe_task.task_id))

    # 4. Historical-span census on the ordinary arm.
    history = torch.load(paths["pslot2_frozen"] / "history.pt", weights_only=True)
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
    ordinary_loss = read_json(ordinary_path / "summary.json")["cumulative_prequential_gaussian_log_loss"]
    primary_arm = arms["pslot2"]
    parity_gap = primary_arm["cumulative_prequential_gaussian_log_loss"] - ordinary_loss
    parity = parity_gap <= PARITY_NATS
    alpha_pass = (primary_arm["alpha_only_k128_ratio"] <= PASS_RATIO
                  and any(v <= PASS_RATIO for v in primary_arm["alpha_only_k128_robustness_ratios"].values())
                  and primary_arm["primary_finite"])
    use = primary_arm["channel_use"]
    carries = (use["route_mass_P_max_step"] >= P_MASS_MIN
               and use["alpha_zeroed_ratio"] >= ALPHA_ZEROED_MIN)
    unused = use["route_mass_P_max_step"] < P_MASS_UNUSED
    full_ok = primary_arm["full_k128_ratio"] <= FULL_FIT_RATIO
    nonvac = all(primary_arm["non_vacuity"].values()) and arms_differ
    if not nonvac:
        branch = "NOT READ: non-vacuity failed"
    elif not parity:
        branch = "D"
    elif alpha_pass and carries:
        branch = "A"
    elif alpha_pass:
        branch = "B*"
    elif unused:
        branch = "U"
    elif full_ok:
        branch = "C"
    else:
        branch = "B"
    descriptions = {
        "A": "in-basis argument is fertile: licenses a frozen multi-world plan with a matched-budget control",
        "B*": "alpha-only passes but P does not carry family computation: accidental",
        "U": "unused: the learner never routed family tasks through P; fertility of P(alpha) untested",
        "C": "future works only through private relearning (full fit within 1.2x); P(alpha) not fertile",
        "B": "P(alpha) not fertile and the full interface is worse than ordinary",
        "D": "restrictive ABI: present-task parity fails; fertility untested",
    }
    report = {
        "frozen_plan": "H39B_PSLOT_PILOT_PLAN.md",
        "status": "EXPLORATORY world-0 pilot",
        "git_commit": git_commit(),
        "protocol": {"world": 0, "supports": [1, 128], "sigma": 0.1,
                     "protocols": [{"tag": t, "optimizer": o, "lr": lr, "steps": s} for o, lr, s, t in protocols],
                     "pass_ratio": PASS_RATIO, "full_fit_ratio": FULL_FIT_RATIO,
                     "parity_nats": PARITY_NATS, "p_mass_min": P_MASS_MIN,
                     "alpha_zeroed_min": ALPHA_ZEROED_MIN, "p_mass_unused": P_MASS_UNUSED,
                     "query_checkpoints": list(CHECKPOINTS)},
        "sources": sources,
        "frozen_control_bit_exact": exact,
        "anchor": {"rows": reproduced, "ordinary_k128_endpoint": ordinary_endpoint,
                   "ordinary_k1_endpoint": ordinary_k1, "ordinary_loss": ordinary_loss},
        "arms": arms,
        "arms_functionally_differ": bool(arms_differ),
        "historical_span": historical,
        "census_final_span_ratio_world0": final_ratio,
        "historical_reading": historical_reading,
        "decision": {"parity_gap_nats": parity_gap, "parity": parity,
                     "alpha_only_pass": alpha_pass, "p_carries": bool(carries),
                     "route_mass_P_max_step": use["route_mass_P_max_step"],
                     "alpha_zeroed_ratio": use["alpha_zeroed_ratio"],
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
              f"{arm['full_k128_ratio']:.3f}, P mass max {arm['channel_use']['route_mass_P_max_step']:.3f}, "
              f"alpha-zeroed ratio {arm['channel_use']['alpha_zeroed_ratio']:.3f}, "
              f"loss gap {arm['cumulative_prequential_gaussian_log_loss'] - ordinary_loss:+.1f}")
    print(f"historical: {historical_reading}")
    print(f"BRANCH {branch}: {descriptions.get(branch, '')}")


if __name__ == "__main__":
    main()
