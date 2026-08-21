"""V6R: localize prospective harm as capacity, identifiability, or findability.

Implements the frozen protocol in ``V6R_ADAPTATION_GEOMETRY_PLAN.md``. Query
labels are observed only at fixed reporting checkpoints. They never enter an
optimizer closure or select a restart, optimizer, checkpoint, or stopping time.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_effective_operator import load_learner
from row.meta_world import MetaFamilySpec, generate_meta_world


ARMS = ("ordinary", "prospective")
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 40, 100, 250, 500, 1000, 2000)
SIGMA = 0.1
SCALE = 2 * SIGMA * SIGMA


def artifact_path(root: Path, arm: str, world: int) -> Path:
    return root / arm / f"world_{world}" / "lifecycle"


def validate_artifact(path: Path, arm: str, world: int) -> dict[str, object]:
    required = ("model.pt", "summary.json", "rho_profile.json", "fingerprint.json")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise SystemExit(f"incomplete V6R source {path}: missing {missing}")
    provenance = json.loads((path / "rho_profile.json").read_text(encoding="utf-8"))
    protocol = provenance.get("v6_arm") or {}
    expected = {
        "arm": arm,
        "freeze_basis_at": None,
        "freeze_slots": None,
        "operator_slots": 12,
        "prospective_steps": 8,
        "prospective_inner_steps": 8,
        "prospective_support": 8,
        "sleeps": [16, 24, 32, 48, 64],
        "lifecycle": True,
    }
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"V6R source mismatch at {path}: {mismatches}")
    fingerprint = json.loads((path / "fingerprint.json").read_text(encoding="utf-8"))
    if int(fingerprint.get("world_seed", -1)) != world:
        raise SystemExit(f"world mismatch at {path}")
    return {
        "path": str(path),
        "git_commit": fingerprint.get("git_commit"),
        "resolved_config_sha256": fingerprint.get("resolved_config_sha256"),
        "protocol": protocol,
    }


def tensors(task, support: int):
    return (
        torch.tensor(task.train_x[:support], dtype=torch.float32),
        torch.tensor(task.train_y[:support], dtype=torch.float32),
        torch.tensor(task.eval_x, dtype=torch.float32),
        torch.tensor(task.eval_y, dtype=torch.float32),
    )


def prepare(base_model, task, support: int, label: str,
            perturbation: np.ndarray | None = None):
    model = copy.deepcopy(base_model)
    probe_id = f"__v6r_{label}_{task.task_id}"
    if probe_id in model.task_codes:
        model.forget_task(probe_id)
    model.begin_task(probe_id)
    code = model.task_codes[probe_id]
    residual = model.task_residuals[probe_id]
    for parameter in model.shared_parameters():
        parameter.requires_grad_(False)
    if perturbation is not None:
        with torch.no_grad():
            split = code.numel()
            code.add_(torch.tensor(
                perturbation[:split].reshape(code.shape), dtype=code.dtype
            ))
            residual.add_(torch.tensor(
                perturbation[split:].reshape(residual.shape), dtype=residual.dtype
            ))
    initial = torch.cat((code.detach().flatten(), residual.detach().flatten())).clone()
    support_x, support_y, query_x, query_y = tensors(task, support)
    return model, probe_id, code, residual, initial, support_x, support_y, query_x, query_y


@torch.no_grad()
def mse(model, task_id: str, x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean((model(x, task_id) - y) ** 2))


def adam_fit(base_model, task, support: int, learning_rate: float, steps: int,
             label: str, checkpoints: tuple[int, ...] = (),
             perturbation: np.ndarray | None = None,
             record_query: bool = True, keep_model: bool = False):
    prepared = prepare(base_model, task, support, label, perturbation)
    (model, probe_id, code, residual, initial, support_x, support_y,
     query_x, query_y) = prepared
    optimizer = torch.optim.Adam([code, residual], lr=learning_rate)
    query_curve = {}
    if record_query and 0 in checkpoints:
        query_curve["0"] = mse(model, probe_id, query_x, query_y)
    initial_support = mse(model, probe_id, support_x, support_y)
    finite = True
    completed = 0
    for update in range(1, steps + 1):
        optimizer.zero_grad()
        loss = torch.mean((model(support_x, probe_id) - support_y) ** 2)
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        loss.backward(inputs=[code, residual])
        optimizer.step()
        completed = update
        if record_query and update in checkpoints:
            query_curve[str(update)] = mse(model, probe_id, query_x, query_y)
    final_support = mse(model, probe_id, support_x, support_y)
    final_query = (mse(model, probe_id, query_x, query_y)
                   if record_query else None)
    final_state = torch.cat((code.detach().flatten(), residual.detach().flatten()))
    result = {
        "optimizer": "adam",
        "learning_rate": learning_rate,
        "requested_updates": steps,
        "completed_updates": completed,
        "optimizer_evaluations": completed,
        "initial_support_mse": initial_support,
        "final_support_mse": final_support,
        "final_query_mse": final_query,
        "query_curve_mse": query_curve,
        "local_displacement": float(torch.linalg.vector_norm(final_state - initial)),
        "finite": bool(
            finite and math.isfinite(final_support)
            and (final_query is None or math.isfinite(final_query))
        ),
    }
    if not keep_model:
        model.forget_task(probe_id)
        return result, None, None
    return result, model, probe_id


def lbfgs_fit(base_model, task, support: int, label: str):
    prepared = prepare(base_model, task, support, label)
    (model, probe_id, code, residual, initial, support_x, support_y,
     query_x, query_y) = prepared
    optimizer = torch.optim.LBFGS(
        [code, residual], lr=1.0, max_iter=500, history_size=100,
        line_search_fn="strong_wolfe",
    )
    initial_support = mse(model, probe_id, support_x, support_y)
    initial_query = mse(model, probe_id, query_x, query_y)

    def closure():
        optimizer.zero_grad()
        loss = torch.mean((model(support_x, probe_id) - support_y) ** 2)
        loss.backward(inputs=[code, residual])
        return loss

    finite = True
    try:
        optimizer.step(closure)
    except (RuntimeError, ValueError):
        finite = False
    final_support = mse(model, probe_id, support_x, support_y)
    final_query = mse(model, probe_id, query_x, query_y)
    final_state = torch.cat((code.detach().flatten(), residual.detach().flatten()))
    state = optimizer.state.get(code) or optimizer.state.get(residual) or {}
    result = {
        "optimizer": "lbfgs",
        "learning_rate": 1.0,
        "requested_updates": 500,
        "completed_updates": int(state.get("n_iter", 0)),
        "optimizer_evaluations": int(state.get("func_evals", 0)),
        "initial_support_mse": initial_support,
        "final_support_mse": final_support,
        "final_query_mse": final_query,
        "query_curve_mse": {"0": initial_query, "final": final_query},
        "local_displacement": float(torch.linalg.vector_norm(final_state - initial)),
        "finite": bool(
            finite and math.isfinite(final_support) and math.isfinite(final_query)
        ),
    }
    model.forget_task(probe_id)
    return result


def standard_fit(base_model, task, label: str):
    # Record every query point because the anchor currency is the cumulative
    # 41-point adaptation trajectory used by score_v6_fertility.adapt_cost.
    checkpoints = tuple(range(41))
    result, _, _ = adam_fit(
        base_model, task, support=1, learning_rate=0.05, steps=40,
        label=label, checkpoints=checkpoints,
    )
    curve = [result["query_curve_mse"][str(step)] for step in checkpoints]
    result["cumulative_scaled_query_cost"] = float(sum(curve)) / SCALE
    return result


def anchor_expected(report: dict, arm: str, world: int) -> list[float]:
    cell = next(
        row for row in report["cells"]
        if row["arm"] == arm and int(row["world"]) == world
    )
    # The corrected V6 scorer was invoked with ``--metric prequential`` and
    # therefore serialized the selected per-task values directly. Accept the
    # richer mapping form too so a report retaining all adapt_cost fields can
    # be checked by the same anchor without changing its meaning.
    return [
        float(task["prequential"] if isinstance(task, dict) else task)
        for task in cell["related"]["1"]
    ]


def operational_equivalence(pairs: list[dict]) -> bool:
    if (len(pairs) != 6
            or len({row["world"] for row in pairs}) != 3
            or not all(math.isfinite(row["ordinary"])
                       and math.isfinite(row["prospective"])
                       and math.isfinite(row["gap"])
                       for row in pairs)):
        return False
    ordinary = np.asarray([row["ordinary"] for row in pairs], dtype=float)
    gaps = np.asarray([row["gap"] for row in pairs], dtype=float)
    mean_ordinary = float(np.mean(ordinary))
    task_ok = np.abs(gaps) <= 0.20 * np.maximum(np.abs(ordinary), 1e-12)
    world_ok = []
    for world in sorted({row["world"] for row in pairs}):
        selected = [row for row in pairs if row["world"] == world]
        world_gap = abs(float(np.mean([row["gap"] for row in selected])))
        world_ordinary = abs(float(np.mean([row["ordinary"] for row in selected])))
        world_ok.append(world_gap <= 0.20 * max(world_ordinary, 1e-12))
    return bool(
        float(np.mean(np.abs(gaps))) <= 0.10 * max(abs(mean_ordinary), 1e-12)
        and int(np.sum(task_ok)) >= 5
        and all(world_ok)
    )


def replicated_worse(pairs: list[dict]) -> bool:
    if (len(pairs) != 6
            or len({row["world"] for row in pairs}) != 3
            or not all(math.isfinite(row["ordinary"])
                       and math.isfinite(row["prospective"])
                       and math.isfinite(row["gap"])
                       for row in pairs)):
        return False
    world_gaps = []
    world_ordinary = []
    for world in sorted({row["world"] for row in pairs}):
        selected = [row for row in pairs if row["world"] == world]
        world_gaps.append(float(np.mean([row["gap"] for row in selected])))
        world_ordinary.append(float(np.mean([row["ordinary"] for row in selected])))
    return bool(
        all(gap > 0 for gap in world_gaps)
        and float(np.mean(world_gaps)) > float(np.std(world_gaps))
        and float(np.mean(world_gaps))
        > 0.10 * max(abs(float(np.mean(world_ordinary))), 1e-12)
    )


def paired_summary(rows: list[dict], method: str, support: int) -> dict[str, object]:
    selected = [
        row for row in rows
        if row["method"] == method and row["support"] == support
    ]
    pairs = []
    for world in sorted({row["world"] for row in selected}):
        for task_index in sorted({
            row["task_index"] for row in selected if row["world"] == world
        }):
            ordinary = next(row for row in selected
                            if row["world"] == world
                            and row["task_index"] == task_index
                            and row["arm"] == "ordinary")
            prospective = next(row for row in selected
                               if row["world"] == world
                               and row["task_index"] == task_index
                               and row["arm"] == "prospective")
            pairs.append({
                "world": world,
                "task_index": task_index,
                "ordinary": ordinary["final_query_scaled"],
                "prospective": prospective["final_query_scaled"],
                "gap": (prospective["final_query_scaled"]
                        - ordinary["final_query_scaled"]),
            })
    world_gaps = [
        float(np.mean([row["gap"] for row in pairs if row["world"] == world]))
        for world in sorted({row["world"] for row in pairs})
    ]
    complete = len(pairs) == 6 and len(world_gaps) == 3
    finite = complete and all(
        math.isfinite(row["ordinary"])
        and math.isfinite(row["prospective"])
        and math.isfinite(row["gap"])
        for row in pairs
    )
    return {
        "method": method,
        "support": support,
        "pairs": pairs,
        "ordinary_mean": float(np.mean([row["ordinary"] for row in pairs])),
        "prospective_mean": float(np.mean([row["prospective"] for row in pairs])),
        "gap_mean": float(np.mean([row["gap"] for row in pairs])),
        "gap_per_world": world_gaps,
        "gap_sd_worlds": float(np.std(world_gaps)),
        "complete": complete,
        "finite": finite,
        "operationally_equivalent": operational_equivalence(pairs),
        "prospective_replicated_worse": replicated_worse(pairs),
    }


def classify(method_summaries: dict[int, dict], standard_gap: float) -> dict[str, object]:
    sparse = method_summaries[1]
    capacity = method_summaries[128]
    reduction = (1.0 - sparse["gap_mean"] / standard_gap
                 if standard_gap != 0 else float("nan"))
    eligible = bool(sparse["complete"] and sparse["finite"]
                    and capacity["complete"] and capacity["finite"]
                    and math.isfinite(standard_gap) and standard_gap > 0)
    if not eligible:
        label = "unresolved_mixed"
    elif capacity["prospective_replicated_worse"]:
        label = "representational_opportunity_loss"
    elif (capacity["operationally_equivalent"]
          and sparse["prospective_replicated_worse"]):
        label = "sparse_identifiability_loss"
    elif (capacity["operationally_equivalent"]
          and sparse["operationally_equivalent"]
          and reduction >= 0.80):
        label = "optimizer_findability_loss"
    else:
        label = "unresolved_mixed"
    return {
        "classification": label,
        "eligible": eligible,
        "standard_gap": standard_gap,
        "sparse_gap_reduction": reduction,
        "k1": sparse,
        "k128": capacity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v6_clean"))
    parser.add_argument("--anchor", type=Path,
                        default=Path("reports/v6_clean_fertility.json"))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v6r_adaptation_geometry.json"))
    args = parser.parse_args()

    if args.worlds != [0, 1, 2]:
        raise SystemExit(
            "the frozen V6R protocol requires exactly --worlds 0 1 2"
        )

    torch.set_num_threads(1)
    sources = []
    paths = {}
    for world in args.worlds:
        for arm in ARMS:
            path = artifact_path(args.root, arm, world)
            paths[(arm, world)] = path
            sources.append(validate_artifact(path, arm, world))

    anchor_report = json.loads(args.anchor.read_text(encoding="utf-8"))
    config = load_config(args.config)
    rows = []
    anchor_rows = []
    tasks_by_world = {}
    for world in args.worlds:
        spec = MetaFamilySpec(
            families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2
        )
        world_config = replace(config.world, seed=world, tasks=spec.total_tasks)
        generated = generate_meta_world(world_config, spec)
        tasks = list(generated.novel_family_tasks)
        if len(tasks) != 2:
            raise SystemExit(f"world {world}: expected 2 novel family tasks, got {len(tasks)}")
        tasks_by_world[world] = tasks

        for arm in ARMS:
            base = load_learner(config, paths[(arm, world)], 12, kind="prospective")
            expected = anchor_expected(anchor_report, arm, world)
            observed = []
            for task_index, task in enumerate(tasks):
                standard = standard_fit(base, task, f"s0_w{world}_t{task_index}_{arm}")
                observed.append(standard["cumulative_scaled_query_cost"])
                row = {
                    "world": world,
                    "task_index": task_index,
                    "task_id": task.task_id,
                    "arm": arm,
                    "method": "standard_adam_005",
                    "support": 1,
                    **standard,
                }
                row["final_query_scaled"] = row["final_query_mse"] / SCALE
                rows.append(row)
            differences = [abs(a - b) for a, b in zip(observed, expected)]
            anchor_rows.append({
                "world": world, "arm": arm, "observed": observed,
                "expected": expected, "max_abs_difference": max(differences),
            })
            if max(differences) > 1e-12:
                raise SystemExit(
                    f"V6R anchor mismatch for {arm} world {world}: {differences}"
                )

    print("V6R anchor reproduced exactly; beginning frozen high-budget fits", flush=True)
    total_pairs = len(args.worlds) * 2
    completed_pairs = 0
    for world in args.worlds:
        for task_index, task in enumerate(tasks_by_world[world]):
            for arm in ARMS:
                base = load_learner(config, paths[(arm, world)], 12,
                                    kind="prospective")
                for support in (1, 128):
                    for method, learning_rate in (
                        ("adam_001", 0.01), ("adam_005", 0.05)
                    ):
                        result, _, _ = adam_fit(
                            base, task, support=support,
                            learning_rate=learning_rate, steps=2000,
                            label=f"{method}_k{support}_w{world}_t{task_index}_{arm}",
                            checkpoints=CHECKPOINTS,
                        )
                        if method == "adam_001" and not result["finite"]:
                            raise SystemExit(
                                f"non-finite primary cell: {arm} world {world} "
                                f"task {task_index} k={support}"
                            )
                        row = {
                            "world": world, "task_index": task_index,
                            "task_id": task.task_id, "arm": arm,
                            "method": method, "support": support, **result,
                        }
                        row["final_query_scaled"] = (
                            row["final_query_mse"] / SCALE
                        )
                        rows.append(row)

                    lbfgs = lbfgs_fit(
                        base, task, support,
                        f"lbfgs_k{support}_w{world}_t{task_index}_{arm}",
                    )
                    row = {
                        "world": world, "task_index": task_index,
                        "task_id": task.task_id, "arm": arm,
                        "method": "lbfgs", "support": support, **lbfgs,
                    }
                    row["final_query_scaled"] = row["final_query_mse"] / SCALE
                    rows.append(row)

                restart_candidates = []
                local_size = (
                    base.initial_residual_state.numel()
                    + base.route_size
                )
                for restart in range(3):
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [61037, world, task_index, restart]
                    ))
                    perturbation = 1e-3 * rng.normal(size=local_size)
                    result, model, probe_id = adam_fit(
                        base, task, support=1, learning_rate=0.01, steps=2000,
                        label=f"restart{restart}_w{world}_t{task_index}_{arm}",
                        perturbation=perturbation, record_query=False,
                        keep_model=True,
                    )
                    restart_candidates.append({
                        "restart": restart, "result": result,
                        "model": model, "probe_id": probe_id,
                    })
                selected = min(
                    restart_candidates,
                    key=lambda candidate: candidate["result"]["final_support_mse"],
                )
                _, _, query_x, query_y = tensors(task, 1)
                selected_query = mse(
                    selected["model"], selected["probe_id"], query_x, query_y
                )
                restart_result = dict(selected["result"])
                restart_result.update({
                    "selected_restart": selected["restart"],
                    "restart_candidates": [
                        {"restart": candidate["restart"], **candidate["result"]}
                        for candidate in restart_candidates
                    ],
                    "restart_support_losses": [
                        candidate["result"]["final_support_mse"]
                        for candidate in restart_candidates
                    ],
                    "final_query_mse": selected_query,
                    "finite": bool(
                        selected["result"]["finite"]
                        and math.isfinite(selected_query)
                    ),
                })
                for candidate in restart_candidates:
                    candidate["model"].forget_task(candidate["probe_id"])
                row = {
                    "world": world, "task_index": task_index,
                    "task_id": task.task_id, "arm": arm,
                    "method": "restart_adam_001", "support": 1,
                    **restart_result,
                }
                row["final_query_scaled"] = selected_query / SCALE
                rows.append(row)

            completed_pairs += 1
            print(
                f"V6R progress {completed_pairs}/{total_pairs} paired future tasks",
                flush=True,
            )

    primary_methods = ("adam_001", "adam_005", "lbfgs")
    paired = {
        method: {
            str(support): paired_summary(rows, method, support)
            for support in (1, 128)
        }
        for method in primary_methods
    }
    paired["restart_adam_001"] = {
        "1": paired_summary(rows, "restart_adam_001", 1)
    }
    standard = paired_summary(rows, "standard_adam_005", 1)
    paired["standard_adam_005"] = {"1": standard}

    method_classifications = {
        method: classify(
            {support: paired[method][str(support)] for support in (1, 128)},
            standard["gap_mean"],
        )
        for method in primary_methods
    }
    primary_label = method_classifications["adam_001"]["classification"]
    robustness_agreement = [
        method for method in ("adam_005", "lbfgs")
        if method_classifications[method]["classification"] == primary_label
    ]
    final_classification = (
        primary_label if robustness_agreement else "unresolved_optimizer_disagreement"
    )

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_plan": "V6R_ADAPTATION_GEOMETRY_PLAN.md",
        "protocol": {
            "worlds": args.worlds,
            "tasks_per_world": 2,
            "supports": [1, 128],
            "standard": {"optimizer": "adam", "lr": 0.05, "steps": 40},
            "primary": {"optimizer": "adam", "lr": 0.01, "steps": 2000},
            "robustness": [
                {"optimizer": "adam", "lr": 0.05, "steps": 2000},
                {"optimizer": "lbfgs", "lr": 1.0, "max_iter": 500,
                 "history_size": 100, "line_search": "strong_wolfe"},
            ],
            "restarts": 3,
            "restart_scale": 1e-3,
            "query_checkpoints": list(CHECKPOINTS),
            "sigma": SIGMA,
        },
        "anchor": anchor_rows,
        "anchor_max_abs_difference": max(
            row["max_abs_difference"] for row in anchor_rows
        ),
        "paired": paired,
        "method_classifications": method_classifications,
        "primary_classification": primary_label,
        "robustness_agreement": robustness_agreement,
        "final_classification": final_classification,
        "sources": sources,
        "rows": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.output)

    print("V6R ADAPTATION GEOMETRY")
    print(f"anchor max abs difference: {result['anchor_max_abs_difference']:.3g}")
    for method in primary_methods:
        classification = method_classifications[method]
        print(
            f"{method:<10} {classification['classification']:<36} "
            f"G1={classification['k1']['gap_mean']:+.4f} "
            f"G128={classification['k128']['gap_mean']:+.4f}"
        )
    print(f"final classification: {final_classification}")


if __name__ == "__main__":
    main()
