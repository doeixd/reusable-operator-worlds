"""SO0: read-only acquisition census over G5R Stages B-D.

Frozen in `SO0_CENSUS_PLAN.md` (POST_E6 program B0). Opened by the Stage D
classification `BUDGET_LIMITED`. Reads existing reports and lifetime artifacts,
copies or computes the five registered tables, and tabulates which budget axes
the existing oracle-route offline cells already isolate. No lifetime, no
training, no new cell, no verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code, write

WORLDS = ("0", "1", "2")
TASKS = 64
THRESHOLD = 0.05
PERSISTENCE_LATER_CHECKPOINTS = 2
PROTOCOL_ID = "SO0-census-v1"

INPUTS = {
    "diagnosis": Path("reports/rotated_g5r_diagnosis.json"),
    "lbfgs_correction": Path("reports/rotated_g5r_diagnosis_lbfgs_correction.json"),
    "interference": Path("reports/rotated_g5r_interference.json"),
    "g5r": Path("reports/rotated_g5r.json"),
}
ONLINE_ARTIFACTS = {
    "O_lr": Path("artifacts/g5r_rotated"),
    "O_or": Path("artifacts/g5r_interference/oracle_online"),
}
TRAJECTORY_CELLS = ("C_hi", "C_lo", "L_lo", "L_hi")
ORACLE_OFFLINE_CELLS = ("C_hi", "C_lo")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_distinct_tasks(batch: int, tasks: int = TASKS) -> float:
    """Expected distinct tasks in a batch drawn with replacement from `tasks`."""
    return tasks * (1.0 - (1.0 - 1.0 / tasks) ** batch)


def first_crossing(trajectory: dict[str, float], threshold: float = THRESHOLD):
    """(first crossing checkpoint or None, persistence label, monotone flag)."""
    points = sorted(((int(k), float(v)) for k, v in trajectory.items()))
    crossing = None
    for index, (checkpoint, value) in enumerate(points):
        if value <= threshold:
            crossing = (index, checkpoint)
            break
    monotone = all(b <= a for (_, a), (_, b) in zip(points, points[1:]))
    if crossing is None:
        return None, "not crossed", monotone
    index, checkpoint = crossing
    later = [v for _, v in points[index + 1 :]]
    if len(later) < PERSISTENCE_LATER_CHECKPOINTS:
        return checkpoint, "crossed, persistence unobservable", monotone
    persistent = all(v <= threshold for v in later[:PERSISTENCE_LATER_CHECKPOINTS])
    return checkpoint, ("persistent" if persistent else "crossed, not persistent"), monotone


def axis_isolation(cells: dict[str, dict]) -> dict:
    """Which budget axes each pair of oracle-route offline cells holds fixed.

    The three axes have two degrees of freedom (example_gradients = updates x
    batch, diversity is a function of batch), so no pair can vary exactly one
    axis. A pair CONTROLS an axis by holding it fixed while the other two move
    together; SO1 needs pairs controlling at least two distinct axes.
    """
    names = sorted(cells)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            fixed = [
                axis
                for axis in ("updates", "diversity", "example_gradients")
                if math.isclose(cells[a][axis], cells[b][axis], rel_tol=1e-9)
            ]
            pairs.append({"pair": [a, b], "axes_held_fixed": fixed, "controls_an_axis": bool(fixed)})
    controlled = sorted({axis for p in pairs for axis in p["axes_held_fixed"]})
    needed = [a for a in ("updates", "diversity", "example_gradients") if a not in controlled]
    return {"pairs": pairs, "axes_controlled_by_some_pair": controlled,
            "axes_not_yet_isolated_by_any_pair": needed}


def budget_row(name: str, updates: int, batch: int) -> dict:
    return {
        "cell": name,
        "updates": updates,
        "batch": batch,
        "diversity": expected_distinct_tasks(batch),
        "example_gradients": updates * batch,
    }


def per_task_end_of_task(path: Path) -> dict[str, float]:
    finals = {}
    with (path / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") == "task_summary":
                finals[row["task_id"]] = float(row["final_nmse"])
    return finals


def route_usage(path: Path, slots: int, steps: int) -> dict:
    routes = json.loads((path / "hard_routes.json").read_text(encoding="utf-8"))
    per_slot = [0] * slots
    per_position = [[0] * slots for _ in range(steps)]
    for route in routes.values():
        for position, slot in enumerate(route):
            per_slot[slot] += 1
            per_position[position][slot] += 1
    return {"routes": len(routes), "usage_counts": per_slot, "position_usage": per_position,
            "active_operators": sum(1 for c in per_slot if c > 0)}


def table_1(diagnosis: dict, correction: dict) -> dict:
    protocol = diagnosis["protocol"]
    grads = protocol["adam_steps"] * protocol["train_examples"]
    rows = []
    for key, cell in sorted(diagnosis["stage_b"]["cells"].items()):
        rows.append({
            "cell": key, "world": cell["world"], "primitive": cell["primitive"],
            "arm": cell["arm"],
            "initial_query_nmse": cell["initial_query_nmse"],
            "final_query_nmse": cell["final_query_nmse"],
            "final_train_nmse": cell["final_train_nmse"],
            "passes": cell["passes"],
            "example_gradients": grads if cell["arm"] != "H-LBFGS" else "NOT COMPARABLE (L-BFGS)",
        })
    for key, cell in sorted(correction["cells"].items()):
        rows.append({
            "cell": key + " (correction)", "world": cell.get("world"),
            "primitive": cell.get("primitive"), "arm": cell.get("arm", "H-LBFGS"),
            "initial_query_nmse": cell.get("initial_query_nmse"),
            "final_query_nmse": cell.get("final_query_nmse"),
            "final_train_nmse": cell.get("final_train_nmse"),
            "passes": cell.get("passes"),
            "example_gradients": "NOT COMPARABLE (L-BFGS)",
        })
    return {
        "note": "Stage B stores endpoints only; no trajectory exists and none is interpolated.",
        "adam_example_gradients_per_cell": grads,
        "arms": diagnosis["stage_b"]["arms"],
        "correction_summary": correction.get("summary"),
        "rows": rows,
    }


def table_2(interference: dict) -> dict:
    protocol = interference["protocol"]
    budgets = {
        "C_hi": budget_row("C_hi", protocol["stage_c_updates"], protocol["stage_c_batch"]),
        "C_lo": budget_row("C_lo", protocol["lifetime_updates"], protocol["lifetime_batch"]),
    }
    rows = {}
    for name in ORACLE_OFFLINE_CELLS:
        rows[name] = {
            "budget": budgets[name],
            "median_by_checkpoint": {
                w: {k: v["median"] for k, v in interference["cells"][name][w]["checkpoints"].items()}
                for w in WORLDS
            },
            "terminal_median": {w: interference["cells"][name][w]["terminal_median"] for w in WORLDS},
        }
    return {
        "note": "C_hi and C_lo differ on all three budget axes and isolate none.",
        "rows": rows,
        "axis_isolation": axis_isolation(budgets),
    }


def table_3(interference: dict) -> dict:
    out = {}
    for name in TRAJECTORY_CELLS:
        out[name] = {}
        for w in WORLDS:
            trajectory = {k: v["median"] for k, v in interference["cells"][name][w]["checkpoints"].items()}
            crossing, label, monotone = first_crossing(trajectory)
            out[name][w] = {
                "trajectory": trajectory, "first_crossing": crossing,
                "persistence": label, "monotone": monotone,
            }
    return out


def table_4(interference: dict) -> dict:
    out = {}
    for name, root in ONLINE_ARTIFACTS.items():
        out[name] = {}
        for w in WORLDS:
            cell = interference["cells"][name][w]
            end_of_task = per_task_end_of_task(root / f"world_{w}")
            terminal = cell["final_per_task"]
            if set(end_of_task) != set(terminal):
                raise RuntimeError(f"{name} w{w}: task sets differ between artifact and report")
            pairs = [(end_of_task[t], terminal[t]) for t in sorted(terminal)]
            worse = sum(1 for e, t in pairs if t > e)
            ratios = sorted(t / e for e, t in pairs if e > 0)
            out[name][w] = {
                "end_of_task_median": cell["end_of_task_median"],
                "terminal_median": cell["terminal_median"],
                "terminal_minus_end_of_task_median": cell["terminal_minus_end_of_task_median"],
                "tasks": len(pairs),
                "fraction_terminal_worse": worse / len(pairs),
                "median_terminal_over_end_of_task": ratios[len(ratios) // 2],
            }
    return out


def table_5(interference: dict, diagnosis: dict) -> dict:
    slots = interference["protocol"]["slots"]
    steps = len(interference["cells"]["C_lo"]["0"]["routing"]["position_usage"])
    out = {"offline": {}, "online": {}, "orthogonality_note":
           "Householder rotations are orthogonal by construction; the Stage A/B "
           "orthogonality_max_abs_error is an implementation check, not a learned diagnostic."}
    for name in ("C_lo", "L_lo", "L_hi"):
        out["offline"][name] = {w: interference["cells"][name][w]["routing"] for w in WORLDS}
    for name, root in ONLINE_ARTIFACTS.items():
        out["online"][name] = {w: route_usage(root / f"world_{w}", slots, steps) for w in WORLDS}
    out["stage_ab_orthogonality_max_abs_error"] = {
        "stage_a_max": max(c["orthogonality_max_abs_error"] for c in diagnosis["stage_a"]["cells"].values()),
        "stage_b_max": max(c["orthogonality_max_abs_error"] for c in diagnosis["stage_b"]["cells"].values()),
    }
    return out


def resolve_expectations(t1, t2, t3, t4) -> dict:
    e1 = not any(p["controls_an_axis"] for p in t2["axis_isolation"]["pairs"])
    c_hi = t3["C_hi"]
    e2 = (
        all(c_hi[w]["persistence"] == "crossed, persistence unobservable" for w in ("1", "2"))
        and c_hi["0"]["persistence"] == "not crossed"
        and all(t3[n][w]["persistence"] == "not crossed" for n in ("C_lo", "L_lo", "L_hi") for w in WORLDS)
    )
    e3 = all(not t3["L_hi"][w]["monotone"] for w in WORLDS) and all(
        t3[n][w]["monotone"] for n in ("C_lo", "C_hi", "L_lo") for w in WORLDS
    )
    e4 = all(t4[n][w]["fraction_terminal_worse"] > 0.9 for n in t4 for w in WORLDS)
    e5 = all(arm["passes"] for arm in t1["arms"].values())
    return {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/so0_census.json"))
    args = parser.parse_args()
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("preregistration check failed")
    if subprocess.run(["python", "tools/check_invalid.py"]).returncode != 0:
        raise SystemExit("invalid-artifact check failed")
    require_clean_code(args.output)

    loaded = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in INPUTS.items()}
    hashes = {str(p): sha256(p) for p in INPUTS.values()}
    for root in ONLINE_ARTIFACTS.values():
        for w in WORLDS:
            for f in ("metrics.jsonl", "hard_routes.json"):
                hashes[str(root / f"world_{w}" / f)] = sha256(root / f"world_{w}" / f)
    if not loaded["interference"].get("complete"):
        raise SystemExit("Stage D report is not complete")

    t1 = table_1(loaded["diagnosis"], loaded["lbfgs_correction"])
    t2 = table_2(loaded["interference"])
    t3 = table_3(loaded["interference"])
    t4 = table_4(loaded["interference"])
    t5 = table_5(loaded["interference"], loaded["diagnosis"])
    out = {
        "frozen_plan": "SO0_CENSUS_PLAN.md",
        "git_commit": git_commit(),
        "protocol": {
            "id": PROTOCOL_ID, "threshold": THRESHOLD, "tasks": TASKS,
            "persistence_later_checkpoints": PERSISTENCE_LATER_CHECKPOINTS,
            "diversity_formula": "tasks * (1 - (1 - 1/tasks) ** batch)",
            "stage_d_classification": loaded["interference"].get("classification"),
            "g5r_verdict": loaded["g5r"]["verdict"].get("G5R"),
            "input_sha256": hashes,
        },
        "T1_isolated_operators": t1,
        "T2_joint_oracle_budget": t2,
        "T3_transition_persistence": t3,
        "T4_online_terminal_vs_end_of_task": t4,
        "T5_routing_diagnostics": t5,
        "expectations": resolve_expectations(t1, t2, t3, t4),
        "so1_input": {
            "oracle_budget_bracket_example_gradients": [
                t2["rows"]["C_lo"]["budget"]["example_gradients"],
                t2["rows"]["C_hi"]["budget"]["example_gradients"],
            ],
            "axes_not_yet_isolated_by_any_pair": t2["axis_isolation"]["axes_not_yet_isolated_by_any_pair"],
        },
        "verdict": None,
        "complete": True,
    }
    write(out, args.output)
    print("expectations:", out["expectations"])
    print("axes SO1 must isolate:", out["so1_input"]["axes_not_yet_isolated_by_any_pair"])
    for name in TRAJECTORY_CELLS:
        print(name, {w: t3[name][w]["persistence"] for w in WORLDS})
    for name in t4:
        print(name, {w: round(t4[name][w]["fraction_terminal_worse"], 3) for w in WORLDS})


if __name__ == "__main__":
    main()
