"""Independent checker for reports/so0_census.json (SO0_CENSUS_PLAN.md acceptance).

Re-reads the source reports and artifacts and verifies that every copied
number in the census equals its source, every computed quantity is finite and
recomputable from the stated formula, the input hashes match the files on
disk, and the expectations are consistent with the tables. Imports nothing
from the census module except its constants.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

CENSUS = Path("reports/so0_census.json")
failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def finite_tree(obj, path="census"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            finite_tree(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            finite_tree(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        check(math.isfinite(obj), f"non-finite at {path}")


def main() -> int:
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    check(census.get("complete") is True, "census not complete")
    check(census.get("verdict") is None, "census must carry no verdict")
    finite_tree(census)

    for path, digest in census["protocol"]["input_sha256"].items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        check(actual == digest, f"input hash changed: {path}")

    interference = json.loads(Path("reports/rotated_g5r_interference.json").read_text(encoding="utf-8"))
    diagnosis = json.loads(Path("reports/rotated_g5r_diagnosis.json").read_text(encoding="utf-8"))
    threshold = census["protocol"]["threshold"]
    tasks = census["protocol"]["tasks"]
    later = census["protocol"]["persistence_later_checkpoints"]

    # T1: copied endpoints.
    for row in census["T1_isolated_operators"]["rows"]:
        if "(correction)" in row["cell"]:
            continue
        src = diagnosis["stage_b"]["cells"][row["cell"]]
        for key in ("initial_query_nmse", "final_query_nmse", "final_train_nmse", "passes"):
            check(row[key] == src[key], f"T1 {row['cell']} {key} differs from source")

    # T2: budgets and copied medians.
    for name, row in census["T2_joint_oracle_budget"]["rows"].items():
        b = row["budget"]
        check(b["example_gradients"] == b["updates"] * b["batch"], f"T2 {name} gradients")
        expected = tasks * (1 - (1 - 1 / tasks) ** b["batch"])
        check(math.isclose(b["diversity"], expected), f"T2 {name} diversity formula")
        for w, medians in row["median_by_checkpoint"].items():
            for ck, value in medians.items():
                check(value == interference["cells"][name][w]["checkpoints"][ck]["median"],
                      f"T2 {name} w{w} ckpt {ck} median differs")

    # T3: recompute crossing and persistence from the copied trajectory.
    for name, worlds in census["T3_transition_persistence"].items():
        for w, cell in worlds.items():
            points = sorted((int(k), v) for k, v in cell["trajectory"].items())
            src = {k: v["median"] for k, v in interference["cells"][name][w]["checkpoints"].items()}
            check(cell["trajectory"] == src, f"T3 {name} w{w} trajectory differs from source")
            crossing = next((i for i, (_, v) in enumerate(points) if v <= threshold), None)
            if crossing is None:
                check(cell["first_crossing"] is None and cell["persistence"] == "not crossed",
                      f"T3 {name} w{w} crossing")
            else:
                check(cell["first_crossing"] == points[crossing][0], f"T3 {name} w{w} crossing point")
                rest = [v for _, v in points[crossing + 1:]]
                if len(rest) < later:
                    check(cell["persistence"] == "crossed, persistence unobservable", f"T3 {name} w{w}")
                else:
                    label = "persistent" if all(v <= threshold for v in rest[:later]) else "crossed, not persistent"
                    check(cell["persistence"] == label, f"T3 {name} w{w} persistence")
            monotone = all(b <= a for (_, a), (_, b) in zip(points, points[1:]))
            check(cell["monotone"] == monotone, f"T3 {name} w{w} monotone")

    # T4: recompute per-task fraction from the artifacts.
    roots = {"O_lr": Path("artifacts/g5r_rotated"), "O_or": Path("artifacts/g5r_interference/oracle_online")}
    for name, worlds in census["T4_online_terminal_vs_end_of_task"].items():
        for w, cell in worlds.items():
            src = interference["cells"][name][w]
            check(cell["terminal_median"] == src["terminal_median"], f"T4 {name} w{w} terminal median")
            check(cell["end_of_task_median"] == src["end_of_task_median"], f"T4 {name} w{w} eot median")
            finals = {}
            with (roots[name] / f"world_{w}" / "metrics.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("record_type") == "task_summary":
                        finals[row["task_id"]] = float(row["final_nmse"])
            terminal = src["final_per_task"]
            worse = sum(1 for t in terminal if terminal[t] > finals[t])
            check(cell["tasks"] == len(terminal), f"T4 {name} w{w} task count")
            check(math.isclose(cell["fraction_terminal_worse"], worse / len(terminal)),
                  f"T4 {name} w{w} fraction_terminal_worse")

    # Expectations consistent with tables.
    t3 = census["T3_transition_persistence"]
    t4 = census["T4_online_terminal_vs_end_of_task"]
    e = census["expectations"]
    check(e["E1"] == (not any(p["controls_an_axis"] for p in census["T2_joint_oracle_budget"]["axis_isolation"]["pairs"])), "E1")
    check(e["E3"] == (all(not t3["L_hi"][w]["monotone"] for w in t3["L_hi"])
                      and all(t3[n][w]["monotone"] for n in ("C_lo", "C_hi", "L_lo") for w in t3[n])), "E3")
    check(e["E4"] == all(t4[n][w]["fraction_terminal_worse"] > 0.9 for n in t4 for w in t4[n]), "E4")

    print("expectations:", e)
    if failures:
        print("SO0 CHECK FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("SO0 CHECK OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
