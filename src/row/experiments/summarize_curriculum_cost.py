"""Descriptive cost accounting for the length curriculum (no verdict).

Reads the committed J1c and J1c-R reports and prices staged formation against
its matched non-staged control in the currencies this project charges:
example-gradients, forward/backward work (which differs because early stages
execute shorter programs), wall-clock seconds, and the TASKS and EXAMPLES the
curriculum consumes that the target distribution never provides.

Tier 0, descriptive: it introduces no threshold and decides nothing. Its
purpose is to state the curriculum's price before the SO2 protocol is frozen.
"""
from __future__ import annotations

import json
from pathlib import Path

from row.experiments.so1_storage import atomic_json, environment, now

J1C = Path("reports/j1c_curriculum.json")
J1CR = Path("reports/j1cr_replication.json")
OUTPUT = Path("reports/curriculum_cost.json")
BATCH = 2
EXAMPLES_PER_TASK = 128
EVAL_EXAMPLES = 256


def stage_rows(cell: dict) -> list[dict]:
    rows = []
    for key in sorted(cell["stages"], key=int):
        stage = cell["stages"][key]
        rows.append({"stage": int(key), "length": stage["length"], "tasks": stage["tasks"],
                     "updates": stage["updates"], "seconds": stage["seconds"],
                     "example_gradients": stage["updates"] * BATCH,
                     # One forward/backward applies `length` operator steps per example.
                     "operator_applications": stage["updates"] * BATCH * stage["length"]})
    return rows


def summarize(report_path: Path, staged_prefix: str, control_prefix: str | None) -> dict:
    report = json.loads(report_path.read_text())
    cells = report["cells"]
    worlds = sorted({k.split("_w")[1] for k in cells})
    out = {"report": report_path.as_posix(), "worlds": {}}
    for w in worlds:
        staged = stage_rows(cells[f"{staged_prefix}_w{w}"])
        row = {"staged_stages": staged,
               "staged_total": {k: sum(s[k] for s in staged)
                                for k in ("updates", "seconds", "example_gradients", "operator_applications")},
               "staged_extra_tasks": sum(s["tasks"] for s in staged if s["stage"] < 3),
               "staged_extra_examples": sum(s["tasks"] * (EXAMPLES_PER_TASK + EVAL_EXAMPLES)
                                            for s in staged if s["stage"] < 3),
               "staged_terminal_median": cells[f"{staged_prefix}_w{w}"]["terminal_median"]}
        if control_prefix is not None:
            control = stage_rows(cells[f"{control_prefix}_w{w}"])
            row["control_total"] = {k: sum(s[k] for s in control)
                                    for k in ("updates", "seconds", "example_gradients", "operator_applications")}
            row["control_terminal_median"] = cells[f"{control_prefix}_w{w}"]["terminal_median"]
            row["ratios_staged_over_control"] = {
                k: row["staged_total"][k] / row["control_total"][k]
                for k in ("updates", "seconds", "example_gradients", "operator_applications")}
        out["worlds"][w] = row
    return out


def main() -> int:
    result = {"status": "DESCRIPTIVE cost accounting; no threshold, no verdict",
              "generated_utc": now(), "environment": environment(),
              "seed5000": summarize(J1C, "STAGED", None),
              "seed3001": summarize(J1CR, "STAGED-R", "NON-STAGED-R")}
    # The seed-5000 control is SO1's reused learned cell: same 65,536 updates, length 3 only.
    for w, row in result["seed5000"]["worlds"].items():
        row["control_total"] = {"updates": 65536, "example_gradients": 65536 * BATCH,
                                "operator_applications": 65536 * BATCH * 3, "seconds": None}
        row["control_terminal_median"] = json.loads(J1C.read_text())["baseline"][w]
        row["ratios_staged_over_control"] = {
            k: row["staged_total"][k] / row["control_total"][k]
            for k in ("updates", "example_gradients", "operator_applications")}
    atomic_json(OUTPUT, result)
    for label in ("seed5000", "seed3001"):
        for w, row in result[label]["worlds"].items():
            ratios = row["ratios_staged_over_control"]
            print(f"{label} w{w}: staged {row['staged_terminal_median']:.4f} vs control "
                  f"{row['control_terminal_median']:.4f}; gradients x{ratios['example_gradients']:.2f}, "
                  f"operator applications x{ratios['operator_applications']:.2f}"
                  + (f", seconds x{ratios['seconds']:.2f}" if ratios.get("seconds") else "")
                  + f"; extra tasks {row['staged_extra_tasks']}, extra examples {row['staged_extra_examples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
