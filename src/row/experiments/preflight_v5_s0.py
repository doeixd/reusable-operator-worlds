"""Preflight for the S0 return-value-gain arm: prove the guards BITE.

This project has twice shipped a guard that printed PASS over zero rows
("CONTROL HOLDS" on an empty path index; task-private compressibility
returning 0.0 at every depth because it scored the wrong set). A check
whose denominator is zero is not a check, so every assertion here
reports the population it was computed over and REFUSES when that
population is empty.

Run it on a two-cell pair before committing a batch to the queue. It
answers "are the invariants this arm depends on actually observable in
the artifact?", not "did the process exit 0".
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

GAP_END = 40
LAST_SLEEP = 32


def signature(directory: Path) -> str | None:
    model = directory / "model.pt"
    if not model.exists():
        return None
    state = torch.load(model, weights_only=True)["model_state_dict"]
    keys = sorted(k for k in state if k.startswith("abstractions."))
    if not keys:
        return "none"
    blob = b"".join(state[k].detach().cpu().numpy().tobytes() for k in keys)
    return hashlib.sha256(blob).hexdigest()[:16]


def summary_of(directory: Path) -> dict:
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


def references(directory: Path) -> dict[str, int]:
    table = summary_of(directory).get("reference_table", {})
    return {k: int(v) for k, v in (table.get("task_reference") or {}).items()}


def rows(directory: Path) -> list[dict]:
    out = []
    for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("record_type") == "prequential":
            out.append(record)
    return out


def per_task(directory: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for record in rows(directory):
        index = record.get("task_index")
        if index is not None:
            out[index] = out.get(index, 0.0) + record["nll"]
    return out


def returning_task_ids(directory: Path) -> dict[int, str]:
    seen: dict[int, str] = {}
    for record in rows(directory):
        index = record.get("task_index")
        if index is not None and index >= GAP_END:
            seen.setdefault(index, record["task_id"])
    return seen


def retired_ids(directory: Path) -> list[int]:
    lifecycle = summary_of(directory).get("lifecycle") or {}
    return [
        int(record["abstraction_id"])
        for record in lifecycle.get("lineage", [])
        if record.get("retired_at_task") is not None
    ]


def births(directory: Path) -> tuple[int, int]:
    lifecycle = summary_of(directory).get("lifecycle") or {}
    lineage = lifecycle.get("lineage", [])
    late = sum(1 for record in lineage if record["born_at_task"] > LAST_SLEEP)
    return late, len(lineage)


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, ok: bool, detail: str) -> None:
        # The detail always carries the denominator, so a check that ran
        # over nothing is visible as such rather than green.
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} {detail}")
        if not ok:
            self.failures.append(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/v5_s0"))
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--gains", type=float, nargs="+", default=[1.0, 1.5])
    parser.add_argument("--reference-s-bar", type=float, default=61.0,
                        help="V4R O4 sealed value; g=1 should land near it")
    args = parser.parse_args()

    horizon = args.total - GAP_END
    report = Report()
    baselines: dict[str, float] = {}

    def cell(gain: float, arm: str) -> Path:
        tag = f"g{int(round(gain * 100)):03d}_s{args.slots}_N{args.total}_{arm}"
        return args.root / tag / f"world_{args.world}" / "lifecycle"

    print(f"S0 PREFLIGHT  N={args.total} (H_R={horizon})  world={args.world} "
          f"slots={args.slots}  gains={args.gains}\n")

    present = [
        g for g in args.gains
        if (cell(g, "retained") / "summary.json").exists()
        and (cell(g, "deleted") / "summary.json").exists()
    ]
    report.check("paired cells present", len(present) == len(args.gains),
                 f"{len(present)} / {len(args.gains)} gains have both arms")
    if not present:
        print("\n  REFUSED: nothing to check.")
        raise SystemExit(1)

    signatures = {g: signature(cell(g, "retained")) for g in present}
    distinct = set(signatures.values())
    report.check("abstraction identical across g", len(distinct) == 1,
                 f"{len(signatures)} gains compared, {len(distinct)} distinct "
                 f"checksum(s): {sorted(distinct)}")

    for gain in present:
        retained, deleted = cell(gain, "retained"), cell(gain, "deleted")
        print(f"\n  --- g={gain} ---")

        for arm, directory in (("retained", retained), ("deleted", deleted)):
            late, born = births(directory)
            report.check(f"no post-gap births [{arm}]", late == 0 and born > 0,
                         f"{late} late / {born} abstractions born")

        window = returning_task_ids(retained)
        report.check("return window size", len(window) == horizon,
                     f"{len(window)} / {horizon} returning tasks scored")

        retained_nll, deleted_nll = per_task(retained), per_task(deleted)
        pre = sum(
            deleted_nll.get(i, 0.0) - retained_nll.get(i, 0.0)
            for i in range(LAST_SLEEP)
        )
        report.check("pre-intervention delta zero", abs(pre) < 1e-9,
                     f"delta={pre:.3e} over {LAST_SLEEP} pre-gap tasks")

        gone = retired_ids(deleted)
        deleted_refs = references(deleted)
        still = sum(
            1 for task_id in returning_task_ids(deleted).values()
            if deleted_refs.get(task_id) in gone
        )
        report.check("retired abstraction unreachable", still == 0 and bool(gone),
                     f"{still} / {len(window)} returning tasks reference "
                     f"retired id(s) {gone}")

        retained_refs = references(retained)
        routed = sum(1 for task_id in window.values() if retained_refs.get(task_id) in gone)
        any_ref = sum(1 for task_id in window.values() if task_id in retained_refs)
        p_reuse = routed / len(window) if window else 0.0
        baseline = baselines.get("p_reuse")
        if gain == 1.0 or baseline is None:
            baselines["p_reuse"] = p_reuse
            report.check("p_reuse baseline recorded", len(window) > 0,
                         f"routed_to_A {routed} / {len(window)} = {p_reuse:.2f}  "
                         f"(any abstraction: {any_ref} / {len(window)})")
        else:
            # Relative, not absolute: the baseline world itself routes
            # only a minority of returning tasks to this one object, so
            # an absolute bound cannot detect gain-induced collapse.
            report.check("p_reuse >= 0.5x baseline", p_reuse >= 0.5 * baseline,
                         f"routed_to_A {routed} / {len(window)} = {p_reuse:.2f} "
                         f"vs baseline {baseline:.2f}  "
                         f"(any abstraction: {any_ref} / {len(window)})")

        scored = [
            i for i in range(GAP_END, args.total)
            if i in retained_nll and i in deleted_nll
        ]
        saving = sum(deleted_nll[i] - retained_nll[i] for i in scored)
        s_bar = saving / len(scored) if scored else 0.0
        report.check("participants scored", len(scored) == horizon,
                     f"{len(scored)} / {horizon} return-window tasks in both arms")
        # Decomposition: is the saving concentrated on tasks that
        # actually route to the carried object, or spread over the whole
        # return window? The two imply different mechanisms.
        on = [i for i in scored if retained_refs.get(window.get(i)) in gone]
        off = [i for i in scored if i not in on]
        mean = lambda idx: (
            sum(deleted_nll[i] - retained_nll[i] for i in idx) / len(idx)
            if idx else float("nan")
        )
        print(f"        s_bar = {s_bar:.1f} nats/use over {len(scored)} tasks")
        print(f"        s_conditional = {mean(on):.1f} over {len(on)} routed"
              f"   s_other = {mean(off):.1f} over {len(off)} unrouted")

        if gain == 1.0:
            near = abs(s_bar - args.reference_s_bar) / args.reference_s_bar
            report.check("g=1 reproduces V4R s_bar", near <= 0.25,
                         f"{s_bar:.1f} vs {args.reference_s_bar:.1f} "
                         f"({near:.0%} apart, 1 world)")

    print()
    if report.failures:
        print(f"  PREFLIGHT FAILED: {len(report.failures)} check(s) — "
              f"{', '.join(report.failures)}")
        raise SystemExit(1)
    print("  PREFLIGHT CLEAN — every check ran over a non-empty population.")


if __name__ == "__main__":
    main()
