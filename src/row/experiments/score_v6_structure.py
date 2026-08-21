"""V6 secondary endpoints: H32 (geometry) and H33 (schema economics).

These are SECONDARY. Phi is the pass (H30), because a prospective
learner may find a representation unlike the teacher's that is
nonetheless better for future learning — treating geometry as the
result is the error V4 and V5 each punished once. What these answer is
the mechanistic question: IF fertility appears, did it appear by
restoring the higher-order structure V5 found missing?

H32: does R_effective rise above the ordinary arm's? V5 measured 0.19
for the frozen-basis learner; the V6 ordinary arm supplies its own
baseline, because unfreezing changed the protocol.

H33: does FACTORIZE now beat matched-budget COMPRESS on the learned
library, where V5 got 0/6? This is the bridge claim. If Phi rises and
H33 still fails, prospective pressure helps adaptation without building
an explicit higher-order library, which would mean meta-learning and
language formation are distinct steps.

Both reuse the V5 instruments unchanged, so the comparison to V5's
numbers is like-for-like.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(module: str, extra: list[str]) -> dict:
    """Run a V5 audit unchanged and return its report."""

    output = Path("reports") / f"_v6_tmp_{module.rsplit('.', 1)[-1]}.json"
    command = [sys.executable, "-m", module, "--output", str(output), *extra]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr[-400:]}
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/v6"))
    parser.add_argument("--arms", nargs="+",
                        default=["ordinary", "replay", "prospective", "supervised"])
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v6_structure.json"))
    args = parser.parse_args()

    worlds = [str(w) for w in args.worlds]
    rows = {}
    for arm in args.arms:
        root = args.root / arm
        if not root.exists():
            continue
        effective = run(
            "row.experiments.audit_effective_operator",
            ["--root", str(args.root), "--conditions", arm,
             "--worlds", *worlds, "--config", "configs/v5_h72.yaml"],
        )
        schema = run(
            "row.experiments.audit_learned_schema",
            ["--root", str(args.root), "--conditions", arm, "--worlds", *worlds],
        )
        rows[arm] = {
            "r_effective": effective.get("mean_r_effective"),
            "null": effective.get("null"),
            "factorize_wins": (schema.get("summary", {}).get(arm, {})
                               .get("factorize_wins")),
            "worlds": (schema.get("summary", {}).get(arm, {}).get("worlds")),
            "covered": (schema.get("summary", {}).get(arm, {})
                        .get("mean_covered_fraction")),
            "errors": [v.get("error") for v in (effective, schema) if "error" in v],
        }

    print("V6 SECONDARY ENDPOINTS (Phi is the pass; these say HOW)\n")
    print(f"  {'arm':<12} {'R_effective':>12} {'null':>7} "
          f"{'FACTORIZE':>11} {'covered':>8}")
    for arm, row in rows.items():
        r = row["r_effective"]
        null = row["null"]
        wins = row["factorize_wins"]
        covered = row["covered"]
        fmt = lambda v, spec: "n/a" if v is None else format(v, spec)
        share = "n/a" if wins is None else f"{wins}/{row['worlds']}"
        print(f"  {arm:<12} {fmt(r, '.3f'):>12} {fmt(null, '.3f'):>7} "
              f"{share:>11} {fmt(covered, '.2f'):>8}")

    ordinary = rows.get("ordinary", {})
    prospective = rows.get("prospective", {})
    print("\n  H32 (prospective raises R_effective above the ordinary arm)")
    if ordinary.get("r_effective") is None or prospective.get("r_effective") is None:
        print("    not evaluable: a required arm did not score")
    else:
        gain = prospective["r_effective"] - ordinary["r_effective"]
        print(f"    {'PASS' if gain > 0 else 'FAIL'}  "
              f"{ordinary['r_effective']:.3f} -> {prospective['r_effective']:.3f} "
              f"({gain:+.3f})")
    print("\n  H33 (FACTORIZE beats matched-budget COMPRESS; V5 gave 0/6)")
    wins = prospective.get("factorize_wins")
    if wins is None:
        print("    not evaluable")
    else:
        print(f"    {'PASS' if wins >= 2 else 'FAIL'}  "
              f"{wins}/{prospective.get('worlds')} worlds")
    print("\n  Reminder: H32 and H33 are mechanistic. A large Phi with a flat")
    print("  R_effective is a legitimate outcome and means the learner found")
    print("  another route to cheap adaptation (review 52).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
