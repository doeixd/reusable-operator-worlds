"""Score O4, the retention amortization law, on the V4R sealed block.

Thresholds transcribed from `V4R_CONFIRMATION_PLAN.md`, frozen at commit
2aec65c and hashed into `tools/check_prereg.py`, before any sealed
retention run existed.

O4 registers a MECHANISM, not just a sign, and it is the one positive
claim in an otherwise negative block:

    RETAIN A  iff  H_R * s_bar  >  lambda * D(A)

with the crossing predicted from independently measured quantities
rather than fitted. Development: predicted 17.1 returning tasks,
observed 17.9.

The protocol is the controlled one. Gap fixed at (32,40); the last sleep
is at the gap, so no replacement abstraction can be born and
`D_retain - D_delete = D(A)` exactly. That control is ASSERTED here, not
assumed: any world with a post-gap birth is excluded and reported,
because without it the carry term is endogenous and the law does not
apply (see PREDICTIONS.md, "Open library: C_reacquire survives, the
DECISION RULE does not").

Scored on the RETURN WINDOW, never end-of-lifetime J: a mid-lifetime
deletion stops the arms being paired the moment it changes what gets
promoted next.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)
RESIDUAL_SCALARS = 1584
CARRY = LN2 * RESIDUAL_SCALARS

# --- frozen thresholds, transcribed from V4R_CONFIRMATION_PLAN.md ---
O4_CROSSING_INTERVAL = (14.0, 22.0)     # returning tasks
O4_SBAR_INTERVAL = (50.0, 75.0)         # nats per returning task
O4_REQUIRE_MONOTONE = True


def per_task(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("record_type") != "prequential":
            continue
        index = record.get("task_index")
        if index is not None:
            out[index] = out.get(index, 0.0) + record["nll"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/v4r_sealed/o4"))
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(400, 430)))
    parser.add_argument("--horizons", type=int, nargs="+", default=[48, 56, 64, 72])
    parser.add_argument("--output", type=Path, default=Path("reports/v4r_sealed_retention.json"))
    args = parser.parse_args()

    print("V4R O4 — RETENTION LAW, sealed block, vs plan @ 2aec65c")
    print(f"  carry = lambda * D(A) = {CARRY:,.0f} nats")
    print(f"  predicted crossing H_R* = carry / s_bar\n")
    print(f"  {'H_R':>4} {'worlds':>7} {'C_reacquire':>12} {'s_bar':>7} {'V_retain':>10} {'verdict':>8}")

    rows, excluded = [], 0
    for total in args.horizons:
        H = total - 40
        values = []
        for world in args.worlds:
            base = args.root / f"N{total}_%s" / f"world_{world}" / "lifecycle"
            r_dir, d_dir = Path(str(base) % "retained"), Path(str(base) % "deleted")
            if not (r_dir / "summary.json").exists() or not (d_dir / "summary.json").exists():
                continue
            # ASSERT the control: no birth after the gap in either arm.
            bad = False
            for d in (r_dir, d_dir):
                lc = json.loads((d / "summary.json").read_text(encoding="utf-8")).get("lifecycle")
                if lc and any(x["born_at_task"] > 32 for x in lc.get("lineage", [])):
                    bad = True
            if bad:
                excluded += 1
                continue
            pr, pd = per_task(r_dir / "metrics.jsonl"), per_task(d_dir / "metrics.jsonl")
            values.append(sum(pd.get(40 + j, 0.0) - pr.get(40 + j, 0.0) for j in range(H)))
        if not values:
            continue
        mean = float(np.mean(values))
        rows.append({"H_R": H, "n": len(values), "c_reacquire": mean,
                     "s_bar": mean / H, "v_retain": mean - CARRY})
        print(f"  {H:>4} {len(values):>7} {mean:>12,.0f} {mean/H:>7.1f} {mean-CARRY:>10,.0f} "
              f"{'RETAIN' if mean > CARRY else 'DELETE':>8}")

    vs = [r["v_retain"] for r in rows]
    monotone = all(b >= a for a, b in zip(vs, vs[1:]))
    sbar = float(np.mean([r["s_bar"] for r in rows]))
    crossing = None
    for a, b in zip(rows, rows[1:]):
        if a["v_retain"] <= 0 < b["v_retain"]:
            span = b["v_retain"] - a["v_retain"]
            crossing = a["H_R"] + (b["H_R"] - a["H_R"]) * (-a["v_retain"]) / span
    predicted = CARRY / sbar if sbar else float("nan")

    ok_mono = monotone or not O4_REQUIRE_MONOTONE
    ok_cross = crossing is not None and O4_CROSSING_INTERVAL[0] <= crossing <= O4_CROSSING_INTERVAL[1]
    ok_sbar = O4_SBAR_INTERVAL[0] <= sbar <= O4_SBAR_INTERVAL[1]

    print(f"\n  excluded for post-gap births: {excluded}")
    print(f"  monotone in H_R                       {'PASS' if ok_mono else 'FAIL'}")
    print(f"  s_bar {sbar:.1f} nats/use (need {O4_SBAR_INTERVAL[0]:.0f}-{O4_SBAR_INTERVAL[1]:.0f})"
          f"        {'PASS' if ok_sbar else 'FAIL'}")
    print(f"  crossing {crossing if crossing is None else round(crossing,1)} "
          f"(need {O4_CROSSING_INTERVAL[0]:.0f}-{O4_CROSSING_INTERVAL[1]:.0f})"
          f"            {'PASS' if ok_cross else 'FAIL'}")
    print(f"  derived prediction H_R* = {predicted:.1f} (unfitted)")
    print(f"\n  O4: {'PASS' if all([ok_mono, ok_cross, ok_sbar]) else 'FAIL'} "
          f"({sum([ok_mono, ok_cross, ok_sbar])}/3)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "plan_commit": "2aec65c", "rows": rows, "s_bar": sbar,
        "crossing": crossing, "predicted": predicted,
        "monotone": monotone, "excluded_post_gap_births": excluded,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
