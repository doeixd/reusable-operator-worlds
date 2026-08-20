"""V5.1: is the amortization relation a LAW or an accounting identity?

`RETAIN iff H_R * s_bar > lambda * D(A)` was confirmed on the V4R sealed
block at a single value of `D(A)`. That confirmation cannot distinguish
a quantitative law from a tautology, because the crossing is by
definition at `carry / s_bar`: fixing `s_bar` and moving `carry` moves
the crossing by arithmetic alone.

The discriminating test manipulates `D(A)` AT THE GENERATOR. Residual
rank sets the abstraction's size exactly linearly -- 99, 198, 396
scalars at rank 1, 2, 4 -- so the law makes a sharp prediction that
nothing in the scoring can force:

    H_R*(rank)  proportional to  D(A; rank)

If the observed crossing at rank 1 is about half that at rank 2, and
rank 4 about double, the relation predicts across regimes it was not
fitted to. If the crossings barely move, the law is an artifact of one
operating point and `s_bar` must be co-varying with rank in a way that
cancels it -- which the scorer therefore reports explicitly.

Scored on the RETURN WINDOW, the only window in which the two arms are
still paired after a mid-lifetime deletion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)
SCALARS = {1: 99, 2: 198, 4: 396}   # residual_u + residual_v + residual_b
PROXY_BITS = 8


def per_task(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("record_type") != "prequential":
            continue
        i = r.get("task_index")
        if i is not None:
            out[i] = out.get(i, 0.0) + r["nll"]
    return out


def crossing_of(rows, carry) -> float | None:
    for a, b in zip(rows, rows[1:]):
        va, vb = a["c"] - carry, b["c"] - carry
        if va <= 0 < vb:
            return a["H"] + (b["H"] - a["H"]) * (-va) / (vb - va)
    return None


def condition_root(root: Path, template: str, rank: int, total: int, arm: str) -> Path:
    return root / template.format(rank=rank, total=total, arm=arm)


def score_rank(root: Path, template: str, rank: int, horizons: list[int],
               worlds: list[int], gap_end: int, last_sleep: int) -> dict:
    carry = LN2 * PROXY_BITS * SCALARS[rank]
    rows, excluded = [], 0
    for total in horizons:
        horizon = total - gap_end
        if horizon <= 0:
            continue
        vals = []
        for world in worlds:
            retained = (condition_root(root, template, rank, total, "retained")
                        / f"world_{world}" / "lifecycle")
            deleted = (condition_root(root, template, rank, total, "deleted")
                       / f"world_{world}" / "lifecycle")
            if not (retained / "summary.json").exists() or not (deleted / "summary.json").exists():
                continue
            bad = False
            for directory in (retained, deleted):
                lifecycle = json.loads((directory / "summary.json").read_text(encoding="utf-8")).get("lifecycle")
                if lifecycle and any(record["born_at_task"] > last_sleep
                                     for record in lifecycle.get("lineage", [])):
                    bad = True
            if bad:
                excluded += 1
                continue
            retained_nll = per_task(retained / "metrics.jsonl")
            deleted_nll = per_task(deleted / "metrics.jsonl")
            vals.append(sum(deleted_nll.get(gap_end + offset, 0.0)
                            - retained_nll.get(gap_end + offset, 0.0)
                            for offset in range(horizon)))
        if vals:
            rows.append({"H": horizon, "n": len(vals), "c": float(np.mean(vals))})
    if not rows:
        return {}
    s_bar = float(np.mean([row["c"] / row["H"] for row in rows]))
    crossing = crossing_of(rows, carry)
    return {
        "carry": carry,
        "s_bar": s_bar,
        "crossing": crossing,
        "predicted": carry / s_bar if s_bar else None,
        "rows": rows,
        "excluded": excluded,
        "scalars": SCALARS[rank],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/v5_causal"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--horizons", type=int, nargs="+", default=[48, 56, 64, 72])
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(500, 510)))
    parser.add_argument("--dir-template", default="r{rank}_N{total}_{arm}",
                        help="condition directory; horizon layout uses N{total}_{arm}")
    parser.add_argument("--gap-end", type=int, default=40)
    parser.add_argument("--last-sleep", type=int, default=32)
    parser.add_argument("--diagnostic", action="store_true",
                        help="banner the run as a development diagnostic, not H19")
    parser.add_argument("--output", type=Path, default=Path("reports/v5_causal.json"))
    args = parser.parse_args()

    title = "V5.1 CAUSAL TEST — does H_R* scale with D(A)?"
    if args.diagnostic:
        title = "DIAGNOSTIC (not H19) — " + title
    print(title)
    print("  prediction: H_R* proportional to residual rank (D(A) is linear in rank)\n")
    results = {}
    for rank in args.ranks:
        scored = score_rank(
            args.root, args.dir_template, rank, args.horizons, args.worlds,
            args.gap_end, args.last_sleep,
        )
        if not scored:
            continue
        results[rank] = scored
        print(f"  rank {rank}: D(A)={scored['scalars']} scalars, carry={scored['carry']:,.0f} nats, "
              f"s_bar={scored['s_bar']:.1f}")
        for row in scored["rows"]:
            print(f"      H_R={row['H']:>2} (n={row['n']:>2})  C_reacquire={row['c']:>7,.0f}  "
                  f"V_retain={row['c']-scored['carry']:>+8,.0f}")
        crossing = scored["crossing"]
        predicted = scored["predicted"]
        crossing_text = "n/a" if crossing is None else f"{crossing:.1f}"
        predicted_text = "n/a" if predicted is None else f"{predicted:.1f}"
        print(f"      crossing {crossing_text}   predicted {predicted_text}   "
              f"excluded {scored['excluded']}\n")

    print("  SCALING CHECK")
    ranks = sorted(k for k in results if results[k]["crossing"] is not None)
    if len(ranks) >= 2:
        lo, hi = ranks[0], ranks[-1]
        d_ratio = SCALARS[hi] / SCALARS[lo]
        h_ratio = results[hi]["crossing"] / results[lo]["crossing"]
        print(f"    D(A) ratio  rank{hi}/rank{lo} = {d_ratio:.2f}")
        print(f"    H_R* ratio  rank{hi}/rank{lo} = {h_ratio:.2f}")
        err = abs(h_ratio - d_ratio) / d_ratio
        print(f"    relative error {err:.1%}   "
              f"{'PROPORTIONAL (law holds)' if err < 0.25 else 'NOT PROPORTIONAL'}")
        sbars = [results[k]["s_bar"] for k in ranks]
        print(f"    s_bar across ranks: {['%.1f' % s for s in sbars]} — if these move with "
              f"rank, the scaling is confounded")
    else:
        print("    insufficient crossings to test scaling")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {str(k): v for k, v in results.items()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
