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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/v5_causal"))
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 4])
    p.add_argument("--horizons", type=int, nargs="+", default=[48, 56, 64, 72])
    p.add_argument("--worlds", type=int, nargs="+", default=list(range(500, 510)))
    p.add_argument("--output", type=Path, default=Path("reports/v5_causal.json"))
    args = p.parse_args()

    print("V5.1 CAUSAL TEST — does H_R* scale with D(A)?")
    print("  prediction: H_R* proportional to residual rank (D(A) is linear in rank)\n")
    results = {}
    for rank in args.ranks:
        carry = LN2 * PROXY_BITS * SCALARS[rank]
        rows, excluded = [], 0
        for total in args.horizons:
            H = total - 40
            vals = []
            for w in args.worlds:
                rd = args.root / f"r{rank}_N{total}_retained" / f"world_{w}" / "lifecycle"
                dd = args.root / f"r{rank}_N{total}_deleted" / f"world_{w}" / "lifecycle"
                if not (rd / "summary.json").exists() or not (dd / "summary.json").exists():
                    continue
                bad = False
                for d in (rd, dd):
                    lc = json.loads((d / "summary.json").read_text(encoding="utf-8")).get("lifecycle")
                    if lc and any(x["born_at_task"] > 32 for x in lc.get("lineage", [])):
                        bad = True
                if bad:
                    excluded += 1
                    continue
                pr, pd = per_task(rd / "metrics.jsonl"), per_task(dd / "metrics.jsonl")
                vals.append(sum(pd.get(40 + j, 0.0) - pr.get(40 + j, 0.0) for j in range(H)))
            if vals:
                rows.append({"H": H, "n": len(vals), "c": float(np.mean(vals))})
        if not rows:
            continue
        sbar = float(np.mean([r["c"] / r["H"] for r in rows]))
        cross = crossing_of(rows, carry)
        results[rank] = {"carry": carry, "s_bar": sbar, "crossing": cross,
                         "predicted": carry / sbar if sbar else None,
                         "rows": rows, "excluded": excluded}
        print(f"  rank {rank}: D(A)={SCALARS[rank]} scalars, carry={carry:,.0f} nats, "
              f"s_bar={sbar:.1f}")
        for r in rows:
            print(f"      H_R={r['H']:>2} (n={r['n']:>2})  C_reacquire={r['c']:>7,.0f}  "
                  f"V_retain={r['c']-carry:>+8,.0f}")
        print(f"      crossing: {'n/a' if cross is None else f'{cross:.1f}'}"
              f"   excluded {excluded}\n")

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
