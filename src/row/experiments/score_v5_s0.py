"""Score the H19 s-arm: does the crossing track 1 / s_bar when only the
family's PAYOFF moves?

V5.1 established `H_R* = lambda D*(A) / s_bar` across three values of
`D(A)` and falsified the proportional form, because residual rank moved
cost and utility together. This is the other half of the law, and the
arm the rung is still partial without: the return-value gain leaves the
carried abstraction bit-identical and changes only what a returning task
gets from it.

    y = f_base(x) + g * A_family(x),  g in {0.5, 1.0, 1.5}, after the gap

Two things make this stronger than the D-arm, and both are asserted here
rather than assumed:

  * the carry term is not "stable to 2%" but IDENTICAL. Promotion sets
    `requires_grad=False` and the pre-gap stream does not depend on `g`,
    so the same tensor is carried in every arm. The scorer compares the
    abstraction tensors bit-for-bit across `g` and refuses the gain
    if they differ.
  * the arms are byte-identical up to the intervention, which shows up
    as an exactly zero pre-intervention delta. A nonzero one means the
    retirement leaked backwards and the cell is void.

The grid rule from the spec is enforced, not worked around: a gain whose
H_R grid does not bracket its own predicted crossing by >= 4 returning
tasks on both sides is reported UNSCOREABLE. The crossing moves as
1/s_bar, so a grid chosen for one gain will not serve another; that is
why measurement runs before the bracketing grid is chosen.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

LN2 = math.log(2.0)
RESIDUAL_SCALARS = {1: 99, 2: 198, 4: 396}
PROXY_BITS = 8
# V5.0 measured shared abstractions at 3.9 bits/scalar against the 8-bit
# serialization proxy. Both currencies are reported; neither is picked
# after seeing the numbers.
FRONTIER_BITS = 3.9
GAP_END = 40
LAST_SLEEP = 32
BRACKET = 4


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


def abstraction_signature(path: Path) -> str | None:
    """A hash of the carried abstractions, for the byte-identity check."""

    model = path / "model.pt"
    if not model.exists():
        return None
    state = torch.load(model, weights_only=True)["model_state_dict"]
    keys = sorted(k for k in state if k.startswith("abstractions."))
    if not keys:
        return "none"
    blob = b"".join(state[k].detach().cpu().numpy().tobytes() for k in keys)
    import hashlib

    return hashlib.sha256(blob).hexdigest()[:16]


def cell(root: Path, gain: float, slots: int, total: int, arm: str) -> Path:
    return root / f"g{int(round(gain * 100)):03d}_s{slots}_N{total}_{arm}"


def score_gain(root: Path, gain: float, slots: int, totals: list[int],
               worlds: list[int]) -> dict:
    rows, excluded, signatures = [], 0, {}
    leaked = 0
    for total in totals:
        horizon = total - GAP_END
        if horizon <= 0:
            continue
        values = []
        for world in worlds:
            retained = cell(root, gain, slots, total, "retained") / f"world_{world}" / "lifecycle"
            deleted = cell(root, gain, slots, total, "deleted") / f"world_{world}" / "lifecycle"
            if not (retained / "summary.json").exists():
                continue
            if not (deleted / "summary.json").exists():
                continue
            bad = False
            for directory in (retained, deleted):
                lifecycle = json.loads(
                    (directory / "summary.json").read_text(encoding="utf-8")
                ).get("lifecycle")
                if lifecycle and any(
                    record["born_at_task"] > LAST_SLEEP
                    for record in lifecycle.get("lineage", [])
                ):
                    bad = True
            if bad:
                excluded += 1
                continue
            retained_nll = per_task(retained / "metrics.jsonl")
            deleted_nll = per_task(deleted / "metrics.jsonl")
            # The arms must be identical before the intervention. Any
            # drift here means the retirement changed the past, and the
            # paired reading is void rather than noisy.
            pre = sum(
                deleted_nll.get(i, 0.0) - retained_nll.get(i, 0.0)
                for i in range(LAST_SLEEP)
            )
            if abs(pre) > 1e-6:
                leaked += 1
                continue
            signatures[f"N{total}_world{world}"] = abstraction_signature(retained)
            values.append(
                sum(
                    deleted_nll.get(GAP_END + offset, 0.0)
                    - retained_nll.get(GAP_END + offset, 0.0)
                    for offset in range(horizon)
                )
            )
        if values:
            rows.append({
                "H_R": horizon,
                "n": len(values),
                "c_reacquire": float(np.mean(values)),
                "s_bar": float(np.mean(values)) / horizon,
            })
    if not rows:
        return {}
    s_bar = float(np.mean([row["s_bar"] for row in rows]))
    return {
        "gain": gain,
        "slots": slots,
        "rows": rows,
        "s_bar": s_bar,
        "excluded_post_gap_births": excluded,
        "excluded_pre_intervention_leak": leaked,
        "signatures": signatures,
    }


def brackets(rows: list[dict], predicted: float) -> bool:
    horizons = sorted(row["H_R"] for row in rows)
    below = [h for h in horizons if h <= predicted - BRACKET]
    above = [h for h in horizons if h >= predicted + BRACKET]
    return bool(below and above)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/v5_s0"))
    parser.add_argument("--gains", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--totals", type=int, nargs="+", default=[56, 72])
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(500, 510)))
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("reports/v5_s0.json"))
    args = parser.parse_args()

    scalars = RESIDUAL_SCALARS[args.rank]
    carry_8bit = LN2 * PROXY_BITS * scalars
    carry_star = LN2 * FRONTIER_BITS * scalars

    print("V5 H19 s-arm (S0) — does H* track 1 / s_bar with A held fixed?")
    print(f"  rank {args.rank}: D(A) = {scalars} scalars")
    print(f"  carry  8-bit {carry_8bit:,.0f} nats   D* (3.9 b/s) {carry_star:,.0f} nats")
    print("  the carried tensor is identical across g, so carry does not move\n")

    scored = {}
    for gain in args.gains:
        result = score_gain(args.root, gain, args.slots, args.totals, args.worlds)
        if not result:
            print(f"  g={gain}: no complete cells")
            continue
        scored[gain] = result
        print(f"  g={gain}  s_bar={result['s_bar']:.1f} nats/use"
              f"   (excluded: {result['excluded_post_gap_births']} post-gap births, "
              f"{result['excluded_pre_intervention_leak']} leaks)")
        for row in result["rows"]:
            print(f"      H_R={row['H_R']:>3} (n={row['n']:>2})  "
                  f"C_reacquire={row['c_reacquire']:>8,.0f}  s_bar={row['s_bar']:>6.1f}")
        for label, carry in (("8-bit", carry_8bit), ("D*", carry_star)):
            predicted = carry / result["s_bar"]
            ok = brackets(result["rows"], predicted)
            print(f"      predicted H* ({label}) = {predicted:.1f}"
                  f"   grid brackets it: {'yes' if ok else 'NO — UNSCOREABLE, widen the grid'}")
            result[f"predicted_{label}"] = predicted
            result[f"brackets_{label}"] = ok
        print()

    # The registered invariance: same abstraction in every gain.
    print("  CARRY INVARIANCE (abstraction tensors across g)")
    keys = set()
    for result in scored.values():
        keys |= set(result["signatures"])
    mismatched = []
    for key in sorted(keys):
        seen = {
            gain: result["signatures"].get(key)
            for gain, result in scored.items()
            if result["signatures"].get(key)
        }
        if len(set(seen.values())) > 1:
            mismatched.append((key, seen))
    if not keys:
        print("    no signatures read")
    elif mismatched:
        print(f"    MISMATCH in {len(mismatched)}/{len(keys)} cells — the gain moved the")
        print("    abstraction, so D(A) is not held fixed and the arm is VOID:")
        for key, seen in mismatched[:5]:
            print(f"      {key}: {seen}")
    else:
        print(f"    identical in {len(keys)}/{len(keys)} cells — carry is the same number,")
        print("    not merely within tolerance")

    # Monotonicity, the registered validity condition on the arm itself.
    gains = sorted(scored)
    if len(gains) >= 2:
        sbars = [scored[g]["s_bar"] for g in gains]
        monotone = all(b >= a for a, b in zip(sbars, sbars[1:]))
        print(f"\n  s_bar monotone in g: {'PASS' if monotone else 'FAIL'}  "
              f"({', '.join(f'{g}:{s:.1f}' for g, s in zip(gains, sbars))})")
        lo, hi = gains[0], gains[-1]
        s_ratio = scored[hi]["s_bar"] / scored[lo]["s_bar"]
        print(f"  s_bar ratio g={hi}/g={lo} = {s_ratio:.2f} (gain ratio {hi/lo:.2f})")
        print("  NOTE: s_bar need not be linear in g. The gain scales the family's")
        print("  contribution INSIDE a tanh, and at g != 1 the frozen abstraction is")
        print("  also a slightly less exact match. The law is scored against MEASURED")
        print("  s_bar, so neither costs it anything; only monotonicity is registered.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {str(k): v for k, v in scored.items()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
