"""Score the V3 sealed block against V3_CONFIRMATION_PLAN.md (bcc8319).

Written before the sealed results were inspected. Every threshold below is
copied from the frozen plan; nothing here may be adjusted after seeing the
data. Interval misses are failures even when signs pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)
BITS = 8

# --- frozen thresholds, transcribed from V3_CONFIRMATION_PLAN.md ---
O1_MIN_POSITIVE, O1_INTERVAL = 27, (35_000.0, 75_000.0)
O2_MIN_POSITIVE, O2_INTERVAL = 25, (400.0, 2_800.0)
O3_MIN_PATTERN, O3_INTERVAL = 28, (0.40, 0.80)
O4_MIN_POSITIVE, O4_INTERVAL = 26, (0.0010, 0.0055)
O5_MIN_REUSE_RATIO = 1.4


def _bits(summary: dict, promotion: dict | None) -> tuple[int, int]:
    shared = BITS * int(summary["shared_parameter_count"])
    task = BITS * int(summary["task_state_scalar_count"])
    if promotion is not None:
        task += int(promotion["reference_bits_total"])
    return shared, task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/v3_sealed"))
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(300, 330)))
    parser.add_argument("--future", type=Path, default=Path("reports/v3_sealed_future.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/v3_sealed.json"))
    args = parser.parse_args()

    two_part, loss_gain, patterns, reductions = [], [], [], []
    structured_reuse, control_reuse = [], []
    structured_library, control_library = [], []
    structured_promoted, structured_refused = [], []
    control_promoted, control_refused = [], []
    for world in args.worlds:
        base = args.root / "structured" / f"world_{world}"
        unpromoted = json.loads((base / "shared_residual" / "summary.json").read_text())
        promoted = json.loads((base / "promoting" / "summary.json").read_text())
        promotion = promoted["promotion"]
        us, ut = _bits(unpromoted, None)
        ps, pt = _bits(promoted, promotion)
        uL = unpromoted["cumulative_prequential_gaussian_log_loss"]
        pL = promoted["cumulative_prequential_gaussian_log_loss"]
        two_part.append((uL + LN2 * (us + ut)) - (pL + LN2 * (ps + pt)))
        loss_gain.append(uL - pL)
        patterns.append(pt < ut and ps > us and (ps + pt) < (us + ut))
        reductions.append(1.0 - (ps + pt) / (us + ut))
        structured_reuse.append(promotion["tasks_reusing_library"])
        structured_library.append(promotion["library_size"])
        structured_promoted.append(promotion["candidates_promoted"])
        structured_refused.append(
            promotion["candidates_considered"] - promotion["candidates_promoted"]
        )

        control = json.loads(
            (args.root / "control" / f"world_{world}" / "promoting" / "summary.json").read_text()
        )["promotion"]
        control_reuse.append(control["tasks_reusing_library"])
        control_library.append(control["library_size"])
        control_promoted.append(control["candidates_promoted"])
        control_refused.append(
            control["candidates_considered"] - control["candidates_promoted"]
        )

    def verdict(name, passed, detail):
        print(f"  {name}: {'PASS' if passed else 'FAIL'} — {detail}")
        return passed

    print("V3 SEALED BLOCK, seeds 300-329, scored against the frozen plan")
    o1_pos = sum(v > 0 for v in two_part)
    o1_mean = float(np.mean(two_part))
    o1 = verdict(
        "O1 two-part gain",
        o1_pos >= O1_MIN_POSITIVE and O1_INTERVAL[0] <= o1_mean <= O1_INTERVAL[1],
        f"positive {o1_pos}/30 (need >={O1_MIN_POSITIVE}), mean {o1_mean:+.0f} "
        f"(interval {O1_INTERVAL[0]:+.0f} to {O1_INTERVAL[1]:+.0f})",
    )
    o2_pos = sum(v > 0 for v in loss_gain)
    o2_mean = float(np.mean(loss_gain))
    o2 = verdict(
        "O2 loss gain",
        o2_pos >= O2_MIN_POSITIVE and O2_INTERVAL[0] <= o2_mean <= O2_INTERVAL[1],
        f"positive {o2_pos}/30 (need >={O2_MIN_POSITIVE}), mean {o2_mean:+.0f} "
        f"(interval {O2_INTERVAL[0]:+.0f} to {O2_INTERVAL[1]:+.0f})",
    )
    o3_count = sum(patterns)
    o3_mean = float(np.mean(reductions))
    o3 = verdict(
        "O3 migration",
        o3_count >= O3_MIN_PATTERN and O3_INTERVAL[0] <= o3_mean <= O3_INTERVAL[1],
        f"three-sign pattern {o3_count}/30 (need >={O3_MIN_PATTERN}), "
        f"mean D_total reduction {o3_mean:.3f} (interval {O3_INTERVAL})",
    )

    o4 = None
    if args.future.exists():
        rows = json.loads(args.future.read_text())
        deltas = [
            r["unpromoted"]["mean_nmse_by_support"]["32"]
            - r["promoted"]["mean_nmse_by_support"]["32"]
            for r in rows
            if "promoted" in r and "unpromoted" in r
        ]
        o4_pos = sum(d > 0 for d in deltas)
        o4_mean = float(np.mean(deltas))
        o4 = verdict(
            "O4 prospective",
            o4_pos >= O4_MIN_POSITIVE and O4_INTERVAL[0] <= o4_mean <= O4_INTERVAL[1],
            f"positive {o4_pos}/{len(deltas)} (need >={O4_MIN_POSITIVE}), "
            f"mean 32-shot improvement {o4_mean:+.5f} (interval {O4_INTERVAL})",
        )
    else:
        print("  O4 prospective: future-block audit not yet run")

    ratio = float(np.mean(structured_reuse)) / max(1e-9, float(np.mean(control_reuse)))
    o5 = verdict(
        "O5 refusal",
        ratio >= O5_MIN_REUSE_RATIO
        and np.mean(structured_library) > np.mean(control_library)
        and np.mean(structured_promoted) > np.mean(structured_refused)
        and np.mean(control_promoted) < np.mean(control_refused),
        f"reuse ratio {ratio:.2f}x (need >={O5_MIN_REUSE_RATIO}), library "
        f"{np.mean(structured_library):.1f} vs {np.mean(control_library):.1f}, "
        f"promote/refuse structured {np.mean(structured_promoted):.1f}/"
        f"{np.mean(structured_refused):.1f}, control {np.mean(control_promoted):.1f}/"
        f"{np.mean(control_refused):.1f}",
    )

    outcomes = {"O1": o1, "O2": o2, "O3": o3, "O4": o4, "O5": o5}
    decided = [v for v in outcomes.values() if v is not None]
    print(f"\n  {sum(decided)}/{len(decided)} registered outcomes pass")
    payload = {
        "plan_commit": "bcc8319",
        "outcomes": outcomes,
        "two_part_gain": two_part,
        "loss_gain": loss_gain,
        "migration_pattern": patterns,
        "total_reduction": reductions,
        "structured": {
            "reuse": structured_reuse,
            "library": structured_library,
            "promoted": structured_promoted,
            "refused": structured_refused,
        },
        "control": {
            "reuse": control_reuse,
            "library": control_library,
            "promoted": control_promoted,
            "refused": control_refused,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
