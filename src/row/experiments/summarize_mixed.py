"""H9a secondary summaries: envelope and two-currency accounting on
mixed-recurrence worlds."""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("artifacts/v2_mixed/canonical"))
    parser.add_argument("--output", type=Path, default=Path("reports/v2_mixed"))
    args = parser.parse_args()
    ln2 = math.log(2)
    worlds = []
    for world_dir in sorted(args.runs.glob("world_*")):
        entry: dict[str, object] = {"world_seed": int(world_dir.name.split("_")[1])}
        losses = {}
        bits = {}
        for model in ("shared_residual", "continuous", "dense"):
            summary_path = world_dir / model / "summary.json"
            if not summary_path.exists():
                break
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            losses[model] = s["cumulative_prequential_gaussian_log_loss"]
            bits[model] = 8 * (
                s["shared_parameter_count"] + s["task_state_scalar_count"]
            )
            entry[f"{model}_novel_32"] = s["novel_composition"]["nmse_by_support"]["32"]
        else:
            envelope = min(losses["continuous"], losses["dense"])
            raw_gain = envelope - losses["shared_residual"]
            entry.update(
                losses=losses,
                retained_bits=bits,
                envelope_minus_shared_raw_nats=raw_gain,
                shared_beats_envelope=raw_gain > 0,
                two_part_gain_vs_envelope=raw_gain
                - ln2
                * (bits["shared_residual"] - min(bits["continuous"], bits["dense"])),
            )
            worlds.append(entry)
    n = len(worlds)
    report = {
        "scope": "canonical mixed-profile Benchmark D, development worlds; H9a secondary",
        "worlds": worlds,
        "envelope_wins": sum(1 for w in worlds if w["shared_beats_envelope"]),
        "n_worlds": n,
        "mean_raw_gain": st.mean(w["envelope_minus_shared_raw_nats"] for w in worlds)
        if worlds
        else None,
        "two_part_wins": sum(1 for w in worlds if w["two_part_gain_vs_envelope"] > 0),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "h9a-envelope.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"envelope wins {report['envelope_wins']}/{n}, mean raw gain "
        f"{report['mean_raw_gain'] and round(report['mean_raw_gain'])}, "
        f"two-part wins {report['two_part_wins']}/{n}"
    )


if __name__ == "__main__":
    main()
