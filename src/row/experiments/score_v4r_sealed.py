"""Score the V4R sealed block (seeds 400-429) against the frozen plan.

Thresholds below are transcribed from `V4R_CONFIRMATION_PLAN.md`, frozen
at commit 2aec65c and hashed into `tools/check_prereg.py`. They were
written before any sealed world was generated. Do not edit them to match
results; a miss is a failure even when the sign is right.

This block is unusual: **the registered prediction is that nothing
pays.** V4R's development census found no structural edit worth making
in the canonical regime, and these worlds test whether that negative
replicates out of sample. A negative is only credible if it was
predicted in advance with the precision a positive would need, so O1 and
O4 carry intervals, not just signs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.census_opportunity import census
from row.task_group_world import TaskGroupSpec

LN2 = math.log(2.0)

# --- frozen thresholds, transcribed from V4R_CONFIRMATION_PLAN.md ---
O1_MIN_COMPRESS_WINS = 27          # of 30
O1_INTERVAL = (1_000.0, 4_000.0)   # mean COMPRESS-minus-FACTORIZE nats
O2_MAX_FACTORIZE_WINS_AT_SMALL_M = 0
O2_M_CEILING = 16
O3_MAX_FORK_PAYS = 2               # of 30 worlds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v4r_sealed/structured"))
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(400, 430)))
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("reports/v4r_sealed.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    spec = TaskGroupSpec(
        groups=2, eta=0.9, future_tasks=8, family_onset=16,
        new_primitive_families=True,
    )

    rows = []
    for world in args.worlds:
        path = args.root / f"world_{world}" / "lifecycle"
        if not (path / "summary.json").exists():
            continue
        rows.append(census(config, path, world, spec, args.slots))

    scored = [r for r in rows if not r.get("reason")]
    compress_wins = sum(1 for r in scored if r["winner"] == "COMPRESS")
    factorize_wins = [r for r in scored if r["winner"] == "FACTORIZE"]
    margins = [r["best_compress_nats"] - r["best_factorize_nats"] for r in scored]
    mean_margin = float(np.mean(margins)) if margins else 0.0
    small_m_factorize = [r for r in factorize_wins if r["library_size"] <= O2_M_CEILING]

    o1_count = compress_wins >= O1_MIN_COMPRESS_WINS
    o1_interval = O1_INTERVAL[0] <= mean_margin <= O1_INTERVAL[1]
    o2 = len(small_m_factorize) <= O2_MAX_FACTORIZE_WINS_AT_SMALL_M

    print("V4R SEALED BLOCK (seeds 400-429) vs V4R_CONFIRMATION_PLAN.md @ 2aec65c")
    print(f"  worlds scored: {len(scored)}")
    print()
    print(f"  O1 count    COMPRESS wins {compress_wins}/{len(scored)} "
          f"(need >= {O1_MIN_COMPRESS_WINS})           {'PASS' if o1_count else 'FAIL'}")
    print(f"  O1 interval mean margin {mean_margin:,.0f} nats "
          f"(need {O1_INTERVAL[0]:,.0f}-{O1_INTERVAL[1]:,.0f})   "
          f"{'PASS' if o1_interval else 'FAIL'}")
    print(f"  O2          FACTORIZE wins at M <= {O2_M_CEILING}: {len(small_m_factorize)} "
          f"(need <= {O2_MAX_FACTORIZE_WINS_AT_SMALL_M})       {'PASS' if o2 else 'FAIL'}")
    print()
    print(f"  library sizes: min {min(r['library_size'] for r in scored)}, "
          f"max {max(r['library_size'] for r in scored)}, "
          f"mean {np.mean([r['library_size'] for r in scored]):.1f}")
    print()
    verdict = all([o1_count, o1_interval, o2])
    print(f"  CENSUS CELLS: {'PASS' if verdict else 'FAIL'} "
          f"({sum([o1_count, o1_interval, o2])}/3 registered criteria met)")
    print("  O3 (FORK) and O4 (retention) are scored by their own runs and are")
    print("  not covered here; this scorer reports the O1/O2 census cells only.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "plan_commit": "2aec65c",
        "worlds_scored": len(scored),
        "compress_wins": compress_wins,
        "mean_margin_nats": mean_margin,
        "factorize_wins_at_small_m": len(small_m_factorize),
        "o1_count_pass": o1_count,
        "o1_interval_pass": o1_interval,
        "o2_pass": o2,
        "cells": rows,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
