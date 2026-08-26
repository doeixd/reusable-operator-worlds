"""E2 lifetimes on the support-split world (E2_COMPOSITION_PLAN.md, Amendment 1).

Injects `SupportSplitFactory` in place of `learned_lifetime.World` — the pattern
`mixed_lifetime` already uses for the meta-recurrence generators — so the
lifetime trains on the frozen program list and never sees a withheld placement.
The split, its diagnostics, and the accepted attempt index are written beside
the artifact BEFORE the lifetime is scored, so the training set is on the record
independently of any result.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from row.config import load_config                                    # noqa: E402
from row.experiments import learned_lifetime                          # noqa: E402
from row.support_split_world import (SupportSplitFactory, SupportSplitSpec,  # noqa: E402
                                     split_programs)

WORLDS = (0, 1, 2)


def run(world: int) -> tuple[int, bool]:
    config = load_config(Path("configs/v1.yaml"))
    output = ROOT / "artifacts" / "e2_support_split" / f"world_{world}"
    config = replace(config, world=replace(config.world, seed=world, reuse_rho=1.0),
                     output_directory=output)
    spec = SupportSplitSpec()
    split = split_programs(config.world, spec)
    output.mkdir(parents=True, exist_ok=True)
    (output / "support_split.json").write_text(json.dumps({
        "spec": spec.as_dict(),
        "withheld_placements": [list(p) for p in split["withheld_placements"]],
        "diagnostics": split["diagnostics"],
        "train_programs": [list(p) for p in split["train"]],
        "strata": {k: [list(p) for p in v] for k, v in split["strata"].items()},
    }, indent=2) + "\n", encoding="utf-8")

    factory = SupportSplitFactory(spec)
    original = learned_lifetime.World
    learned_lifetime.World = factory                                  # type: ignore[assignment]
    try:
        learned_lifetime.run(config, kind="discrete")
    finally:
        learned_lifetime.World = original                             # type: ignore[assignment]
    return world, (output / "summary.json").exists()


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        for world, ok in pool.map(run, WORLDS):
            print(f"world {world}: summary={ok}", flush=True)
    print("E2_LIFETIMES_DONE")
