"""Sealed export-confirmation lifetimes, seeds 800-829.

`EXPORT_CONFIRMATION_PLAN.md`, frozen at 4b1f8cd before any world in the band
existed. One support-split discrete lifetime per world carries every estimand.

Refuses to start unless the preregistration check passes, and refuses to
overwrite a world that already has a summary, so the block cannot be silently
re-run under different code.
"""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from row.config import load_config                                        # noqa: E402
from row.experiments import learned_lifetime                              # noqa: E402
from row.support_split_world import (SupportSplitFactory, SupportSplitSpec,  # noqa: E402
                                     split_programs)

SEEDS = tuple(range(800, 830))


def run(world: int) -> tuple[int, bool]:
    output = ROOT / "artifacts" / "export_confirmation" / f"world_{world}"
    if (output / "summary.json").exists():
        return world, True                       # resumable; never overwritten
    config = load_config(Path("configs/v1.yaml"))
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
    learned_lifetime.World = factory                                      # type: ignore[assignment]
    try:
        learned_lifetime.run(config, kind="discrete")
    finally:
        learned_lifetime.World = original                                 # type: ignore[assignment]
    return world, (output / "summary.json").exists()


if __name__ == "__main__":
    if subprocess.run([sys.executable, "tools/check_prereg.py"], cwd=ROOT).returncode != 0:
        raise SystemExit("prereg check failed; the sealed block does not open")
    done = 0
    with ProcessPoolExecutor(max_workers=3) as pool:
        for world, ok in pool.map(run, SEEDS):
            done += int(ok)
            print(f"world {world}: summary={ok}", flush=True)
    print(f"EXPORT_CONFIRMATION_LIFETIMES_DONE {done}/{len(SEEDS)}")
