"""H39 world-0 pilot launcher (H39_PILOT_PLAN.md, Amendment 1).

Three cells, one writer each, bounded pool of three. Every cell carries
its complete intervention record; a mismatched existing artifact is
refused by the runner. Exits nonzero if any cell fails.
"""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMON = [
    sys.executable, "-m", "row.experiments.mixed_lifetime",
    "--config", "configs/v5_h72.yaml", "--world-seed", "0",
    "--r-meta", "1.0", "--meta-families", "4", "--meta-tasks-per-family", "16",
    "--meta-subspace-rank", "2", "--family-onset", "8", "--operator-slots", "12",
    "--sleeps", "16", "24", "32", "48", "64", "--lifecycle", "--arm", "ordinary",
    "--prospective-steps", "8", "--prospective-inner-steps", "8",
]
CELLS = {
    "ordinary_history": ["--model", "prospective", "--snapshot-history"],
    # factorized_grouped withdrawn (plan Amendment 2): ill-posed for an
    # unseen-family future. Primary is the pooled a=2 oracle form.
    "factorized_pooled2": ["--model", "factorized", "--schema-grouping", "pooled",
                           "--schema-dim", "2", "--snapshot-history"],
    "factorized_pooled": ["--model", "factorized", "--schema-grouping", "pooled",
                          "--schema-dim", "8", "--snapshot-history"],
}
# H39b cells (H39B_PSLOT_PILOT_PLAN.md)
CELLS.update({
    "pslot2": ["--model", "pslot", "--slot-args", "2", "--snapshot-history"],
    "pslot8": ["--model", "pslot", "--slot-args", "8", "--snapshot-history"],
    "pslot2_frozen": ["--model", "pslot", "--slot-args", "2", "--freeze-args",
                      "--snapshot-history"],
})
# H39c K-sweep cells (H39C_KSWEEP_PLAN.md): name -> (world, extra args)
SWEEP = {}
for _w in (0, 1, 2):
    for _k in (2, 4, 8, 16):
        SWEEP[f"ksweep_p{_k}/world_{_w}"] = (_w, ["--model", "pslot", "--slot-args", str(_k),
                                                 "--snapshot-history"])
    SWEEP[f"ksweep_g8/world_{_w}"] = (_w, ["--model", "pslot", "--slot-args", "8",
                                           "--freeze-matrices", "--snapshot-history"])
# H39d cells (H39D_CAPACITY_PLAN.md)
for _w in (0, 1, 2):
    for _k in (32, 64):
        SWEEP[f"cap_p{_k}/world_{_w}"] = (_w, ["--model", "pslot", "--slot-args", str(_k),
                                              "--snapshot-history"])
    for _k in (16, 32):
        SWEEP[f"cap_m2k{_k}/world_{_w}"] = (_w, ["--model", "pslot", "--slot-args", str(_k),
                                                "--pslot-count", "2", "--snapshot-history"])
# H47 B1 cells (H47_MEMBERSHIP_PLAN.md Amendment 1); M = cap_m2k32, reused
for _w in (0, 1, 2):
    _base = ["--model", "pslot", "--slot-args", "32", "--pslot-count", "2", "--snapshot-history"]
    SWEEP[f"b1_larb/world_{_w}"] = (_w, _base + ["--route-policy", "mask_arbitrary"])
    SWEEP[f"b1_hearly/world_{_w}"] = (_w, _base + ["--route-policy", "anneal", "--anneal-start", "8",
                                                  "--anneal-commit", "24", "--anneal-final", "0.1"])
    SWEEP[f"b1_hlate/world_{_w}"] = (_w, _base + ["--route-policy", "anneal", "--anneal-start", "40",
                                                 "--anneal-commit", "56", "--anneal-final", "0.1"])
# H47 B2 cells (H47_MEMBERSHIP_PLAN.md Amendments 2-3): the G = 2 world
for _w in (0, 1, 2):
    _base = ["--model", "pslot", "--slot-args", "32", "--pslot-count", "2", "--snapshot-history",
             "--schema-groups", "2"]
    SWEEP[f"b2_m/world_{_w}"] = (_w, list(_base))
    SWEEP[f"b2_ltrue/world_{_w}"] = (_w, _base + ["--route-policy", "mask_group"])
    SWEEP[f"b2_hearly/world_{_w}"] = (_w, _base + ["--route-policy", "anneal", "--anneal-start", "8",
                                                  "--anneal-commit", "24", "--anneal-final", "0.1"])
    SWEEP[f"b2_hlate/world_{_w}"] = (_w, _base + ["--route-policy", "anneal", "--anneal-start", "40",
                                                 "--anneal-commit", "56", "--anneal-final", "0.1"])
# H48b width sweep (H48B_WIDTH_SWEEP_PLAN.md): G = 2 world, K in {2,4,8,16}; K = 32 reuses b2_m / b2_ltrue
for _w in (0, 1, 2):
    for _k in (2, 4, 8, 16):
        _base = ["--model", "pslot", "--slot-args", str(_k), "--pslot-count", "2", "--snapshot-history",
                 "--schema-groups", "2"]
        SWEEP[f"w_m{_k}/world_{_w}"] = (_w, list(_base))
        SWEEP[f"w_l{_k}/world_{_w}"] = (_w, _base + ["--route-policy", "mask_group"])
REQUIRED = ("model.pt", "summary.json", "rho_profile.json", "fingerprint.json",
            "config.yaml", "history.pt")


def run(name: str) -> tuple[str, int]:
    if name in SWEEP:
        world, extra = SWEEP[name]
        out = ROOT / "artifacts" / "h39c" / name / "lifecycle"
        common = list(COMMON)
        common[common.index("--world-seed") + 1] = str(world)
    else:
        world, extra = 0, CELLS[name]
        out = ROOT / "artifacts" / "h39_pilot" / name / "world_0" / "lifecycle"
        common = COMMON
    log = ROOT / "tools" / f"h39_{name.replace('/', '_')}.log"
    with log.open("a", encoding="utf-8") as handle:
        code = subprocess.run(
            common + extra + ["--output", str(out)],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        ).returncode
        handle.write(f"exit={code}\n")
    missing = [f for f in REQUIRED if not (out / f).exists()]
    if missing:
        return name, 1
    return name, code


def main() -> int:
    names = sys.argv[1:] or list(CELLS)
    if names == ["sweep"]:
        names = [n for n in SWEEP if n.startswith("ksweep_")]
    elif names == ["capacity"]:
        names = [n for n in SWEEP if n.startswith("cap_")]
    elif names == ["b1"]:
        names = [n for n in SWEEP if n.startswith("b1_")]
    elif names == ["b2gate"]:
        names = [n for n in SWEEP if n.startswith("b2_m/") or n.startswith("b2_ltrue/")]
    elif names == ["width"]:
        names = [n for n in SWEEP if n.startswith("w_")]
    elif names == ["b2h"]:
        names = [n for n in SWEEP if n.startswith("b2_hearly/") or n.startswith("b2_hlate/")]
    with ProcessPoolExecutor(max_workers=3) as pool:  # slots=12: cap 3 (memory)
        results = list(pool.map(run, names))
    failed = [n for n, c in results if c != 0]
    for name, code in results:
        print(f"{name}: exit={code}")
    if failed:
        print(f"FAILED cells: {failed}")
        return 1
    print("H39_PILOT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
