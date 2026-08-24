"""Preregistration-as-code: verify the audit trail mechanically.

Checks:
1. Every file listed in FROZEN has no changes since its freeze commit.
2. Every artifact/report path cited in a STATUS annotation of the V2 spec
   exists on disk (or is explicitly marked as external).

Run from the repository root: `python tools/check_prereg.py`.
Exit code 0 = trail verifies; 1 = violation (printed).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Freeze commit = the commit after which the file may no longer change.
# EXPERIMENT_PLAN.md was a living document during development by design;
# its invariant is stillness from the last pre-confirmation edit (d655ce0,
# which precedes the confirmation freeze e0b0552) onward. The checker's
# first run caught exactly this distinction and the manifest was corrected
# with this justification (PROGRESS.md, 2026-08-18).
FROZEN = {
    "neural_library_learning_v1_experimental_spec.md": "e1be00a",
    "EXPERIMENT_PLAN.md": "d655ce0",
    "CONFIRMATION_PLAN.md": "e0b0552",
    "V2_CONFIRMATION_PLAN.md": "085b1a3",
    "V3_CONFIRMATION_PLAN.md": "bcc8319",
    "V4R_CONFIRMATION_PLAN.md": "2aec65c",
    "V5_CONFIRMATION_PLAN.md": "1ed227d",
    "V6R_ADAPTATION_GEOMETRY_PLAN.md": "4c1bfaa",
    "H39_EXISTENCE_PLAN.md": "16906ff",
    "H39_PILOT_PLAN.md": "108df94",
    "H39B_PSLOT_PILOT_PLAN.md": "a6f9b4a",
    "H39C_KSWEEP_PLAN.md": "061b912",
    "H39D_CAPACITY_PLAN.md": "05ccaa0",
    "H39_CONFIRMATION_PLAN.md": "5ee3f5d",
    "H39_NEXT_STEPS_PLAN.md": "eec41b2",
    "H47_MEMBERSHIP_PLAN.md": "1d3902a",
    "H48B_WIDTH_SWEEP_PLAN.md": "525baac",
    "H49_DISCOVERABILITY_PLAN.md": "5302c92",
    "H50_REORGANIZATION_PLAN.md": "0b1b5e9",
    "H51_REORGANIZABILITY_PLAN.md": "48a99e9",
}

STATUS_PATH_PATTERN = re.compile(
    r"(?:artifacts|reports)/[A-Za-z0-9_./-]+"
)


def frozen_files_unchanged() -> list[str]:
    violations = []
    for path, commit in FROZEN.items():
        result = subprocess.run(
            ["git", "diff", "--stat", commit, "HEAD", "--", path],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip():
            violations.append(
                f"FROZEN FILE CHANGED since {commit}: {path}\n{result.stdout}"
            )
    return violations


def status_paths_exist() -> list[str]:
    violations = []
    spec = Path("row_v2_experimental_spec.md").read_text(encoding="utf-8")
    blocks = re.findall(r"\*\*[^*]*STATUS[^*]*\*\*[^#]*?(?=\n\n|\Z)", spec)
    cited: set[str] = set()
    for block in blocks:
        for match in STATUS_PATH_PATTERN.findall(block):
            cited.add(match.rstrip(".,;:)"))
    for path in sorted(cited):
        candidate = Path(path.rstrip("/"))
        if candidate.suffix in ("", "/"):
            if not candidate.exists():
                violations.append(f"STATUS cites missing directory: {path}")
        elif not candidate.exists():
            base = candidate if candidate.suffix else candidate.parent
            if not base.exists():
                violations.append(f"STATUS cites missing path: {path}")
    return violations


def main() -> int:
    violations = frozen_files_unchanged() + status_paths_exist()
    if violations:
        print("PREREGISTRATION CHECK FAILED")
        for violation in violations:
            print(" -", violation)
        return 1
    print(
        f"prereg check OK: {len(FROZEN)} frozen files unchanged; "
        "all STATUS-cited paths exist"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
