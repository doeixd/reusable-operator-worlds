"""Fail if any artifact listed in the invalid manifest still exists.

Documentation of an invalidation is not a safeguard. This is: run it in
the same breath as `check_prereg.py` and a resurrected artifact set
becomes a non-zero exit rather than a number someone quotes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "INVALID_MANIFEST.md"
PATTERN = re.compile(r"^\s{4}(artifacts/[A-Za-z0-9_./-]+)", re.MULTILINE)


def main() -> int:
    if not MANIFEST.exists():
        print("no invalid manifest; nothing to check")
        return 0
    root = MANIFEST.resolve().parents[1]
    listed = PATTERN.findall(MANIFEST.read_text(encoding="utf-8"))
    if not listed:
        # A checker that parses nothing passes everything. The first
        # version of this file omitted re.MULTILINE and reported "0
        # withdrawn paths, none present" over a manifest listing six.
        print("MANIFEST PARSED NO PATHS — the checker is vacuous")
        return 1
    present = [path for path in listed if (root / path).exists()]
    if present:
        print("INVALID ARTIFACTS PRESENT — these were withdrawn and must not "
              "be scored:")
        for path in present:
            print(f"  {path}")
        return 1
    print(f"invalid-artifact check OK: {len(listed)} withdrawn paths, none present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
