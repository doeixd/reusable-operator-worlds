"""Fail closed if a plan in scope lacks its discriminating-power section.

Companion to `check_prereg.py` (which verifies that frozen files are still) and
`check_invalid.py` (which verifies that withdrawn artifacts are absent). This
one verifies that plans written under the 2026-09-22 design-adequacy protocol
say how their data could have come out the other way.

It checks PRESENCE AND STRUCTURE, never correctness: it cannot tell whether a
false-fire rate is right, only whether it was measured and written down.

Scope is an explicit list, not a glob. Historical plans are not retrofitted -
they were written under different rules and rewriting them would falsify the
record. Add a plan here when it is written.

Run from the repository root: `python tools/check_adequacy.py`.
Exit 0 = every plan in scope carries the section; 1 = violation (printed).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SECTION = '# Discriminating power'

#: Plans written under the protocol. New plans are added here at freeze time.
#: `DESIGN_ADEQUACY.md` is deliberately NOT here: it is the protocol, not a plan,
#: and special-casing it to pass would leave a checker that can never fail - the
#: `check_invalid.py` defect that printed a clean pass over a manifest of six.
IN_SCOPE = ()

#: Each numbered item the section must address, and a pattern that evidences it.
REQUIRED = (
    ('the decision rule', re.compile(r'rule', re.I)),
    ('the null and effect samplers', re.compile(r'null', re.I)),
    ('a measured false-fire rate', re.compile(r'false[- ]fire', re.I)),
    ('a measured detection rate', re.compile(r'detect', re.I)),
    ('what the checks do not cover', re.compile(r'do(es)? not (catch|cover)', re.I)),
)


def section_of(text):
    """The plan's discriminating-power section, or None."""
    start = text.find(SECTION)
    if start < 0:
        return None
    rest = text[start + len(SECTION):]
    end = re.search(r'^# ', rest, re.MULTILINE)
    return rest[:end.start()] if end else rest


def check(paths=IN_SCOPE):
    problems = []
    for name in paths:
        path = Path(name)
        if not path.exists():
            problems.append(f'{name}: in scope but missing from the repository')
            continue
        text = path.read_text(encoding='utf-8')
        body = section_of(text)
        if body is None:
            problems.append(f'{name}: no "{SECTION}" section')
            continue
        if not body.strip():
            problems.append(f'{name}: "{SECTION}" section is empty')
            continue
        for label, pattern in REQUIRED:
            if not pattern.search(body):
                problems.append(f'{name}: section does not address {label}')
    return problems


def main():
    if not IN_SCOPE:
        # Not a pass. A guard that parses nothing must say so rather than print
        # a reassuring OK (the `check_invalid.py` re.MULTILINE defect).
        print('adequacy check INACTIVE: no plans in scope yet. Add each new plan to '
              'IN_SCOPE at freeze time; the evaluator is exercised by '
              'tests/test_check_adequacy.py.')
        return 0
    problems = check()
    if problems:
        for problem in problems:
            print(problem)
        print(f'adequacy check FAILED: {len(problems)} problem(s)')
        return 1
    print(f'adequacy check OK: {len(IN_SCOPE)} plan(s) in scope carry a discriminating-power section')
    return 0


if __name__ == '__main__':
    sys.exit(main())
