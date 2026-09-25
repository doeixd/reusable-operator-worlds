"""Fail closed if a plan in scope lacks its necessity or discriminating-power section.

Companion to `check_prereg.py` (which verifies that frozen files are still) and
`check_invalid.py` (which verifies that withdrawn artifacts are absent). This
one verifies that plans written under the 2026-09-22 design-adequacy protocol
say how their TASK could have failed to elicit the behaviour (`# Necessity`)
and how their DATA could have come out the other way
(`# Discriminating power`).

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

SECTIONS = ('# Necessity', '# Discriminating power')

#: Plans that are FROZEN OR FREEZE-READY under the protocol. A draft that still
#: says its own rates are unmeasured does not belong here; the checker would
#: correctly refuse it, and adding it would create pressure to weaken the
#: checker rather than finish the plan.
#: `DESIGN_ADEQUACY.md` is deliberately NOT here: it is the protocol, not a plan,
#: and special-casing it to pass would leave a checker that can never fail - the
#: `check_invalid.py` defect that printed a clean pass over a manifest of six.
IN_SCOPE = (
    'N1_ANCHOR_SUPPLY_PLAN.md',
    'N1B_ANCHOR_DOSE_PLAN.md',
    'N1C_ANCHOR_COVERAGE_PLAN.md',
    'O1_ONLINE_ANCHOR_TIER1_PLAN.md',
    'O2_ONLINE_ANCHOR_RELIABILITY_PLAN.md',
)

#: Explicit sentinel for a plan with no behaviour to elicit (an observational
#: census). Must be accompanied by a stated reason; see `check`.
NOT_APPLICABLE = re.compile(r'necessity gate does not apply', re.I)

#: Each numbered item a section must address, and a pattern that evidences it.
REQUIRED = {
    '# Necessity': (
        ('the arm that refuses the behaviour', re.compile(r'refus', re.I)),
        ('a measured refusal cost and its scale', re.compile(r'scale|contribution', re.I)),
        ('the impostors scored', re.compile(r'impostor', re.I)),
        ('the difficulty band and its direction', re.compile(r'band', re.I)),
    ),
    '# Discriminating power': (
        ('the decision rule', re.compile(r'rule', re.I)),
        ('the null and effect samplers', re.compile(r'null', re.I)),
        ('a measured false-fire rate', re.compile(r'false[- ]fire', re.I)),
        ('a measured detection rate', re.compile(r'detect', re.I)),
        ('what the checks do not cover', re.compile(r'do(es)? not (catch|cover)', re.I)),
    ),
}


def section_of(text, heading):
    """A plan's named section, up to the next top-level heading, or None.

    The heading must be matched AT THE START OF A LINE. A plain `find` also
    matches the heading's name mentioned in prose - both plans in IN_SCOPE say
    "carries `# Necessity` and `# Discriminating power` sections" in their
    preamble - and then returns the few lines before the first real heading,
    which contain none of the required items. That defect made the checker
    report 15 problems against two plans that satisfy it, and is exactly the
    class of bug the `check_invalid.py` missing-`re.MULTILINE` failure was.
    """
    match = re.search(r'^%s\s*$' % re.escape(heading), text, re.MULTILINE)
    if match is None:
        return None
    rest = text[match.end():]
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
        for heading in SECTIONS:
            body = section_of(text, heading)
            if body is None:
                problems.append(f'{name}: no "{heading}" section')
                continue
            if not body.strip():
                problems.append(f'{name}: "{heading}" section is empty')
                continue
            if heading == '# Necessity' and NOT_APPLICABLE.search(body):
                # An observational census over frozen artifacts trains nothing
                # and has no behaviour to elicit, so the gate genuinely does not
                # apply. The sentinel must be explicit and must carry a reason,
                # so that "N/A" cannot be reached by omission.
                if not re.search(r'because|since|no learner|frozen artifacts', body, re.I):
                    problems.append(f'{name}: "{heading}" claims NOT APPLICABLE without a reason')
                continue
            for label, pattern in REQUIRED[heading]:
                if not pattern.search(body):
                    problems.append(f'{name}: "{heading}" does not address {label}')
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
