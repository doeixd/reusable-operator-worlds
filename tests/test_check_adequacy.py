"""Tests for the adequacy checker's EVALUATOR, not its constants.

`IN_SCOPE` is empty until plans are written under the protocol, so the checker
prints INACTIVE in normal operation. That makes these tests the only thing
standing between the evaluator and silent rot - the depth-five lesson, where a
gate's tests checked two constants and a per-block indexing bug shipped.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

import check_adequacy as ca  # noqa: E402

COMPLETE = """# A plan

Preamble.

# Discriminating power

The decision rule is `k >= 3 of 36 cells above their own floor`.
Null sampler: cells drawn from the control arm. Effect sampler: cells at the
registered effect size.
Measured false-fire rate 0.02; detection rate 0.91.
Graded axis passes at 1.4; the triage partitions 0..36.
What the checks do not cover here: whether the two arms share a coordinate
system, which is argued in the construction section.

# Next section

Body.
"""


def written(text, name='PLAN.md'):
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / name
    path.write_text(text, encoding='utf-8')
    return path


class TestSectionExtraction(unittest.TestCase):
    def test_section_stops_at_the_next_heading(self):
        body = ca.section_of(COMPLETE)
        self.assertIn('false-fire rate', body)
        self.assertNotIn('Next section', body)

    def test_absent_section_is_none(self):
        self.assertIsNone(ca.section_of('# A plan\n\nNo such section.\n'))


class TestCheck(unittest.TestCase):
    def test_complete_section_passes(self):
        self.assertEqual(ca.check([written(COMPLETE)]), [])

    def test_missing_section_fails(self):
        problems = ca.check([written('# A plan\n\nNothing here.\n')])
        self.assertEqual(len(problems), 1)
        self.assertIn('no "# Discriminating power" section', problems[0])

    def test_empty_section_fails(self):
        problems = ca.check([written('# A plan\n\n# Discriminating power\n\n# Next\n\nx\n')])
        self.assertTrue(any('is empty' in p for p in problems))

    def test_each_required_item_is_detected_when_missing(self):
        """Drop one required item at a time; each omission must be named."""
        cases = {
            'a measured false-fire rate': 'false-fire rate 0.02; ',
            'a measured detection rate': 'detection rate 0.91.',
            'what the checks do not cover': 'What the checks do not cover here: whether the two arms share a coordinate\nsystem, which is argued in the construction section.',
        }
        for label, fragment in cases.items():
            text = COMPLETE.replace(fragment, '')
            self.assertIn(fragment, COMPLETE, fragment)
            problems = ca.check([written(text)])
            self.assertTrue(any(label in p for p in problems),
                            f'omitting {label!r} was not caught: {problems}')

    def test_missing_file_in_scope_fails(self):
        problems = ca.check(['NO_SUCH_PLAN.md'])
        self.assertTrue(any('missing from the repository' in p for p in problems))

    def test_inactive_scope_does_not_report_a_pass(self):
        """An empty scope must not print a reassuring OK."""
        self.assertEqual(ca.IN_SCOPE, (), 'add plans to IN_SCOPE, and update this test')
        import io
        import contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ca.main()
        self.assertEqual(code, 0)
        self.assertIn('INACTIVE', out.getvalue())
        self.assertNotIn('OK', out.getvalue())


if __name__ == '__main__':
    unittest.main()
