from __future__ import annotations

import unittest

from row.experiments.summarize_stage2 import DEVELOPMENT_WORLDS


class SummarizeStageTwoTests(unittest.TestCase):
    def test_development_worlds_are_disjoint_from_confirmation(self) -> None:
        self.assertEqual(DEVELOPMENT_WORLDS, tuple(range(3, 10)))
        self.assertTrue(set(DEVELOPMENT_WORLDS).isdisjoint(range(100, 130)))


if __name__ == "__main__":
    unittest.main()
