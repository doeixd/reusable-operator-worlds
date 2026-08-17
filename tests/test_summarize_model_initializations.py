from __future__ import annotations

import unittest

from row.experiments.sweep_model_initializations import SECOND_SEEDS


class SummarizeModelInitializationsTests(unittest.TestCase):
    def test_second_seeds_are_distinct_from_canonical(self) -> None:
        self.assertEqual(SECOND_SEEDS["continuous"], 4001)
        self.assertEqual(SECOND_SEEDS["dense"], 3001)


if __name__ == "__main__":
    unittest.main()
