from __future__ import annotations

import unittest

from row.experiments.plot_robustness import _effect_matrix


class PlotRobustnessTests(unittest.TestCase):
    def test_requires_one_record_per_world_and_condition(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected one"):
            _effect_matrix({"paired_world_effects": []}, "effect")


if __name__ == "__main__":
    unittest.main()
