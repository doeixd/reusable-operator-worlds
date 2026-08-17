from __future__ import annotations

import unittest

from row.experiments.plot_structural_controls import _validated_rows


class PlotStructuralControlsTests(unittest.TestCase):
    def test_requires_all_development_worlds(self) -> None:
        with self.assertRaisesRegex(ValueError, "worlds 0-9"):
            _validated_rows({"paired_world_effects": [{"world_seed": 0}]})


if __name__ == "__main__":
    unittest.main()
