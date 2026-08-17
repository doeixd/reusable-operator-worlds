from __future__ import annotations

import unittest

from row.experiments.compare_rho import _interpolated_crossing


class CompareRhoTests(unittest.TestCase):
    def test_interpolates_first_sign_change(self) -> None:
        rows = [
            {"x": 0.0, "effect": -2.0},
            {"x": 0.5, "effect": -1.0},
            {"x": 1.0, "effect": 3.0},
        ]
        self.assertAlmostEqual(
            _interpolated_crossing(rows, "x", "effect"), 0.625
        )

    def test_returns_none_without_sign_change(self) -> None:
        rows = [{"x": 0.0, "effect": -2.0}, {"x": 1.0, "effect": -1.0}]
        self.assertIsNone(_interpolated_crossing(rows, "x", "effect"))


if __name__ == "__main__":
    unittest.main()
