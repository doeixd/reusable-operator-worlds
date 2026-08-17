from __future__ import annotations

import unittest

from row.experiments.summarize_rho_replication import _crossing


class SummarizeRhoReplicationTests(unittest.TestCase):
    def test_crossing_interpolates_in_x_space(self) -> None:
        rows = [
            {"x": 0.4, "effect": -3.0},
            {"x": 0.8, "effect": 1.0},
        ]
        self.assertAlmostEqual(_crossing(rows, "x", "effect"), 0.7)


if __name__ == "__main__":
    unittest.main()
