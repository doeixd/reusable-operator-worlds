import math
import unittest

from row.experiments import census_route_margin_normalization as rm


def cell(run, world, stream, stage3, route_margin=1.0, self_margin=1.0, recovered=rm.ROUTES):
    return {"run": run, "world": world, "stream": stream, "route_margin": route_margin,
            "self_margin": self_margin, "stage2_terminal": 0.03, "stage3_terminal": stage3,
            "passed": stage3 <= rm.THRESHOLD, "recovered_routes": recovered,
            "self_margin_min": self_margin * 0.5, "self_margin_max": self_margin * 1.5}


def twentyfour(margins=None, outcomes=None, self_margins=None):
    """SO2 (1 stream x 3 worlds) + SO3 and SO4 (3 streams x 7 worlds) = 24 cells."""
    layout = [("SO2", w, 0) for w in (0, 1, 2)]
    layout += [(run, w, s) for run, w in rm.THREE_STREAM_WORLDS for s in (0, 1, 2)]
    cells = []
    for i, (run, w, s) in enumerate(layout):
        cells.append(cell(run, w, s,
                          outcomes[i] if outcomes else (0.01 if i % 2 else 0.5),
                          route_margin=margins[i] if margins else 1.0 + 0.5 * i,
                          self_margin=self_margins[i] if self_margins else 0.5 + 0.1 * i))
    return cells


def concordant_cells(worlds_concordant: int, key="route_margin"):
    """Build 24 cells where exactly `worlds_concordant` of the 7 trios are concordant."""
    cells = twentyfour()
    for index, (run, w) in enumerate(rm.THREE_STREAM_WORLDS):
        trio = [c for c in cells if c["run"] == run and c["world"] == w]
        # stream 1 is the worst outcome in every trio
        for c in trio:
            c["stage3_terminal"] = {0: 0.01, 1: 0.90, 2: 0.02}[c["stream"]]
            c["passed"] = c["stage3_terminal"] <= rm.THRESHOLD
        low, high = (0.10, 0.90) if index < worlds_concordant else (0.90, 0.10)
        for c in trio:
            c[key] = {1: low, 0: high, 2: (low + high) / 2}[c["stream"]]
    return cells


class SelfMarginTests(unittest.TestCase):
    def test_recovery_gate_fires(self):
        cells = twentyfour()
        cells[5]["recovered_routes"] = 9
        problems = rm.guards(cells, rm.analyse(cells))
        self.assertTrue(any("recovered only 9/16" in p for p in problems))

    def test_non_positive_and_constant_self_margin_refused(self):
        cells = twentyfour()
        cells[0]["self_margin"] = 0.0
        self.assertTrue(any("not positive" in p for p in rm.guards(cells, rm.analyse(cells))))
        flat = twentyfour(self_margins=[1.0] * 24)
        self.assertTrue(any("constant across cells" in p for p in rm.guards(flat, rm.analyse(flat))))


class ConcordanceTests(unittest.TestCase):
    def test_counts_and_exact_null(self):
        for expected in (0, 3, 5, 7):
            a = rm.concordance(concordant_cells(expected), "route_margin")
            self.assertTrue(a["available"])
            self.assertEqual((a["k"], a["n"]), (expected, 7))
        # Exact Binomial(7, 1/3) upper tail at k=5; the plan quotes it rounded as 0.045.
        self.assertAlmostEqual(rm.binomial_upper_tail(5, 7, 1 / 3), 0.0452674897119342, places=12)
        self.assertAlmostEqual(rm.binomial_upper_tail(0, 7, 1 / 3), 1.0, places=12)
        self.assertAlmostEqual(rm.concordance(concordant_cells(7), "route_margin")["expected"], 7 / 3)

    def test_ties_are_reported_not_broken(self):
        cells = concordant_cells(7)
        trio = [c for c in cells if c["run"] == "SO3" and c["world"] == 3]
        for c in trio:
            c["route_margin"] = 0.5
        a = rm.concordance(cells, "route_margin")
        self.assertIn("SO3_w3", a["ties"])
        self.assertTrue(any("tied margins" in p for p in rm.guards(cells, rm.analyse(cells))))

    def test_missing_stream_makes_test_a_unavailable(self):
        cells = [c for c in concordant_cells(7) if not (c["run"] == "SO4" and c["world"] == 9 and c["stream"] == 2)]
        a = rm.concordance(cells, "route_margin")
        self.assertFalse(a["available"])


class TriageTests(unittest.TestCase):
    def analysis_for(self, cells):
        return rm.analyse(cells)

    def test_survives_on_test_a(self):
        analysis = self.analysis_for(concordant_cells(6))
        self.assertGreaterEqual(analysis["test_a_route_margin"]["k"], 5)
        self.assertEqual(rm.triage(analysis), "SURVIVES")

    def test_survives_on_test_b_alone(self):
        # Test A at k=2 (below 3 is DISSOLVES only if B is also weak), B strongly negative.
        outcomes = [0.60 - 0.02 * i for i in range(24)]
        self_margins = [0.5 + 0.1 * i for i in range(24)]
        cells = twentyfour(outcomes=outcomes, self_margins=self_margins)
        analysis = self.analysis_for(cells)
        b = analysis["test_b_self_margin"]
        self.assertLessEqual(b["permutation_fraction"], 0.05)
        self.assertGreaterEqual(abs(b["spearman_pooled"]), 0.5)
        self.assertEqual(rm.triage(analysis), "SURVIVES")

    def test_dissolves_needs_both_weak(self):
        cells = concordant_cells(2)
        # Break any self_margin relation: alternate high/low against the outcome.
        for i, c in enumerate(cells):
            c["self_margin"] = 1.0 + (0.5 if i % 2 else -0.4)
        analysis = self.analysis_for(cells)
        self.assertLessEqual(analysis["test_a_route_margin"]["k"], 3)
        self.assertLess(abs(analysis["test_b_self_margin"]["spearman_pooled"]), 0.3)
        self.assertEqual(rm.triage(analysis), "DISSOLVES")

    def test_mixed_between_the_branches(self):
        cells = concordant_cells(4)          # k = 4: neither >= 5 nor <= 3
        for i, c in enumerate(cells):
            c["self_margin"] = 1.0 + (0.5 if i % 2 else -0.4)
        analysis = self.analysis_for(cells)
        self.assertEqual(analysis["test_a_route_margin"]["k"], 4)
        self.assertEqual(rm.triage(analysis), "MIXED")


if __name__ == "__main__":
    unittest.main()
