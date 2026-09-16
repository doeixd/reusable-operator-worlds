import math
import unittest

from row.experiments import census_library_geometry as geo


def cell(run, world, stream, stage3, **measures):
    base = {"min_pair": 0.5, "mean_pair": 0.9, "route_margin": 0.4, "effective_rank": 6.0,
            "slot_norm_spread": 1.5}
    base.update(measures)
    return {"run": run, "world": world, "stream": stream, "stage2_terminal": 0.03,
            "stage3_terminal": stage3, "passed": stage3 <= geo.THRESHOLD, **base}


def twentyfour(min_pairs=None, stage3=None):
    """24 cells in the census's own layout; optional per-index overrides."""
    layout = [("SO2", w, 0) for w in (0, 1, 2)]
    layout += [("SO3", w, s) for w in (3, 4, 5) for s in (0, 1, 2)]
    layout += [("SO4", w, s) for w in (6, 7, 8, 9) for s in (0, 1, 2)]
    cells = []
    for i, (run, w, s) in enumerate(layout):
        outcome = stage3[i] if stage3 else (0.01 if i % 2 else 0.5)
        mp = min_pairs[i] if min_pairs else 0.2 + 0.05 * i
        # Every measure must VARY: the non-vacuity guard refuses a constant one.
        cells.append(cell(run, w, s, outcome, min_pair=mp, mean_pair=mp + 0.4,
                          route_margin=0.2 + 0.01 * i, effective_rank=4.0 + 0.1 * i,
                          slot_norm_spread=1.2 + 0.05 * i))
    return cells


class MeasureTests(unittest.TestCase):
    def test_duplicate_slot_drives_min_pair_to_zero(self):
        import torch

        class FakeOp(torch.nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale

            def forward(self, x):
                return x * self.scale

        class FakeModel(torch.nn.Module):
            def __init__(self, scales):
                super().__init__()
                self.library = torch.nn.ModuleList([FakeOp(s) for s in scales])
                self.task_codes = {}

        x = torch.eye(4)
        distinct = geo.slot_outputs(FakeModel([1.0, 2.0, 4.0]), x)
        self.assertEqual(distinct.shape[0], 3)
        # A duplicated slot means two identical rows, so the minimum pairwise distance is 0.
        duplicated = geo.slot_outputs(FakeModel([1.0, 1.0, 4.0]), x)
        pairs = [float(torch.linalg.vector_norm(duplicated[i] - duplicated[j]))
                 for i, j in ((0, 1), (0, 2), (1, 2))]
        self.assertAlmostEqual(min(pairs), 0.0, places=12)
        self.assertGreater(min(float(torch.linalg.vector_norm(distinct[i] - distinct[j]))
                               for i, j in ((0, 1), (0, 2), (1, 2))), 0.0)


class GuardTests(unittest.TestCase):
    def test_all_guards_pass_on_a_well_formed_census(self):
        cells = twentyfour()
        self.assertEqual(geo.guards(cells, geo.analyse(cells)), [])

    def test_wrong_count_non_finite_and_min_exceeding_mean(self):
        cells = twentyfour()
        self.assertTrue(any("expected 24" in p for p in geo.guards(cells[:-1], geo.analyse(cells[:-1]))))
        broken = twentyfour()
        broken[0] = dict(broken[0], route_margin=float("nan"))
        self.assertTrue(any("non-finite" in p for p in geo.guards(broken, geo.analyse(broken))))
        inverted = twentyfour()
        inverted[0] = dict(inverted[0], min_pair=2.0, mean_pair=1.0)
        self.assertTrue(any("min_pair exceeds mean_pair" in p for p in geo.guards(inverted, geo.analyse(inverted))))

    def test_constant_measure_and_narrow_spread_are_refused(self):
        constant = twentyfour(min_pairs=[0.5] * 24)
        problems = geo.guards(constant, geo.analyse(constant))
        self.assertTrue(any("constant across cells" in p for p in problems))
        narrow = twentyfour(min_pairs=[0.50 + 0.001 * i for i in range(24)])
        problems = geo.guards(narrow, geo.analyse(narrow))
        self.assertTrue(any("cannot discriminate at this scale" in p for p in problems))


class W8Tests(unittest.TestCase):
    def w8(self, values, key="min_pair"):
        cells = twentyfour()
        for c in cells:
            if c["run"] == "SO4" and c["world"] == 8:
                c[key] = values[c["stream"]]
                if key == "min_pair":
                    c["mean_pair"] = values[c["stream"]] + 0.4
        return geo.w8_ordering(cells, key)

    def test_lower_is_worse_ordering(self):
        # s1 had the WORST stage-3 and s0 the best, so min_pair must be lowest at s1, highest at s0.
        self.assertTrue(self.w8({1: 0.05, 2: 0.30, 0: 0.60})["orders_w8"])
        self.assertFalse(self.w8({1: 0.60, 2: 0.30, 0: 0.05})["orders_w8"])

    def test_slot_norm_spread_is_inverted(self):
        # Higher spread is less favourable, so the worst-outcome stream must be highest.
        self.assertTrue(self.w8({1: 9.0, 2: 3.0, 0: 1.5}, "slot_norm_spread")["orders_w8"])
        self.assertFalse(self.w8({1: 1.5, 2: 3.0, 0: 9.0}, "slot_norm_spread")["orders_w8"])


class TriageTests(unittest.TestCase):
    def cells_for(self, rho_sign=1.0, order_w8=True):
        """Stage-3 outcome tracking min_pair (or not), with w8 ordered (or not)."""
        stage3, min_pairs = [], []
        for i in range(24):
            mp = 0.1 + 0.03 * i
            min_pairs.append(mp)
            stage3.append(0.01 + 0.02 * i * rho_sign if rho_sign > 0 else 0.60 - 0.02 * i)
        cells = twentyfour(min_pairs=min_pairs, stage3=stage3)
        # `w8_ordering` reads the registered constant W8_STREAM_ORDER = (worst, best) =
        # (1, 0), the real SO4 world-8 outcome (s1 0.1731 worst, s0 0.0161 best). Pin the
        # synthetic outcomes to that ordering so the test exercises the constant rather
        # than deriving a different one.
        worst, best = geo.W8_STREAM_ORDER
        w8 = [c for c in cells if c["run"] == "SO4" and c["world"] == 8]
        middle = next(c["stream"] for c in w8 if c["stream"] not in (worst, best))
        outcome_by_stream = {worst: 0.90, middle: 0.30, best: 0.01}
        favourable = {worst: 0.01, middle: 0.50, best: 0.99}
        for c in w8:
            c["stage3_terminal"] = outcome_by_stream[c["stream"]]
            c["passed"] = c["stage3_terminal"] <= geo.THRESHOLD
            c["min_pair"] = favourable[c["stream"]] if order_w8 else 1.0 - favourable[c["stream"]]
            c["mean_pair"] = c["min_pair"] + 0.4
        return cells

    def test_geometry_explains_needs_both_correlation_and_w8(self):
        cells = self.cells_for(order_w8=True)
        analysis = geo.analyse(cells)
        m = analysis["measures"]["min_pair"]
        self.assertGreaterEqual(abs(m["spearman_pooled"]), 0.5)
        self.assertTrue(m["w8"]["orders_w8"])
        self.assertEqual(geo.triage(analysis), "GEOMETRY-EXPLAINS")

    def test_strong_correlation_but_w8_unordered_is_not_explains(self):
        analysis = geo.analyse(self.cells_for(order_w8=False))
        self.assertNotEqual(geo.triage(analysis), "GEOMETRY-EXPLAINS")


if __name__ == "__main__":
    unittest.main()
