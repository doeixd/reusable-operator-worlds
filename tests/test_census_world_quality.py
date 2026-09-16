import math
import unittest

from row.experiments import census_world_quality as census


def stages(s1, s2, eot2):
    return {"1": {"terminal_median": s1}, "2": {"terminal_median": s2, "end_of_task_median": eot2}}


def so2_report(values=((0.2, 0.1, 0.3, 0.02), (0.3, 0.15, 0.4, 0.2), (0.25, 0.12, 0.35, 0.03))):
    cells = {}
    for w, (s1, s2, eot2, s3) in zip((0, 1, 2), values):
        cells[f"STAGED_w{w}"] = {"stages": stages(s1, s2, eot2) | {"3": {"terminal_median": s3}}}
    return {"classification": "SO2_FAILS", "git_commit": "aaa", "cells": cells}


def prefix_report(worlds, label, terminal, arm_suffix=""):
    """`terminal[(w, s)]` gives the stage-3 terminal median."""
    prefixes, cells = {}, {}
    for w in worlds:
        for s in (0, 1, 2):
            prefixes[f"w{w}_s{s}"] = {"stages": stages(0.2 + 0.01 * s, 0.1 + 0.01 * s, 0.3)}
            cells[f"w{w}_s{s}{arm_suffix}"] = {"terminal_median": terminal[(w, s)]}
    return {"classification": {"program": label}, "git_commit": "bbb", "prefixes": prefixes, "cells": cells}


def so3_report(terminal=None):
    terminal = terminal or {(w, s): 0.02 for w in (3, 4, 5) for s in (0, 1, 2)}
    return prefix_report((3, 4, 5), "SO3_FAILS", terminal, "_BASE")


def so4_report(terminal=None):
    terminal = terminal or {(w, s): 0.02 for w in (6, 7, 8, 9) for s in (0, 1, 2)}
    return prefix_report((6, 7, 8, 9), "SO4_FAILS", terminal)


class CollectTests(unittest.TestCase):
    def test_twentyfour_cells_in_run_order(self):
        rows = census.collect(so2_report(), so3_report(), so4_report())
        self.assertEqual(len(rows), 24)
        self.assertEqual([sum(r["run"] == k for r in rows) for k in ("SO2", "SO3", "SO4")], [3, 9, 12])
        self.assertEqual(rows[0]["run"], "SO2")
        self.assertEqual((rows[0]["stage1_terminal"], rows[0]["stage2_terminal"]), (0.2, 0.1))
        self.assertAlmostEqual(rows[0]["stage1_to_stage2_ratio"], 0.5)
        self.assertTrue(rows[0]["passed"])          # 0.02 <= 0.05
        self.assertFalse(rows[1]["passed"])         # 0.20 > 0.05
        self.assertEqual(rows[-1]["run"], "SO4")
        self.assertEqual((rows[-1]["world"], rows[-1]["stream"]), (9, 2))

    def test_so3_reads_base_arm_only(self):
        terminal = {(w, s): 0.9 for w in (3, 4, 5) for s in (0, 1, 2)}
        report = so3_report(terminal)
        report["cells"]["w3_s0_LR_HALF"] = {"terminal_median": 0.001}  # must be ignored
        rows = [r for r in census.collect(so2_report(), report, so4_report()) if r["run"] == "SO3"]
        self.assertEqual({r["stage3_terminal"] for r in rows}, {0.9})


class GuardTests(unittest.TestCase):
    def reports(self):
        return {"SO2": so2_report(), "SO3": so3_report(), "SO4": so4_report()}

    def test_all_guards_pass(self):
        reports = self.reports()
        rows = census.collect(reports["SO2"], reports["SO3"], reports["SO4"])
        self.assertEqual(census.guards(rows, reports), [])

    def test_wrong_cell_count_and_bad_label_and_non_finite(self):
        reports = self.reports()
        rows = census.collect(reports["SO2"], reports["SO3"], reports["SO4"])
        self.assertTrue(any("cells, expected" in p for p in census.guards(rows[:-1], reports)))
        wrong = self.reports()
        wrong["SO4"]["classification"] = {"program": "SO4_PASSES"}
        self.assertTrue(any("classification" in p for p in census.guards(rows, wrong)))
        broken = list(rows)
        broken[0] = dict(broken[0], stage2_terminal=float("nan"))
        self.assertTrue(any("non-finite" in p for p in census.guards(broken, reports)))


class AnalysisTests(unittest.TestCase):
    def rows(self, pairs):
        """pairs: list of (stage2_terminal, stage3_terminal); other fields tracked to stage2."""
        return [{"run": "SO4", "world": 6 + i // 3, "stream": i % 3, "stage1_terminal": s2 * 2,
                 "stage2_terminal": s2, "stage2_end_of_task": s2 * 3, "stage1_to_stage2_ratio": 0.5,
                 "stage3_terminal": s3, "passed": s3 <= census.THRESHOLD}
                for i, (s2, s3) in enumerate(pairs)]

    def test_spearman_orders(self):
        self.assertAlmostEqual(census.spearman([1, 2, 3, 4], [2, 3, 4, 9]), 1.0)
        self.assertAlmostEqual(census.spearman([1, 2, 3, 4], [9, 4, 3, 2]), -1.0)

    def test_spearman_handles_ties(self):
        # `argsort(argsort(x))` invents a strict order for ties: it returned +1.0 for a
        # constant predictor and 1.0 for the single-tie case below. Average ranks fix both.
        self.assertTrue(math.isnan(census.spearman([0.1] * 5, [1, 2, 3, 4, 5])))
        self.assertTrue(math.isnan(census.spearman([1, 2, 3, 4, 5], [7] * 5)))
        self.assertAlmostEqual(census.spearman([1, 1, 2, 3], [1, 2, 3, 4]), 0.9486832980505138, places=12)
        self.assertAlmostEqual(census.spearman([1, 1, 2, 2], [1, 2, 3, 4]), 0.8944271909999159, places=12)

    def test_prefix_predictive(self):
        pairs = [(0.01, 0.01), (0.02, 0.02), (0.03, 0.03), (0.04, 0.04), (0.05, 0.045),
                 (0.20, 0.30), (0.22, 0.40), (0.24, 0.50), (0.26, 0.60), (0.28, 0.70)]
        analysis = census.analyse(self.rows(pairs))
        self.assertEqual(census.triage(analysis), "PREFIX-PREDICTIVE")
        self.assertGreater(analysis["predictors"]["stage2_terminal"]["separation"]["ratio"], 2.0)

    def test_stage3_localized(self):
        # Stage-2 VARIES but carries no rank information about the outcome: failing cells
        # sit at the LOW end as often as the high end (rho +0.13, permutation 0.36, no
        # separation). This is the realistic null.
        pairs = [(0.05, 0.60), (0.06, 0.01), (0.07, 0.50), (0.08, 0.02), (0.09, 0.40),
                 (0.10, 0.03), (0.11, 0.70), (0.12, 0.04), (0.13, 0.80), (0.14, 0.02)]
        analysis = census.analyse(self.rows(pairs))
        self.assertEqual(census.triage(analysis), "STAGE3-LOCALIZED")

    def test_weak_relation_without_separation_is_mixed(self):
        # Every failing cell one step ABOVE a passing one: a real but weak rank relation
        # (rho +0.43) with no 2x separation. Neither decisive branch should claim it.
        pairs = [(0.05, 0.01), (0.06, 0.60), (0.07, 0.02), (0.08, 0.50), (0.09, 0.03),
                 (0.10, 0.40), (0.11, 0.04), (0.12, 0.70), (0.13, 0.02), (0.14, 0.80)]
        analysis = census.analyse(self.rows(pairs))
        self.assertEqual(census.triage(analysis), "MIXED")

    def test_constant_predictor_is_undefined_not_perfect(self):
        # A constant predictor carries no information; argsort(argsort(x)) would have
        # reported +1.0 for it. `analyse` must yield nan and triage must not call it
        # PREFIX-PREDICTIVE.
        pairs = [(0.10, 0.01), (0.10, 0.60), (0.10, 0.02), (0.10, 0.50),
                 (0.10, 0.03), (0.10, 0.40), (0.10, 0.04), (0.10, 0.70)]
        analysis = census.analyse(self.rows(pairs))
        self.assertTrue(math.isnan(analysis["predictors"]["stage2_terminal"]["spearman_pooled"]))
        self.assertNotEqual(census.triage(analysis), "PREFIX-PREDICTIVE")

    def test_permutation_fraction_is_deterministic_and_bounded(self):
        rows = self.rows([(0.01, 0.01), (0.02, 0.02), (0.03, 0.30), (0.04, 0.40),
                          (0.05, 0.045), (0.20, 0.60), (0.22, 0.02), (0.24, 0.70)])
        first = census.analyse(rows)["predictors"]["stage2_terminal"]["permutation_fraction"]
        second = census.analyse(rows)["predictors"]["stage2_terminal"]["permutation_fraction"]
        self.assertEqual(first, second)
        self.assertGreater(first, 0.0)
        self.assertLessEqual(first, 1.0)

    def test_mixed_outcome_worlds_are_identified(self):
        pairs = [(0.01, 0.01), (0.02, 0.02), (0.03, 0.03),      # world 6: all pass
                 (0.10, 0.01), (0.11, 0.90), (0.12, 0.02)]      # world 7: mixed
        analysis = census.analyse(self.rows(pairs))
        self.assertEqual(analysis["mixed_outcome_worlds"], ["SO4_w7"])
        self.assertEqual(analysis["mixed_cells"], 3)


if __name__ == "__main__":
    unittest.main()
