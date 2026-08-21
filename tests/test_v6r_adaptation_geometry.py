"""Decision-rule tests for the frozen V6R adaptation-geometry audit."""

import unittest

from row.experiments.audit_v6r_adaptation_geometry import (
    anchor_expected,
    classify,
    operational_equivalence,
    replicated_worse,
)


def pairs(ordinary: float, gaps: list[float]) -> list[dict]:
    return [
        {
            "world": index // 2,
            "task_index": index % 2,
            "ordinary": ordinary,
            "prospective": ordinary + gap,
            "gap": gap,
        }
        for index, gap in enumerate(gaps)
    ]


def summary(ordinary: float, gaps: list[float]) -> dict:
    rows = pairs(ordinary, gaps)
    return {
        "complete": True,
        "finite": True,
        "gap_mean": sum(gaps) / len(gaps),
        "operationally_equivalent": operational_equivalence(rows),
        "prospective_replicated_worse": replicated_worse(rows),
    }


class DecisionRuleTests(unittest.TestCase):
    def test_anchor_accepts_selected_metric_and_richer_report_schemas(self):
        base = {"arm": "ordinary", "world": 0, "related": {"1": [1.5, 2.5]}}
        self.assertEqual(
            anchor_expected({"cells": [base]}, "ordinary", 0), [1.5, 2.5]
        )
        rich = {
            **base,
            "related": {"1": [{"prequential": 1.5}, {"prequential": 2.5}]},
        }
        self.assertEqual(
            anchor_expected({"cells": [rich]}, "ordinary", 0), [1.5, 2.5]
        )

    def test_operational_equivalence_requires_all_registered_guards(self):
        self.assertTrue(operational_equivalence(
            pairs(10.0, [0.5, -0.5, 0.4, -0.4, 0.3, -0.3])
        ))
        self.assertFalse(operational_equivalence(
            pairs(10.0, [4.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        ))
        self.assertFalse(operational_equivalence(
            pairs(10.0, [0.1] * 5)
        ))

    def test_replicated_worse_requires_three_consistent_worlds(self):
        self.assertTrue(replicated_worse(pairs(10.0, [2.0] * 6)))
        self.assertFalse(replicated_worse(
            pairs(10.0, [2.0, 2.0, -1.0, -1.0, 2.0, 2.0])
        ))

    def test_classifies_representational_opportunity_loss_first(self):
        result = classify({
            1: summary(10.0, [2.0] * 6),
            128: summary(10.0, [2.0] * 6),
        }, standard_gap=3.0)
        self.assertEqual(
            result["classification"], "representational_opportunity_loss"
        )

    def test_classifies_sparse_identifiability_loss(self):
        result = classify({
            1: summary(10.0, [2.0] * 6),
            128: summary(10.0, [0.2, -0.2] * 3),
        }, standard_gap=3.0)
        self.assertEqual(
            result["classification"], "sparse_identifiability_loss"
        )

    def test_classifies_optimizer_findability_loss(self):
        result = classify({
            1: summary(10.0, [0.2, -0.2] * 3),
            128: summary(10.0, [0.2, -0.2] * 3),
        }, standard_gap=3.0)
        self.assertEqual(
            result["classification"], "optimizer_findability_loss"
        )

    def test_nonfinite_or_incomplete_protocol_is_ineligible(self):
        sparse = summary(10.0, [0.2, -0.2] * 3)
        sparse["finite"] = False
        result = classify({
            1: sparse,
            128: summary(10.0, [0.2, -0.2] * 3),
        }, standard_gap=3.0)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["classification"], "unresolved_mixed")


if __name__ == "__main__":
    unittest.main()
