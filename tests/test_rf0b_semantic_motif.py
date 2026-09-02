import unittest

import numpy as np

from row.experiments.audit_rf0b_semantic_motif import (
    classify,
    empty_occurrences,
    motif_score,
    phi,
    synthetic_controls,
)
from row.experiments.score_rf0b import validate_motif_arm


def synthetic_regenerated(depth=8):
    motif = (0, 1, 2)
    programs, routes, carries, sites = [], [], [], []
    for index in range(128):
        program = [3] * depth
        carry = index < 64
        site = index % (depth - 2) if carry else -1
        if carry:
            program[site:site + 3] = motif
        programs.append(tuple(program))
        routes.append(list(program))
        carries.append(carry)
        sites.append(site)
    return {"motif": motif, "programs": programs, "routes": routes,
            "carries": carries, "sites": sites}


class RF0bSemanticMotifTests(unittest.TestCase):
    def test_joint_score_is_not_mean_position_accuracy(self):
        data = synthetic_regenerated()
        predictions = np.asarray(data["routes"], dtype=np.int64)
        score = motif_score(data, predictions.reshape(-1))
        self.assertEqual(score["exact_recovery"], 1.0)
        self.assertEqual(score["relative_position_accuracy"], [1.0, 1.0, 1.0])
        self.assertEqual(score["modal_decoded_motif"], [0, 1, 2])
        # Break a different motif coordinate on every planted task.
        for index in range(64):
            site = data["sites"][index]
            predictions[index, site + index % 3] = 5
        score = motif_score(data, predictions.reshape(-1))
        self.assertEqual(score["exact_recovery"], 0.0)
        self.assertTrue(all(value > 0.6 for value in score["relative_position_accuracy"]))

    def test_empty_occurrences_builds_only_declared_route_fields(self):
        data = synthetic_regenerated(depth=4)
        rows = empty_occurrences(data["programs"], data["routes"],
                                 data["carries"], data["sites"], 4)
        self.assertEqual(rows.rows, 512)
        self.assertEqual(rows.native.shape, (512, 0))
        self.assertEqual(rows.trace.shape, (512, 0))
        self.assertEqual(int(rows.planted.sum()), 64 * 3)

    def test_phi_reports_degenerate_binary_vector_as_null(self):
        self.assertIsNone(phi(np.ones(10), np.arange(10) % 2))
        self.assertAlmostEqual(phi(np.arange(10) % 2, np.arange(10) % 2), 1.0)

    def test_synthetic_controls_pass(self):
        controls = synthetic_controls()
        self.assertTrue(controls["passed"])
        self.assertEqual(controls["positive_exact_recovery"], 1.0)
        self.assertLessEqual(controls["negative_exact_recovery"], 0.10)

    def test_decision_ladder_detects_horizon(self):
        worlds = {}
        for world in range(3):
            worlds[str(world)] = {"structurally_scoreable": True, "depths": {
                "8": {"Z": {"exact_recovery": 0.50}, "delta_sem": 0.30,
                      "Z_minus_R_exact": 0.30, "permutation_pass": True},
                "10": {"Z": {"exact_recovery": 0.25}, "delta_sem": 0.18,
                       "Z_minus_R_exact": 0.22, "permutation_pass": True},
            }}
        decision = classify(worlds, {"passed": True})
        self.assertEqual(decision["classification"],
                         "SEMANTIC MOTIF HORIZON BETWEEN 8 AND 10")

    def test_decision_ladder_detects_full_ceiling(self):
        worlds = {}
        for world in range(3):
            cell = {"Z": {"exact_recovery": 0.50}, "delta_sem": 0.30,
                    "Z_minus_R_exact": 0.30, "permutation_pass": True}
            worlds[str(world)] = {"structurally_scoreable": True,
                                  "depths": {"8": dict(cell), "10": dict(cell)}}
        decision = classify(worlds, {"passed": True})
        self.assertEqual(decision["classification"],
                         "SEMANTIC CANONICALIZATION CEILING EXISTS")

    def test_independent_scorer_recomputes_joint_rates(self):
        record = {
            "planted_tasks": 64,
            "unplanted_tasks": 64,
            "exact_count": 32,
            "exact_recovery": 0.5,
            "relative_position_accuracy": [0.5, 0.5, 0.5],
            "independence_product": 0.125,
            "exact_minus_independence": 0.375,
            "pairwise_phi": {"0_1": 0.0, "0_2": None, "1_2": 1.0},
            "modal_decoded_motif": [0, 1, 2],
            "modal_count": 32,
            "modal_recurrence": 0.5,
            "modal_equals_teacher_motif": True,
            "unplanted_teacher_motif_occurrence": 0.1,
        }
        validate_motif_arm(record, "test")
        record["exact_recovery"] = 0.4
        with self.assertRaises(ValueError):
            validate_motif_arm(record, "test")


if __name__ == "__main__":
    unittest.main()
