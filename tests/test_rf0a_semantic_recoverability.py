import unittest

import numpy as np

from row.experiments.audit_rf0a_semantic_recoverability import (
    ARMS,
    OccurrenceData,
    balanced_accuracy,
    classify,
    feature_matrix,
    fit_ridge,
    permute_within_position,
    role_vector,
    select_probe,
    synthetic_controls,
)


def toy_data(labels=None) -> OccurrenceData:
    rows = 24
    if labels is None:
        labels = np.tile(np.arange(6), 4)
    symbols = np.tile(np.arange(12), 2)
    roles = np.stack([role_vector(index % 4, 4) for index in range(rows)])
    fingerprints = np.stack([symbols, symbols ** 2], axis=1).astype(np.float64)
    neighbours = np.zeros((rows, 26), dtype=np.float64)
    neighbours[:, 0] = 1.0
    native = np.arange(rows * 9, dtype=np.float64).reshape(rows, 9) / 100.0
    trace = np.arange(rows * 7, dtype=np.float64).reshape(rows, 7) / 100.0
    return OccurrenceData(
        labels=np.asarray(labels, dtype=np.int64),
        raw_symbols=symbols,
        roles=roles,
        fingerprints=fingerprints,
        neighbours=neighbours,
        native=native,
        trace=trace,
        positions=np.tile(np.arange(4), 6),
        task_indices=np.repeat(np.arange(6), 4),
        planted=np.zeros(rows, dtype=bool),
    )


class RF0aSemanticRecoverabilityTests(unittest.TestCase):
    def test_role_vector_boundaries_and_depth_one(self):
        np.testing.assert_array_equal(role_vector(0, 4)[:3], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(role_vector(1, 4)[:3], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(role_vector(3, 4)[:3], [0.0, 0.0, 1.0])
        single = role_vector(0, 1)
        self.assertTrue(np.all(np.isfinite(single)))
        self.assertEqual(single[3], 0.0)
        self.assertEqual(single[4], 0.0)

    def test_feature_arms_are_label_blind(self):
        original = toy_data()
        relabelled = toy_data(np.roll(original.labels, 1))
        for arm in ARMS:
            np.testing.assert_array_equal(
                feature_matrix(original, arm), feature_matrix(relabelled, arm)
            )

    def test_ridge_recovers_linearly_separable_classes(self):
        labels = np.repeat(np.arange(6), 10)
        x = np.eye(6)[labels]
        # The contiguous split omits classes, and must fail closed.
        with self.assertRaises(ValueError):
            select_probe(x[:30], labels[:30], x[30:], labels[30:])

    def test_ridge_with_all_classes(self):
        labels = np.tile(np.arange(6), 10)
        x = np.eye(6)[labels]
        selected, path = select_probe(x[:30], labels[:30], x[30:], labels[30:])
        self.assertIn(selected, {1e-4, 1e-2, 1.0, 1e2})
        self.assertEqual(len(path), 4)
        model = fit_ridge(x[:30], labels[:30], selected)
        score, _, confusion = balanced_accuracy(labels[30:], model.predict(x[30:]))
        self.assertEqual(score, 1.0)
        self.assertEqual(sum(map(sum, confusion)), 30)

    def test_position_permutation_preserves_each_position_multiset(self):
        data = toy_data()
        shuffled = permute_within_position(
            data.labels, data.positions, np.random.default_rng(123)
        )
        for position in range(4):
            indices = data.positions == position
            self.assertEqual(
                sorted(data.labels[indices].tolist()), sorted(shuffled[indices].tolist())
            )

    def test_registered_synthetic_controls(self):
        controls = synthetic_controls()
        self.assertTrue(controls["positive"]["passed"])
        self.assertGreaterEqual(controls["positive"]["delta"], 0.40)
        self.assertTrue(controls["negative"]["passed"])

    def test_decision_ladder_prefers_role_conditioned_rung(self):
        worlds = {}
        for world in range(3):
            arms = {}
            for arm in ARMS:
                score = 0.20
                if arm == "Z":
                    score = 0.35
                elif arm == "ZR":
                    score = 0.70
                arms[arm] = {"deep": {
                    "8": {"balanced_accuracy": score},
                    "10": {"balanced_accuracy": score},
                }}
            worlds[str(world)] = {
                "arms": arms,
                "contrasts": {
                    "8": {"ZR_minus_Z": 0.35},
                    "10": {"ZR_minus_Z": 0.35},
                },
                "non_vacuity": {
                    "base_scoreable": True,
                    "permutation_pass": {"8": True, "10": True},
                },
            }
        controls = {"positive": {"passed": True}, "negative": {"passed": True}}
        decision = classify(worlds, controls)
        self.assertEqual(decision["classification"],
                         "ROLE-CONDITIONED LOCAL SEMANTICS")


if __name__ == "__main__":
    unittest.main()
