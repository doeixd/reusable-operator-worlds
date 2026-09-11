import unittest

import numpy as np
from scipy.stats import spearmanr

from row.experiments import audit_j0_library_quality as j0
from row.experiments import score_j0_library_quality as scorer


def cells(q, g, random=2.0, enum=0.5):
    names = [f"{k}_w{w}" for k in j0.LIBRARIES for w in j0.WORLDS]
    return {n: {"q": qi, "g": gi, "medians": {"random": random, "enum": enum}} for n, qi, gi in zip(names, q, g)}


class J0Tests(unittest.TestCase):
    def test_independent_spearman_matches_scipy_with_ties(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a, b = rng.integers(0, 6, 30).astype(float), rng.normal(size=30)
            self.assertAlmostEqual(scorer.spearman(a, b), float(spearmanr(a, b).statistic), places=12)

    def test_classification_ladder(self):
        q = np.linspace(-6, 0, 30)
        rising = q + np.random.default_rng(1).normal(scale=0.3, size=30)
        stats = j0.statistics(cells(q, rising))
        self.assertGreater(stats["pooled_rho"], 0.5)
        self.assertLess(stats["permutation_p"], 0.05)
        self.assertEqual(j0.classify(cells(q, rising), stats, True), "CF2_SUPPORTED")
        self.assertEqual(j0.classify(cells(q, -rising), j0.statistics(cells(q, -rising)), True), "CF2_REVERSED")
        flat = np.random.default_rng(2).permutation(np.tile([0.0, 1.0, 2.0], 10))
        flat_stats = {"pooled_rho": 0.05, "permutation_p": 0.4, "within_world_rho": {"0": 0, "1": 0, "2": 0}}
        self.assertEqual(j0.classify(cells(q, flat), flat_stats, True), "CF2_FLAT")
        self.assertEqual(j0.classify(cells(q, rising), stats, False), "HARNESS_FAILED")
        self.assertEqual(j0.classify(cells(q, rising, random=0.1), stats, True), "HARNESS_FAILED")


if __name__ == "__main__":
    unittest.main()
