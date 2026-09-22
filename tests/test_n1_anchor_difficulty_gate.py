import unittest

import numpy as np
import torch

from row.experiments.audit_h47_baselines import ari_nmi
from row.experiments import n1_anchor_difficulty_gate as gate


class ARIComparabilityTests(unittest.TestCase):
    """ARI is the statistic BECAUSE it is chance-corrected; check that it is."""

    def test_perfect_agreement_is_one(self):
        labels = [0, 0, 1, 1, 2, 2]
        self.assertAlmostEqual(ari_nmi(labels, labels)[0], 1.0, places=9)

    def test_relabelling_does_not_change_ari(self):
        """Slot indices and primitive indices live in different spaces."""
        truth = [0, 0, 1, 1, 2, 2]
        relabelled = [7, 7, 3, 3, 11, 11]
        self.assertAlmostEqual(ari_nmi(truth, relabelled)[0], 1.0, places=9)

    def test_random_assignment_is_near_zero_at_both_depths(self):
        """The whole point: 12 routes and 1,728 routes must score the same at chance.

        A raw agreement rate would give 1/12 against 1/1728 and make depth-1 look
        clustered by construction. ARI must not.
        """
        rng = np.random.default_rng(3)
        n = 400
        truth = rng.integers(0, 6, n).tolist()
        shallow = rng.integers(0, 12, n).tolist()
        deep = rng.integers(0, 12 ** 3, n).tolist()
        self.assertLess(abs(ari_nmi(truth, shallow)[0]), 0.05)
        self.assertLess(abs(ari_nmi(truth, deep)[0]), 0.05)

    def test_constant_assignment_scores_zero_not_one(self):
        """Every task routed to one slot is no clustering at all."""
        truth = [0, 1, 2, 3, 4, 5] * 4
        constant = [0] * 24
        self.assertLessEqual(ari_nmi(truth, constant)[0], 1e-9)


class GateRuleTests(unittest.TestCase):
    def test_rule_requires_every_world(self):
        """One world clustering more at depth 3 must fail the premise."""
        rows = [{'depth1_more_clustered': True}, {'depth1_more_clustered': True},
                {'depth1_more_clustered': False}]
        self.assertFalse(all(r['depth1_more_clustered'] for r in rows))

    def test_selected_slots_returns_one_slot_per_step(self):
        from row.config import load_config
        from row.experiments.audit_rotated_g5r_interference import world_config
        from row.experiments.audit_so1_budget_bracket import build_fast
        from row.experiments.l0d_depth4_execution_gate import DepthLibrary

        cfg = world_config(load_config('configs/v1.yaml'), 0)
        library = DepthLibrary(build_fast(cfg))

        class FakeTask:
            def __init__(self):
                generator = torch.Generator().manual_seed(5)
                self.train_x = torch.randn(8, 16, generator=generator).numpy()
                self.train_y = torch.randn(8, 16, generator=generator).numpy()

        for depth in (1, 2):
            routes = gate.selected_slots(library, [FakeTask(), FakeTask()], depth)
            self.assertEqual(len(routes), 2)
            for route in routes:
                self.assertEqual(len(route), depth)
                self.assertTrue(all(0 <= slot < library.slots for slot in route))


if __name__ == '__main__':
    unittest.main()
