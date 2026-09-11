import unittest

import torch

from row.config import load_config
from row.experiments import audit_j1_search_loop as j1
from row.experiments.audit_rotated_g5r_interference import _assignment, offline_cell, world_config
from row.experiments.audit_so1_budget_bracket import build_fast, checkpoints_for
from row.rotated_world import generate_rotated_world


class J1Tests(unittest.TestCase):
    def test_loop_with_hook_off_is_bitwise_offline_cell(self):
        torch.set_num_threads(1)
        cfg = world_config(load_config("configs/v1.yaml"), 1)
        world = generate_rotated_world(cfg.world)
        reference = offline_cell(cfg, world, _assignment(cfg, world), oracle=True, updates=16, batch=2,
                                 cell_index=j1.STREAM, checkpoints_requested=checkpoints_for(16),
                                 build=build_fast, retain_per_task=True)
        mine = j1.train_cell(1, "oracle", updates=16)
        self.assertEqual(mine["final_per_task"], reference["final_per_task"])
        self.assertEqual(mine["trajectory"], {k: v["median"] for k, v in reference["checkpoints"].items()})

    def test_sham_is_a_permutation_of_the_same_routes(self):
        routes = {f"t{i}": (i % 12, 0, 1) for i in range(64)}
        shuffled = j1.permuted(routes, 1, 3)
        self.assertEqual(sorted(shuffled.values()), sorted(routes.values()))
        self.assertNotEqual(shuffled, routes)
        self.assertEqual(shuffled, j1.permuted(routes, 1, 3))

    def test_ari(self):
        self.assertAlmostEqual(j1.ari([0, 0, 1, 1], [5, 5, 7, 7]), 1.0)
        self.assertAlmostEqual(j1.ari([0, 0, 1, 1], [0, 1, 0, 1]), -0.5)

    def test_classification(self):
        so1 = {j1.LEARNED_KEY: {str(w): {"terminal_median": 0.95} for w in j1.WORLDS}}
        def cell(m):
            return {"terminal_median": m, "finite": True, "pinned_one_hot": True, "shared_relative_change": 1.0,
                    "reload_exact": True, "rounds": [{"argmin_ok": True}]}
        cells = {f"J1_w{w}": cell(m) for w, m in zip(j1.WORLDS, (0.7, 0.01, 0.02))}
        cells |= {f"SHAM_w{w}": cell(0.9) for w in j1.WORLDS}
        self.assertEqual(j1.classify(cells, so1, True), "J1_ACQUIRES")
        self.assertEqual(j1.classify(cells, so1, False), "HARNESS_FAILED")
        cells |= {f"J1_w{w}": cell(m) for w, m in zip(j1.WORLDS, (0.7, 0.3, 0.4))}
        self.assertEqual(j1.classify(cells, so1, True), "J1_IMPROVES")
        cells |= {f"J1_w{w}": cell(0.8) for w in j1.WORLDS}
        self.assertEqual(j1.classify(cells, so1, True), "J1_FAILS")


if __name__ == "__main__":
    unittest.main()
