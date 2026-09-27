import unittest
from pathlib import Path

import numpy as np
import torch

from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o5_reroute_sleep as o5
from row.experiments import score_o5_reroute_sleep as scorer
from row.experiments.audit_j2a_staged_library import enum_route
from row.experiments.audit_so1r_route_only import FrozenLibrary


class O5Tests(unittest.TestCase):
    def test_labels_partition_and_agree(self):
        for r in range(9):
            for b in range(5):
                self.assertEqual(o5.label(r, b), scorer.label(r, b))
        self.assertEqual((o5.label(6, 1), o5.label(5, 1), o5.label(2, 0), o5.label(8, 2)),
                         ('COLLAPSE_REPAIRED', 'PARTIAL', 'NO_REPAIR', 'HARMS'))

    def test_cells(self):
        self.assertEqual(len(o5.cells()), 87)

    def test_enumeration_matches_brute_force_at_every_depth(self):
        cfg3, model, st, canon = o3.load_shuffled_terminal(o5.terminal_path('O3', 20, 0), 20, 0)
        lib = FrozenLibrary(model)
        x = torch.randn(16, cfg3.world.state_dim, generator=torch.Generator().manual_seed(0))
        for d in (1, 2):
            lib.steps = d
            target = [3, 7][:d]
            with torch.no_grad():
                y = lib.hard(x, target)
            self.assertEqual(list(enum_route(lib, x, y)), target)

    def test_reroute_changes_code_to_searched_route(self):
        cfg3, model, st, canon = o3.load_shuffled_terminal(o5.terminal_path('O4', 911, 0), 911, 0)
        _, _, _, plan, _ = o2.build_stream('SHUFFLED', 911, 0)
        pool = o2d.reservoir(st, 911, 0, 64)
        changed = o5.reroute(model, st, plan, pool)
        self.assertGreater(changed, 0)   # a collapsed cell: stale routes exist


if __name__ == '__main__':
    unittest.main()
