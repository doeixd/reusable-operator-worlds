import unittest

from row.experiments import o2g_stage3_collapse as o2g
from row.experiments import score_o2g_stage3_collapse as scorer
from row.experiments.audit_j1c_curriculum import library_sha


class O2GTests(unittest.TestCase):
    def test_labels_agree_and_partition(self):
        for c in range(7):
            for h in range(37):
                self.assertEqual(o2g.label(c, h), scorer.label(c, h))
        self.assertEqual((o2g.label(5, 3), o2g.label(6, 4), o2g.label(2, 0), o2g.label(3, 0)),
                         ('SYSTEMATIC', 'MIXED', 'STOCHASTIC', 'MIXED'))

    def test_registered_collapsed_cells(self):
        self.assertEqual(sorted(o2g.collapsed_o2_cells()), [(14, 1), (14, 2), (16, 1)])

    def test_rerun_seeds_distinct(self):
        seeds = {o2g.rerun_seed(w, s, k) for w, s, k in o2g.cells()}
        self.assertEqual(len(seeds), 42)

    def test_stage2_reloads_strictly(self):
        m = o2g.load_stage2(14, 1)
        self.assertEqual(len(library_sha(m)), 64)


if __name__ == '__main__':
    unittest.main()
