import unittest

from row.experiments import d2_depth5_formation as d2
from row.experiments import d4_stream as d4
from row.experiments import dn_stream as dn


class D2Tests(unittest.TestCase):
    def test_dn_reproduces_d4_at_depth_4(self):
        a = d4.build_stream(31, 2)
        b = dn.build_stream(4, 31, 2)
        self.assertEqual([t.task_id for t in a[2]], [t.task_id for t in b[2]])
        self.assertEqual(a[3], b[3])
        self.assertEqual(a[0], b[0])

    def test_depth5_stream(self):
        cfg, _, st, plan, canonical = dn.build_stream(5, 37, 0)
        self.assertEqual(len(st), d2.N_STREAM)
        self.assertEqual({d: sum(v == d for v in plan.values()) for d in range(1, 6)}, {1: 60, 2: 64, 3: 64, 4: 64, 5: 64})
        self.assertTrue(all(plan[t.task_id] == 5 for t in canonical))
        self.assertEqual(cfg.discrete_model.task_steps, 5)
        self.assertGreaterEqual(len({plan[t.task_id] for t in st[:20]}), 2)

    def test_cells_labels_summary(self):
        self.assertEqual(len(d2.cells()), 63)
        self.assertEqual(d2.WORLDS, tuple(range(37, 44)))
        self.assertNotIn(d2.DRY_WORLD, d2.WORLDS)
        self.assertEqual((d2.label(18, 2), d2.label(17, 0), d2.label(11, 0), d2.label(21, 3)),
                         ('TRANSFERS', 'PARTIAL', 'DOES_NOT_TRANSFER', 'HARMS'))
        recs = {f'{a}_w{w}_s{s}': {'terminal_median': {'RS5': 0.01, 'SLEEP5': 1.5, 'SHUFFLED5': 2.0}[a],
                                   'after_reroute_only_median': 0.1, 'reroute_seconds': 200.0}
                for a, w, s in d2.cells()}
        out = d2.summarize(recs)
        self.assertEqual((out['label'], out['n_better_vs_sleep'], out['h']), ('TRANSFERS', 21, 0))


if __name__ == '__main__':
    unittest.main()
