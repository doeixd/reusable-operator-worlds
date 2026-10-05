import unittest

from row.experiments import d1_depth4_formation as d1
from row.experiments import d4_stream as d4


class D1Tests(unittest.TestCase):
    def test_stream_composition(self):
        cfg, world, st, plan, canonical = d4.build_stream(30, 0)
        self.assertEqual(len(st), 252)
        self.assertEqual({d: sum(v == d for v in plan.values()) for d in (1, 2, 3, 4)}, {1: 60, 2: 64, 3: 64, 4: 64})
        self.assertEqual(len(canonical), 64)
        self.assertTrue(all(plan[t.task_id] == 4 for t in canonical))
        self.assertEqual(len({t.task_id for t in st}), 252)
        self.assertGreaterEqual(len({plan[t.task_id] for t in st[:20]}), 2)
        self.assertEqual(cfg.discrete_model.task_steps, 4)
        _, _, st1, _, _ = d4.build_stream(30, 1)
        self.assertNotEqual([t.task_id for t in st], [t.task_id for t in st1])   # streams reorder

    def test_cells_and_labels(self):
        self.assertEqual(len(d1.cells()), 84)
        self.assertEqual(d1.WORLDS, tuple(range(30, 37)))
        self.assertNotIn(d1.DRY_WORLD, d1.WORLDS)
        seen = set()
        for k in range(22):
            for h in range(22):
                seen.add(d1.label(k, h))
        self.assertEqual(seen, {'HARMS', 'TRANSFERS', 'PARTIAL', 'DOES_NOT_TRANSFER'})
        self.assertEqual((d1.label(18, 2), d1.label(17, 0), d1.label(11, 0), d1.label(21, 3)),
                         ('TRANSFERS', 'PARTIAL', 'DOES_NOT_TRANSFER', 'HARMS'))

    def test_summary_harm_and_nan(self):
        recs = {f'{a}_w{w}_s{s}': {'terminal_median': {'RW_SLEEP4': 0.01, 'SLEEP4': 0.02, 'RW4': 0.03,
                                                        'SHUFFLED4': 0.2}[a]} for a, w, s in d1.cells()}
        out = d1.summarize(recs)
        self.assertEqual((out['label'], out['h'], out['n_better_vs_sleep']), ('TRANSFERS', 0, 21))
        for w, s in [(30, 0), (30, 1), (30, 2)]:
            recs[f'RW_SLEEP4_w{w}_s{s}']['terminal_median'] = float('nan')
        out = d1.summarize(recs)
        self.assertEqual((out['label'], out['h']), ('HARMS', 3))


if __name__ == '__main__':
    unittest.main()
