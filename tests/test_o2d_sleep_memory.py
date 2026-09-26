import unittest

from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import score_o2d_sleep_memory as scorer


class O2DTests(unittest.TestCase):
    def test_labels_partition_and_agree(self):
        for k in range(22):
            for b in range(10):
                self.assertEqual(o2d.label(k, b), scorer.label(k, b))
        self.assertEqual((o2d.label(18, 1), o2d.label(17, 0), o2d.label(20, 2)),
                         ('SLEEP_MEMORY_SUFFICES', 'SHORT', 'HARMS'))

    def test_reservoir_sizes_and_nesting_free_selection(self):
        _, _, stream, _, _ = o2.build_stream('SHUFFLED', 13, 0)
        for m in (4, 16, 64):
            pool = o2d.reservoir(stream, 13, 0, m)
            self.assertEqual(len(pool), 188 * m)
            self.assertEqual({p[2] for p in pool}, {t.task_id for t in stream})
        a = o2d.reservoir(stream, 13, 0, 16)
        b = o2d.reservoir(stream, 13, 0, 16)
        self.assertTrue(all(x[2] == y[2] and (x[0] == y[0]).all() for x, y in zip(a, b)))

    def test_cells(self):
        self.assertEqual(len(o2d.cells()), 63)


if __name__ == '__main__':
    unittest.main()
