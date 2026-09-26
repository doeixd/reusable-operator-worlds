import unittest

import numpy as np

from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import score_o2c_consolidation as scorer


class O2CTests(unittest.TestCase):
    def test_labels_partition_and_agree(self):
        for r in range(9):
            for b in range(10):
                self.assertEqual(o2c.label(r, b), scorer.label(r, b))
                self.assertIn(o2c.label(r, b), ('RESCUES', 'HARMS', 'NO_RESCUE'))
        self.assertEqual((o2c.label(5, 1), o2c.label(4, 1), o2c.label(8, 2)), ('RESCUES', 'NO_RESCUE', 'HARMS'))

    def test_cells(self):
        self.assertEqual(len(o2c.cells()), 42)

    def test_replay_seed_matches_lifetime_default(self):
        self.assertEqual(o2c.replay_seed(13, 0, 5000), 5001)
        self.assertEqual(o2c.replay_seed(13, 1, 5000), o2.replay_seed_for(13, 1))

    def test_reconstruction_is_deterministic_and_sized(self):
        _, _, stream, _, _ = o2.build_stream('SHUFFLED', 13, 0)
        a = o2c.reconstruct_buffer(stream, 5001, 8)
        b = o2c.reconstruct_buffer(stream, 5001, 8)
        self.assertEqual(len(a), 752)
        self.assertTrue(all(x[2] == y[2] and np.array_equal(x[0], y[0]) for x, y in zip(a, b)))
        c = o2c.reconstruct_buffer(stream, 5002, 8)
        self.assertEqual([x[2] for x in a], [x[2] for x in c])          # task order is the stream's
        self.assertFalse(all(np.array_equal(x[0], y[0]) for x, y in zip(a, c)))  # the seed picks the examples


if __name__ == '__main__':
    unittest.main()
