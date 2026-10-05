import unittest

import torch

from row.experiments import b1_branch_2x2 as b1


class B1Tests(unittest.TestCase):
    def test_label_partition(self):
        self.assertEqual({b1.label(k, 192) for k in range(193)}, {'LEARNABLE', 'PARTIAL', 'NOT_LEARNABLE'})
        self.assertEqual((b1.label(154, 192), b1.label(153, 192), b1.label(96, 192), b1.label(95, 192)),
                         ('LEARNABLE', 'PARTIAL', 'PARTIAL', 'NOT_LEARNABLE'))

    def test_best_pair_recovers_planted_pair(self):
        torch.manual_seed(0)
        E = torch.rand(400, 30) + 1.0
        E[:200, 7] = 0.0     # half the examples are fit exactly by route 7
        E[200:, 19] = 0.0    # the other half by route 19
        a, b, n = b1.best_pair(E)
        self.assertEqual({a, b}, {7, 19})
        self.assertGreaterEqual(n, 2)

    def test_summary_counts(self):
        tasks = [{'variant': 'INPUT' if i % 2 else 'MID', 'REFUSAL': 1.0, 'OS_OP': 0.01, 'OS_LP': 0.02, 'LS_OP': 0.03,
                  'LS_LP': 0.04 if i < 160 else float('nan'), 'LS_OP_structure_exact': True,
                  'LS_LP_pair_matches_oracle': True} for i in range(192)]
        out = b1.summarize({30: {'tasks': tasks}})
        self.assertEqual(out['ALL']['LS_LP']['passes'], 160)
        self.assertEqual(out['label'], 'LEARNABLE')


if __name__ == '__main__':
    unittest.main()
