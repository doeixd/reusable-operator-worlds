import unittest

from row.experiments import o4_sealed_confirmation as o4
from row.experiments import score_o4_sealed_confirmation as scorer


class O4Tests(unittest.TestCase):
    def test_cells_and_sealed_worlds(self):
        c = o4.cells()
        self.assertEqual(len(c), 45 + 45 + 15)
        self.assertEqual({w for _, w, _ in c}, set(range(900, 915)))
        self.assertNotIn(o4.DRY_WORLD, range(900, 930))

    def test_label_boundary_and_agreement(self):
        recs = {f'{a}_w{w}_s{s}': {'terminal_median': 1.9 if a == 'PLAIN' else 0.01} for a, w, s in o4.cells()}
        self.assertEqual(o4.summarize(recs)['label'], 'CONFIRMED')
        fails = [k for k in recs if k.startswith('SLEEP')][:4]
        for k in fails:
            recs[k]['terminal_median'] = 0.2
        s = o4.summarize(recs)
        self.assertEqual((s['passes']['SLEEP'], s['label']), (41, 'CONFIRMED'))
        recs[[k for k in recs if k.startswith('SLEEP')][4]]['terminal_median'] = float('nan')
        s = o4.summarize(recs)
        self.assertEqual((s['passes']['SLEEP'], s['label']), (40, 'NOT_CONFIRMED'))
        self.assertEqual(scorer.label(41, False), 'CONFIRMED')
        self.assertEqual(scorer.label(40, False), 'NOT_CONFIRMED')
        recs['PLAIN_w900_s0']['terminal_median'] = 0.01
        self.assertEqual(o4.summarize(recs)['label'], 'FLOOR_FAILED')
        self.assertEqual(scorer.label(45, True), 'FLOOR_FAILED')


if __name__ == '__main__':
    unittest.main()
