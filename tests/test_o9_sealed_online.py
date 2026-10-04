import unittest

from row.experiments import o9_sealed_online as o9
from row.experiments import score_o9_sealed_online as scorer


class O9Tests(unittest.TestCase):
    def test_cells_and_constants_match_plan(self):
        c = o9.cells()
        self.assertEqual(len(c), 150)
        self.assertEqual(o9.WORLDS, tuple(range(930, 945)))
        self.assertEqual(list(o9.WORLDS), scorer.WORLDS)
        self.assertEqual([a for a, _, _ in c[:45]], ['REROUTE_WAKE'] * 45)   # decisive arm first
        self.assertEqual(sum(a == 'PLAIN' for a, _, _ in c), 15)
        self.assertNotIn(o9.DRY_WORLD, range(900, 960))
        self.assertFalse(set(o9.WORLDS) & set(range(900, 930)))
        self.assertEqual((o9.CONFIRM_AT, o9.ATTRIBUTE_AT, o9.THRESHOLD, o9.MIN_FREE_GIB), (41, 30, 0.05, 2.0))
        self.assertEqual({a for a, _, _ in o9.dry_cells()}, set(o9.ARMS))   # dry run covers every arm

    def test_labels_partition_and_agree_with_scorer(self):
        for k in range(46):
            for plain in (0, 1):
                self.assertEqual(o9.label(k, plain > 0), scorer.primary(k, plain))
            self.assertEqual(o9.attribution(k), scorer.secondary(k))
        self.assertEqual((o9.label(41, False), o9.label(40, False), o9.label(45, True)),
                         ('CONFIRMED', 'NOT_CONFIRMED', 'FLOOR_FAILED'))
        self.assertEqual((o9.attribution(30), o9.attribution(29)), ('ATTRIBUTED', 'NOT_ATTRIBUTED'))

    def _records(self, rw=0.01, sh=0.2, rs=0.01, plain=1.9):
        recs = {}
        for a, w, s in o9.cells():
            m = {'REROUTE_WAKE': rw, 'SHUFFLED': sh, 'REROUTE_SLEEP': rs, 'PLAIN': plain}[a]
            recs[f'{a}_w{w}_s{s}'] = {'terminal_median': m, 'terminal_stale': 0}
        return recs

    def test_summary_rules_and_nan(self):
        recs = self._records()
        out = o9.summarize(recs)
        self.assertEqual((out['label'], out['attribution'], out['n_better']), ('CONFIRMED', 'ATTRIBUTED', 45))
        for w in o9.WORLDS[:2]:
            for s in o9.STREAMS:
                recs[f'REROUTE_WAKE_w{w}_s{s}']['terminal_median'] = float('nan')
        out = o9.summarize(recs)
        self.assertEqual((out['passes']['REROUTE_WAKE'], out['n_better'], out['label']), (39, 39, 'NOT_CONFIRMED'))
        recs = self._records()
        recs['PLAIN_w930_s0']['terminal_median'] = 0.01
        self.assertEqual(o9.summarize(recs)['label'], 'FLOOR_FAILED')
        recs = self._records(sh=0.01)   # ties are not better
        self.assertEqual(o9.summarize(recs)['attribution'], 'NOT_ATTRIBUTED')


if __name__ == '__main__':
    unittest.main()
