import unittest

from row.experiments import o11_sealed_instream_sleep as o11
from row.experiments import score_o11_sealed_instream_sleep as scorer


class O11Tests(unittest.TestCase):
    def test_cells_and_constants(self):
        c = o11.cells()
        self.assertEqual(len(c), 195)
        self.assertEqual(o11.WORLDS, tuple(range(945, 960)))
        self.assertEqual(list(o11.WORLDS), scorer.WORLDS)
        self.assertFalse(set(o11.WORLDS) & set(range(900, 945)))
        self.assertNotIn(o11.DRY_WORLD, range(900, 960))
        self.assertEqual({a for a, _, _ in o11.dry_cells()}, set(o11.ARMS))
        self.assertEqual([a for a, _, _ in c[:45]], ['REROUTE_WAKE'] * 45)
        self.assertEqual(o11.PARENT, {'RW_SLEEP': 'REROUTE_WAKE', 'SLEEP': 'SHUFFLED'})

    def test_labels_agree(self):
        for k in range(46):
            for p in (0, 1):
                self.assertEqual(o11.label(k, p > 0), scorer.primary(k, p))
            self.assertEqual(o11.attribution(k), scorer.secondary(k))

    def _recs(self, rws=0.01, sleep=0.02):
        m = {'REROUTE_WAKE': 0.03, 'SHUFFLED': 0.1, 'PLAIN': 1.9, 'RW_SLEEP': rws, 'SLEEP': sleep}
        return {f'{a}_w{w}_s{s}': {'terminal_median': m[a]} for a, w, s in o11.cells()}

    def test_summary(self):
        out = o11.summarize(self._recs())
        self.assertEqual((out['label'], out['attribution']), ('CONFIRMED', 'ATTRIBUTED'))
        self.assertEqual(o11.summarize(self._recs(sleep=0.01))['attribution'], 'NOT_ATTRIBUTED')   # ties
        recs = self._recs()
        for w in o11.WORLDS[:2]:
            for s in o11.STREAMS:
                recs[f'RW_SLEEP_w{w}_s{s}']['terminal_median'] = float('nan')
        out = o11.summarize(recs)
        self.assertEqual((out['passes']['RW_SLEEP'], out['n_better'], out['label']), (39, 39, 'NOT_CONFIRMED'))


if __name__ == '__main__':
    unittest.main()
