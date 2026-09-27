import unittest

from row.experiments import o6_sealed_reroute as o6
from row.experiments import score_o6_sealed_reroute as scorer


def records(value=0.01):
    return {f'{a}_w{w}_s{s}': {'terminal_median': 1.9 if a == 'PLAIN' else value} for a, w, s in o6.cells()}


class O6Tests(unittest.TestCase):
    def test_cells_and_sealed_worlds(self):
        c = o6.cells()
        self.assertEqual(len(c), 45 * 3 + 15)
        self.assertEqual({w for _, w, _ in c}, set(range(915, 930)))
        self.assertFalse({w for _, w, _ in c} & set(range(900, 915)))   # O4/O5-contaminated worlds excluded
        self.assertNotIn(o6.DRY_WORLD, range(900, 930))
        self.assertEqual(o6.WORLDS, tuple(scorer.WORLDS))
        self.assertEqual({a: list(s) for a, s in o6.ARM_STREAMS.items()}, scorer.ARM_STREAMS)

    def test_primary_boundary_uses_reroute_arm(self):
        recs = records()
        s = o6.summarize(recs)
        self.assertEqual(s['label'], 'CONFIRMED')
        rs = [k for k in recs if k.startswith('REROUTE_SLEEP')]
        for k in rs[:4]:
            recs[k]['terminal_median'] = 0.2
        self.assertEqual((o6.summarize(recs)['passes']['REROUTE_SLEEP'], o6.summarize(recs)['label']),
                         (41, 'CONFIRMED'))
        recs[rs[4]]['terminal_median'] = float('nan')
        self.assertEqual((o6.summarize(recs)['passes']['REROUTE_SLEEP'], o6.summarize(recs)['label']),
                         (40, 'NOT_CONFIRMED'))
        # SLEEP failing everywhere must not change the primary label
        recs2 = records()
        for k in recs2:
            if k.startswith('SLEEP'):
                recs2[k]['terminal_median'] = 0.9
        self.assertEqual(o6.summarize(recs2)['label'], 'CONFIRMED')
        recs2['PLAIN_w915_s0']['terminal_median'] = 0.01
        self.assertEqual(o6.summarize(recs2)['label'], 'FLOOR_FAILED')
        for k, f in ((41, False), (40, False), (45, True)):
            self.assertEqual(scorer.label(k, f), 'FLOOR_FAILED' if f else ('CONFIRMED' if k >= 41 else 'NOT_CONFIRMED'))

    def test_attribution_counts_strict_improvement_and_nan(self):
        recs = records()   # all ties: nothing is better
        self.assertEqual((o6.summarize(recs)['n_better'], o6.summarize(recs)['attribution']), (0, 'NOT_ATTRIBUTED'))
        pairs = [(w, s) for w in o6.WORLDS for s in o6.STREAMS]
        for w, s in pairs[:30]:
            recs[f'SLEEP_w{w}_s{s}']['terminal_median'] = 0.02
        self.assertEqual((o6.summarize(recs)['n_better'], o6.summarize(recs)['attribution']), (30, 'ATTRIBUTED'))
        w, s = pairs[0]
        recs[f'REROUTE_SLEEP_w{w}_s{s}']['terminal_median'] = float('nan')
        self.assertEqual((o6.summarize(recs)['n_better'], o6.summarize(recs)['attribution']), (29, 'NOT_ATTRIBUTED'))
        self.assertFalse(scorer.better(float('nan'), 0.5))
        self.assertFalse(scorer.better(0.01, 0.01))
        self.assertFalse(scorer.better(0.01, float('nan')))   # conservative: a NaN SLEEP never counts for RS
        self.assertEqual((scorer.attribution(30), scorer.attribution(29)), ('ATTRIBUTED', 'NOT_ATTRIBUTED'))

    def test_dependent_order_and_validate(self):
        self.assertEqual(o6.DEPENDENT, ('REROUTE_SLEEP', 'SLEEP'))
        rec = {'arm': 'REROUTE_SLEEP', 'world': 915, 'stream': 0, 'scored_tasks': 64,
               'terminal_per_task': {str(i): 0.0 for i in range(64)}, 'extra_updates': 8192,
               'pool_size': 188 * 64, 'swap': True, 'library_sha256': 'b', 'library_sha256_before': 'a'}
        o6.validate(rec)
        for field, bad in (('swap', False), ('pool_size', 10), ('library_sha256', 'a')):
            with self.assertRaises(ValueError):
                o6.validate(rec | {field: bad})


if __name__ == '__main__':
    unittest.main()
