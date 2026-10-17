import unittest

from row.config import load_config
from row.experiments import g1s2_tanh_budget as g2
from row.experiments import g1t_tanh_formation as g1t
from row.experiments import o2_online_reliability as o2
from row.models import tanh_form


def _records(terminals, d1=None, ari=None, anchor_ok=True):
    """`terminals`: point tag -> slept terminal median."""
    d1, ari = d1 or {}, ari or {}
    anchor = {'WAKE': {'t': 0.5}, 'SLEEP': {'t': 0.18}}
    out = {}
    for u in g2.WAKE_UPDATES:
        out[g2.key_of('WAKE', u)] = {'terminal_median': 0.5, 'd1_ari': 1.0,
                                     'terminal_per_task': anchor['WAKE'] if u == 1 and anchor_ok else {'t': 0.9}}
        curve = {}
        for n in g2.CHECKPOINTS:
            tag = g2.point_tag(u, n)
            per = anchor['SLEEP'] if (u, n) == (1, g2.CHECKPOINTS[0]) and anchor_ok else {'t': 0.9}
            curve[str(n)] = {'terminal_median': terminals[tag], 'terminal_per_task': per, 'd1_ari': ari.get(tag, 1.0),
                             'depth': {'1': {'median': d1.get(tag, 0.02)}, '2': {'median': 0.05},
                                       '3': {'median': terminals[tag]}}}
        out[g2.key_of('SLEEP', u)] = {'curve': curve}
    return out, anchor


def _terms(**over):
    base = {g2.point_tag(u, n): 0.18 for u in g2.WAKE_UPDATES for n in g2.CHECKPOINTS}
    base.update(over)
    return base


class BudgetTests(unittest.TestCase):
    def tearDown(self):
        g2.restore()

    def test_budget_applies_and_restores(self):
        g2.apply_budget(2)
        d = o2.stage_setup(70, 3, 5000)[0].discrete_model
        self.assertEqual((d.global_learning_rate, d.updates_per_example), (0.003, 2))
        self.assertEqual(tanh_form.INIT_GAIN, 1.0)
        g2.apply_budget(1)   # re-application does not nest wrappers
        d = o2.stage_setup(70, 1, 5000)[0].discrete_model
        self.assertEqual((d.global_learning_rate, d.updates_per_example), (0.003, 1))
        g2.restore()
        self.assertIs(o2.stage_setup, g1t._ORIGINAL['stage_setup'])
        base = load_config('configs/v1.yaml').discrete_model
        self.assertEqual(o2.stage_setup(70, 3, 5000)[0].discrete_model.global_learning_rate,
                         base.global_learning_rate)

    def test_gates_pass(self):
        out = g2.gates()
        self.assertTrue(out['all_pass'], out)

    def test_sleep_curve_calls_every_checkpoint(self):
        g2.apply_budget(1)
        cfg3 = o2.stage_setup(70, 3, 5000)[0]
        import numpy as np
        pool = [(np.zeros(16, np.float32), np.ones(16, np.float32), 'a') for _ in range(4)]
        m = o2.planned_model(cfg3, {})
        m.begin_task('a')
        seen = []
        g2.sleep_curve(cfg3, m, pool, ['a'], [0], (2, 5), seen.append)
        self.assertEqual(seen, [2, 5])


class ReadingTests(unittest.TestCase):
    def test_labels(self):
        rec, a = _records(_terms())
        self.assertEqual(g2.reading(rec, a)['label'], 'NO_GAIN')
        rec, a = _records(_terms(u2_s65536=0.08))
        r = g2.reading(rec, a)
        self.assertEqual((r['label'], r['chosen']), ('IMPROVES', 'u2_s65536'))
        rec, a = _records(_terms(u1_s32768=0.04, u2_s65536=0.08))
        r = g2.reading(rec, a)
        self.assertEqual((r['label'], r['chosen']), ('FITS', 'u1_s32768'))

    def test_identity_anchor_and_compounding(self):
        rec, a = _records(_terms(u1_s65536=0.03), ari={'u1_s65536': 0.5})
        r = g2.reading(rec, a)
        self.assertNotEqual(r['chosen'], 'u1_s65536')
        rec, a = _records(_terms(), anchor_ok=False)
        self.assertEqual(g2.reading(rec, a)['label'], 'ANCHOR_MISS')
        rec, a = _records(_terms(), ari={t: 0.2 for t in _terms()})
        self.assertEqual(g2.reading(rec, a)['label'], 'IDENTITY_LOST')
        rec, a = _records(_terms(u1_s8192=0.2), d1={'u1_s8192': 0.025})
        self.assertAlmostEqual(g2.reading(rec, a)['points']['u1_s8192']['compounding'], 8.0)
        rec, a = _records(_terms(u1_s65536=0.09, u1_s32768=0.12))
        r = g2.reading(rec, a)
        self.assertAlmostEqual(r['last_doubling_ratio']['u1'], 0.75)
        self.assertEqual(r['still_falling'], {'u1': True, 'u2': False})

    def test_g1s_anchor_reads_chosen_cells(self):
        a = g2.g1s_anchor()
        self.assertEqual((len(a['WAKE']), len(a['SLEEP'])), (64, 64))


if __name__ == '__main__':
    unittest.main()
