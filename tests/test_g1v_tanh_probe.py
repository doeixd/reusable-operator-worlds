import unittest

from row.experiments import g1r_tanh_reroute as gr
from row.experiments import g1v_tanh_probe as gv
from row.experiments import o2_online_reliability as o2
from row.experiments import o5_reroute_sleep as o5
from row.models.tanh_form import FrozenTanhLibrary


def _point(median, ari=1.0):
    return {'terminal_median': median, 'terminal_per_task': {f't{i}': median for i in range(64)}, 'd1_ari': ari,
            'depth': {d: {'median': median, 'n': 1} for d in '123'}}


def _records(finals, aris=None):
    aris = aris or {}
    out = {}
    for w, v in zip(gv.WORLDS, finals):
        out[gv.key_of('WAKE', w)] = {'terminal_median': 0.9}
        out[gv.key_of('RR_SLEEP', w)] = {
            'curve': {str(n): _point(v if n == gv.REGISTERED else 0.2, aris.get(w, 1.0)) for n in gv.CHECKPOINTS},
            'routes_changed': 40, 'formed_s1': {'median': {'1': 0.9, '3': 0.85}}}
    return out


class ReadingTests(unittest.TestCase):
    def test_labels_and_denominator(self):
        r = gv.reading(_records([0.04, 0.03, 0.045, 0.049, 0.08]))
        self.assertEqual((r['label'], r['forming_worlds'], r['denominator'], r['below_loose']), ('FORMS', 4, 5, 5))
        self.assertEqual(gv.reading(_records([0.04, 0.06, 0.07, 0.08, 0.12]))['label'], 'FAILS')
        self.assertEqual(gv.reading(_records([0.04, 0.03, 0.07, 0.08, 0.12]))['label'], 'MIXED')

    def test_registered_point_not_best_and_identity(self):
        recs = _records([0.06] * 5)
        for w in gv.WORLDS:   # an earlier checkpoint below the bar does not count
            recs[gv.key_of('RR_SLEEP', w)]['curve']['8192'] = _point(0.01)
        self.assertEqual(gv.reading(recs)['label'], 'FAILS')
        r = gv.reading(_records([0.01] * 5, aris={w: 0.5 for w in gv.WORLDS}))
        self.assertEqual(r['label'], 'FAILS')

    def test_g1t_pairing(self):
        r = gv.reading(_records([0.06] * 5))
        self.assertEqual(sorted(r['g1t_paired_rw_sleep']), ['71', '72'])


class BindingTests(unittest.TestCase):
    def tearDown(self):
        gr.restore()

    def test_gates_pass_and_binding(self):
        out = gv.gates((71, 73))
        self.assertTrue(out['all_pass'], out)
        gr.use_reroute_form(gv.WAKE_UPDATES)
        self.assertIs(o5.FrozenLibrary, FrozenTanhLibrary)
        d = o2.stage_setup(73, 3, 5000)[0].discrete_model
        self.assertEqual((d.global_learning_rate, d.updates_per_example), (0.003, 2))


if __name__ == '__main__':
    unittest.main()
