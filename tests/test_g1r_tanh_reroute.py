import unittest

from row.experiments import g1r_tanh_reroute as gr
from row.experiments import o5_reroute_sleep as o5
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.models import tanh_form
from row.models.tanh_form import FrozenTanhLibrary

TASKS = [f't{i}' for i in range(64)]


def _point(median, per=None, ari=1.0):
    return {'terminal_median': median, 'terminal_per_task': per or {t: median for t in TASKS}, 'd1_ari': ari,
            'depth': {d: {'median': 0.05, 'n': 1} for d in '123'}}


def _ref(n_collapsed):
    """G1S2-like reference: the first n_collapsed[u] tasks collapsed at 65,536."""
    ref = {}
    for u in gr.WAKES:
        last = {t: (1.0 if i < n_collapsed[u] else 0.08) for i, t in enumerate(TASKS)}
        ref[u] = {'wake': {t: 0.5 for t in TASKS},
                  'curve': {str(n): (last if n == gr.CHECKPOINTS[-1] else {t: 0.2 for t in TASKS})
                            for n in gr.CHECKPOINTS}}
    return ref


def _records(ref, rescued_frac, rr_best, ari=1.0, break_anchor=False):
    out = {}
    for u in gr.WAKES:
        wake = _point(0.5, dict(ref[u]['wake']))
        if break_anchor and u == 1:
            wake = _point(0.5, {t: 0.51 for t in TASKS})
        curve = {str(n): _point(0.1, dict(ref[u]['curve'][str(n)])) for n in gr.CHECKPOINTS}
        collapsed = [t for t, v in ref[u]['curve'][str(gr.CHECKPOINTS[-1])].items() if v > gr.COLLAPSED]
        k = round(rescued_frac * len(collapsed))
        rr = {t: (0.1 if t in collapsed[:k] else (0.9 if t in collapsed else 0.08)) for t in TASKS}
        out[gr.key_of('SLEEP_RR', u)] = {'wake': wake, 'curve': curve, 'rr': _point(0.1, rr), 'rr_sleep': _point(0.1),
                                         'routes_changed': 5}
        rcurve = {str(n): _point(rr_best if n == gr.CHECKPOINTS[-1] else 0.3, ari=ari) for n in gr.CHECKPOINTS}
        out[gr.key_of('RR_SLEEP', u)] = {'wake': _point(0.5, dict(ref[u]['wake'])), 'curve': rcurve,
                                         'rerouted': _point(0.4), 'routes_changed': 7}
    return out


class ReadingTests(unittest.TestCase):
    def test_labels(self):
        ref = _ref({1: 20, 2: 34})
        self.assertEqual(gr.reading(_records(ref, 0.9, 0.03), ref)['label'], 'FITS')
        r = gr.reading(_records(ref, 0.9, 0.08), ref)
        self.assertEqual(r['label'], 'STALE_ONLY')
        self.assertAlmostEqual(r['stale_fraction'], (18 + 31) / 54)
        self.assertEqual(r['collapsed_tasks'], {'u1': 20, 'u2': 34})
        self.assertEqual(gr.reading(_records(ref, 0.2, 0.08), ref)['label'], 'NOT_STALE')

    def test_identity_and_anchor(self):
        ref = _ref({1: 20, 2: 34})
        self.assertEqual(gr.reading(_records(ref, 0.2, 0.03, ari=0.5), ref)['label'], 'NOT_STALE')
        self.assertEqual(gr.reading(_records(ref, 0.9, 0.03, break_anchor=True), ref)['label'], 'ANCHOR_MISS')

    def test_reference_reads_g1s2(self):
        ref = gr.g1s2_reference()
        last = ref[1]['curve'][str(gr.CHECKPOINTS[-1])]
        self.assertEqual(sum(v > gr.COLLAPSED for v in last.values()), 20)
        self.assertEqual(sum(v > gr.COLLAPSED for v in ref[2]['curve'][str(gr.CHECKPOINTS[-1])].values()), 34)


class BindingTests(unittest.TestCase):
    def tearDown(self):
        gr.restore()

    def test_binding_and_restore(self):
        gr.use_reroute_form(1)
        self.assertIs(o5.FrozenLibrary, FrozenTanhLibrary)
        gr.restore()
        self.assertIs(o5.FrozenLibrary, FrozenLibrary)
        self.assertEqual(tanh_form.INIT_GAIN, 1.0)

    @unittest.skipUnless(gr.wake_terminal(1).exists(), 'saved G1S2 terminal not present')
    def test_gates_pass(self):
        self.assertTrue(gr.gates()['all_pass'])


if __name__ == '__main__':
    unittest.main()
