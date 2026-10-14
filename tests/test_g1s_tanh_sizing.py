import unittest

import torch

from row.config import load_config
from row.experiments import g1s_tanh_sizing as g1s
from row.experiments import g1t_tanh_formation as g1t
from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.models import tanh_form
from row.models.tanh_form import tanh_gain_scale


def _records(slept, ari=None, anchor_ok=True):
    """`slept`: tag -> slept terminal median. WAKE cells and the baseline per-task anchor are filled in."""
    ari = ari or {}
    anchor = {'WAKE': {'t': 0.5}, 'SLEEP': {'t': 0.4}}
    out = {}
    for s in g1s.SETTINGS:
        t = g1s.setting_tag(s)
        per = anchor if s == g1s.BASELINE and anchor_ok else {'WAKE': {'t': 0.9}, 'SLEEP': {'t': 0.9}}
        out[g1s.key_of('WAKE', s)] = {'terminal_median': 0.6, 'terminal_per_task': per['WAKE'], 'd1_ari': 1.0}
        out[g1s.key_of('SLEEP', s)] = {'terminal_median': slept[t], 'terminal_per_task': per['SLEEP'],
                                       'd1_ari': ari.get(t, 1.0),
                                       'depth': {d: {'median': 0.1, 'n': 1} for d in '123'}}
    return out, anchor


class SettingTests(unittest.TestCase):
    def tearDown(self):
        g1s.restore()

    def test_gain_and_lr_take_effect_and_baseline_is_g1t(self):
        cfg = load_config('configs/v1.yaml')
        g1s.apply_setting((0.01, 2.0))
        model = o2.planned_model(cfg, {})
        sv = torch.linalg.svdvals(model.library[3].A.detach())
        self.assertTrue(torch.allclose(sv, torch.full_like(sv, 2.0), atol=1e-5))
        self.assertAlmostEqual(float(model.library[3].alpha.detach()), tanh_gain_scale(2.0), places=6)
        stage_cfg = o2.stage_setup(70, 3, 5000)[0]
        self.assertEqual(stage_cfg.discrete_model.global_learning_rate, 0.01)
        self.assertEqual(ll._training_values(stage_cfg, o2.KIND)[0], 0.01)   # what wake and sleep read
        g1s.apply_setting(g1s.BASELINE)
        sized = o2.planned_model(cfg, {}).state_dict()
        g1s.restore()
        g1t.use_form()
        plain = o2.planned_model(cfg, {}).state_dict()
        self.assertTrue(all(torch.equal(v, sized[k]) for k, v in plain.items()))
        self.assertEqual(o2.stage_setup(70, 3, 5000)[0], g1t.form_stage_setup(70, 3, 5000)[0])

    def test_restore(self):
        g1s.apply_setting((0.003, 2.0))
        g1s.restore()
        self.assertEqual(tanh_form.INIT_GAIN, 1.0)
        self.assertIs(o2.stage_setup, g1t._ORIGINAL['stage_setup'])

    def test_gate_fails_when_gain_is_not_applied(self):
        original = g1s.apply_setting

        def broken(setting):
            original(setting)
            tanh_form.INIT_GAIN = 1.0
        g1s.apply_setting = broken
        try:
            out = g1s.gates(((0.001, 1.0), (0.001, 2.0)))
        finally:
            g1s.apply_setting = original
        self.assertTrue(out['S1']['lr0.001_g1'])
        self.assertFalse(out['S1']['lr0.001_g2'])
        self.assertFalse(out['all_pass'])


class ReadingTests(unittest.TestCase):
    def _slept(self, **over):
        base = {g1s.setting_tag(s): 0.42 for s in g1s.SETTINGS}
        base.update(over)
        return base

    def test_labels(self):
        rec, anchor = _records(self._slept())
        self.assertEqual(g1s.reading(rec, anchor)['label'], 'NO_GAIN')
        rec, anchor = _records(self._slept(**{'lr0.003_g1': 0.12}))
        r = g1s.reading(rec, anchor)
        self.assertEqual((r['label'], r['chosen']), ('IMPROVES', 'lr0.003_g1'))
        rec, anchor = _records(self._slept(**{'lr0.003_g1': 0.12, 'lr0.01_g2': 0.02}))
        r = g1s.reading(rec, anchor)
        self.assertEqual((r['label'], r['chosen']), ('FITS', 'lr0.01_g2'))

    def test_identity_filter_and_anchor(self):
        slept = self._slept(**{'lr0.01_g2': 0.02})
        rec, anchor = _records(slept, ari={'lr0.01_g2': 0.5})
        r = g1s.reading(rec, anchor)
        self.assertNotEqual(r['chosen'], 'lr0.01_g2')   # an impostor fit is never chosen
        self.assertEqual(r['label'], 'NO_GAIN')
        rec, anchor = _records(slept, ari={t: 0.1 for t in slept})
        self.assertEqual(g1s.reading(rec, anchor)['label'], 'IDENTITY_LOST')
        rec, anchor = _records(slept, anchor_ok=False)
        self.assertEqual(g1s.reading(rec, anchor)['label'], 'ANCHOR_MISS')

    def test_g1t_anchor_reads_world_70(self):
        a = g1s.g1t_anchor()
        self.assertEqual(len(a['WAKE']), 64)
        self.assertEqual(len(a['SLEEP']), 64)

    def test_cells_order_wake_first(self):
        c = g1s.cells()
        self.assertEqual(len(c), 12)
        self.assertEqual({a for a, _ in c[:6]}, {'WAKE'})


if __name__ == '__main__':
    unittest.main()
