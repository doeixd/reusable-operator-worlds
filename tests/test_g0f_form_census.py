import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from row.experiments import g0f_form_census as g0f
from row.experiments import score_g0f_form_census as sc
from row.form_world import FormPrimitive, form_library, tanh_gain_scale


def rec(key, form, knob, w, s1=0.98, s2=1.6, ari1=0.6, ari3=0.0, var=1.0, s6=0.001):
    return {'candidate': key, 'form': form, 'knob': knob, 'K': 6, 'world': w,
            's1': {'1': {'accuracy': [s1] * 24, 'minority_fraction': [0.5] * 24},
                   '3': {'accuracy': [s1] * 24, 'minority_fraction': [0.5] * 24}},
            's2': [{'runner_up_nmse': s2, 'second_wrong_nmse': s2, 'wrong_below_0.05': 0}] * 32,
            's3': {'ari_depth1': ari1, 'ari_depth3_mean': ari3, 'slots_used_depth1': 12},
            's4': {'output_variance_median': var, 'output_variance_max_over_min': 1.2, 'norm_ratio_median': 1.0,
                   'norm_ratio_max': 1.05, 'finite': True},
            's6': [{'primitive': k, 'heldout_nmse': s6, 'fit_nmse': s6} for k in range(6)], 'seconds': 1.0}


class FormTests(unittest.TestCase):
    def test_forms(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4000, 16))
        fold = FormPrimitive.random('fold', 4, 70, 0, 16)
        self.assertTrue(np.allclose(np.linalg.norm(fold(x), axis=1), np.linalg.norm(x, axis=1)))
        self.assertFalse(np.allclose(fold(x), x @ fold.Q.T))           # it actually folds
        t = FormPrimitive.random('tanh', 2.0, 70, 0, 16)
        self.assertAlmostEqual(float(np.var(t(x))), 1.0, delta=0.1)
        self.assertAlmostEqual(tanh_gain_scale(1e-4) * 1e-4, 1.0, places=4)
        a, b = form_library('gate', 2.0, 70, 6, 16), form_library('gate', 2.0, 70, 6, 16)
        self.assertTrue(all(np.array_equal(p.A, q.A) and np.array_equal(p.Q, q.Q) for p, q in zip(a, b)))
        self.assertFalse(np.array_equal(form_library('gate', 2.0, 71, 1, 16)[0].A, a[0].A))
        with self.assertRaises(ValueError):
            FormPrimitive.random('fold', 1.5, 70, 0, 16)

    def test_learner_reproduces_teacher_at_init(self):
        import torch
        for form, knob in (('tanh', 2.0), ('gate', 4.0), ('fold', 2.0)):
            p = FormPrimitive.random(form, knob, 70, 0, 16)
            x = np.random.default_rng(1).normal(size=(64, 16))
            with torch.no_grad():
                got = g0f.FormOperator(p)(torch.tensor(x)).numpy()
            self.assertTrue(np.allclose(got, p(x), atol=1e-10), form)
        lib = g0f.untrained(g0f.BASE, 16)
        x = np.random.default_rng(2).normal(size=(64, 16))
        with torch.no_grad():
            self.assertTrue(np.allclose(g0f.FormOperator(lib[0])(torch.tensor(x)).numpy(), lib[0](x), atol=1e-10))

    def test_grid(self):
        self.assertEqual(len(g0f.CANDIDATES), sc.CANDIDATES)
        self.assertEqual(g0f.key_of(g0f.BASE), sc.ANCHOR)
        self.assertEqual(g0f.WORLDS, sc.WORLDS)


class RuleTests(unittest.TestCase):
    def test_judge(self):
        base = sc.stats([rec('b', 'rot', .35, w) for w in sc.WORLDS])
        self.assertFalse(sc.judge(base)['passes'])
        self.assertTrue(sc.judge(dict(base, s1=0.90))['passes'])
        self.assertFalse(sc.judge(dict(base, s1=0.9001))['passes'])
        self.assertFalse(sc.judge(dict(base, s1=0.8, ari1=[0.6, 0.29, 0.6]))['passes'])
        self.assertFalse(sc.judge(dict(base, s1=0.8, ari1=[0.6, 0.31, 0.6], ari3=[0, 0.4, 0]))['passes'])
        self.assertFalse(sc.judge(dict(base, s1=0.8, var_ratio=4.1))['passes'])
        self.assertFalse(sc.judge(dict(base, s1=0.8, s6=0.0101))['passes'])
        self.assertTrue(sc.judge(dict(base, s1=0.8, s6=0.01))['passes'])

    def _setup(self, records, anchor):
        tmp = Path(tempfile.mkdtemp())
        plan = tmp / 'plan.md'
        plan.write_text('plan')
        (tmp / 'g0.json').write_text(json.dumps({'records': anchor}))
        (tmp / 'own.json').write_text(json.dumps({'records': anchor}))
        rep = {'protocol': {'input_sha256': {plan.as_posix(): hashlib.sha256(b'plan').hexdigest()}}, 'records': records}
        (tmp / 'rep.json').write_text(json.dumps(rep))
        return tmp, plan

    def _all(self, overrides):
        recs = []
        for form, knob in g0f.CANDIDATES:
            key = g0f.key_of((form, knob))
            recs += [rec(key, form, knob, w, **overrides.get(key, {})) for w in sc.WORLDS]
        return recs

    def test_selection_gate_refusal(self):
        recs = self._all({'fold2': {'s1': 0.8, 'ari1': 0.5}, 'tanh4': {'s1': 0.7, 'ari1': 0.4},
                          'gate4': {'s1': 0.6, 'ari1': 0.9, 's6': 0.5}})
        anchor = [r for r in recs if r['candidate'] == sc.ANCHOR]
        tmp, plan = self._setup(recs, anchor)
        kw = dict(plan=plan, g0_baseline=tmp / 'g0.json', baseline=tmp / 'own.json')
        out = sc.score(tmp / 'rep.json', **kw)
        self.assertEqual((out['label'], out['chosen'], out['passing']), ('SELECTED', 'fold2', ['fold2', 'tanh4']))
        none = self._all({})
        tmp2, plan2 = self._setup(none, [r for r in none if r['candidate'] == sc.ANCHOR])
        self.assertEqual(sc.score(tmp2 / 'rep.json', plan=plan2, g0_baseline=tmp2 / 'g0.json',
                                  baseline=tmp2 / 'own.json')['label'], 'NO_FORM')
        moved = copy.deepcopy(anchor)
        moved[0]['s2'][0] = dict(moved[0]['s2'][0], runner_up_nmse=1.7)
        (tmp / 'g0.json').write_text(json.dumps({'records': moved}))
        self.assertEqual(sc.score(tmp / 'rep.json', **kw)['label'], 'UNINTERPRETABLE')
        plan.write_text('edited')
        with self.assertRaises(ValueError):
            sc.score(tmp / 'rep.json', **kw)


if __name__ == '__main__':
    unittest.main()
