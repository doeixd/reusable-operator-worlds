import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from row.experiments import g0_generator_census as g0
from row.experiments import score_g0_generator_census as sc


def rec(key, a, g, k, m, w, s1=0.98, s2=1.6, ari1=0.6, ari3=0.0, var=1.0):
    return {'candidate': key, 'alpha': a, 'gain': g, 'K': k, 'shared': m, 'world': w,
            's1': {'1': {'accuracy': [s1] * 24, 'minority_fraction': [0.5] * 24},
                   '3': {'accuracy': [s1] * 24, 'minority_fraction': [0.5] * 24}},
            's2': [{'runner_up_nmse': s2, 'second_wrong_nmse': s2, 'wrong_below_0.05': 0}] * 32,
            's3': {'ari_depth1': ari1, 'ari_depth3_mean': ari3, 'slots_used_depth1': 12},
            's4': {'output_variance_median': var, 'output_variance_max_over_min': 1.2, 'norm_ratio_median': 1.0,
                   'norm_ratio_max': 1.05, 'finite': True}, 'seconds': 1.0}


class G0Tests(unittest.TestCase):
    def test_grid(self):
        c = g0.candidates()
        self.assertEqual(len(c), 39)
        self.assertIn(g0.BASE, c)
        self.assertEqual(g0.key_of(g0.BASE), sc.BASE_KEY)
        self.assertEqual(g0.WORLDS, sc.WORLDS)

    def test_judge(self):
        base = sc.stats([rec('b', .35, 1, 6, 0, w) for w in sc.WORLDS])
        self.assertFalse(sc.judge(base)['passes'])
        self.assertTrue(sc.judge(dict(base, s1=0.90))['passes'])
        self.assertFalse(sc.judge(dict(base, s1=0.9001))['passes'])
        self.assertTrue(sc.judge(dict(base, s2=0.5))['s2_in_band'])
        self.assertFalse(sc.judge(dict(base, s2=0.009))['s2_in_band'])
        self.assertFalse(sc.judge(dict(base, s1=0.8, ari1=[0.6, 0.29, 0.6]))['passes'])
        self.assertFalse(sc.judge(dict(base, s1=0.8, ari1=[0.6, 0.31, 0.6], ari3=[0, 0.4, 0]))['passes'])
        self.assertFalse(sc.judge(dict(base, s1=0.8, var_ratio=4.1))['passes'])
        self.assertEqual(sc.judge(dict(base, s1=0.8, s2=0.3))['moves'], 2)

    def _report(self, records, baseline_records):
        tmp = Path(tempfile.mkdtemp())
        plan = tmp / 'plan.md'
        plan.write_text('plan')
        (tmp / 'base.json').write_text(json.dumps({'records': baseline_records}))
        rep = {'protocol': {'input_sha256': {plan.as_posix(): hashlib.sha256(b'plan').hexdigest()}},
               'records': records, 'summary': g0.summarize(records)}
        (tmp / 'rep.json').write_text(json.dumps(rep))
        return tmp, plan

    def test_selection_and_gate(self):
        base = [rec(sc.BASE_KEY, .35, 1, 6, 0, w) for w in sc.WORLDS]
        k12 = [rec('a0.35_g1.0_K12_m0.0', .35, 1, 12, 0, w, s1=0.8, s2=0.3) for w in sc.WORLDS]
        a1 = [rec('a1.0_g1.0_K6_m0.0', 1.0, 1, 6, 0, w, s1=0.85) for w in sc.WORLDS]
        m4 = [rec('a0.35_g1.0_K6_m4.0', .35, 1, 6, 4, w, s1=0.86, s2=0.3) for w in sc.WORLDS]
        tmp, plan = self._report(base + k12 + a1 + m4, base)
        out = sc.score(tmp / 'rep.json', tmp / 'base.json', plan)
        self.assertEqual((out['label'], out['chosen']), ('SELECTED', 'a0.35_g1.0_K6_m4.0'))
        self.assertEqual(out['best_s1_mover'], 'a1.0_g1.0_K6_m0.0')
        self.assertTrue(out['summary_agrees'])
        changed = copy.deepcopy(base)
        changed[0]['s2'][0] = dict(changed[0]['s2'][0], runner_up_nmse=1.5)
        tmp, plan = self._report(changed + k12, base)
        self.assertEqual(sc.score(tmp / 'rep.json', tmp / 'base.json', plan)['label'], 'UNINTERPRETABLE')
        tmp, plan = self._report(base, base)
        self.assertEqual(sc.score(tmp / 'rep.json', tmp / 'base.json', plan)['label'], 'NO_CANDIDATE')
        plan.write_text('changed')
        with self.assertRaises(ValueError):
            sc.score(tmp / 'rep.json', tmp / 'base.json', plan)


if __name__ == '__main__':
    unittest.main()
