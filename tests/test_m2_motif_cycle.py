import unittest

import numpy as np

from row.experiments import m2_motif_cycle as m
from row.experiments import score_m2_motif_cycle as sc


def synthetic_report(seed, scenario):
    rng = np.random.default_rng(seed)
    recs = m._synthetic(rng, 9, *m.DISCRIMINATION_SCENARIOS[scenario])
    cells = {}
    for (i, rec), key in zip(sorted(recs.items()), sorted(sc.CELLS)):
        for b in rec['branches'].values():
            b['deep'] = {f't{j}': v for j, v in b['deep'].items()}
            b['shallow'] = {f's{j}': list(b['shallow'].values())[0] for j in range(m.ms.N_SHALLOW)}
        cells[key] = rec
    proto = {'solved': m.SOLVED, 'formed': m.FORMED, 'need_streams': [m.NEED_FORMED, m.NEED_INC, m.NEED_CYC],
             'scale': 1}
    return {'complete': True, 'cells': cells, 'protocol': proto, 'protocol_sha256': 'x',
            'summary': m.summarize(cells)}


class SummarizeTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(synthetic_report(0, 'effect_sizing')['summary']['label'], 'CYCLE_SOLVES')
        self.assertEqual(synthetic_report(0, 'null_incumbent_also')['summary']['label'], 'NO_NECESSITY')
        self.assertEqual(synthetic_report(0, 'null_weak_tasks')['summary']['label'], 'PARTIAL')
        self.assertEqual(m.summarize(m._synthetic(np.random.default_rng(0), 9, 0.0, 1.0, 1.0, 0.0, 0.0))['label'],
                         'FORMATION_BROKEN')

    def test_solved_bar_is_strict(self):
        rec = {'branches': {'CYC': {'deep': {0: m.SOLVED, 1: m.SOLVED - 1e-9}}}}
        self.assertEqual(m.solved_fraction(rec, 'CYC'), 0.5)

    def test_eight_of_nine(self):
        rng = np.random.default_rng(1)
        recs = m._synthetic(rng, 9, 1.0, 1.0, 1.0, 0.0, 0.0)
        for j in range(m.N_DEEP):
            recs[0]['branches']['CYC']['deep'][j] = 1.5
        self.assertEqual(m.summarize(recs)['label'], 'CYCLE_SOLVES')
        for j in range(m.N_DEEP):
            recs[1]['branches']['CYC']['deep'][j] = 1.5
        self.assertEqual(m.summarize(recs)['label'], 'PARTIAL')

    def test_discrimination_runs(self):
        out = m.discrimination(draws=50)
        self.assertEqual(out['effect_sizing']['CYCLE_SOLVES'], 1.0)
        self.assertEqual(out['null_weak_tasks']['CYCLE_SOLVES'], 0.0)


class ScorerTest(unittest.TestCase):
    def test_agrees_with_runner(self):
        for seed, scen in enumerate(m.DISCRIMINATION_SCENARIOS):
            rep = synthetic_report(seed, scen)
            res = sc.score(rep)
            self.assertTrue(res['agrees_with_runner'], scen)

    def test_refuses_incomplete_and_wrong_constants(self):
        rep = synthetic_report(0, 'effect_sizing')
        rep['cells'].pop('w84_s0')
        with self.assertRaises(ValueError):
            sc.score(rep)
        rep = synthetic_report(0, 'effect_sizing')
        rep['protocol']['solved'] = 0.1
        with self.assertRaises(ValueError):
            sc.score(rep)

    def test_refuses_short_branch(self):
        rep = synthetic_report(0, 'effect_sizing')
        rep['cells']['w84_s0']['branches']['CYC']['deep'].pop('t0')
        with self.assertRaises(ValueError):
            sc.score(rep)


if __name__ == '__main__':
    unittest.main()
