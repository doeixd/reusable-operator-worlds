import unittest

import numpy as np
import torch

from row.experiments import m3_fork_cycle as m
from row.experiments import score_m3_fork_cycle as sc


def synthetic_report(seed, scenario):
    recs = m._synthetic(np.random.default_rng(seed), 9, *m.DISCRIMINATION_SCENARIOS[scenario])
    cells = {}
    for (_, rec), key in zip(sorted(recs.items()), sorted(sc.CELLS)):
        for b in rec['branches'].values():
            b['deep'] = {f't{j}': v for j, v in b['deep'].items()}
            b['shallow'] = {f's{j}': b['shallow'][0] for j in range(m.ms.N_SHALLOW)}
        cells[key] = rec
    proto = {'solved': m.SOLVED, 'formed': m.FORMED,
             'need_streams': [m.NEED_FORMED, m.NEED_INC, m.NEED_SOLVE, m.HARM_MAX], 'scale': 1}
    return {'complete': True, 'cells': cells, 'protocol': proto, 'protocol_sha256': 'x', 'summary': m.summarize(cells)}


class SummarizeTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(synthetic_report(0, 'effect_sizing')['summary']['label'], 'REPAIRED_CYCLE_SOLVES')
        self.assertEqual(synthetic_report(0, 'null_incumbent_also')['summary']['label'], 'NO_NECESSITY')
        rng = np.random.default_rng(0)
        self.assertEqual(m.summarize(m._synthetic(rng, 9, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0))['label'], 'FORMATION_BROKEN')
        # FCYC fails in 3 streams where CYC worked: harm 3 >= 2
        self.assertEqual(m.summarize(m._synthetic(rng, 9, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0))['label'], 'REPAIR_HARMS')

    def test_harm_and_rescue_counts(self):
        recs = m._synthetic(np.random.default_rng(1), 9, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        s = m.summarize(recs)
        self.assertEqual((s['rescue'], s['harm']), (9, 0))

    def test_discrimination_runs(self):
        out = m.discrimination(draws=30)
        self.assertEqual(out['effect_sizing']['REPAIRED_CYCLE_SOLVES'], 1.0)


class ForkTest(unittest.TestCase):
    def test_op_apply_matches_frozen_library(self):
        from row.experiments.explore_m3_slot_census import load_wake
        from row.experiments.audit_so1r_route_only import FrozenLibrary
        try:
            _, model, *_ = load_wake(84, 1)
        except FileNotFoundError:
            self.skipTest('M2 wake artifacts absent')
        lib = FrozenLibrary(model)
        z = torch.randn(7, 16, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            for k in range(lib.slots):
                self.assertLess(float((m.op_apply(model.library[k], z) - m.m2.slot_apply(lib, z, k)).abs().max()),
                                1e-5)


    def test_fit_slot_on_frozen_parameters(self):
        # the wake terminal arrives with shared parameters frozen by the novel-composition probe (dry run 2026-10-29)
        from row.models.learned_models import RotatedLearnedOperator  # noqa: F401  (import path check)
        from row.experiments.explore_m3_slot_census import load_wake
        try:
            _, model, *_ = load_wake(84, 1)
        except FileNotFoundError:
            self.skipTest('M2 wake artifacts absent')
        op = model.library[0]
        op.requires_grad_(False)
        before = op.V.detach().clone()
        steps, m.FIT_STEPS = m.FIT_STEPS, 3
        try:
            m.fit_slot(op, [torch.randn(8, 16)], [torch.randn(8, 16)])
        finally:
            m.FIT_STEPS = steps
        self.assertFalse(torch.equal(before, op.V.detach()))


class ScorerTest(unittest.TestCase):
    def test_agrees_with_runner(self):
        for seed, scen in enumerate(m.DISCRIMINATION_SCENARIOS):
            self.assertTrue(sc.score(synthetic_report(seed, scen))['agrees_with_runner'], scen)

    def test_refuses(self):
        rep = synthetic_report(0, 'effect_sizing')
        rep['cells'].pop('w87_s0')
        with self.assertRaises(ValueError):
            sc.score(rep)
        rep = synthetic_report(0, 'effect_sizing')
        rep['protocol']['need_streams'] = [7, 7, 7, 2]
        with self.assertRaises(ValueError):
            sc.score(rep)


if __name__ == '__main__':
    unittest.main()
