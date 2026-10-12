import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from row.experiments import b2_depth2 as bd
from row.experiments import b2d2_sizing as sz
from row.experiments import loop_stream as ls
from row.experiments import score_b2_depth2 as sc


class B2Depth2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.world, cls.st, cls.plan, cls.canonical, cls.ltasks, cls.meta = ls.stream(61, 0, 4, 2)
        cls.loop_ids = {t.task_id for t in cls.ltasks}
        cls.cand_ids = {t.task_id for t in cls.st if cls.plan[t.task_id] == 2}

    def test_labels(self):
        self.assertEqual(bd.label(6, 1.0, 1.0, 1.0, 9), 'FORMATION_BROKEN')
        self.assertEqual(bd.label(7, 0.9, 0.8, 0.95, 8), 'DISCOVERS')
        self.assertEqual(bd.label(9, 0.89, 1.0, 1.0, 9), 'NOT_DISCOVERED')
        self.assertEqual(bd.label(9, 1.0, 0.79, 1.0, 9), 'NOT_DISCOVERED')
        self.assertEqual(bd.label(9, 1.0, 1.0, 0.94, 9), 'NOT_DISCOVERED')
        self.assertEqual(bd.label(9, 1.0, 1.0, 1.0, 7), 'NOT_DISCOVERED')
        self.assertFalse(bd.passing(0.5, bd.LOOP_BAR))
        self.assertTrue(bd.passing(0.4999, bd.LOOP_BAR))

    def test_candidates_and_arms(self):
        self.assertEqual(len(self.cand_ids), 64 + 4)
        self.assertTrue(self.loop_ids <= self.cand_ids)
        o = sz.build('ONESTEP2', self.cfg, self.plan, self.cand_ids, self.loop_ids)
        g = sz.build('GIVEN2', self.cfg, self.plan, self.cand_ids, self.loop_ids)
        self.assertEqual(set(o.loop_plan), self.cand_ids)
        self.assertEqual(set(g.loop_plan), self.loop_ids)
        self.assertEqual((o.state_halt, o.step_bias, o.detach_halt_input, o.halt_decay, o.halt_init, o.body_depth),
                         (True, True, True, sz.HALT_DECAY, sz.HALT_INIT, 2))
        self.assertEqual((g.state_halt, g.step_bias, g.detach_halt_input, g.halt_decay, g.halt_init, g.body_depth),
                         (True, False, True, None, None, 2))
        t = next(iter(self.loop_ids))
        o.begin_task(t)
        torch.testing.assert_close(o.halt_b[t].detach(), torch.tensor([4.0] + [-4.0] * (bd.K - 1)))

    def test_install_sets_route_and_form(self):
        t = next(iter(self.loop_ids))
        m = sz.build('ONESTEP2', self.cfg, self.plan, self.cand_ids, self.loop_ids)
        m.begin_task(t)
        sz.install_count(m, t, (3, 7), 1)
        self.assertEqual(torch.argmax(m.task_codes[t].detach(), -1)[:2].tolist(), [3, 7])
        x = torch.randn(8, self.cfg.world.state_dim)
        m.eval()
        with torch.no_grad():
            coeff = torch.nn.functional.one_hot(torch.argmax(m.task_codes[t], -1), m.operator_slots).float()
            torch.testing.assert_close(m(x, t), m._route(x, coeff, 2))

    def test_independent_scorer_agrees_and_refuses(self):
        rng = np.random.default_rng(1)
        tmp = Path(tempfile.mkdtemp())
        plan = tmp / 'plan.md'
        plan.write_text('plan')
        (tmp / 'cells').mkdir()
        records = {}
        for a in bd.ARMS:
            for w in bd.WORLDS:
                for s in bd.STREAMS:
                    cands = [f't{i}' for i in range(bd.N_CAND)]
                    loops = cands[:bd.N_LOOP]
                    rec = {'arm': a, 'world': w, 'stream': s, 'loop_task_ids': loops,
                           'canonical_per_task': {f'c{i}': float(rng.random() * 0.04) for i in range(64)},
                           'loop_per_task': {t: float(rng.random() * 0.6) for t in loops},
                           'anchor2_per_task': {t: 0.01 for t in cands[bd.N_LOOP:]},
                           'wake_anchor2_rooted': 0}
                    for k in ('canonical', 'loop'):
                        rec[f'{k}_median'] = float(np.median(list(rec[f'{k}_per_task'].values())))
                    for k in ('anchor2_median', 'anchor1_median', 'wake_canonical_median', 'wake_loop_median',
                              'wake_anchor2_median', 'reroute_canonical_median', 'count_accuracy_median',
                              'wake_count_accuracy_median'):
                        rec[k] = 0.5
                    if a == 'ONESTEP2':
                        rec['decisions'] = {t: {'loop': bool(rng.random() < (0.95 if t in loops else 0.02))}
                                            for t in cands}
                        rec['loops_kept'] = sum(rec['decisions'][t]['loop'] for t in loops)
                        rec['straight_kept_count'] = sum(not v['loop'] for t, v in rec['decisions'].items()
                                                         if t not in loops)
                        rec['wake_state_norms'] = {t: float(rng.random() + (1.5 if t in loops else 0)) for t in cands}
                        rec['wake_state_auc'] = sz.bh.auc([rec['wake_state_norms'][t] for t in loops],
                                                          [v for t, v in rec['wake_state_norms'].items() if t not in loops])
                        rec['wake_state_norm_median'] = {'loop': 1.0, 'straight': 0.1}
                    key = f'{a}_w{w}_s{s}'
                    records[key] = rec
                    (tmp / 'cells' / f'{key}.json').write_text(json.dumps(
                        {'stamp': {'protocol_sha256': 'x'}, 'complete': True, 'record': rec}))
        summary = bd.summarize(records)
        report = {'protocol_sha256': 'x', 'cells': records, 'summary': summary,
                  'protocol': {'input_sha256': {plan.as_posix(): hashlib.sha256(b'plan').hexdigest()}}}
        (tmp / 'report.json').write_text(json.dumps(report))
        out = sc.score(tmp, tmp / 'report.json', plan)
        self.assertTrue(out['agrees_with_runner'])
        self.assertEqual(out['label'], summary['label'])
        report['summary'] = dict(summary, n_terminal=summary['n_terminal'] - 1)
        (tmp / 'report.json').write_text(json.dumps(report))
        self.assertFalse(sc.score(tmp, tmp / 'report.json', plan)['agrees_with_runner'])
        plan.write_text('changed')
        with self.assertRaises(ValueError):
            sc.score(tmp, tmp / 'report.json', plan)


if __name__ == '__main__':
    unittest.main()
