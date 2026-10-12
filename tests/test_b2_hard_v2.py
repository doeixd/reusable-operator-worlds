import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from row.experiments import b2_hard_v2 as bv
from row.experiments import loop_stream as ls
from row.experiments import score_b2_hard_v2 as sc


class B2HardV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.world, cls.st, cls.plan, cls.canonical, cls.ltasks, cls.meta = ls.stream(57, 0, 4)
        cls.loop_ids = {t.task_id for t in cls.ltasks}
        cls.cand_ids = {t.task_id for t in cls.st if cls.plan[t.task_id] == 1}

    def test_labels(self):
        self.assertEqual(bv.label(6, 1.0, 1.0, 1.0, 9), 'FORMATION_BROKEN')
        self.assertEqual(bv.label(7, 0.9, 0.8, 0.95, 8), 'DISCOVERS')
        self.assertEqual(bv.label(9, 0.89, 1.0, 1.0, 9), 'NOT_DISCOVERED')
        self.assertEqual(bv.label(9, 1.0, 0.79, 1.0, 9), 'NOT_DISCOVERED')
        self.assertEqual(bv.label(9, 1.0, 1.0, 0.94, 9), 'NOT_DISCOVERED')
        self.assertEqual(bv.label(9, 1.0, 1.0, 1.0, 7), 'NOT_DISCOVERED')
        self.assertEqual(bv.roots_label(0, 1), 'NO_ROOTS_IN_CONTROL')
        self.assertEqual(bv.roots_label(0, 2), 'ROOTS_PREVENTED')
        self.assertEqual(bv.roots_label(1, 5), 'ROOTS_NOT_PREVENTED')
        self.assertEqual(bv.rooted({'0': [1, 1], '1': [2, 3], '2': [2, 1], '3': []}), [1])

    def test_arm_constructions(self):
        o = bv.build('ONESTEP', self.cfg, self.plan, self.cand_ids, self.loop_ids)
        z = bv.build('ZERO', self.cfg, self.plan, self.cand_ids, self.loop_ids)
        g = bv.build('GIVEN', self.cfg, self.plan, self.cand_ids, self.loop_ids)
        self.assertEqual(set(o.loop_plan), self.cand_ids)
        self.assertEqual(set(z.loop_plan), self.cand_ids)
        self.assertEqual(set(g.loop_plan), self.loop_ids)
        for m in (o, z):
            self.assertEqual((m.state_halt, m.step_bias, m.detach_halt_input, m.halt_decay),
                             (True, True, True, bv.HALT_DECAY))
        self.assertEqual((o.halt_init, z.halt_init, g.halt_init), (bv.HALT_INIT, None, None))
        self.assertEqual((g.state_halt, g.step_bias, g.detach_halt_input, g.halt_decay), (True, False, True, None))
        t = next(iter(self.cand_ids))
        o.begin_task(t)
        z.begin_task(t)
        torch.testing.assert_close(o.halt_b[t].detach(), torch.tensor([4.0] + [-4.0] * (bv.K - 1)))
        self.assertEqual(float(z.halt_b[t].detach().abs().sum()), 0.0)

    def test_independent_scorer_agrees_and_refuses(self):
        rng = np.random.default_rng(1)
        tmp = Path(tempfile.mkdtemp())
        plan = tmp / 'plan.md'
        plan.write_text('plan')
        (tmp / 'cells').mkdir()
        records = {}
        for a in bv.ARMS:
            for w in bv.WORLDS:
                for s in bv.STREAMS:
                    cands = [f't{i}' for i in range(bv.N_CAND)]
                    loops = cands[:bv.N_LOOP]
                    root = a == 'ZERO' and s == 0
                    rec = {'arm': a, 'world': w, 'stream': s, 'loop_task_ids': loops,
                           'canonical_per_task': {f'c{i}': float(rng.random() * 0.04) for i in range(64)},
                           'loop_per_task': {t: float(rng.random() * 0.5) for t in loops},
                           'wake_anchor_best_L_by_op': {str(op): ([2] * 10 if root and op == 0 else [1] * 10)
                                                        for op in range(6)}}
                    rec['wake_rooted_ops'] = bv.rooted(rec['wake_anchor_best_L_by_op'])
                    for k in ('canonical', 'loop'):
                        rec[f'{k}_median'] = float(np.median(list(rec[f'{k}_per_task'].values())))
                    for k in ('anchor_median', 'wake_canonical_median', 'wake_loop_median', 'wake_anchor_median',
                              'reroute_canonical_median', 'count_accuracy_median', 'wake_count_accuracy_median'):
                        rec[k] = 0.5
                    if a != 'GIVEN':
                        rec['decisions'] = {t: {'loop': bool(rng.random() < (0.95 if t in loops else 0.02))}
                                            for t in cands}
                        rec['loops_kept'] = sum(rec['decisions'][t]['loop'] for t in loops)
                        rec['straight_kept_count'] = sum(not v['loop'] for t, v in rec['decisions'].items()
                                                         if t not in loops)
                        rec['wake_state_norms'] = {t: float(rng.random() + (1.5 if t in loops else 0)) for t in cands}
                        rec['wake_state_auc'] = bv.bh.auc([rec['wake_state_norms'][t] for t in loops],
                                                          [v for t, v in rec['wake_state_norms'].items() if t not in loops])
                        rec['wake_state_norm_median'] = {'loop': 1.0, 'straight': 0.1}
                    key = f'{a}_w{w}_s{s}'
                    records[key] = rec
                    (tmp / 'cells' / f'{key}.json').write_text(json.dumps(
                        {'stamp': {'protocol_sha256': 'x'}, 'complete': True, 'record': rec}))
        summary = bv.summarize(records)
        self.assertEqual(summary['label'], 'DISCOVERS')
        self.assertEqual(summary['rooted_cells'], {'ONESTEP': 0, 'ZERO': 3, 'GIVEN': 0})
        self.assertEqual(summary['roots_label'], 'ROOTS_PREVENTED')
        report = {'protocol_sha256': 'x', 'cells': records, 'summary': summary,
                  'protocol': {'input_sha256': {plan.as_posix(): hashlib.sha256(b'plan').hexdigest()}}}
        (tmp / 'report.json').write_text(json.dumps(report))
        out = sc.score(tmp, tmp / 'report.json', plan)
        self.assertTrue(out['agrees_with_runner'])
        # a tampered summary disagrees
        report['summary'] = dict(summary, roots_label='ROOTS_NOT_PREVENTED')
        (tmp / 'report.json').write_text(json.dumps(report))
        self.assertFalse(sc.score(tmp, tmp / 'report.json', plan)['agrees_with_runner'])
        plan.write_text('changed')
        with self.assertRaises(ValueError):
            sc.score(tmp, tmp / 'report.json', plan)


if __name__ == '__main__':
    unittest.main()
