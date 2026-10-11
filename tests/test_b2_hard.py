import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from row.experiments import b2_hard as bh
from row.experiments import census_b2h_data_test as cb
from row.experiments import loop_stream as ls
from row.experiments import score_b2_hard as sc
from row.experiments.audit_so1r_route_only import FrozenLibrary


class B2HardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.world, cls.st, cls.plan, cls.canonical, cls.ltasks, cls.meta = ls.stream(53, 0, 4)
        cls.loop_ids = {t.task_id for t in cls.ltasks}
        cls.cand_ids = {t.task_id for t in cls.st if cls.plan[t.task_id] == 1}

    def test_labels(self):
        self.assertEqual(bh.label(6, 1.0, 1.0, 1.0, 9), 'FORMATION_BROKEN')
        self.assertEqual(bh.label(7, 0.9, 0.8, 0.95, 8), 'DISCOVERS')
        self.assertEqual(bh.label(9, 0.89, 1.0, 1.0, 9), 'NOT_DISCOVERED')
        self.assertEqual(bh.label(9, 1.0, 0.79, 1.0, 9), 'NOT_DISCOVERED')
        self.assertEqual(bh.label(9, 1.0, 1.0, 0.94, 9), 'NOT_DISCOVERED')
        self.assertEqual(bh.label(9, 1.0, 1.0, 1.0, 7), 'NOT_DISCOVERED')

    def test_arm_constructions(self):
        a = bh.build('ALLLOOP', self.cfg, self.plan, self.cand_ids, self.loop_ids)
        c = bh.build('ALLCOUNT', self.cfg, self.plan, self.cand_ids, self.loop_ids)
        g = bh.build('GIVEN', self.cfg, self.plan, self.cand_ids, self.loop_ids)
        self.assertEqual(set(a.loop_plan), self.cand_ids)
        self.assertEqual(set(c.loop_plan), self.cand_ids)
        self.assertEqual(set(g.loop_plan), self.loop_ids)
        self.assertEqual(len(self.cand_ids), 60 + 4)
        self.assertEqual((a.state_halt, a.step_bias, a.detach_halt_input, a.halt_decay), (True, True, True, bh.HALT_DECAY))
        self.assertEqual((c.state_halt, c.step_bias), (False, False))
        self.assertEqual((g.state_halt, g.step_bias, g.detach_halt_input, g.halt_decay), (True, False, True, None))

    def test_fit_forms_matches_census_and_installs(self):
        m = bh.build('ALLLOOP', self.cfg, self.plan, self.cand_ids, self.loop_ids)
        for t in self.st:
            m.begin_task(t.task_id)
        lib = FrozenLibrary(m)
        for t in (self.ltasks[0], next(x for x in self.st if x.task_id in self.cand_ids - self.loop_ids)):
            X, Y = np.asarray(t.train_x[:64], dtype=np.float64), np.asarray(t.train_y[:64], dtype=np.float64)
            bc, bl = bh.fit_forms(lib, X, Y, 7)
            cc, cl = cb.forms(lib, X, Y, X, Y, 7)
            self.assertEqual((bc[0], bc[1], bc[2]), (cc[0], cc[2], cc[3]))
            self.assertEqual((bl[0], bl[1]), (cl[0], cl[2]))
            # installed count form reproduces its support loss in evaluation
            bh.install_count(m, t.task_id, bc[1], bc[2])
            m.eval()
            with torch.no_grad():
                pred = m(torch.tensor(X, dtype=torch.float32), t.task_id).numpy()
            self.assertAlmostEqual(float(np.mean((pred - Y) ** 2)), bc[0], places=4)
            bh.install_loop(m, t.task_id, bl[1], bl[2])
            with torch.no_grad():
                pred = m(torch.tensor(X, dtype=torch.float32), t.task_id).numpy()
            self.assertAlmostEqual(float(np.mean((pred - Y) ** 2)), bl[0], places=3)

    def test_independent_scorer_agrees_and_refuses(self):
        rng = np.random.default_rng(1)
        tmp = Path(tempfile.mkdtemp())
        plan = tmp / 'plan.md'
        plan.write_text('plan')
        (tmp / 'cells').mkdir()
        records = {}
        for a in bh.ARMS:
            for w in bh.WORLDS:
                for s in bh.STREAMS:
                    cands = [f't{i}' for i in range(bh.N_CAND)]
                    loops = cands[:bh.N_LOOP]
                    rec = {'arm': a, 'world': w, 'stream': s, 'loop_task_ids': loops,
                           'canonical_per_task': {f'c{i}': float(rng.random() * 0.04) for i in range(64)},
                           'loop_per_task': {t: float(rng.random() * (0.5 if a != 'ALLCOUNT' else 2)) for t in loops}}
                    for k in ('canonical', 'loop'):
                        rec[f'{k}_median'] = float(np.median(list(rec[f'{k}_per_task'].values())))
                    for k in ('anchor_median', 'wake_canonical_median', 'wake_loop_median', 'wake_anchor_median',
                              'count_accuracy_median', 'wake_count_accuracy_median'):
                        rec[k] = 0.5
                    if a != 'GIVEN':
                        rec['decisions'] = {t: {'loop': bool(a == 'ALLLOOP' and rng.random() < (0.95 if t in loops else 0.02))}
                                            for t in cands}
                        rec['loops_kept'] = sum(rec['decisions'][t]['loop'] for t in loops)
                        rec['straight_kept_count'] = sum(not v['loop'] for t, v in rec['decisions'].items() if t not in loops)
                    if a == 'ALLLOOP':
                        rec['wake_state_norms'] = {t: float(rng.random() + (1.5 if t in loops else 0)) for t in cands}
                        rec['wake_state_auc'] = bh.auc([rec['wake_state_norms'][t] for t in loops],
                                                       [v for t, v in rec['wake_state_norms'].items() if t not in loops])
                        rec['wake_state_norm_median'] = {'loop': 1.0, 'straight': 0.1}
                    key = f'{a}_w{w}_s{s}'
                    records[key] = rec
                    (tmp / 'cells' / f'{key}.json').write_text(json.dumps(
                        {'stamp': {'protocol_sha256': 'x'}, 'complete': True, 'record': rec}))
        summary = bh.summarize(records)
        self.assertEqual(summary['label'], 'DISCOVERS')
        report = {'protocol_sha256': 'x', 'cells': records, 'summary': summary,
                  'protocol': {'input_sha256': {plan.as_posix(): hashlib.sha256(b'plan').hexdigest()}}}
        (tmp / 'report.json').write_text(json.dumps(report))
        out = sc.score(tmp, tmp / 'report.json', plan)
        self.assertTrue(out['agrees_with_runner'])
        plan.write_text('changed')
        with self.assertRaises(ValueError):
            sc.score(tmp, tmp / 'report.json', plan)


if __name__ == '__main__':
    unittest.main()
