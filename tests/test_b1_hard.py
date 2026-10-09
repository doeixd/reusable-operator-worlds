import unittest

import numpy as np
import torch

from row.experiments import b1_hard as bh
from row.experiments import b1_online as b1o
from row.experiments import branch_stream as bs
from row.models.branch_gated import BranchGatedLearner


class B1HardTests(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(bh.auc([2, 3], [0, 1]), 1.0)
        self.assertEqual(bh.auc([0, 1], [2, 3]), 0.0)
        self.assertEqual(bh.auc([1, 1], [1, 1]), 0.5)

    def test_labels(self):
        self.assertEqual(bh.label(6, 1.0, 1.0, 1.0, 9), 'FORMATION_BROKEN')
        self.assertEqual(bh.label(7, 0.9, 0.8, 0.95, 8), 'DISCOVERS')
        self.assertEqual(bh.label(9, 0.89, 1.0, 1.0, 9), 'NOT_DISCOVERED')
        self.assertEqual(bh.label(9, 1.0, 0.79, 1.0, 9), 'NOT_DISCOVERED')
        self.assertEqual(bh.label(9, 1.0, 1.0, 0.94, 9), 'NOT_DISCOVERED')
        self.assertEqual(bh.label(9, 1.0, 1.0, 1.0, 7), 'NOT_DISCOVERED')

    def test_learner_options(self):
        cfg, world, st, plan, canonical, btasks, meta = bs.stream(44, 0, 4)
        allplan = {t.task_id: 'input' for t in st}
        const = b1o.build(BranchGatedLearner, cfg, plan, branch_plan=allplan, branch_seed=1, state_gate=False)
        params = const.begin_task(st[0].task_id)
        self.assertEqual(len(params), 3)
        self.assertNotIn(st[0].task_id, const.gate_w)
        x = torch.randn(5, cfg.world.state_dim)
        g = const.gate(x, st[0].task_id)
        self.assertTrue(torch.all(g == g[0]))   # state-independent
        decay = b1o.build(BranchGatedLearner, cfg, plan, branch_plan=allplan, branch_seed=1, gate_decay=0.1)
        params = decay.begin_task(st[0].task_id)
        self.assertIsInstance(params[-1], dict)
        self.assertEqual(params[-1]['weight_decay'], 0.1)
        self.assertIs(params[-1]['params'][0], decay.gate_w[st[0].task_id])
        default = b1o.build(BranchGatedLearner, cfg, plan, branch_plan=allplan, branch_seed=1)
        self.assertEqual(len(default.begin_task(st[0].task_id)), 4)   # B1-online's construction

    def test_independent_scorer_agrees_and_refuses(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from row.experiments import score_b1_hard as sc
        rng = np.random.default_rng(0)
        tmp = Path(tempfile.mkdtemp())
        plan = tmp / 'plan.md'
        plan.write_text('plan')
        (tmp / 'cells').mkdir()
        records = {}
        for a in bh.ARMS:
            for w in bh.WORLDS:
                for s in bh.STREAMS:
                    ids = [f't{i}' for i in range(236)]
                    branch = ids[:48]
                    rec = {'arm': a, 'world': w, 'stream': s, 'branch_task_ids': branch,
                           'decisions': {t: bool(a == 'ALLGATE' and rng.random() < (0.9 if t in branch else 0.03))
                                         for t in ids},
                           'canonical_per_task': {f'c{i}': float(rng.random() * 0.06) for i in range(64)},
                           'branch_per_task': {t: float(rng.random()) for t in branch}}
                    rec['wake_gate_norms'] = {t: float(rng.random() + (1.5 if t in branch else 0)) for t in ids}
                    rec['wake_gate_auc'] = bh.auc([rec['wake_gate_norms'][t] for t in branch],
                                                  [v for t, v in rec['wake_gate_norms'].items() if t not in branch])
                    rec['canonical_median'] = float(np.median(list(rec['canonical_per_task'].values())))
                    rec['branch_median'] = float(np.median(list(rec['branch_per_task'].values())))
                    rec['branch_recall'] = sum(rec['decisions'][t] for t in branch)
                    rec['straight_kept_single'] = sum(not v for t, v in rec['decisions'].items() if t not in branch)
                    key = f'{a}_w{w}_s{s}'
                    records[key] = rec
                    (tmp / 'cells' / f'{key}.json').write_text(json.dumps(
                        {'stamp': {'protocol_sha256': 'x'}, 'complete': True, 'record': rec}))
        report = {'protocol_sha256': 'x', 'cells': records, 'summary': bh.summarize(records, b1o_cells=tmp / 'none'),
                  'protocol': {'input_sha256': {plan.as_posix(): hashlib.sha256(b'plan').hexdigest()}}}
        (tmp / 'report.json').write_text(json.dumps(report))
        out = sc.score(tmp, tmp / 'report.json', plan)
        self.assertTrue(out['agrees_with_runner'])
        self.assertEqual(out['label'], report['summary']['label'])
        plan.write_text('changed')
        with self.assertRaises(ValueError):
            sc.score(tmp, tmp / 'report.json', plan)


if __name__ == '__main__':
    unittest.main()
