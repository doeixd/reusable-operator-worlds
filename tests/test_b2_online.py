import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from row.experiments import b1_online as b1o
from row.experiments import b2_online as b2
from row.experiments import loop_stream as ls
from row.experiments import score_b2_online as sc
from row.models.loop_gated import LoopLearner


class B2OnlineTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(b2.label(6, 9, 9), 'FORMATION_BROKEN')
        self.assertEqual(b2.label(7, 8, 8), 'LOOPS_ONLINE')
        self.assertEqual(b2.label(9, 7, 9), 'LOOPS_AFTER_SLEEP')
        self.assertEqual(b2.label(9, 9, 7), 'NO_LOOPS')

    def test_refit_recovers_a_planted_loop(self):
        """Targets made by the learner's OWN hard loop (body slot 3, a random halting hyperplane, or a fixed count of 2
        for the control): the support-only re-fit must install a loop that reproduces them (near-zero support error),
        and the control must install a state-independent count."""
        cfg, world, st, plan, canonical, ltasks, meta = ls.stream(44, 0, 2)
        t = ltasks[0]
        d = cfg.world.state_dim
        for arm, state_halt in (('LOOP', True), ('CONSTCOUNT', False)):
            m = b1o.build(LoopLearner, cfg, plan, loop_plan={t.task_id: 'state'}, state_halt=state_halt)
            m.begin_task(t.task_id)
            xs = torch.tensor(t.train_x[:64], dtype=torch.float32)
            b2.set_body(m, t.task_id, 3)
            with torch.no_grad():
                if state_halt:
                    m.halt_w[t.task_id].copy_(torch.randn(d, generator=torch.Generator().manual_seed(7)))
                else:
                    m.halt_b[t.task_id].copy_(torch.tensor([10.0, 10.0, -10.0, -10.0, -10.0, -10.0]))
            m.eval()
            with torch.no_grad():
                ys = m(xs, t.task_id).clone()
                m.task_codes[t.task_id].zero_()
                m.halt_b[t.task_id].zero_()
                if state_halt:
                    m.halt_w[t.task_id].zero_()
            pool = [(x, y, t.task_id) for x, y in zip(xs.numpy(), ys.numpy())]
            before = b2.hard_support(m, t.task_id, xs, ys)
            self.assertEqual(b2.refit_loops(m, [t], pool, 44, arm), 1)
            after = b2.hard_support(m, t.task_id, xs, ys)
            self.assertLess(after, 0.05 * before)
            self.assertEqual(int(torch.argmax(m.task_codes[t.task_id][0])), 3)
            if not state_halt:
                self.assertTrue(torch.all(m.iteration_counts(xs, t.task_id) == 2))

    def test_independent_scorer_agrees_and_refuses(self):
        rng = np.random.default_rng(1)
        tmp = Path(tempfile.mkdtemp())
        plan = tmp / 'plan.md'
        plan.write_text('plan')
        (tmp / 'cells').mkdir()
        records = {}
        for a in b2.ARMS:
            for w in b2.WORLDS:
                for s in b2.STREAMS:
                    scale = {'LOOP': 0.15, 'CONSTCOUNT': 1.0, 'REFUSAL': 1.0}[a]
                    rec = {'arm': a, 'world': w, 'stream': s,
                           'canonical_per_task': {f'c{i}': float(rng.random() * 0.06) for i in range(64)},
                           'loop_per_task': {f'l{i}': float(scale * (0.5 + rng.random())) for i in range(48)},
                           'wake_loop_median': float(scale * (0.5 + rng.random())),
                           'wake_canonical_median': 0.1, 'count_accuracy_median': 0.9,
                           'wake_count_accuracy_median': 0.8}
                    rec['canonical_median'] = float(np.median(list(rec['canonical_per_task'].values())))
                    rec['loop_median'] = float(np.median(list(rec['loop_per_task'].values())))
                    rec['loop_below_threshold'] = 0
                    key = f'{a}_w{w}_s{s}'
                    records[key] = rec
                    (tmp / 'cells' / f'{key}.json').write_text(json.dumps(
                        {'stamp': {'protocol_sha256': 'x'}, 'complete': True, 'record': rec}))
        report = {'protocol_sha256': 'x', 'cells': records, 'summary': b2.summarize(records),
                  'protocol': {'input_sha256': {plan.as_posix(): hashlib.sha256(b'plan').hexdigest()}}}
        (tmp / 'report.json').write_text(json.dumps(report))
        out = sc.score(tmp, tmp / 'report.json', plan)
        self.assertTrue(out['agrees_with_runner'])
        plan.write_text('changed')
        with self.assertRaises(ValueError):
            sc.score(tmp, tmp / 'report.json', plan)


if __name__ == '__main__':
    unittest.main()
