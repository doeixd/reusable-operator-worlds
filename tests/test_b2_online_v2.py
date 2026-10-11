import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from row.experiments import b2_online_v2 as b2
from row.experiments import score_b2_online_v2 as sc


class B2OnlineV2Tests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(b2.label(6, 9, 9), 'FORMATION_BROKEN')
        self.assertEqual(b2.label(7, 8, 8), 'LOOPS_ONLINE')
        self.assertEqual(b2.label(9, 7, 9), 'LOOPS_AFTER_SLEEP')
        self.assertEqual(b2.label(9, 9, 7), 'NO_LOOPS')
        self.assertEqual(b2.mechanism_label(8), 'DETACH_PROTECTS')
        self.assertEqual(b2.mechanism_label(7), 'DETACH_NOT_SHOWN')

    def test_arm_constructions(self):
        """LOOP detaches the predicate input; LOOP_ATTACHED is v1's LOOP; CONSTCOUNT has no state halting."""
        import row.experiments.loop_stream as ls
        from row.experiments import b1_online as b1o
        from row.models.loop_gated import LoopLearner
        cfg, world, st, plan, canonical, ltasks, meta = ls.stream(50, 0, 2)
        ids = {t.task_id: 'state' for t in ltasks}
        for arm, sh, det in (('LOOP', True, True), ('LOOP_ATTACHED', True, False), ('CONSTCOUNT', False, False)):
            m = b1o.build(LoopLearner, cfg, plan, loop_plan=ids, max_iterations=b2.K,
                          state_halt=(arm != 'CONSTCOUNT'), detach_halt_input=(arm == 'LOOP'))
            self.assertEqual((m.state_halt, m.detach_halt_input), (sh, det), arm)

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
                    scale = {'LOOP': 0.15, 'LOOP_ATTACHED': 0.3, 'CONSTCOUNT': 1.0}[a]
                    cs = {'LOOP': 0.04, 'LOOP_ATTACHED': 0.12, 'CONSTCOUNT': 0.04}[a]
                    rec = {'arm': a, 'world': w, 'stream': s,
                           'canonical_per_task': {f'c{i}': float(rng.random() * cs) for i in range(64)},
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
        summary = b2.summarize(records)
        self.assertEqual(summary['mechanism_label'], 'DETACH_PROTECTS')
        report = {'protocol_sha256': 'x', 'cells': records, 'summary': summary,
                  'protocol': {'loop_bar': b2.LOOP_BAR,
                               'input_sha256': {plan.as_posix(): hashlib.sha256(b'plan').hexdigest()}}}
        (tmp / 'report.json').write_text(json.dumps(report))
        out = sc.score(tmp, tmp / 'report.json', plan)
        self.assertTrue(out['agrees_with_runner'])
        plan.write_text('changed')
        with self.assertRaises(ValueError):
            sc.score(tmp, tmp / 'report.json', plan)


if __name__ == '__main__':
    unittest.main()
