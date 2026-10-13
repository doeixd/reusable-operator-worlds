import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from row.experiments import o2_online_reliability as o2
from row.experiments import sc0_anchor_dose as sc
from row.experiments import score_sc0_anchor_dose as ss


class SC0Tests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(sc.label(False, {4: 5, 8: 5, 16: 5}), 'UNINTERPRETABLE')
        self.assertEqual(sc.label(True, {4: 5, 8: 5, 16: 5}), 'NO_OPPORTUNITY')
        self.assertEqual(sc.label(True, {4: 1, 8: 3, 16: 4}), 'OPPORTUNITY')
        self.assertEqual(sc.label(True, {4: 0, 8: 0, 16: 3}), 'GRADED')
        self.assertEqual(sc.label(True, {4: 4, 8: 5, 16: 5}), 'GRADED')
        self.assertFalse(sc.passing(0.05))
        self.assertTrue(sc.passing(0.0499))
        self.assertFalse(sc.passing(float('nan')))

    def test_dose_streams_nested_and_dose60_is_o2(self):
        _, _, full, plan, _ = o2.build_stream('SHUFFLED', 64, 1)
        streams = {n: sc.build_dose_stream(n, 64, 1) for n in (0, 4, 8, 16, 60)}
        self.assertEqual([t.task_id for t in streams[60][2]], [t.task_id for t in full])
        kept = {n: set(v[5]) for n, v in streams.items()}
        self.assertTrue(kept[0] <= kept[4] <= kept[8] <= kept[16] <= kept[60])
        for n, v in streams.items():
            tasks, p = v[2], v[3]
            self.assertEqual(len(tasks), 128 + n)
            self.assertEqual(sum(1 for d in p.values() if d == 1), n)
            others = [t.task_id for t in tasks if p[t.task_id] != 1]
            self.assertEqual(others, [t.task_id for t in full if plan[t.task_id] != 1])
        self.assertEqual(len(sc.cells()), 17)

    def test_scorer_agrees_and_refuses(self):
        rng = np.random.default_rng(3)
        tmp = Path(tempfile.mkdtemp())
        plan = tmp / 'plan.md'
        plan.write_text('plan')
        (tmp / 'cells').mkdir()
        records = {}
        for n, w, s in sc.cells():
            level = 0.01 if n >= 8 else (0.2 if n == 0 else 0.03 + 0.04 * (s % 2))
            rec = {'dose': n, 'world': w, 'stream': s, 'stream_tasks': 128 + n, 'anchor_ids': [f'a{i}' for i in range(n)],
                   'canonical_per_task': {f'c{i}': float(level * (0.5 + rng.random())) for i in range(64)}}
            for k in ('wake_canonical_median', 'reroute_canonical_median'):
                rec[k] = 0.5
            rec['canonical_median'] = float(np.median(list(rec['canonical_per_task'].values())))
            key = f'D{n}_w{w}_s{s}'
            records[key] = rec
            (tmp / 'cells' / f'{key}.json').write_text(json.dumps(
                {'stamp': {'protocol_sha256': 'x'}, 'complete': True, 'record': rec}))
        summary = sc.summarize(records)
        report = {'protocol_sha256': 'x', 'cells': records, 'summary': summary,
                  'protocol': {'input_sha256': {plan.as_posix(): hashlib.sha256(b'plan').hexdigest()}}}
        (tmp / 'report.json').write_text(json.dumps(report))
        out = ss.score(tmp, tmp / 'report.json', plan)
        self.assertTrue(out['agrees_with_runner'])
        self.assertEqual(out['label'], summary['label'])
        report['summary'] = dict(summary, label='NO_OPPORTUNITY' if summary['label'] != 'NO_OPPORTUNITY' else 'GRADED')
        (tmp / 'report.json').write_text(json.dumps(report))
        self.assertFalse(ss.score(tmp, tmp / 'report.json', plan)['agrees_with_runner'])
        plan.write_text('changed')
        with self.assertRaises(ValueError):
            ss.score(tmp, tmp / 'report.json', plan)


if __name__ == '__main__':
    unittest.main()
