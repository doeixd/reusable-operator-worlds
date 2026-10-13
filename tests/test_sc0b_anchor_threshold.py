import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from row.experiments import sc0_anchor_dose as sc0
from row.experiments import sc0b_anchor_threshold as sb
from row.experiments import score_sc0b_anchor_threshold as ss


class SC0bTests(unittest.TestCase):
    def test_labels_and_threshold(self):
        self.assertEqual(sb.label(False, {24: 5, 32: 5, 40: 5}), 'UNINTERPRETABLE')
        self.assertEqual(sb.label(True, {24: 0, 32: 5, 40: 4}), 'LENGTHS_SUFFICE')
        self.assertEqual(sb.label(True, {24: 0, 32: 4, 40: 5}), 'GRADED')
        self.assertEqual(sb.label(True, {24: 0, 32: 0, 40: 1}), 'LENGTHS_INSUFFICIENT')
        self.assertEqual(sb.label(True, {24: 2, 32: 3, 40: 3}), 'GRADED')
        self.assertEqual(sb.stream_threshold({4: False, 16: True, 24: False, 32: True, 40: True}), 32)
        self.assertEqual(sb.stream_threshold({4: False, 40: False}), None)
        self.assertEqual(sb.stream_threshold({4: True, 40: True}), 4)

    def test_cells_nest_with_sc0(self):
        self.assertEqual(len(sb.cells()), 16)
        self.assertEqual(sb.cells()[0], (16, 64, 0))
        kept = {n: set(sc0.build_dose_stream(n, 64, 2)[5]) for n in (16, 24, 32, 40)}
        self.assertTrue(kept[16] <= kept[24] <= kept[32] <= kept[40])

    def _fixture(self, anchor_same=True):
        rng = np.random.default_rng(5)
        tmp = Path(tempfile.mkdtemp())
        plan = tmp / 'plan.md'
        plan.write_text('plan')
        (tmp / 'cells').mkdir()

        def rec(n, s, level):
            r = {'dose': n, 'world': 64, 'stream': s, 'stream_tasks': 128 + n, 'anchor_ids': [f'a{i}' for i in range(n)],
                 'canonical_per_task': {f'c{i}': float(level * (0.5 + rng.random())) for i in range(64)},
                 'library_sha256': f'lib{n}{s}', 'wake_canonical_median': 1.0, 'reroute_canonical_median': 0.5}
            r['canonical_median'] = float(np.median(list(r['canonical_per_task'].values())))
            return r

        sc0_cells = {f'D{n}_w64_s{s}': rec(n, s, 0.01 if (n == 16 and s in (0, 2)) else 1.4)
                     for n in sc0.DOSES for s in sb.STREAMS}
        sc0_path = tmp / 'sc0.json'
        sc0_path.write_text(json.dumps({'cells': sc0_cells}))
        records = {'D16_w64_s0': dict(sc0_cells['D16_w64_s0']) if anchor_same else rec(16, 0, 0.02)}
        for n, w, s in sb.cells()[1:]:
            records[f'D{n}_w{w}_s{s}'] = rec(n, s, 0.01 if (n >= 32 or s != 1) else 1.4)
        for key, r in records.items():
            (tmp / 'cells' / f'{key}.json').write_text(json.dumps(
                {'stamp': {'protocol_sha256': 'x'}, 'complete': True, 'record': r}))
        summary = sb.summarize(records, sc0_cells)
        report = {'protocol_sha256': 'x', 'cells': records, 'summary': summary,
                  'protocol': {'input_sha256': {plan.as_posix(): hashlib.sha256(b'plan').hexdigest(),
                                                sc0_path.as_posix(): hashlib.sha256(sc0_path.read_bytes()).hexdigest()}}}
        (tmp / 'report.json').write_text(json.dumps(report))
        return tmp, plan, sc0_path, report, summary

    def test_scorer_agrees_and_refuses(self):
        tmp, plan, sc0_path, report, summary = self._fixture()
        self.assertEqual(summary['label'], 'LENGTHS_SUFFICE')
        self.assertEqual(summary['per_stream_threshold']['1'], 32)
        self.assertEqual(summary['per_stream_threshold']['0'], 16)
        out = ss.score(tmp, tmp / 'report.json', plan, sc0_path)
        self.assertTrue(out['agrees_with_runner'])
        report['summary'] = dict(summary, label='GRADED')
        (tmp / 'report.json').write_text(json.dumps(report))
        self.assertFalse(ss.score(tmp, tmp / 'report.json', plan, sc0_path)['agrees_with_runner'])
        plan.write_text('changed')
        with self.assertRaises(ValueError):
            ss.score(tmp, tmp / 'report.json', plan, sc0_path)

    def test_anchor_mismatch_is_uninterpretable(self):
        tmp, plan, sc0_path, _, summary = self._fixture(anchor_same=False)
        self.assertEqual(summary['label'], 'UNINTERPRETABLE')
        out = ss.score(tmp, tmp / 'report.json', plan, sc0_path)
        self.assertEqual(out['label'], 'UNINTERPRETABLE')
        self.assertTrue(out['agrees_with_runner'])


if __name__ == '__main__':
    unittest.main()
