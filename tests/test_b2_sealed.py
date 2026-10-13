import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from row.experiments import b2_sealed as bs
from row.experiments import score_b2_sealed as ss


def fake_record(arm, w, s, rng, formed=True, rooted=False, loop_level=0.28):
    loops = [f'L{i}' for i in range(48)]
    straight = [f'S{i}' for i in range(60)]
    level = 0.01 if formed else 1.4
    r = {'arm': arm, 'world': w, 'stream': s, 'candidates': 108, 'loop_task_ids': loops,
         'canonical_per_task': {f'c{i}': float(level * (0.5 + rng.random())) for i in range(64)},
         'loop_per_task': {t: float(loop_level * (0.8 + 0.4 * rng.random())) for t in loops},
         'anchor_per_task': {t: 0.004 for t in straight}, 'library_sha256': f'lib-{arm}-{w}-{s}',
         'wake_rooted_ops': [3] if rooted else [], 'wake_canonical_median': 0.07, 'wake_loop_median': 0.8,
         'reroute_canonical_median': 0.05, 'count_accuracy_median': 0.86, 'anchor_median': 0.004}
    r['canonical_median'] = float(np.median(list(r['canonical_per_task'].values())))
    r['loop_median'] = float(np.median(list(r['loop_per_task'].values())))
    if arm != 'GIVEN':
        r['wake_state_norms'] = {**{t: 2.5 + rng.random() for t in loops}, **{t: 0.4 * rng.random() for t in straight}}
        r['decisions'] = {**{t: {'loop': True} for t in loops}, **{t: {'loop': False} for t in straight}}
        pos = np.array([r['wake_state_norms'][t] for t in loops])
        neg = np.array([r['wake_state_norms'][t] for t in straight])
        r['wake_state_auc'] = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
        r['loops_kept'], r['straight_kept_count'] = 48, 60
    else:
        r['decisions'] = None
    return r


class B2SealedTests(unittest.TestCase):
    def test_labels_and_cells(self):
        self.assertEqual(bs.label(False, 30, 1.0, 1.0, 1.0, 30), 'UNINTERPRETABLE')
        self.assertEqual(bs.label(True, 26, 1.0, 1.0, 1.0, 30), 'FORMATION_BROKEN')
        self.assertEqual(bs.label(True, 27, 0.95, 0.85, 0.96, 27), 'CONFIRMED')
        self.assertEqual(bs.label(True, 30, 0.89, 0.85, 0.96, 30), 'NOT_DISCOVERED')
        self.assertEqual(bs.label(True, 30, 0.95, 0.85, 0.96, 26), 'NOT_DISCOVERED')
        self.assertEqual(bs.roots_label(0, 1), 'NO_ROOTS_IN_CONTROL')
        self.assertEqual(bs.roots_label(1, 2), 'ROOTS_PREVENTED')
        self.assertEqual(bs.roots_label(2, 5), 'ROOTS_NOT_PREVENTED')
        c = bs.cells()
        self.assertEqual(len(c), 61)
        self.assertEqual(c[0], ('ONESTEP', 57, 0))
        self.assertEqual([x[0] for x in c[1:31]], ['ONESTEP'] * 30)
        self.assertEqual(sum(1 for a, w, s in c if a == 'GIVEN' and s == 0 and 1100 <= w <= 1114), 15)
        self.assertTrue(all(1100 <= w <= 1114 for a, w, s in c[1:]))
        self.assertEqual(bs.FAILURES_TO_STOP, 4)

    def _fixture(self, anchor_same=True, given_formed=15, onestep_failed=0, drop=()):
        rng = np.random.default_rng(7)
        tmp = Path(tempfile.mkdtemp())
        plan = tmp / 'plan.md'
        plan.write_text('plan')
        (tmp / 'cells').mkdir()
        dev_anchor = fake_record('ONESTEP', 57, 0, rng)
        dev = tmp / 'dev.json'
        dev.write_text(json.dumps({'cells': {'ONESTEP_w57_s0': dev_anchor}}))
        records = {}
        for i, (a, w, s) in enumerate(bs.cells()):
            key = bs.key_of(a, w, s)
            if key in drop:
                continue
            if i == 0:
                r = dict(dev_anchor) if anchor_same else fake_record(a, w, s, rng)
            elif a == 'ONESTEP':
                r = fake_record(a, w, s, rng, formed=i > onestep_failed)
            elif a == 'ZERO':
                r = fake_record(a, w, s, rng, rooted=w % 3 == 0)
            else:
                r = fake_record(a, w, s, rng, formed=(w - 1100) < given_formed)
            records[key] = r
            (tmp / 'cells' / f'{key}.json').write_text(json.dumps(
                {'stamp': {'protocol_sha256': 'x'}, 'complete': True, 'record': r}))
        summary = bs.summarize(records, dev_anchor)
        report = {'protocol_sha256': 'x', 'cells': records, 'summary': summary,
                  'protocol': {'input_sha256': {plan.as_posix(): hashlib.sha256(b'plan').hexdigest(),
                                                dev.as_posix(): hashlib.sha256(dev.read_bytes()).hexdigest()}}}
        (tmp / 'report.json').write_text(json.dumps(report))
        return tmp, plan, dev, report, summary

    def test_confirmed_and_scorer_agrees(self):
        tmp, plan, dev, report, summary = self._fixture()
        self.assertEqual(summary['label'], 'CONFIRMED')
        self.assertEqual(summary['roots_label'], 'ROOTS_PREVENTED')
        out = ss.score(tmp, tmp / 'report.json', plan, dev)
        self.assertTrue(out['agrees_with_runner'], out)
        report['summary'] = dict(summary, label='NOT_DISCOVERED')
        (tmp / 'report.json').write_text(json.dumps(report))
        self.assertFalse(ss.score(tmp, tmp / 'report.json', plan, dev)['agrees_with_runner'])
        plan.write_text('changed')
        with self.assertRaises(ValueError):
            ss.score(tmp, tmp / 'report.json', plan, dev)

    def test_gates_and_broken(self):
        _, _, _, _, s = self._fixture(anchor_same=False)
        self.assertEqual(s['label'], 'UNINTERPRETABLE')
        _, _, _, _, s = self._fixture(given_formed=11)
        self.assertEqual(s['label'], 'UNINTERPRETABLE')
        tmp, plan, dev, _, s = self._fixture(onestep_failed=4)
        self.assertEqual((s['k_formation'], s['label']), (26, 'FORMATION_BROKEN'))
        self.assertTrue(ss.score(tmp, tmp / 'report.json', plan, dev)['agrees_with_runner'])

    def test_early_stopped_run_scores(self):
        dropped = tuple(bs.key_of('ONESTEP', w, 1) for w in range(1105, 1115))
        tmp, plan, dev, _, s = self._fixture(onestep_failed=4, drop=dropped)
        self.assertEqual((s['onestep_cells_complete'], s['label']), (20, 'FORMATION_BROKEN'))
        out = ss.score(tmp, tmp / 'report.json', plan, dev)
        self.assertEqual(out['label'], 'FORMATION_BROKEN')
        self.assertTrue(out['agrees_with_runner'])

    def test_confirmed_requires_every_cell(self):
        with self.assertRaises(ValueError):
            self._fixture(drop=(bs.key_of('ONESTEP', 1114, 1),))


if __name__ == '__main__':
    unittest.main()
