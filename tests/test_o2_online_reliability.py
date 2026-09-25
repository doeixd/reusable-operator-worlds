import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from row.experiments import o1_online_anchor as o1
from row.experiments import o2_online_reliability as o2
from row.experiments import score_o2_online_reliability as scorer
from row.experiments.audit_j1c_curriculum import stage_setup


class StreamTests(unittest.TestCase):
    def test_stream_zero_is_o1_bitwise(self):
        for arm in ('SHUFFLED', 'MIXED_L1'):
            for w in o1.WORLDS:
                a = [t.task_id for t in o2.build_stream(arm, w, 0)[2]]
                b = [t.task_id for t in o1.build_stream(arm, w)[2]]
                self.assertEqual(a, b, (arm, w))

    def test_streams_reorder_the_same_multiset(self):
        orders = [[t.task_id for t in o2.build_stream('SHUFFLED', 13, s)[2]] for s in o2.STREAMS]
        self.assertEqual(len({tuple(o) for o in orders}), 3)
        self.assertEqual(len({tuple(sorted(o)) for o in orders}), 1)

    def test_every_registered_single_lifetime_stream_interleaves(self):
        for arm in ('SHUFFLED', 'MIXED_L1'):
            for w in o2.WORLDS:
                for s in o2.STREAMS:
                    _, _, stream, plan, canonical = o2.build_stream(arm, w, s)
                    self.assertEqual(len(stream), o2.EXPECTED_STREAM[arm])
                    self.assertEqual(len(canonical), 64)
                    self.assertGreater(len({plan[t.task_id] for t in stream[:20]}), 1, (arm, w, s))


class SeedAndCellTests(unittest.TestCase):
    def test_replay_seeds(self):
        self.assertIsNone(o2.replay_seed_for(13, 0))
        seeds = {o2.replay_seed_for(w, s) for w in o2.WORLDS for s in (1, 2)}
        self.assertEqual(len(seeds), 14)
        self.assertTrue(all(isinstance(x, int) for x in seeds))

    def test_registered_cells(self):
        cells = o2.cells()
        self.assertEqual(len(cells), 3 * 7 * 3 + 7)
        self.assertEqual(len(set(cells)), len(cells))
        self.assertEqual(cells[0][0], 'SHUFFLED')
        self.assertEqual({s for a, w, s in cells if a == 'PLAIN'}, {0})
        self.assertEqual(set(o2.WORLDS) & {10, 11, 12}, set())

    def test_lean_changes_only_diagnostics(self):
        cfg = stage_setup(13, 3, o2.MODEL_SEED)[0]
        lean = o2.run_cfg(cfg)
        self.assertEqual(dataclasses.replace(lean, evaluation=cfg.evaluation), cfg)
        self.assertEqual(lean.evaluation.lifetime_checkpoints, ())


class DecisionTests(unittest.TestCase):
    def test_labels_partition_and_agree_with_scorer(self):
        for k in range(22):
            self.assertIn(o2.label(k), ('RELIABLE', 'INTERMEDIATE', 'UNRELIABLE'))
            self.assertEqual(o2.label(k), scorer.label(k))
        self.assertEqual((o2.label(19), o2.label(18), o2.label(17), o2.label(16)),
                         ('RELIABLE', 'INTERMEDIATE', 'INTERMEDIATE', 'UNRELIABLE'))

    def _records(self, shuffled_pass=21, plain=2.0, nan_cell=None):
        recs, n = {}, 0
        for a, w, s in o2.cells():
            v = plain if a == 'PLAIN' else 0.9
            if a == 'SHUFFLED':
                v = 0.01 if n < shuffled_pass else 0.9
                n += 1
            key = f'{a}_w{w}_s{s}'
            recs[key] = {'terminal_median': float('nan') if key == nan_cell else v}
        return recs

    def test_summary(self):
        self.assertEqual(o2.summarize(self._records(19))['primary'], 'ORDER_FREE_RELIABLE')
        self.assertEqual(o2.summarize(self._records(17))['primary'], 'ORDER_FREE_INTERMEDIATE')
        self.assertEqual(o2.summarize(self._records(16))['primary'], 'ORDER_FREE_UNRELIABLE')

    def test_non_finite_counts_as_not_passing(self):
        s = o2.summarize(self._records(19, nan_cell='SHUFFLED_w13_s0'))
        self.assertEqual(s['passes']['SHUFFLED'], 18)
        self.assertEqual(s['non_finite_cells'], ['SHUFFLED_w13_s0'])

    def test_floor_clause(self):
        s = o2.summarize(self._records(21, plain=0.01))
        self.assertEqual(s['primary'], 'FLOOR_FAILED')
        self.assertTrue(all(v == 'FLOOR_FAILED' for v in s['labels'].values()))


class ValidationTests(unittest.TestCase):
    def _rec(self, arm='SHUFFLED', **kw):
        n = o2.EXPECTED_STREAM[arm]
        base = {'arm': arm, 'world': 13, 'stream': 0, 'scored_tasks': 64,
                'terminal_per_task': {str(i): 0.01 for i in range(64)}, 'stream_tasks': n, 'trained_tasks': n,
                'depth_histogram': dict(o2.EXPECTED_DEPTHS[arm]), 'route_lengths_match_plan': True,
                'routes_checked': n, 'anchor_abs_error': 0.0,
                'first_20_stream_depths': [1, 3] * 10 if arm in ('SHUFFLED', 'MIXED_L1') else None}
        base.update(kw)
        return base

    def test_valid_record_passes(self):
        for arm in o2.ARMS:
            o2.validate_cell(self._rec(arm))

    def test_refusals(self):
        bad = [dict(scored_tasks=188), dict(trained_tasks=64), dict(depth_histogram={'1': 0, '2': 0, '3': 64}),
               dict(route_lengths_match_plan=False), dict(routes_checked=64), dict(anchor_abs_error=1e-3),
               dict(first_20_stream_depths=[3] * 20), dict(stream=3)]
        for kw in bad:
            with self.assertRaises(ValueError, msg=str(kw)):
                o2.validate_cell(self._rec('SHUFFLED', **kw))
        with self.assertRaises(ValueError):
            o2.validate_cell(self._rec('PLAIN', stream=1))


class RestartTests(unittest.TestCase):
    def _seed(self, tmp, tamper=False):
        from row.experiments.so1_storage import atomic_json, fingerprint
        root, report = Path(tmp) / 'run', Path(tmp) / 'r.json'
        sha = fingerprint(o2.protocol(root) | {'scale': 1})
        (root / 'cells').mkdir(parents=True)
        for a, w, s in o2.cells():
            rec = ValidationTests._rec(ValidationTests(), a) | {'world': w, 'stream': s, 'terminal_median': 0.01,
                                                                 'seconds': 1.0}
            saved = {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': rec,
                     'record_sha256': fingerprint(rec), 'finished_utc': 'x'}
            if tamper and (a, w, s) == ('SHUFFLED', 13, 0):
                saved['record']['terminal_median'] = 0.9
            atomic_json(root / 'cells' / f'{a}_w{w}_s{s}.json', saved)
        return root, report

    def test_completed_cells_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report = self._seed(tmp)
            with mock.patch.object(o2, 'host_ok', lambda: (99.0, True)), \
                 mock.patch.object(o2, 'run_cell', side_effect=AssertionError('recomputed')):
                o2.run(o2.cells(), root=root, output=report, jobs=1, require_gates=False)
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'complete')
            # every seeded cell, PLAIN included, sits at 0.01, so the floor clause must fire
            self.assertEqual(json.loads(report.read_text())['summary']['primary'], 'FLOOR_FAILED')

    def test_tampered_cell_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report = self._seed(tmp, tamper=True)
            with mock.patch.object(o2, 'host_ok', lambda: (99.0, True)):
                with self.assertRaises(ValueError):
                    o2.run(o2.cells(), root=root, output=report, jobs=1, require_gates=False)

    def test_launch_refuses_without_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report = self._seed(tmp)
            with mock.patch.object(o2, 'ROOT', Path(tmp) / 'nogates'), \
                 mock.patch.object(o2, 'host_ok', lambda: (99.0, True)):
                with self.assertRaises((FileNotFoundError, RuntimeError)):
                    o2.run(o2.cells(), root=root, output=report, jobs=1)


if __name__ == '__main__':
    unittest.main()
