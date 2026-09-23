import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from row.experiments import o1_online_anchor as o1
from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.audit_so1_budget_bracket import build_fast


class StreamTests(unittest.TestCase):
    def test_streams_hold_the_registered_tasks(self):
        for world in o1.WORLDS:
            for arm, n in (('SHUFFLED', 188), ('MIXED_L1', 124)):
                _, _, stream, plan, canonical = o1.build_stream(arm, world)
                self.assertEqual(len(stream), n, (arm, world))
                self.assertEqual(len(canonical), 64)
                ids = {t.task_id for t in stream}
                self.assertTrue({t.task_id for t in canonical} <= ids)
                depths = {d: sum(1 for v in plan.values() if v == d) for d in (1, 2, 3)}
                expected = {1: 60, 2: 64 if arm == 'SHUFFLED' else 0, 3: 64}
                self.assertEqual(depths, expected, (arm, world))

    def test_shuffled_is_the_staged_multiset_in_another_order(self):
        _, _, stream, _, _ = o1.build_stream('SHUFFLED', 10)
        staged = []
        for stage in (1, 2, 3):
            staged += [t.task_id for t in stage_setup(10, stage, o1.MODEL_SEED)[1].tasks]
        self.assertEqual(sorted(t.task_id for t in stream), sorted(staged))
        self.assertNotEqual([t.task_id for t in stream], staged, 'the stream must not be the curriculum order')

    def test_streams_interleave_rather_than_stage(self):
        for arm in ('SHUFFLED', 'MIXED_L1'):
            _, _, stream, plan, _ = o1.build_stream(arm, 11)
            head = [plan[t.task_id] for t in stream[:20]]
            self.assertGreater(len(set(head)), 1, arm)

    def test_stream_is_deterministic(self):
        a = [t.task_id for t in o1.build_stream('MIXED_L1', 12)[2]]
        b = [t.task_id for t in o1.build_stream('MIXED_L1', 12)[2]]
        self.assertEqual(a, b)


class LearnerTests(unittest.TestCase):
    def test_planned_learner_matches_the_fast_learner_at_init(self):
        cfg, _, _, _ = stage_setup(10, 3, o1.MODEL_SEED)
        self.assertEqual(library_sha(build_fast(cfg)), library_sha(o1.planned_model(cfg, {})))

    def test_plan_sets_depth_through_the_argumentless_begin_task(self):
        cfg, _, _, _ = stage_setup(10, 3, o1.MODEL_SEED)
        model = o1.planned_model(cfg, {'a': 1, 'b': 3})
        model.begin_task('a')
        model.begin_task('b')
        model.begin_task('probe')
        self.assertEqual((model.depth_of('a'), model.depth_of('b'), model.depth_of('probe')), (1, 3, 3))


class ValidationAndSummaryTests(unittest.TestCase):
    def _rec(self, arm, world, v, **kw):
        base = {'arm': arm, 'world': world, 'terminal_median': v, 'scored_tasks': 64,
                'stream_tasks': o1.EXPECTED_STREAM[arm], 'route_lengths_match_plan': True}
        base.update(kw)
        return base

    def test_wrong_scored_set_is_refused(self):
        with self.assertRaises(ValueError):
            o1.validate_cell(self._rec('SHUFFLED', 10, 0.01, scored_tasks=188))

    def test_route_length_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            o1.validate_cell(self._rec('MIXED_L1', 10, 0.01, route_lengths_match_plan=False))

    def test_decision(self):
        recs = {}
        for w in o1.WORLDS:
            recs[f'SHUFFLED_w{w}'] = self._rec('SHUFFLED', w, 0.9)
            recs[f'MIXED_L1_w{w}'] = self._rec('MIXED_L1', w, 0.01 if w != 12 else 0.9)
        self.assertEqual(o1.summarize(recs)['decision'], 'LIVE')
        for w in o1.WORLDS:
            recs[f'MIXED_L1_w{w}'] = self._rec('MIXED_L1', w, 0.9)
        self.assertEqual(o1.summarize(recs)['decision'], 'NOT_LIVE')


class RestartTests(unittest.TestCase):
    def _seed(self, tmp, tamper=False):
        from row.experiments.so1_storage import atomic_json, fingerprint
        root, report = Path(tmp) / 'run', Path(tmp) / 'r.json'
        with mock.patch.object(o1, 'ROOT', root):
            sha = fingerprint(o1.protocol())
        (root / 'cells').mkdir(parents=True)
        for arm in o1.ARMS:
            for w in o1.WORLDS:
                rec = {'arm': arm, 'world': w, 'terminal_median': 0.01, 'scored_tasks': 64,
                       'stream_tasks': o1.EXPECTED_STREAM[arm], 'route_lengths_match_plan': True,
                       'seconds': 1.0}
                saved = {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': rec,
                         'record_sha256': fingerprint(rec), 'finished_utc': 'x'}
                if tamper and arm == 'SHUFFLED' and w == 10:
                    saved['record']['terminal_median'] = 0.9
                atomic_json(root / 'cells' / f'{arm}_w{w}.json', saved)
        return root, report

    def test_completed_cells_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report = self._seed(tmp)
            with mock.patch.object(o1, 'ROOT', root), mock.patch.object(o1, 'OUTPUT', report), \
                 mock.patch.object(o1, 'host_ok', lambda: (99.0, True)), \
                 mock.patch.object(o1, 'run_cell', side_effect=AssertionError('recomputed')):
                o1.run([(a, w) for a in o1.ARMS for w in o1.WORLDS], jobs=1)
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'complete')

    def test_tampered_cell_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report = self._seed(tmp, tamper=True)
            with mock.patch.object(o1, 'ROOT', root), mock.patch.object(o1, 'OUTPUT', report), \
                 mock.patch.object(o1, 'host_ok', lambda: (99.0, True)):
                with self.assertRaises(ValueError):
                    o1.run([(a, w) for a in o1.ARMS for w in o1.WORLDS], jobs=1)


if __name__ == '__main__':
    unittest.main()
