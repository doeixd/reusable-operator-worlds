import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from row.experiments import n1_anchor_supply as n1
from row.experiments import n1c_anchor_coverage as nc


class ConstructionTests(unittest.TestCase):
    def test_every_arm_holds_its_registered_count_and_coverage(self):
        for world in nc.WORLDS:
            for arm, (count, coverage) in nc.EXPECTED.items():
                _, canonical, chosen, pool, excluded = nc.build_pool(arm, world)
                self.assertEqual(len(pool), 188, (arm, world))
                self.assertEqual(sum(len(t.train_x) for t in pool), 24064, (arm, world))
                self.assertEqual(len(canonical), 64)
                self.assertEqual(len(chosen), count, (arm, world))
                self.assertTrue(all(t.depth == 1 for t in chosen))
                ops = {t.program[0] for t in chosen}
                self.assertEqual(len(ops), coverage, (arm, world))
                if coverage == 5:
                    self.assertNotIn(excluded, ops)

    def test_primary_contrast_differs_only_in_coverage(self):
        """COVER6_K6 and COVER5_K6: same count, one covers the excluded op."""
        for world in nc.WORLDS:
            _, _, full, _, x = nc.build_pool('COVER6_K6', world)
            _, _, part, _, x2 = nc.build_pool('COVER5_K6', world)
            self.assertEqual(x, x2)
            self.assertEqual(len(full), len(part))
            self.assertIn(x, {t.program[0] for t in full})
            self.assertNotIn(x, {t.program[0] for t in part})
            # Five of the six anchors are shared; they differ in one slot.
            shared = {t.task_id for t in full} & {t.task_id for t in part}
            self.assertEqual(len(shared), 5)

    def test_both_incomplete_arms_exclude_the_same_operation(self):
        for world in nc.WORLDS:
            self.assertEqual(nc.build_pool('COVER5_K6', world)[4],
                             nc.build_pool('COVER5_K18', world)[4])

    def test_zero_anchors_would_be_n1_sham(self):
        cfg, world_obj, canonical = n1.canonical_tasks(0)
        fillers = n1.extra_length3_tasks(cfg, world_obj, nc.FILLERS, 0)
        pool = [] + canonical + fillers[:nc.FILLERS]
        self.assertEqual([t.task_id for t in pool],
                         [t.task_id for t in n1.arm_tasks('SHAM', 0)[3]])

    def test_selection_is_deterministic(self):
        a = [t.task_id for t in nc.build_pool('COVER5_K18', 1)[2]]
        b = [t.task_id for t in nc.build_pool('COVER5_K18', 1)[2]]
        self.assertEqual(a, b)

    def test_short_prefix_trains_and_draws_match_across_arms(self):
        torch.set_num_threads(1)
        draws = []
        for arm in nc.ARMS:
            cfg, canonical, _, pool, _ = nc.build_pool(arm, 0)
            _, drawn = n1.train_pooled(cfg, pool, canonical, 8,
                                       np.random.SeedSequence([n1.POOL_STREAM, 0]))
            draws.append(drawn)
        self.assertTrue(all(d == draws[0] for d in draws))


class SplitAndTriageTests(unittest.TestCase):
    def test_split_partitions_the_canonical_tasks(self):
        _, canonical, _, _, x = nc.build_pool('COVER5_K6', 0)
        per_task = {t.task_id: 0.5 for t in canonical}
        s = nc.split_by_excluded(per_task, canonical, x)
        self.assertEqual(s['tasks_using_excluded'] + s['tasks_not_using_excluded'], 64)
        self.assertGreater(s['tasks_using_excluded'], 0)

    def _records(self, values, free=None):
        out = {}
        for arm, per in values.items():
            for w, v in enumerate(per):
                out[f'{arm}_w{w}'] = {'arm': arm, 'world': w, 'terminal_median': v,
                                      'excluded_operation': 2, 'near_threshold': 0.03 <= v <= 0.08,
                                      'split': {'median_not_using_excluded': free if free is not None else v}}
        return out

    def test_every_primary_outcome_is_reachable(self):
        ok, bad = [0.01] * 3, [1.0] * 3
        cases = {('ok', 'bad'): 'COVERAGE_MATTERS', ('ok', 'ok'): 'COVERAGE_NOT_NEEDED_AT_K6',
                 ('bad', 'bad'): 'K6_INSUFFICIENT', ('bad', 'ok'): 'ANOMALOUS'}
        pick = {'ok': ok, 'bad': bad}
        for (a, b), label in cases.items():
            t = nc.triage(self._records({'COVER6_K6': pick[a], 'COVER5_K6': pick[b],
                                         'COVER5_K18': bad}))
            self.assertEqual(t['primary'], label)

    def test_local_versus_global_split(self):
        t = nc.triage(self._records({'COVER6_K6': [0.01] * 3, 'COVER5_K6': [1.0] * 3,
                                     'COVER5_K18': [1.0] * 3}, free=0.01))
        self.assertEqual(set(t['descriptive_split'].values()), {'LOCAL'})
        t = nc.triage(self._records({'COVER6_K6': [0.01] * 3, 'COVER5_K6': [1.0] * 3,
                                     'COVER5_K18': [1.0] * 3}, free=0.9))
        self.assertEqual(set(t['descriptive_split'].values()), {'GLOBAL'})


class RestartTests(unittest.TestCase):
    def _seed(self, tmp, tamper=False):
        from row.experiments.so1_storage import atomic_json, fingerprint
        root, report = Path(tmp) / 'run', Path(tmp) / 'r.json'
        with mock.patch.object(nc, 'ROOT', root):
            sha = fingerprint(nc.protocol())
        (root / 'cells').mkdir(parents=True)
        for arm, (count, coverage) in nc.EXPECTED.items():
            for w in nc.WORLDS:
                rec = {'arm': arm, 'world': w, 'updates': 65536, 'anchors': count,
                       'operations_covered': list(range(coverage)), 'excluded_operation': 5,
                       'pool_tasks': 188, 'pool_examples': 24064, 'scored_tasks': 64,
                       'terminal_median': 0.01, 'below_0.05': 60, 'per_task': {},
                       'split': {'tasks_using_excluded': 30, 'tasks_not_using_excluded': 34,
                                 'median_using_excluded': 0.01, 'median_not_using_excluded': 0.01},
                       'library_sha256': 'x', 'first_64_draws': [], 'near_threshold': False,
                       'seconds': 1.0}
                saved = {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': rec,
                         'record_sha256': fingerprint(rec), 'finished_utc': 'x'}
                if tamper and arm == 'COVER5_K6' and w == 1:
                    saved['record']['terminal_median'] = 0.9
                atomic_json(root / 'cells' / f'{arm}_w{w}.json', saved)
        return root, report

    def test_completed_cells_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report = self._seed(tmp)
            with mock.patch.object(nc, 'ROOT', root), mock.patch.object(nc, 'OUTPUT', report), \
                 mock.patch.object(nc, 'host_ok', lambda: (99.0, True)), \
                 mock.patch.object(nc, 'run_cell', side_effect=AssertionError('recomputed')):
                nc.run([(a, w) for a in nc.ARMS for w in nc.WORLDS], jobs=1)
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'complete')

    def test_tampered_cell_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report = self._seed(tmp, tamper=True)
            with mock.patch.object(nc, 'ROOT', root), mock.patch.object(nc, 'OUTPUT', report), \
                 mock.patch.object(nc, 'host_ok', lambda: (99.0, True)):
                with self.assertRaises(ValueError):
                    nc.run([(a, w) for a in nc.ARMS for w in nc.WORLDS], jobs=1)


if __name__ == '__main__':
    unittest.main()
