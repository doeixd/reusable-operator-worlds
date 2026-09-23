import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from row.experiments import n1_anchor_supply as n1
from row.experiments import n1b_anchor_dose as nb


def _ids_and_arrays(pool):
    return [t.task_id for t in pool], [t.train_x for t in pool], [t.train_y for t in pool]


class EndpointIdentityTests(unittest.TestCase):
    """k = 124 and k = 0 must BE N1's committed INTERLEAVED and SHAM pools."""

    def _same(self, a, b):
        ia, xa, ya = _ids_and_arrays(a)
        ib, xb, yb = _ids_and_arrays(b)
        self.assertEqual(ia, ib)
        for u, v in zip(xa, xb):
            self.assertTrue(np.array_equal(u, v))
        for u, v in zip(ya, yb):
            self.assertTrue(np.array_equal(u, v))

    def test_all_anchors_is_n1_interleaved(self):
        self._same(nb.build_pool('ENDPOINT_ALL', 0)[3], n1.arm_tasks('INTERLEAVED', 0)[3])

    def test_no_anchors_is_n1_sham(self):
        self._same(nb.build_pool('ENDPOINT_NONE', 0)[3], n1.arm_tasks('SHAM', 0)[3])

    def test_prefix_training_is_bitwise_identical_to_n1(self):
        torch.set_num_threads(1)
        cfg, canonical, _, pool = nb.build_pool('ENDPOINT_ALL', 0)
        _, _, canon_n1, pool_n1 = n1.arm_tasks('INTERLEAVED', 0)
        seq = lambda: np.random.SeedSequence([n1.POOL_STREAM, 0])
        a, _ = n1.train_pooled(cfg, pool, canonical, 32, seq())
        b, _ = n1.train_pooled(cfg, pool_n1, canon_n1, 32, seq())
        self.assertEqual(n1.library_sha(a), n1.library_sha(b))


class ConstructionTests(unittest.TestCase):
    def test_every_arm_holds_the_registered_pool(self):
        for arm, (count, depths) in nb.EXPECTED_ANCHORS.items():
            _, canonical, chosen, pool = nb.build_pool(arm, 1)
            self.assertEqual(len(pool), 188, arm)
            self.assertEqual(sum(len(t.train_x) for t in pool), 24064, arm)
            self.assertEqual(len(chosen), count, arm)
            self.assertEqual(len(canonical), 64, arm)
            if depths is not None:
                self.assertEqual({str(d): sum(1 for t in chosen if t.depth == d) for d in (1, 2)},
                                 depths, arm)

    def test_fillers_are_length_three_and_distinct(self):
        _, _, chosen, pool = nb.build_pool('DOSE_8', 2)
        fillers = [t for t in pool if t.task_id.startswith('sham_')]
        self.assertEqual(len(fillers), 116)
        self.assertTrue(all(t.depth == 3 for t in fillers))
        self.assertEqual(len({t.program for t in fillers}), 116)

    def test_dose_subsets_are_nested_and_in_n1_order(self):
        anchors = n1.anchor_tasks(0)
        small = [t.task_id for t in nb.select_anchors('DOSE_8', anchors, 0)]
        large = [t.task_id for t in nb.select_anchors('DOSE_32', anchors, 0)]
        self.assertTrue(set(small) <= set(large), 'DOSE_8 must be a subset of DOSE_32')
        order = [t.task_id for t in anchors]
        self.assertEqual(small, sorted(small, key=order.index))


class TriageTests(unittest.TestCase):
    def _records(self, values):
        out = {}
        for arm, per_world in values.items():
            for w, v in enumerate(per_world):
                out[f'{arm}_w{w}'] = {'arm': arm, 'world': w, 'terminal_median': v,
                                      'near_threshold': 0.03 <= v <= 0.08}
        return out

    def test_persistence_rule_and_length_outcomes(self):
        r = self._records({'L1_ONLY': [0.01] * 3, 'L2_ONLY': [0.9] * 3,
                           'DOSE_8': [0.9] * 3, 'DOSE_32': [0.01] * 3})
        t = nb.triage(r)
        self.assertEqual(t['length_verdict'], 'L1_SUFFICES_ONLY')
        self.assertEqual(t['dose_k_star'], 32)
        self.assertFalse(t['dose_non_monotone'])

    def test_non_monotone_is_flagged_not_hidden(self):
        r = self._records({'L1_ONLY': [0.01] * 3, 'L2_ONLY': [0.01] * 3,
                           'DOSE_8': [0.01] * 3, 'DOSE_32': [0.9] * 3})
        t = nb.triage(r)
        self.assertEqual(t['dose_k_star'], 124)
        self.assertTrue(t['dose_non_monotone'])
        self.assertEqual(t['length_verdict'], 'BOTH_SUFFICE')

    def test_near_threshold_cells_are_listed(self):
        r = self._records({'L1_ONLY': [0.04, 0.01, 0.01], 'L2_ONLY': [0.9] * 3,
                           'DOSE_8': [0.9] * 3, 'DOSE_32': [0.01] * 3})
        self.assertEqual(nb.triage(r)['near_threshold_cells'], ['L1_ONLY_w0'])


class RestartTests(unittest.TestCase):
    def _record(self, arm, world):
        count, depths = nb.EXPECTED_ANCHORS[arm]
        return {'arm': arm, 'world': world, 'updates': 65536, 'anchors': count,
                'anchor_depths': depths or {'1': count, '2': 0}, 'pool_tasks': 188,
                'pool_examples': 24064, 'scored_tasks': 64, 'terminal_median': 0.01,
                'below_0.05': 60, 'per_task': {}, 'library_sha256': 'x',
                'first_64_draws': [], 'near_threshold': False, 'seconds': 1.0}

    def _seed_run(self, tmp, tamper=False):
        from row.experiments.so1_storage import atomic_json, fingerprint
        root, report = Path(tmp) / 'run', Path(tmp) / 'r.json'
        with mock.patch.object(nb, 'ROOT', root):
            sha = fingerprint(nb.protocol())
        (root / 'cells').mkdir(parents=True)
        for arm in nb.ARMS:
            for w in nb.WORLDS:
                rec = self._record(arm, w)
                saved = {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': rec,
                         'record_sha256': fingerprint(rec), 'finished_utc': 'x'}
                if tamper and arm == 'DOSE_8' and w == 0:
                    saved['record']['terminal_median'] = 0.5
                atomic_json(root / 'cells' / f'{arm}_w{w}.json', saved)
        return root, report

    def test_completed_cells_are_reused_and_nothing_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report = self._seed_run(tmp)
            with mock.patch.object(nb, 'ROOT', root), mock.patch.object(nb, 'OUTPUT', report), \
                 mock.patch.object(nb, 'host_ok', lambda: (99.0, True)), \
                 mock.patch.object(nb, 'run_cell', side_effect=AssertionError('recomputed')):
                nb.run([(a, w) for a in nb.ARMS for w in nb.WORLDS], jobs=1)
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'complete')

    def test_tampered_cell_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report = self._seed_run(tmp, tamper=True)
            with mock.patch.object(nb, 'ROOT', root), mock.patch.object(nb, 'OUTPUT', report), \
                 mock.patch.object(nb, 'host_ok', lambda: (99.0, True)):
                with self.assertRaises(ValueError):
                    nb.run([(a, w) for a in nb.ARMS for w in nb.WORLDS], jobs=1)


if __name__ == '__main__':
    unittest.main()
