import unittest

from row.experiments import n1_anchor_supply as n1


class ConstructionTests(unittest.TestCase):
    """The arms must be built as Amendments 1 and 3 registered them."""

    def test_interleaved_and_sham_are_pool_matched(self):
        _, _, canon_i, pool_i = n1.arm_tasks('INTERLEAVED', 0)
        _, _, canon_s, pool_s = n1.arm_tasks('SHAM', 0)
        self.assertEqual(len(pool_i), 188)
        self.assertEqual(len(pool_s), 188)
        self.assertEqual(sum(len(t.train_x) for t in pool_i),
                         sum(len(t.train_x) for t in pool_s))
        self.assertEqual(len(canon_i), 64)
        self.assertEqual(len(canon_s), 64)

    def test_interleaved_carries_the_anchors_and_sham_does_not(self):
        _, _, _, pool_i = n1.arm_tasks('INTERLEAVED', 0)
        _, _, _, pool_s = n1.arm_tasks('SHAM', 0)
        self.assertEqual({d: sum(1 for t in pool_i if t.depth == d) for d in (1, 2, 3)},
                         {1: 60, 2: 64, 3: 64})
        self.assertEqual({d: sum(1 for t in pool_s if t.depth == d) for d in (1, 2, 3)},
                         {1: 0, 2: 0, 3: 188})

    def test_sham_extra_programs_are_distinct_and_not_canonical(self):
        """124 distinct length-3 programs, none of them a canonical task."""
        cfg, world, canonical, pool = n1.arm_tasks('SHAM', 0)
        canonical_programs = {t.program for t in canonical}
        extra = [t for t in pool if t.task_id.startswith('sham_')]
        self.assertEqual(len(extra), 124)
        self.assertEqual(len({t.program for t in extra}), 124)
        self.assertEqual(set(t.program for t in extra) & canonical_programs, set())

    def test_none_is_the_published_sixty_four_task_floor(self):
        _, _, canonical, pool = n1.arm_tasks('NONE', 0)
        self.assertEqual(len(pool), 64)
        self.assertEqual([t.task_id for t in pool], [t.task_id for t in canonical])

    def test_every_arm_is_scored_on_the_same_canonical_tasks(self):
        reference = [t.task_id for t in n1.arm_tasks('NONE', 0)[2]]
        for arm in ('INTERLEAVED', 'SHAM'):
            self.assertEqual([t.task_id for t in n1.arm_tasks(arm, 0)[2]], reference)

    def test_budget_is_the_registered_invariant(self):
        self.assertEqual(n1.UPDATES, 65536)
        self.assertEqual(n1.UPDATES, sum(u for _, _, u, _, _ in n1.STAGES))


class ValidationTests(unittest.TestCase):
    def _record(self, **kw):
        base = {'arm': 'INTERLEAVED', 'world': 0, 'updates': 65536, 'pool_tasks': 188,
                'pool_examples': 24064, 'scored_tasks': 64,
                'depth_histogram': {'1': 60, '2': 64, '3': 64},
                'terminal_median': 0.01, 'below_0.05': 60, 'per_task': {},
                'library_sha256': 'x', 'first_64_draws': [], 'seconds': 1.0}
        base.update(kw)
        return base

    def test_wrong_pool_size_is_refused(self):
        with self.assertRaises(ValueError):
            n1.validate_cell(self._record(pool_tasks=120))

    def test_sham_with_anchors_is_refused(self):
        with self.assertRaises(ValueError):
            n1.validate_cell(self._record(arm='SHAM',
                                          depth_histogram={'1': 60, '2': 64, '3': 64}))

    def test_scoring_on_the_wrong_set_is_refused(self):
        with self.assertRaises(ValueError):
            n1.validate_cell(self._record(scored_tasks=188))


class TriageTests(unittest.TestCase):
    """The registered rule, including Amendment 1's uninterpretable guard."""

    def _cells(self, interleaved, sham):
        out = {}
        for world in (0, 1, 2):
            out[f'INTERLEAVED_w{world}'] = {'arm': 'INTERLEAVED', 'world': world,
                                            'terminal_median': interleaved[world]}
            out[f'SHAM_w{world}'] = {'arm': 'SHAM', 'world': world,
                                     'terminal_median': sham[world]}
        return out

    def test_suffice_needs_two_of_three_worlds(self):
        t = n1.triage(self._cells([0.01, 0.02, 0.9], [0.95, 0.95, 0.95]))
        self.assertEqual(t['verdict'], 'ANCHORS_SUFFICE')
        t = n1.triage(self._cells([0.01, 0.9, 0.9], [0.95, 0.95, 0.95]))
        self.assertNotEqual(t['verdict'], 'ANCHORS_SUFFICE')

    def test_partial_fires_on_the_ratio(self):
        t = n1.triage(self._cells([0.07, 0.08, 0.09], [0.95, 0.95, 0.95]))
        self.assertEqual(t['verdict'], 'ANCHORS_PARTIAL')

    def test_low_sham_is_uninterpretable_not_insufficient(self):
        """A ratio test whose denominator collapses gives a false negative."""
        t = n1.triage(self._cells([0.30, 0.31, 0.32], [0.33, 0.34, 0.35]))
        self.assertEqual(t['verdict'], 'PARTIAL_UNINTERPRETABLE')
        self.assertEqual(t['uninterpretable_worlds'], [0, 1, 2])

    def test_null_reads_insufficient(self):
        t = n1.triage(self._cells([0.94, 0.95, 0.93], [0.95, 0.94, 0.96]))
        self.assertEqual(t['verdict'], 'ANCHORS_INSUFFICIENT')



class RestartTests(unittest.TestCase):
    """Relaunch must reuse a completed cell and must refuse a tampered one."""

    def _drive(self, tmp, calls):
        import json
        from pathlib import Path
        from unittest import mock
        root, report = Path(tmp) / 'run', Path(tmp) / 'report.json'
        record = {'arm': 'NONE', 'world': 0, 'updates': 65536, 'pool_tasks': 64,
                  'pool_examples': 8192, 'scored_tasks': 64,
                  'depth_histogram': {'1': 0, '2': 0, '3': 64},
                  'terminal_median': 0.93, 'below_0.05': 0, 'per_task': {},
                  'library_sha256': 'x', 'first_64_draws': [], 'seconds': 1.0}

        def fake_cell(arm, world):
            calls.append((arm, world))
            return dict(record, arm=arm, world=world)

        with mock.patch.object(n1, 'ROOT', root), mock.patch.object(n1, 'OUTPUT', report), \
             mock.patch.object(n1, 'run_cell', fake_cell), \
             mock.patch.object(n1, 'host_ok', lambda: (99.0, True)), \
             mock.patch.object(n1, 'WORLDS', (0,)), mock.patch.object(n1, 'ARMS', ('NONE',)):
            n1.run([('NONE', 0)])
        return root, report

    def test_second_launch_reuses_the_completed_cell(self):
        import tempfile
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            self._drive(tmp, calls)
            self.assertEqual(len(calls), 1)
            self._drive(tmp, calls)          # relaunch, same protocol
            self.assertEqual(len(calls), 1, 'the completed cell was recomputed')

    def test_tampered_cell_is_refused(self):
        import json, tempfile
        from pathlib import Path
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._drive(tmp, calls)
            path = root / 'cells' / 'NONE_w0.json'
            saved = json.loads(path.read_text())
            saved['record']['terminal_median'] = 0.0001      # rewrite the result
            path.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                self._drive(tmp, calls)

if __name__ == '__main__':
    unittest.main()
