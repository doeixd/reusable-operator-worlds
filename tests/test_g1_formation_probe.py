import unittest

import numpy as np

from row.experiments import audit_j1c_curriculum as j1c
from row.experiments import g1_formation_probe as g1
from row.experiments import o2_online_reliability as o2


def _records(rw, plain=None):
    plain = plain or {}
    out = {}
    for m in g1.SHARED:
        for w in g1.WORLDS:
            out[g1.key_of('RW_SLEEP', m, w)] = {'terminal_median': rw[(m, w)], 'formed': {}}
            out[g1.key_of('REROUTE_WAKE', m, w)] = {'terminal_median': 1.0}
            out[g1.key_of('PLAIN', m, w)] = {'terminal_median': plain.get((m, w), 1.2)}
    return out


class G1ProbeTests(unittest.TestCase):
    def tearDown(self):
        g1.restore_generator()

    def test_binding_reaches_o2_and_restores(self):
        self.assertIs(o2.stage_setup, j1c.stage_setup)
        g1.use_generator(4.0)
        self.assertIsNot(o2.stage_setup, j1c.stage_setup)
        g1.restore_generator()
        self.assertIs(o2.stage_setup, j1c.stage_setup)

    def test_stage_setup_reduces_bitwise_at_m0(self):
        for stage in (1, 3):
            cfg_a, world_a, upd_a, seq_a = j1c.stage_setup(71, stage, 5000)
            cfg_b, world_b, upd_b, seq_b = g1.general_stage_setup(71, stage, 5000, shared=0.0)
            self.assertEqual(cfg_a, cfg_b)
            self.assertEqual(upd_a, upd_b)
            self.assertEqual(seq_a.entropy, seq_b.entropy)
            self.assertEqual(seq_a.spawn_key, seq_b.spawn_key)
            for a, b in zip(world_a.tasks, world_b.tasks, strict=True):
                self.assertEqual(a.task_id, b.task_id)
                self.assertTrue(np.array_equal(a.train_y, b.train_y) and np.array_equal(a.eval_y, b.eval_y))
        _, w4, _, _ = g1.general_stage_setup(71, 1, 5000, shared=4.0)
        self.assertFalse(np.array_equal(w4.tasks[0].train_y, world_a.tasks[0].train_y))

    def test_reading_labels(self):
        ok = {(m, w): 0.01 for m in g1.SHARED for w in g1.WORLDS}
        self.assertEqual(g1.reading(_records(ok))['label'], 'FORMS')
        bad4 = {**ok, **{(4.0, w): 1.5 for w in g1.WORLDS}}
        self.assertEqual(g1.reading(_records(bad4))['label'], 'FAILS')
        mixed = {**ok, (4.0, 70): 1.5}
        self.assertEqual(g1.reading(_records(mixed))['label'], 'MIXED')
        # a world whose m = 0 anchor fails leaves the denominator
        miss = {**mixed, (0.0, 70): 0.2}
        r = g1.reading(_records(miss))
        self.assertEqual((r['label'], r['denominator']), ('FORMS', 2))
        none = {(m, w): 0.2 for m in g1.SHARED for w in g1.WORLDS}
        self.assertEqual(g1.reading(_records(none))['label'], 'ANCHOR_MISS')
        self.assertFalse(g1.reading(_records(ok, {(4.0, 71): 0.01}))['anchors_needed_at_m4'])
        self.assertFalse(g1.passing(float('nan')))
        self.assertFalse(g1.passing(0.05))


if __name__ == '__main__':
    unittest.main()


class G1ScorerTests(unittest.TestCase):
    def _write(self, root, tamper=None):
        import json
        from pathlib import Path
        from row.experiments import score_g1_formation_probe as sc
        from row.experiments.so1_storage import digest, fingerprint
        root = Path(root)
        (root / 'cells').mkdir()
        formed = {'d1_ari': 0.9, 'identifiability_median': 2.0, 'near_tie_tasks': 0,
                  'formed_runner_up_nmse_median': 0.1, 'winner_query_nmse_median': 0.02,
                  'winner_query_below_0.05': 60}
        cells = {}
        for arm in sc.ARMS:
            for m in sc.SHARED:
                for w in sc.WORLDS:
                    rec = {'terminal_median': 0.01 if arm == 'RW_SLEEP' else 1.2, 'formed': formed,
                           'd1_ari_wake': 0.5}
                    cells[sc.key(arm, m, w)] = rec
                    saved = {'stamp': {'protocol_sha256': 'abc'}, 'complete': True, 'record': rec,
                             'record_sha256': fingerprint(rec)}
                    if tamper == sc.key(arm, m, w):
                        saved['record'] = dict(rec, terminal_median=0.5)
                    (root / 'cells' / f'{sc.key(arm, m, w)}.json').write_text(json.dumps(saved))
        rep = {'protocol': {'input_sha256': {sc.PLAN.as_posix(): digest(sc.PLAN)}}, 'protocol_sha256': 'abc',
               'gates': {'all_pass': True, 'E1': {'passes': True}, 'E2': {'passes': True}},
               'cells': cells, 'reading': {'label': 'FORMS'}}
        (root / 'report.json').write_text(json.dumps(rep))
        return root

    def test_recomputes_and_refuses(self):
        import tempfile
        from row.experiments import score_g1_formation_probe as sc
        with tempfile.TemporaryDirectory() as d:
            root = self._write(d)
            out = sc.score(root / 'report.json', root / 'cells')
            self.assertEqual((out['label'], out['m4_passes'], out['runner_agrees']), ('FORMS', 3, True))
        with tempfile.TemporaryDirectory() as d:
            root = self._write(d, tamper=sc.key('RW_SLEEP', 4.0, 71))
            with self.assertRaises(ValueError):
                sc.score(root / 'report.json', root / 'cells')
