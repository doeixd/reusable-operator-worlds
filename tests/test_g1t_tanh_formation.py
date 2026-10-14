import unittest

import numpy as np
import torch

from row.config import load_config
from row.experiments import audit_j1c_curriculum as j1c
from row.experiments import audit_so2_online_gate as so2
from row.experiments import g1t_tanh_formation as g1t
from row.experiments import o2_online_reliability as o2
from row.experiments import o8_instream_reroute as o8
from row.form_world import form_library
from row.models.tanh_form import FrozenTanhLibrary, TanhFormLearner, build_tanh_learner, tanh_gain_scale


def _records(sleep, ari=None, plain=1.9):
    ari = ari or {}
    out = {}
    for w in g1t.WORLDS:
        out[g1t.key_of('RW_SLEEP', w)] = {'terminal_median': sleep[w], 'formed': {'d1_ari': ari.get(w, 0.99)},
                                          'formed_s1': {'median': {'1': 0.9}}}
        out[g1t.key_of('REROUTE_WAKE', w)] = {'terminal_median': 1.0}
        out[g1t.key_of('SHUFFLED', w)] = {'terminal_median': 1.0}
        out[g1t.key_of('SLEEP', w)] = {'terminal_median': 0.01}
        out[g1t.key_of('STAGED', w)] = {'terminal_median': 0.01}
        out[g1t.key_of('PLAIN', w)] = {'terminal_median': plain}
    return out


class TanhLearnerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config('configs/v1.yaml')

    def test_init_depends_on_seed_only_and_keeps_variance(self):
        a, b = build_tanh_learner(self.cfg), build_tanh_learner(self.cfg)
        for pa, pb in zip(a.library.parameters(), b.library.parameters(), strict=True):
            self.assertTrue(torch.equal(pa, pb))
        op = a.library[0]
        self.assertTrue(torch.allclose(op.A @ op.A.T, torch.eye(16), atol=1e-5))   # gain 1, orthogonal
        self.assertAlmostEqual(float(op.alpha), tanh_gain_scale(1.0), places=6)
        z = FrozenTanhLibrary(a).candidates(torch.randn(4000, 16, generator=torch.Generator().manual_seed(1)))
        self.assertLess(abs(float(z.var()) - 1.0), 0.05)
        dets = {round(float(torch.linalg.det(op.rotation.matrix()))) for op in a.library}
        self.assertEqual(dets, {-1, 1})   # slots cover both components of O(d)

    def test_planned_depth_and_search_view_bitwise(self):
        m = build_tanh_learner(self.cfg, plan={'one': 1, 'two': 2}, task_steps=3)
        for t in ('one', 'two', 'three'):
            m.begin_task(t)
            with torch.no_grad():
                m.task_codes[t].normal_()
        m.eval()
        x = torch.randn(32, 16)
        for t, depth in (('one', 1), ('two', 2), ('three', 3)):
            route = m.hard_routes()[t]
            self.assertEqual(len(route), depth)
            lib = FrozenTanhLibrary(m)
            lib.steps = depth
            self.assertTrue(torch.equal(lib.hard(x, route), m(x, t)))

    def test_scale_in_no_decay_group_and_state_roundtrip(self):
        m = build_tanh_learner(self.cfg)
        names = [n for n, _ in m.named_parameters() if n.endswith('.alpha')]
        self.assertEqual(len(names), 12)
        from row.experiments import learned_lifetime as ll
        groups = ll._shared_optimizer(m, 0.001, 1e-4).param_groups
        no_decay = [g for g in groups if g['weight_decay'] == 0.0]
        self.assertEqual(sum(len(g['params']) for g in no_decay), 12)
        m.begin_task('t')
        fresh = build_tanh_learner(self.cfg)
        fresh.begin_task('t')
        fresh.load_state_dict(m.state_dict(), strict=True)

    def test_slot_can_fit_teacher_primitive(self):
        """Non-vacuity: the slot family contains the teacher form, so gradient fitting must get far below 1."""
        teacher = form_library('tanh', 2.0, 70, 6, 16)[0]
        rng = np.random.default_rng(0)
        x = rng.normal(size=(256, 16))
        xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(teacher(x), dtype=torch.float32)
        op = build_tanh_learner(self.cfg).library[0]
        opt = torch.optim.Adam(op.parameters(), lr=0.01)
        for _ in range(400):
            opt.zero_grad()
            loss = torch.mean((op(xt) - yt) ** 2) / torch.var(yt)
            loss.backward()
            opt.step()
        self.assertLess(float(loss), 0.05)


class G1TRunnerTests(unittest.TestCase):
    def tearDown(self):
        g1t.restore()

    def test_binding_reaches_pipeline_and_restores(self):
        self.assertIs(o2.stage_setup, j1c.stage_setup)
        g1t.use_form()
        self.assertIs(o2.stage_setup, g1t.form_stage_setup)
        self.assertIsInstance(o2.planned_model(load_config('configs/v1.yaml'), {}), TanhFormLearner)
        self.assertIsInstance(so2.build_fast(o2.stage_setup(70, 2, 5000)[0]), TanhFormLearner)
        self.assertIs(o8.FrozenLibrary, FrozenTanhLibrary)
        g1t.restore()
        self.assertIs(o2.stage_setup, j1c.stage_setup)
        self.assertIsNot(o8.FrozenLibrary, FrozenTanhLibrary)

    def test_form_stage_matches_committed_inputs_not_targets(self):
        for stage in (1, 3):
            cfg_a, world_a, upd_a, seq_a = j1c.stage_setup(71, stage, 5000)
            cfg_b, world_b, upd_b, seq_b = g1t.form_stage_setup(71, stage, 5000)
            self.assertEqual(cfg_a, cfg_b)
            self.assertEqual(upd_a, upd_b)
            self.assertEqual(seq_a.entropy, seq_b.entropy)
            for a, b in zip(world_a.tasks, world_b.tasks, strict=True):
                self.assertEqual((a.task_id, a.program), (b.task_id, b.program))
                self.assertTrue(np.array_equal(a.train_x, b.train_x) and np.array_equal(a.eval_x, b.eval_x))
                self.assertFalse(np.array_equal(a.train_y, b.train_y))

    def test_reading_labels(self):
        rot = {w: 0.01 for w in g1t.WORLDS}
        ok = {w: 0.01 for w in g1t.WORLDS}
        self.assertEqual(g1t.reading(_records(ok), rot)['label'], 'FORMS')
        self.assertEqual(g1t.reading(_records({w: 1.5 for w in g1t.WORLDS}), rot)['label'], 'FAILS')
        self.assertEqual(g1t.reading(_records({**ok, 70: 1.5}), rot)['label'], 'MIXED')
        # an identity-free pass is an impostor and does not form
        r = g1t.reading(_records(ok, ari={70: 0.45}), rot)
        self.assertEqual((r['label'], r['impostor_worlds']), ('MIXED', [70]))
        r = g1t.reading(_records(ok, ari={w: 0.45 for w in g1t.WORLDS}), rot)
        self.assertEqual(r['label'], 'FAILS')
        # a world whose rotated reference failed leaves the denominator
        r = g1t.reading(_records({**ok, 70: 1.5}), {**rot, 70: 0.3})
        self.assertEqual((r['label'], r['denominator']), ('FORMS', 2))
        self.assertEqual(g1t.reading(_records(ok), {w: 0.3 for w in g1t.WORLDS})['label'], 'ANCHOR_MISS')
        self.assertFalse(g1t.passing(float('nan')))
        self.assertEqual(g1t.reading(_records(ok, plain=0.01), rot)['plain_passes'], 3)

    def test_rotated_reference_is_g1_m0(self):
        ref = g1t.rotated_reference()
        self.assertEqual(sorted(ref), list(g1t.WORLDS))
        self.assertTrue(all(v < 0.05 for v in ref.values()))


if __name__ == '__main__':
    unittest.main()
