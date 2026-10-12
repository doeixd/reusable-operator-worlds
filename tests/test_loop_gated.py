import unittest

import numpy as np
import torch

from row.experiments import b1_online as b1o
from row.experiments import loop_stream as ls
from row.models.branch_gated import BranchGatedLearner
from row.models.loop_gated import LoopLearner


class LoopLearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.world, cls.st, cls.plan, cls.canonical, cls.ltasks, cls.meta = ls.stream(44, 0, 4)

    def build(self, cls, **kw):
        return b1o.build(cls, self.cfg, self.plan, **kw)

    def test_stream(self):
        self.assertEqual(len(self.st), 188 + 4)
        for t in self.ltasks:
            self.assertEqual(self.plan[t.task_id], 1)
            P = self.world.library[self.meta[t.task_id]['P']]
            y, _ = ls.teacher_loop(P, np.array(self.meta[t.task_id]['w']), t.eval_x)
            np.testing.assert_array_equal(y, t.eval_y)

    def test_switch_is_parent(self):
        torch.manual_seed(0)
        a = self.build(BranchGatedLearner)
        torch.manual_seed(0)
        b = self.build(LoopLearner)
        for t in self.st[:5]:
            a.begin_task(t.task_id)
            b.begin_task(t.task_id)
        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(sa.keys(), sb.keys())
        for k in sa:
            self.assertTrue(torch.equal(sa[k], sb[k]), k)
        x = torch.randn(7, self.cfg.world.state_dim)
        for mode in (True, False):
            a.train(mode)
            b.train(mode)
            for t in self.st[:5]:
                self.assertTrue(torch.equal(a(x, t.task_id), b(x, t.task_id)))

    def test_eval_semantics_and_train_limits(self):
        m = self.build(LoopLearner, loop_plan={t.task_id: 'state' for t in self.ltasks})
        t = self.ltasks[0].task_id
        params = m.begin_task(t)
        self.assertEqual(len(params), 3)
        d = self.cfg.world.state_dim
        w = torch.randn(d)
        with torch.no_grad():
            m.halt_w[t].copy_(w)
        x = torch.randn(64, d)
        m.eval()
        coeff = torch.nn.functional.one_hot(torch.argmax(m.task_codes[t], -1), m.operator_slots).float()
        expected, k = [], []
        for row in x:
            z, n = row.unsqueeze(0), 0
            for _ in range(m.max_iterations):
                if not float(z[0].detach() @ w) > 0:
                    break
                z = m._route(z, coeff, 1)
                n += 1
            expected.append(z[0])
            k.append(n)
        with torch.no_grad():
            torch.testing.assert_close(m(x, t), torch.stack(expected))
            self.assertEqual(m.iteration_counts(x, t).tolist(), k)
        m.train()
        with torch.no_grad():
            m.halt_w[t].zero_()
            m.halt_b[t].fill_(-60.0)   # always stop: identity
            torch.testing.assert_close(m(x, t), x)
            m.halt_b[t].fill_(60.0)    # never stop: K soft applications
            z = x
            c = m._route_coefficients(m.task_codes[t])
            for _ in range(m.max_iterations):
                z = m._route(z, c, 1)
            torch.testing.assert_close(m(x, t), z)

    def test_constant_halting_control(self):
        m = self.build(LoopLearner, loop_plan={t.task_id: 'state' for t in self.ltasks}, state_halt=False)
        t = self.ltasks[0].task_id
        params = m.begin_task(t)
        self.assertEqual(len(params), 2)
        self.assertNotIn(t, m.halt_w)
        m.eval()
        k = m.iteration_counts(torch.randn(20, self.cfg.world.state_dim), t)
        self.assertTrue(torch.all(k == k[0]))   # state-independent
        with torch.no_grad():
            m.halt_b[t].copy_(torch.tensor([5.0, 5.0, 5.0, -5.0, 5.0, 5.0]))
        x = torch.randn(9, self.cfg.world.state_dim)
        self.assertTrue(torch.all(m.iteration_counts(x, t) == 3))   # a learned fixed count
        coeff = torch.nn.functional.one_hot(torch.argmax(m.task_codes[t], -1), m.operator_slots).float()
        with torch.no_grad():
            torch.testing.assert_close(m(x, t), m._route(m._route(m._route(x, coeff, 1), coeff, 1), coeff, 1))


    def test_detach_halt_input(self):
        """Detached predicate input: same evaluation; in training the halting still learns but the library gradient
        loses exactly the path through the predicate."""
        t = self.ltasks[0].task_id
        torch.manual_seed(3)
        m = self.build(LoopLearner, loop_plan={t: 'state'})
        m.begin_task(t)
        with torch.no_grad():
            m.halt_w[t].normal_()
            m.task_codes[t].normal_()
        x = torch.randn(9, self.cfg.world.state_dim)

        def run():
            m.eval()
            with torch.no_grad():
                e = m(x, t).clone()
            m.train()
            m.zero_grad()
            m(x, t).pow(2).sum().backward()
            return e, torch.cat([p.grad.flatten() for p in m.library.parameters()]).clone(), m.halt_w[t].grad.clone()

        e0, g0, h0 = run()
        m.detach_halt_input = True
        e1, g1, h1 = run()
        self.assertTrue(torch.equal(e0, e1))
        self.assertFalse(torch.equal(g0, g1))
        self.assertGreater(float(h1.norm()), 0.0)
        self.assertTrue(torch.allclose(h0, h1))   # the halting parameters' own gradient is unchanged

    def test_step_bias_nests_count_and_decay_group(self):
        """step_bias: h = 0 with biases [+, -, ...] is the single route (count 1) in evaluation and nearly so in
        training; the predicate is h.z_j + b_j; halt_decay puts h alone in a weight-decay group."""
        m = self.build(LoopLearner, loop_plan={t.task_id: 'state' for t in self.ltasks}, step_bias=True,
                       halt_decay=0.1, detach_halt_input=True)
        t = self.ltasks[0].task_id
        params = m.begin_task(t)
        self.assertEqual(tuple(m.halt_b[t].shape), (m.max_iterations,))
        self.assertIs(params[1], m.halt_b[t])
        self.assertEqual(params[2], {'params': [m.halt_w[t]], 'weight_decay': 0.1})
        d = self.cfg.world.state_dim
        x = torch.randn(32, d)
        coeff = torch.nn.functional.one_hot(torch.argmax(m.task_codes[t], -1), m.operator_slots).float()
        with torch.no_grad():
            m.halt_b[t].copy_(torch.tensor([60.0] + [-60.0] * (m.max_iterations - 1)))
            m.eval()
            torch.testing.assert_close(m(x, t), m._route(x, coeff, 1))
            self.assertEqual(set(m.iteration_counts(x, t).tolist()), {1})
            m.train()
            soft = m._route(x, m._route_coefficients(m.task_codes[t]), 1)
            torch.testing.assert_close(m(x, t), soft)
            w = torch.randn(d)
            m.halt_w[t].copy_(w)
            z = torch.randn(5, d)
            torch.testing.assert_close(m.halt_logit(z, t, 2), z @ w + m.halt_b[t][2])
        # defaults unchanged: one bias, h in the plain list
        m0 = self.build(LoopLearner, loop_plan={t: 'state'})
        p0 = m0.begin_task(t)
        self.assertEqual(tuple(m0.halt_b[t].shape), (1,))
        self.assertEqual(len(p0), 3)
        self.assertIs(p0[1], m0.halt_w[t])

    def test_halt_init_starts_at_one_step(self):
        """halt_init=c: per-step biases start at [+c, -c, ...]; at c = 60 a fresh candidate is the single route in
        evaluation and in training; the default (None) keeps zeros."""
        t = self.ltasks[0].task_id
        d = self.cfg.world.state_dim
        x = torch.randn(16, d)
        for kw in ({'state_halt': True, 'step_bias': True, 'detach_halt_input': True}, {'state_halt': False}):
            m = self.build(LoopLearner, loop_plan={t: 'state'}, halt_init=60.0, **kw)
            m.begin_task(t)
            K = m.max_iterations
            torch.testing.assert_close(m.halt_b[t].detach(), torch.tensor([60.0] + [-60.0] * (K - 1)))
            with torch.no_grad():
                m.eval()
                coeff = torch.nn.functional.one_hot(torch.argmax(m.task_codes[t], -1), m.operator_slots).float()
                torch.testing.assert_close(m(x, t), m._route(x, coeff, 1))
                m.train()
                torch.testing.assert_close(m(x, t), m._route(x, m._route_coefficients(m.task_codes[t]), 1))
            m0 = self.build(LoopLearner, loop_plan={t: 'state'}, **kw)
            m0.begin_task(t)
            self.assertEqual(float(m0.halt_b[t].detach().abs().sum()), 0.0)
            self.assertEqual(tuple(m0.halt_b[t].shape), (K,))

if __name__ == '__main__':
    unittest.main()
