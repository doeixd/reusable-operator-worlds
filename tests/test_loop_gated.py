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


if __name__ == '__main__':
    unittest.main()
