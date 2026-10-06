import unittest

import torch

from row.experiments import b1_online as b1o
from row.experiments import branch_stream as bs
from row.experiments import o2_online_reliability as o2
from row.models.branch_gated import BranchGatedLearner
from row.models.online_variable_depth import PlannedDepthRotatedLearner


class B1OnlineTests(unittest.TestCase):
    def test_empty_branch_plan_is_the_straight_line_learner(self):
        cfg, world, st, plan, canonical = o2.build_stream('SHUFFLED', 44, 0)
        base = b1o.build(PlannedDepthRotatedLearner, cfg, plan)
        gated = b1o.build(BranchGatedLearner, cfg, plan, branch_plan={}, branch_seed=1)
        self.assertEqual([n for n, _ in base.named_parameters()], [n for n, _ in gated.named_parameters()])
        for t in st[:5]:
            self.assertIsInstance(gated.begin_task(t.task_id), torch.nn.Parameter)
            base.begin_task(t.task_id)
        x = torch.randn(8, cfg.world.state_dim)
        base.eval(); gated.eval()
        for t in st[:5]:
            self.assertTrue(torch.equal(base(x, t.task_id), gated(x, t.task_id)))

    def test_branch_task_returns_its_parameters_and_hard_gate_routes(self):
        cfg, world, st, plan, canonical, btasks, meta = bs.stream(44, 0, 4)
        m = b1o.build(BranchGatedLearner, cfg, plan, branch_plan={t.task_id: 'input' for t in btasks}, branch_seed=1)
        params = m.begin_task(btasks[0].task_id)
        self.assertEqual(len(params), 4)
        m.gate_b[btasks[0].task_id].data.fill_(100.0)   # gate pinned to route 1
        m.eval()
        x = torch.randn(8, cfg.world.state_dim)
        route1 = m._route(x, m._route_coefficients(m.task_codes[btasks[0].task_id]), m.depth_of(btasks[0].task_id))
        self.assertTrue(torch.equal(m(x, btasks[0].task_id), route1))

    def test_stream_and_labels(self):
        cfg, world, st, plan, canonical, btasks, meta = bs.stream(44, 0, b1o.N_BRANCH)
        self.assertEqual((len(st), len(btasks), len(canonical)), (236, 48, 64))
        self.assertEqual({len(t.program.primitive_ids) for t in btasks}, {2})
        self.assertEqual((b1o.label(8, 7, 9), b1o.label(7, 9, 9), b1o.label(9, 6, 9)),
                         ('ONLINE_BRANCHES', 'NO_BRANCH_GAIN', 'FORMATION_BROKEN'))


if __name__ == '__main__':
    unittest.main()
