import unittest

import numpy as np
import torch

from row.experiments import census_so2_interference as census
from row.experiments.audit_j1c_curriculum import stage_setup
from row.experiments.audit_so1_budget_bracket import build_fast


class TransplantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg, world, _, _ = stage_setup(1, 2, 5000)
        self.tasks = list(world.tasks[:3])
        self.model = build_fast(self.cfg)
        for task in self.tasks:
            code = self.model.begin_task(task.task_id)
            with torch.no_grad():
                code.normal_()  # distinct routes, so the comparison is not trivially zero

    def test_own_library_reproduces_forward_bitwise(self):
        host = census.transplant(self.model, self.model, self.cfg)
        self.model.eval()
        host.eval()
        x = torch.tensor(np.random.default_rng(3).normal(size=(8, self.cfg.world.state_dim)), dtype=torch.float32)
        with torch.no_grad():
            for task in self.tasks:
                self.assertTrue(torch.equal(self.model(x, task.task_id), host(x, task.task_id)))

    def test_other_library_changes_forward(self):
        # Same seed (a different seed changes which slots carry d vs d-1 reflections,
        # so its library would not even load); perturb the library instead.
        other = build_fast(self.cfg)
        with torch.no_grad():
            for parameter in other.library.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        host = census.transplant(other, self.model, self.cfg)
        self.model.eval()
        host.eval()
        x = torch.tensor(np.random.default_rng(3).normal(size=(8, self.cfg.world.state_dim)), dtype=torch.float32)
        with torch.no_grad():
            self.assertFalse(torch.equal(self.model(x, self.tasks[0].task_id), host(x, self.tasks[0].task_id)))

    def test_spearman_orders(self):
        self.assertAlmostEqual(census.spearman([0, 1, 2, 3], [1, 2, 3, 9]), 1.0)
        self.assertAlmostEqual(census.spearman([0, 1, 2, 3], [9, 3, 2, 1]), -1.0)


if __name__ == "__main__":
    unittest.main()
