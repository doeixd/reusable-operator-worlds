import unittest

import numpy as np
import torch

from row.experiments import d4_stream as d4
from row.experiments import deep_reroute as dr
from row.experiments.audit_j2a_staged_library import enum_route
from row.experiments.audit_so1r_route_only import FrozenLibrary


class DeepRerouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cfg, _, st, plan, canonical = d4.build_stream(30, 0)
        cls.model = d4.planned_model(cfg, plan)
        task = canonical[0]
        cls.x = torch.tensor(task.train_x[:64], dtype=torch.float32)
        cls.y = torch.tensor(task.train_y[:64], dtype=torch.float32)

    def test_split_equals_all_route_values_and_argmin(self):
        lib = FrozenLibrary(self.model)
        for depth in (2, 3, 4):
            lib.steps = depth
            full = lib.all_route_support_mse(self.x, self.y)
            for chunk in (7, 1024):
                split = dr.split_route_mse(lib, self.x, self.y, depth, chunk=chunk)
                self.assertEqual(split.shape, full.shape)
                self.assertLessEqual(float(torch.max(torch.abs(split - full))), 1e-5)
                self.assertEqual(int(torch.argmin(split)), int(torch.argmin(full)))
            self.assertEqual(dr.exhaustive_route(lib, self.x, self.y, depth), [int(r) for r in enum_route(lib, self.x, self.y)])

    def test_depth5_route_is_the_split_argmin_and_its_mse_is_direct(self):
        lib = FrozenLibrary(self.model)
        route = dr.exhaustive_route(lib, self.x, self.y, 5)
        self.assertEqual(len(route), 5)
        lib.steps = 5
        values = dr.split_route_mse(lib, self.x, self.y, 5)
        self.assertEqual(values.numel(), 12 ** 5)
        with torch.no_grad():
            direct = float(torch.mean((lib.hard(self.x, route) - self.y) ** 2))
        self.assertLessEqual(abs(direct - float(values.min())), 1e-5)


if __name__ == '__main__':
    unittest.main()
