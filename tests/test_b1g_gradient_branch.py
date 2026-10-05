import unittest

import torch

from row.experiments import b1g_gradient_branch as g
from row.experiments import dn_stream as dn
from row.experiments.audit_so1r_route_only import FrozenLibrary


class B1gTests(unittest.TestCase):
    def test_switch_recovers_single_route_bitwise(self):
        cfg, _, st, plan, _ = dn.build_stream(4, 30, 0)
        lib = FrozenLibrary(dn.planned_model(4, cfg, plan))
        torch.manual_seed(0)
        x, c1, c2 = torch.randn(32, 16), torch.randn(3, 12), torch.randn(3, 12)
        p = torch.nn.functional.one_hot(torch.argmax(c1, -1), 12).float()
        with torch.no_grad():
            pinned = g.gated_predict(lib, x, c1, c2, torch.zeros(16), torch.tensor([50.0]), 0.1, False, hard=True)
            self.assertTrue(torch.equal(pinned, g.soft_forward(lib, x, p)))

    def test_label_partition(self):
        self.assertEqual({g.label(k, 192) for k in range(193)}, {'GRADIENT_FINDS', 'PARTIAL', 'GRADIENT_FAILS'})
        self.assertEqual((g.label(154, 192), g.label(153, 192), g.label(95, 192)),
                         ('GRADIENT_FINDS', 'PARTIAL', 'GRADIENT_FAILS'))


if __name__ == '__main__':
    unittest.main()
