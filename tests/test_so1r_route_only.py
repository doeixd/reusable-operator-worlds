import unittest

import torch

from row.experiments import audit_so1r_route_only as so1r


class FakeModel:
    def __init__(self, seed=0):
        from row.config import load_config
        from row.experiments.audit_rotated_g5r_interference import world_config
        from row.experiments.audit_so1_budget_bracket import build_fast
        self.inner = build_fast(world_config(load_config("configs/v1.yaml"), seed))
        self.library = self.inner.library
        self.task_steps = self.inner.task_steps


def summary(oracle=.005, enum=.005, opt=.005, random=1.5, eligible=None, vacuous=False):
    med = {"oracle": oracle, "enum": enum, "opt": opt, "random": random, "k0": 1.7}
    return {"medians": med, "passes": {k: v <= .05 for k, v in med.items()},
            "eligible": oracle <= .05 if eligible is None else eligible,
            "oracle_bitwise": not vacuous, "enum_support_le_oracle": True, "opt_codes_moved": True,
            "opt_median_support_drop": .9, "k0_differs_from_opt": True}


class SO1RTests(unittest.TestCase):
    def test_enumeration_order_matches_hard_forward(self):
        torch.manual_seed(0)
        library = so1r.FrozenLibrary(FakeModel())
        x, y = torch.randn(8, 16), torch.randn(8, 16)
        support = library.all_route_support_mse(x, y)
        self.assertEqual(support.numel(), 12 ** 3)
        for route in ([0, 0, 0], [3, 7, 11], [11, 0, 5]):
            index = (route[0] * 12 + route[1]) * 12 + route[2]
            self.assertEqual(so1r.unflatten(index, 12, 3), route)
            direct = torch.mean((library.hard(x, route) - y) ** 2)
            self.assertTrue(torch.allclose(support[index], direct, rtol=1e-5, atol=1e-7))

    def test_hard_forward_matches_model_bitwise(self):
        model = FakeModel()
        code = model.inner.begin_task("t")
        with torch.no_grad():
            code.fill_(-100.0)
            code[0, 4] = code[1, 9] = code[2, 2] = 100.0
        model.inner.eval()
        x = torch.randn(5, 16)
        with torch.no_grad():
            self.assertTrue(torch.equal(model.inner(x, "t"), so1r.FrozenLibrary(model).hard(x, [4, 9, 2])))

    def test_classification_ladder(self):
        base = {f"L{i}": summary() for i in range(4)}
        base["W0a"] = summary(oracle=.7, enum=.7, opt=.9)
        self.assertEqual(so1r.classify(base), "ROUTES_RECOVERABLE")
        self.assertEqual(so1r.classify(dict(base, L1=summary(opt=.9))), "SEARCH_ONLY")
        self.assertEqual(so1r.classify(dict(base, L2=summary(enum=.3, opt=.3))), "NOT_IDENTIFIABLE")
        self.assertEqual(so1r.classify(dict(base, L0=summary(random=.01))), "HARNESS_FAILED")
        self.assertEqual(so1r.classify(dict(base, L3=summary(vacuous=True))), "HARNESS_FAILED")
        self.assertEqual(so1r.classify({"L0": summary(), "W": summary(oracle=.7)}), "HARNESS_FAILED")


if __name__ == "__main__":
    unittest.main()
