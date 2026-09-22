import unittest

import torch

from row.config import load_config
from row.experiments.audit_rotated_g5r_interference import world_config
from row.experiments.audit_so1_budget_bracket import build_fast
from row.models.variable_depth import VariableDepthRotatedLearner


def _build(depth=3):
    cfg = world_config(load_config('configs/v1.yaml'), 0)
    fast = build_fast(cfg)
    variable = VariableDepthRotatedLearner(
        d=cfg.world.state_dim,
        operator_slots=fast.operator_slots,
        operator_rank=fast.library[0].U.shape[-1],
        task_steps=depth,
        alpha=0.2,
        initial_temperature=fast.initial_temperature,
        final_temperature=fast.final_temperature,
        seed=cfg.discrete_model.seed,
    )
    variable.load_state_dict(fast.state_dict(), strict=False)
    return cfg, fast, variable


class BitwiseReductionTests(unittest.TestCase):
    """The reduction must be bitwise BY CONSTRUCTION, not by tolerance."""

    def test_full_depth_task_matches_the_committed_learner_bitwise(self):
        cfg, fast, variable = _build()
        x = torch.randn(9, cfg.world.state_dim, generator=torch.Generator().manual_seed(2))
        fast.begin_task('t')
        variable.begin_task('t', depth=3)
        with torch.no_grad():
            torch.nn.init.normal_(fast.task_codes['t'], std=0.7)
            variable.task_codes['t'].copy_(fast.task_codes['t'])
        for module in (fast, variable):
            module.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(fast.forward(x, 't'), variable.forward(x, 't')))

    def test_unset_depth_defaults_to_task_steps_and_is_bitwise(self):
        cfg, fast, variable = _build()
        x = torch.randn(5, cfg.world.state_dim, generator=torch.Generator().manual_seed(4))
        fast.begin_task('u')
        variable.begin_task('u')          # no depth given
        with torch.no_grad():
            torch.nn.init.normal_(fast.task_codes['u'], std=0.5)
            variable.task_codes['u'].copy_(fast.task_codes['u'])
        fast.eval(); variable.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(fast.forward(x, 'u'), variable.forward(x, 'u')))
        self.assertEqual(variable.depth_of('u'), 3)
        self.assertTrue(variable.uniform_depth())

    def test_depth_one_applies_exactly_one_step(self):
        """A depth-1 task must equal the full model truncated after one step."""
        cfg, fast, variable = _build()
        x = torch.randn(7, cfg.world.state_dim, generator=torch.Generator().manual_seed(6))
        variable.begin_task('short', depth=1)
        with torch.no_grad():
            torch.nn.init.normal_(variable.task_codes['short'], std=0.6)
        variable.eval()
        with torch.no_grad():
            once = variable.forward(x, 'short')
        # Rebuild as a genuine one-step model with the same first-row code.
        _, _, one_step = _build(depth=1)
        one_step.load_state_dict(variable.state_dict(), strict=False)
        one_step.begin_task('short')
        with torch.no_grad():
            one_step.task_codes['short'].copy_(variable.task_codes['short'][:1])
        one_step.eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(once, one_step.forward(x, 'short'), atol=1e-6))

    def test_shallow_task_leaves_trailing_code_rows_untouched(self):
        cfg, _, variable = _build()
        x = torch.randn(4, cfg.world.state_dim, generator=torch.Generator().manual_seed(8))
        y = torch.randn(4, cfg.world.state_dim, generator=torch.Generator().manual_seed(9))
        code = variable.begin_task('s', depth=1)
        variable.train()
        loss = torch.nn.functional.mse_loss(variable.forward(x, 's'), y)
        loss.backward()
        self.assertIsNotNone(code.grad)
        self.assertGreater(float(code.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(code.grad[1:].abs().sum()), 0.0)

    def test_depth_outside_range_is_refused(self):
        _, _, variable = _build()
        with self.assertRaises(ValueError):
            variable.begin_task('bad', depth=0)
        with self.assertRaises(ValueError):
            variable.begin_task('worse', depth=4)

    def test_hard_routes_are_truncated_to_their_own_depth(self):
        _, _, variable = _build()
        variable.begin_task('a', depth=1)
        variable.begin_task('b', depth=2)
        variable.begin_task('c', depth=3)
        routes = variable.hard_routes()
        self.assertEqual(len(routes['a']), 1)
        self.assertEqual(len(routes['b']), 2)
        self.assertEqual(len(routes['c']), 3)
        self.assertFalse(variable.uniform_depth())


if __name__ == '__main__':
    unittest.main()
