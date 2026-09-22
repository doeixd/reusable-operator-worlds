import unittest

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_rotated_g5r_interference import world_config
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.audit_so1r_route_only import unflatten
from row.experiments.l0d_depth4_execution_gate import DepthLibrary
from row.experiments import sg0_headroom_gate as sg0


class SG0EstimandTests(unittest.TestCase):
    """The plan's equivalence checks, written before launch."""

    @classmethod
    def setUpClass(cls):
        cfg = world_config(load_config('configs/v1.yaml'), 0)
        cls.library = DepthLibrary(build_fast(cfg))
        generator = torch.Generator().manual_seed(11)
        cls.x = torch.randn(17, 16, generator=generator)
        cls.y = torch.randn(17, 16, generator=generator)

    def test_full_reduction_reproduces_all_route_support_mse(self):
        """Reducing the matrix over every example must equal the existing evaluator."""
        errors = sg0.per_example_squared_error(self.library, self.x, self.y, 3)
        expected = self.library.all_route_support_mse_depth(self.x, self.y, 3)
        self.assertEqual(errors.shape, (12 ** 3, 17))
        self.assertTrue(torch.allclose(errors.mean(dim=1), expected, rtol=1e-6, atol=1e-6))
        self.assertEqual(int(torch.argmin(errors.mean(dim=1))), int(torch.argmin(expected)))

    def test_matrix_row_matches_direct_hard_execution(self):
        """A single route's error from the matrix must equal hard() on the same route."""
        errors = sg0.per_example_squared_error(self.library, self.x, self.y, 3)
        index = 777
        route = unflatten(index, self.library.slots, 3)
        with torch.no_grad():
            direct = torch.mean((self.library.hard(self.x, route) - self.y) ** 2)
        self.assertAlmostEqual(float(errors[index].mean()), float(direct), places=6)

    def test_subset_mean_equals_recomputation_on_that_subset(self):
        """The subset-mean shortcut must equal re-executing on the subset alone."""
        columns = torch.tensor([0, 3, 4, 9, 15], dtype=torch.long)
        errors = sg0.per_example_squared_error(self.library, self.x, self.y, 3)
        subset = sg0.per_example_squared_error(self.library, self.x[columns], self.y[columns], 3)
        self.assertTrue(torch.allclose(errors[:, columns].mean(dim=1), subset.mean(dim=1),
                                       rtol=1e-6, atol=1e-6))

    def test_argmin_route_breaks_ties_to_lowest_index(self):
        """The registered tie rule, checked on a deliberately tied input."""
        tied = torch.ones(5, 4)
        index, values = sg0.argmin_route(tied)
        self.assertEqual(index, 0)
        self.assertEqual(values.shape, (5,))

    def test_identical_routes_give_exactly_zero_regret(self):
        """Validation must reject a row claiming identical routes and nonzero regret."""
        row = {'task': 0, 'program': [0, 1, 2], 'r_hat': [0, 0, 0], 'r_star': [0, 0, 0],
               'r_hat_index': 0, 'r_star_index': 0, 'routes_differ': False,
               'nmse_hat_qb': 0.5, 'nmse_star_qb': 0.5, 'regret': 0.1, 'null_floor': 0.0,
               'regret_above_floor': True, 'identifiability': 1.0, 'best_support_mse': 1.0,
               'near_tie_size': 1, 'near_tie_disagreement': 0.0, 'enum_seconds': 0.1}
        record = {'rows': [dict(row, task=i, program=[i, 1, 2]) for i in range(sg0.TASKS)]}
        with self.assertRaises(ValueError):
            sg0.validate_cell(record)

    def test_dry_run_cell_is_the_registered_decisive_cell(self):
        self.assertEqual(sg0.DRY_RUN_CELL, ('STAGED5000', 0, 3, 128))
        self.assertIn(sg0.DRY_RUN_CELL[0], sg0.STAGED)

    def test_full_grid_has_the_registered_denominators(self):
        """36 staged and 36 control cells at depths 3-4, per the plan."""
        cells = sg0.grid(depths=(3, 4), supports=(128, 8, 2), worlds=(0, 1, 2))
        staged = [c for c in cells if c[0] in sg0.STAGED]
        control = [c for c in cells if c[0] in sg0.CONTROL]
        self.assertEqual(len(staged), 36)
        self.assertEqual(len(control), 36)
        self.assertEqual(len(set(cells)), len(cells))

    def test_partial_run_refuses_to_report_a_triage(self):
        """A dry run must not present itself as the registered 36-cell triage."""
        records = {'a': {'staged': True, 'cell_counts_toward_k': True, 'median_regret': 1.0,
                         'routes_differ_count': 3}}
        summary = sg0.summarize(records, 'dry-run')
        self.assertEqual(summary['triage'], 'NOT APPLICABLE')
        self.assertEqual(summary['staged_cells'], 1)


class SG0QuantityTests(unittest.TestCase):
    """The plan lists quantities the runner must record; diff them against the code."""

    REQUIRED = {'regret', 'null_floor', 'identifiability', 'near_tie_disagreement',
                'near_tie_size', 'nmse_hat_qb', 'nmse_star_qb', 'enum_seconds',
                'r_hat', 'r_star', 'program'}

    def test_every_planned_quantity_is_produced(self):
        import inspect
        source = inspect.getsource(sg0.measure_cell)
        missing = {q for q in self.REQUIRED if f"'{q}'" not in source}
        self.assertEqual(missing, set(), f'runner does not record: {sorted(missing)}')

    def test_cell_record_carries_library_hash_and_seeds(self):
        import inspect
        protocol_source = inspect.getsource(sg0.protocol)
        for field in ('split_seed', 'bootstrap_seed', 'bootstrap_draws', 'near_tie_eps', 'tie_rule'):
            self.assertIn(field, protocol_source)
        self.assertIn('library_sha256', inspect.getsource(sg0.measure_cell))



class SG0LabelTests(unittest.TestCase):
    """A partial grid must never be labelled as the registered full grid."""

    def test_only_the_registered_grid_is_full(self):
        self.assertTrue(sg0.is_full_grid((3, 4), (128, 8, 2), (0, 1, 2)))
        self.assertFalse(sg0.is_full_grid((3,), (128, 8, 2), (0, 1, 2)))
        self.assertFalse(sg0.is_full_grid((3, 4), (128,), (0, 1, 2)))
        self.assertFalse(sg0.is_full_grid((3, 4), (128, 8, 2), (0, 1)))

    def test_depth_three_only_grid_is_not_the_full_denominator(self):
        cells = sg0.grid(depths=(3,), supports=(128, 8, 2), worlds=(0, 1, 2))
        staged = [c for c in cells if c[0] in sg0.STAGED]
        self.assertEqual(len(staged), 18)
        self.assertNotEqual(len(staged), 36)

if __name__ == '__main__':
    unittest.main()
