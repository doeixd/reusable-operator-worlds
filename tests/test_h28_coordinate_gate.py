"""Oracle construction tests; no trained models or sealed worlds."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('h28_coordinate_gate',
    Path(__file__).resolve().parents[1]/'tools/h28_coordinate_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class CoordinateGateTests(unittest.TestCase):
    def setUp(self):
        self.library, self.splits, self.matrices = gate.fixture()

    def test_deterministic_fixture_and_disjoint_streams(self):
        library, splits, matrices = gate.fixture()
        for name in self.splits:
            np.testing.assert_array_equal(self.splits[name], splits[name])
        self.assertFalse(np.array_equal(splits['support'], splits['query'][:16]))
        for name in matrices:
            np.testing.assert_array_equal(self.matrices[name], matrices[name])
        for left, right in zip(self.library, library):
            np.testing.assert_array_equal(left.U, right.U)

    def test_full_nonlinear_conjugacy(self):
        z = self.splits['query']
        for matrix in self.matrices.values():
            inverse = gate.checked_inverse(matrix)
            for primitive in self.library:
                np.testing.assert_allclose(gate.realized(primitive, z@matrix.T, matrix, inverse),
                                           primitive(z)@matrix.T, atol=1e-12, rtol=1e-12)

    def test_weight_only_wrong_for_mixing_but_correct_for_permutation(self):
        z = self.splits['query']
        for name, matrix in self.matrices.items():
            inverse = gate.checked_inverse(matrix)
            got = gate.weight_only(self.library[0], z@matrix.T, matrix, inverse)@inverse.T
            error = gate.discrepancy(got, self.library[0](z))
            if name in ('identity', 'permutation'):
                self.assertLess(error, 1e-20)
            else:
                self.assertGreater(error, gate.TOL)

    def test_bad_coordinate_matrices_rejected(self):
        for matrix in (np.zeros((2, 2)), np.ones((2, 3)), np.diag([1., 3.]), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                gate.checked_inverse(matrix)

    def test_transported_metric_ignores_observation_scaling(self):
        z = self.splits['query']
        prediction, target = z+.1, z
        expected = gate.discrepancy(prediction, target)
        for matrix in self.matrices.values():
            inverse = gate.checked_inverse(matrix)
            self.assertAlmostEqual(gate.discrepancy((prediction@matrix.T)@inverse.T,
                                                    (target@matrix.T)@inverse.T), expected)

    def test_zero_or_nonfinite_targets_rejected(self):
        for target in (np.zeros((2, 2)), np.full((2, 2), np.nan)):
            with self.assertRaises(ValueError):
                gate.discrepancy(np.ones((2, 2)), target)

    def test_analytic_jacobian_matches_independent_difference(self):
        for primitive in self.library:
            np.testing.assert_allclose(gate.derivative_at_zero(primitive),
                                       gate.finite_difference(primitive), atol=1e-8, rtol=1e-8)

    def test_nonconjugate_trace_witness(self):
        jacobian = gate.derivative_at_zero(self.library[0])
        self.assertGreater(abs(np.trace(1.1*jacobian)-np.trace(jacobian)), 1.)
        for matrix in self.matrices.values():
            self.assertAlmostEqual(np.trace(matrix@jacobian@gate.checked_inverse(matrix)), np.trace(jacobian))

    def test_identity_core_cannot_hide_computation(self):
        x = self.splits['query']
        for matrix in self.matrices.values():
            inverse = gate.checked_inverse(matrix)
            np.testing.assert_allclose((x@inverse.T)@matrix.T, x, atol=1e-12)

    def test_program_order_matters(self):
        x = self.splits['query']
        forward, reverse = x.copy(), x.copy()
        for k in gate.PROGRAMS[0]:
            forward = self.library[k](forward)
        for k in reversed(gate.PROGRAMS[0]):
            reverse = self.library[k](reverse)
        self.assertGreater(gate.discrepancy(forward, reverse), gate.TOL)

    def test_gate_payload_and_complete_boundaries(self):
        result = gate.run_checks()
        self.assertEqual(len(result['rows']), 2*4*2*3)
        self.assertEqual(result['payload']['core_float64_bytes'], 6*(16*8+8*16+8+1)*8)
        self.assertTrue(all(n == 16*16*8 for n in result['payload']['adapter_float64_bytes'].values()))
        self.assertFalse(result['economic_value_measured'])


if __name__ == '__main__':
    unittest.main()
