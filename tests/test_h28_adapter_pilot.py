"""Development-only checks for the oracle-core adapter pilot."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('h28_adapter_pilot',
    Path(__file__).resolve().parents[1] / 'tools/h28_adapter_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class AdapterPilotTests(unittest.TestCase):
    def test_oracle_is_zero_on_observed_inputs(self):
        library, matrices, data, _ = pilot.fixture()
        for matrix in matrices.values():
            observed = data['query'] @ matrix.T
            for primitive in library:
                np.testing.assert_allclose(
                    pilot.predict(primitive, observed, matrix),
                    pilot.observed_target(primitive, data['query'], matrix),
                    atol=1e-12, rtol=1e-12)

    def test_true_angles_fit_support_better_than_zero(self):
        library, matrices, data, angles = pilot.fixture()
        for index, context in enumerate(('context_1', 'context_2')):
            matrix = matrices[context]
            support = data['support'] @ matrix.T
            true_loss = np.mean([
                pilot.loss(pilot.predict(library[op], support, matrix),
                           pilot.observed_target(library[op], data['support'], matrix))
                for op in pilot.SUPPORT_OPS])
            zero_loss = np.mean([
                pilot.loss(pilot.predict(library[op], support, pilot.givens(np.zeros(4))),
                           pilot.observed_target(library[op], data['support'], matrix))
                for op in pilot.SUPPORT_OPS])
            self.assertLess(true_loss, zero_loss)
            self.assertEqual(len(angles[index]), 4)

    def test_smoke_has_no_query_selection(self):
        result = pilot.run(smoke=True)
        self.assertEqual(result['contexts'], ('identity',))
        self.assertFalse(result['query_used_for_fit'])
        self.assertFalse(result['core_changed'])
        self.assertEqual(len(result['records']), 6)


if __name__ == '__main__':
    unittest.main()
