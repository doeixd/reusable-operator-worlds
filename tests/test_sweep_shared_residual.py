import unittest

from row.experiments.sweep_shared_residual import _record, _rho_label


class SweepSharedResidualTests(unittest.TestCase):
    def test_rho_label(self) -> None:
        self.assertEqual(_rho_label(0.75), "0p75")

    def test_record_rejects_escape_hatch_violation(self) -> None:
        summary = {
            "residual_diagnostics": {
                "maximum_task_functional_residual_to_parent_update_ratio": 1.0
            }
        }
        with self.assertRaisesRegex(ValueError, "escape guard"):
            _record(summary, 2, 0.5)


if __name__ == "__main__":
    unittest.main()
