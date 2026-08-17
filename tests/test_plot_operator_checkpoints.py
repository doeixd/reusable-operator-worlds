from __future__ import annotations

import unittest

from row.experiments.plot_operator_checkpoints import validate_report
from row.experiments.sweep_operator_checkpoints import CHECKPOINTS


class PlotOperatorCheckpointsTests(unittest.TestCase):
    def test_validation_requires_all_development_worlds(self) -> None:
        report = {
            "records": [
                {"world_seed": world, "model": model}
                for world in range(10)
                for model in ("continuous", "discrete")
            ],
            "checkpoint_summaries": [
                {"tasks_completed": checkpoint} for checkpoint in CHECKPOINTS
            ],
        }
        validate_report(report)
        report["records"].pop()
        with self.assertRaisesRegex(ValueError, "worlds 0-9"):
            validate_report(report)


if __name__ == "__main__":
    unittest.main()
