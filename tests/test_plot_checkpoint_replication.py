from __future__ import annotations

import unittest

from row.experiments.plot_checkpoint_replication import validate_report


class PlotCheckpointReplicationTests(unittest.TestCase):
    def test_validation_rejects_incomplete_worlds(self) -> None:
        with self.assertRaisesRegex(ValueError, "worlds 0-9"):
            validate_report({"records": [], "checkpoint_summaries": []})


if __name__ == "__main__":
    unittest.main()
