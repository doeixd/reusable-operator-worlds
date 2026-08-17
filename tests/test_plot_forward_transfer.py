from __future__ import annotations

import unittest

from row.experiments.plot_forward_transfer import _model_bins


class PlotForwardTransferTests(unittest.TestCase):
    def test_requires_eight_bins_and_ten_worlds(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete"):
            _model_bins({"task_index_bins": []}, "continuous")


if __name__ == "__main__":
    unittest.main()
