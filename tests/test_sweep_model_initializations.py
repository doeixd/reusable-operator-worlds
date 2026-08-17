from __future__ import annotations

import unittest
from pathlib import Path

from row.config import load_config
from row.experiments.sweep_model_initializations import _seeded_config


class SweepModelInitializationsTests(unittest.TestCase):
    def test_changes_only_selected_model_seed_and_run_envelope(self) -> None:
        base = load_config(Path("configs/v1.yaml"))
        config = _seeded_config(base, "continuous", 4, 4001, Path("artifact"))
        self.assertEqual(config.world.seed, 4)
        self.assertEqual(config.continuous_model.seed, 4001)
        self.assertEqual(config.dense_model, base.dense_model)
        self.assertEqual(config.evaluation.lifetime_checkpoints, ())


if __name__ == "__main__":
    unittest.main()
