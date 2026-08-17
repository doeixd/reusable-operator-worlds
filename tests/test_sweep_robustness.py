from __future__ import annotations

import unittest
from pathlib import Path

from row.config import load_config
from row.experiments.sweep_robustness import CONDITIONS, _condition_config


class SweepRobustnessTests(unittest.TestCase):
    def test_reverse_preserves_replay_and_changes_order(self) -> None:
        base = load_config(Path("configs/v1.yaml"))
        config, order = _condition_config(
            base, "reverse", "continuous", 2, Path("artifact")
        )
        self.assertEqual(order, "reverse")
        self.assertEqual(config.continuous_model.replay_ratio, 1.0)
        self.assertEqual(config.world.seed, 2)

    def test_replay_conditions_are_symmetric_across_models(self) -> None:
        base = load_config(Path("configs/v1.yaml"))
        for condition, expected in (("replay0", 0.0), ("replay4", 4.0)):
            for model in ("continuous", "dense"):
                config, order = _condition_config(
                    base, condition, model, 0, Path("artifact")
                )
                selected = (
                    config.continuous_model if model == "continuous" else config.dense_model
                )
                self.assertEqual(order, "forward")
                self.assertEqual(selected.replay_ratio, expected)
        self.assertEqual(set(CONDITIONS), {"reverse", "replay0", "replay4"})


if __name__ == "__main__":
    unittest.main()
