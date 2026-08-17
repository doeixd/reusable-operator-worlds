from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path

from row.config import load_config
from row.experiments.sweep_forward_transfer import _fresh_task_loss
from row.world import World


class SweepForwardTransferTests(unittest.TestCase):
    def test_fresh_task_protocol_is_deterministic_and_finite(self) -> None:
        base = load_config(Path("configs/v1.yaml"))
        config = replace(
            base,
            world=replace(
                base.world,
                tasks=2,
                examples_per_task=3,
                evaluation_examples=4,
            ),
        )
        task = World.generate(config.world).tasks[0]
        first = _fresh_task_loss(config, "dense", task)
        second = _fresh_task_loss(config, "dense", task)
        self.assertTrue(math.isfinite(first))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
