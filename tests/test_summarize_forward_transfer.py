from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from row.config import load_config
from row.experiments.summarize_forward_transfer import _route_context, _task_index_bins
from row.world import World


class SummarizeForwardTransferTests(unittest.TestCase):
    def test_route_context_has_no_prior_exposure_for_first_task(self) -> None:
        base = load_config(Path("configs/v1.yaml"))
        world = World.generate(replace(base.world, tasks=3))
        exposure, similarity = _route_context(world, 0)
        self.assertEqual(exposure, 0)
        self.assertEqual(similarity, 0.0)

    def test_task_bins_preserve_world_replication(self) -> None:
        rows = [
            {
                "model": model,
                "world_seed": world,
                "task_index": task,
                "forward_transfer_gaussian_log_loss": float(task + world),
            }
            for model in ("continuous", "dense")
            for world in range(10)
            for task in range(64)
        ]
        bins = _task_index_bins(rows)
        self.assertEqual(len(bins), 16)
        self.assertEqual(len(bins[0]["world_means"]), 10)


if __name__ == "__main__":
    unittest.main()
