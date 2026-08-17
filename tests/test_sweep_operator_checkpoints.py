from __future__ import annotations

import unittest

from row.experiments.sweep_operator_checkpoints import (
    CHECKPOINTS,
    METRICS,
    _aggregate,
    _operator_curve,
)


def _curve(offset: float) -> list[dict[str, float | int | None]]:
    return [
        {
            "tasks_completed": checkpoint,
            **{
                metric: (
                    None
                    if metric == "true_route_future_programs_nmse_mean"
                    and checkpoint == 64
                    else offset + checkpoint / 1000
                )
                for metric in METRICS
            },
        }
        for checkpoint in CHECKPOINTS
    ]


class SweepOperatorCheckpointsTests(unittest.TestCase):
    def test_requires_new_diagnostic_in_every_checkpoint(self) -> None:
        summary = {
            "novel_composition_checkpoints": [
                {"tasks_completed": checkpoint} for checkpoint in CHECKPOINTS
            ]
        }
        with self.assertRaisesRegex(ValueError, "predates"):
            _operator_curve(summary)

    def test_aggregate_preserves_world_replication_and_nullable_future(self) -> None:
        records = [
            {"world_seed": 0, "model": "continuous", "operator_checkpoints": _curve(0.1)},
            {"world_seed": 0, "model": "discrete", "operator_checkpoints": _curve(0.2)},
        ]
        report = _aggregate(records)
        final = report["checkpoint_summaries"][-1]["models"]["continuous"]
        self.assertIsNone(final["mean_true_route_future_programs_nmse_mean"])
        self.assertEqual(
            final["per_world_one_to_one_mean_primitive_distance"][0]["world_seed"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
