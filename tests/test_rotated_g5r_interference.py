import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from row.config import load_config
from row.experiments.audit_rotated_g5r_diagnosis import assign_slots
from row.experiments.audit_rotated_g5r_interference import (
    classify,
    offline_cell,
    oracle_routes,
    pin_code,
    pinned_is_one_hot,
    run_online_oracle,
    score,
    score_online_artifact,
)
from row.rotated_world import generate_rotated_world


def tiny_config(directory: Path | None = None):
    base = load_config("configs/v1.yaml")
    config = replace(
        base,
        world=replace(
            base.world,
            state_dim=4,
            teacher_rank=2,
            teacher_primitives=2,
            program_length=2,
            tasks=2,
            examples_per_task=2,
            evaluation_examples=4,
        ),
        discrete_model=replace(
            base.discrete_model,
            operator_slots=4,
            operator_rank=2,
            task_steps=2,
            replay_examples_per_task=1,
        ),
        evaluation=replace(
            base.evaluation,
            support_points=(0, 1, 2),
            lifetime_checkpoints=(),
            checkpoint_novel_tasks=1,
            extended_diagnostics=False,
        ),
    )
    if directory is not None:
        config = replace(config, output_directory=directory)
    return config


class RotatedG5RInterferenceTests(unittest.TestCase):
    def test_pinned_code_is_exactly_one_hot_across_the_temperature_range(self):
        code = torch.nn.Parameter(torch.zeros(3, 12))
        pin_code(code, (4, 0, 11))
        self.assertFalse(code.requires_grad)
        for temperature in (1.0, 0.5, 0.1):
            self.assertTrue(pinned_is_one_hot(code, temperature))
        self.assertEqual(tuple(int(v) for v in torch.argmax(code, dim=-1)), (4, 0, 11))
        with self.assertRaises((RuntimeError, IndexError)):
            pin_code(torch.zeros(2, 3), (0, 5))  # slot out of range

    def test_classification_ladder_truth_table(self):
        self.assertEqual(classify(False, False, False), "BUDGET_LIMITED")
        self.assertEqual(classify(False, True, True), "BUDGET_LIMITED")
        self.assertEqual(classify(True, False, True), "ONLINE_INTERFERENCE")
        self.assertEqual(classify(True, True, False), "ROUTE_INFERENCE")
        self.assertEqual(classify(True, False, False), "BOTH_INDEPENDENTLY_SUFFICIENT")
        self.assertEqual(classify(True, True, True), "INTERACTION_ONLY")

    def test_offline_oracle_cell_keeps_routes_and_moves_shared_parameters(self):
        config = tiny_config()
        world = generate_rotated_world(config.world)
        assignment = assign_slots(world.library, 4, config.discrete_model.seed, 4)
        result = offline_cell(
            config, world, assignment, oracle=True, updates=3, batch=2, cell_index=0,
            checkpoints_requested=(0, 1, 3),
        )
        self.assertTrue(result["finite"])
        self.assertTrue(result["pinned_routes_preserved"])
        self.assertTrue(result["pinned_one_hot_at_1.0"])
        self.assertGreater(result["shared_relative_change"], 0.0)
        self.assertIsNone(result["code_relative_change"])
        self.assertEqual(set(result["checkpoints"]), {"0", "1", "3"})
        self.assertEqual(len(result["final_per_task"]), 2)

    def test_offline_learned_cell_moves_codes_and_is_reproducible(self):
        config = tiny_config()
        world = generate_rotated_world(config.world)
        assignment = assign_slots(world.library, 4, config.discrete_model.seed, 4)
        first = offline_cell(
            config, world, assignment, oracle=False, updates=3, batch=2, cell_index=1
        )
        second = offline_cell(
            config, world, assignment, oracle=False, updates=3, batch=2, cell_index=1
        )
        self.assertTrue(first["finite"])
        self.assertGreater(first["code_relative_change"], 0.0)
        self.assertGreater(first["shared_relative_change"], 0.0)
        self.assertIsNone(first["pinned_routes_preserved"])
        self.assertEqual(first["terminal_median"], second["terminal_median"])

    def test_online_oracle_lifetime_pins_routes_persists_and_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config(Path(directory) / "artifact")
            world = generate_rotated_world(config.world)
            assignment = assign_slots(world.library, 4, config.discrete_model.seed, 4)
            path = run_online_oracle(config, world, assignment)
            self.assertTrue((path / "g5r_interference_stamp.json").exists())
            # Resume is a no-op with a matching stamp.
            self.assertEqual(run_online_oracle(config, world, assignment), path)
            result = score_online_artifact(
                config, world, path, oracle_assignment=assignment
            )
            self.assertTrue(result["anchor_passes"])
            self.assertTrue(result["pinned_routes_preserved"])
            self.assertTrue(result["pinned_one_hot_at_1.0"])
            self.assertGreater(result["shared_relative_change"], 0.0)
            self.assertIn("end_of_task_median", result)
            routes = oracle_routes(world, assignment)
            self.assertEqual(len(routes), 2)

    def test_scorer_uses_argmax_routes_in_eval_mode(self):
        config = tiny_config()
        world = generate_rotated_world(config.world)
        from row.experiments.audit_rotated_g5r_interference import build_model

        model = build_model(config)
        for task in world.tasks:
            model.begin_task(task.task_id)
        model.train()
        result = score(model, world)
        self.assertFalse(model.training)
        self.assertEqual(len(result["per_task"]), 2)


if __name__ == "__main__":
    unittest.main()
