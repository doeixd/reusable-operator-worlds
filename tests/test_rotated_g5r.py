"""Matched-family G5R learner and artifact-path checks."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from row.config import load_config
from row.experiments.audit_rotated_g5r import _load_or_run
from row.rotated_world import generate_rotated_world


class TestRotatedG5R(unittest.TestCase):
    def test_tiny_lifetime_persists_reloads_and_resumes_matched_learner(self):
        base = load_config("configs/v1.yaml")
        with tempfile.TemporaryDirectory() as directory:
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
                    operator_slots=2,
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
                output_directory=Path(directory) / "artifact",
            )
            world = generate_rotated_world(config.world)
            first, summary = _load_or_run(config, world)
            second, resumed = _load_or_run(config, world)
            self.assertEqual(summary["model"], "rotated_discrete")
            self.assertEqual(resumed["model"], "rotated_discrete")
            self.assertEqual(first.state_dict().keys(), second.state_dict().keys())
            self.assertTrue((config.output_directory / "fingerprint.json").exists())
            self.assertTrue((config.output_directory / "g5r_stamp.json").exists())


if __name__ == "__main__":
    unittest.main()
