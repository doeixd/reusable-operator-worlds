from __future__ import annotations

import unittest
from pathlib import Path

from row.config import load_config
from row.experiments.tune_development import _configured_run


class TuneDevelopmentTests(unittest.TestCase):
    def test_configures_hypernetwork_without_changing_other_models(self) -> None:
        base = load_config(Path("configs/v1.yaml"))
        configured = _configured_run(
            base, "hypernetwork", 2, 3e-4, 5e-3, Path("artifact")
        )
        self.assertEqual(configured.world.seed, 2)
        self.assertEqual(configured.hypernetwork_model.global_learning_rate, 3e-4)
        self.assertEqual(configured.hypernetwork_model.task_learning_rate, 5e-3)
        self.assertEqual(configured.continuous_model, base.continuous_model)

    def test_dense_task_code_dimension_is_an_explicit_control(self) -> None:
        base = load_config(Path("configs/v1.yaml"))
        configured = _configured_run(
            base,
            "dense",
            0,
            1e-3,
            5e-2,
            Path("artifact"),
            dense_hidden_width=32,
            dense_task_embedding_dim=24,
        )
        self.assertEqual(configured.dense_model.hidden_width, 32)
        self.assertEqual(configured.dense_model.task_embedding_dim, 24)


if __name__ == "__main__":
    unittest.main()
