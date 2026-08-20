"""Config validation that V5.1 depends on."""

from __future__ import annotations

import unittest
from dataclasses import replace

from row.config import GatedModelConfig, SharedResidualModelConfig, VariationalModelConfig


class ResidualRankCapTests(unittest.TestCase):
    def test_shared_residual_default_stays_two(self) -> None:
        self.assertEqual(SharedResidualModelConfig().residual_rank, 2)

    def test_shared_residual_allows_rank_four(self) -> None:
        config = replace(SharedResidualModelConfig(), residual_rank=4)
        self.assertEqual(config.residual_rank, 4)

    def test_shared_residual_rejects_rank_five(self) -> None:
        with self.assertRaises(ValueError):
            replace(SharedResidualModelConfig(), residual_rank=5)

    def test_variational_and_gated_validate_rank_four(self) -> None:
        VariationalModelConfig.validate(
            replace(VariationalModelConfig(), residual_rank=4)
        )
        GatedModelConfig.validate(replace(GatedModelConfig(), residual_rank=4))
