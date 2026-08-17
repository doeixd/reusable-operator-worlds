"""ROW learner implementations."""

from row.models.numpy_mlp import ScratchResidualMLP
from row.models.torch_oracle import LearnedOperator, OracleCompositor

__all__ = ["LearnedOperator", "OracleCompositor", "ScratchResidualMLP"]
