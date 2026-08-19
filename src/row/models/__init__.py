"""ROW learner implementations."""

from row.models.learned_models import (
    ContinuousBasisLearner,
    DenseLearner,
    DiscreteLibraryLearner,
    HypernetworkLearner,
    PresenceGatedDiscreteLibraryLearner,
    SharedParentResidualLearner,
    VariationalSharedResidualLearner,
)
from row.models.gated_models import GatedInnovationLearner
from row.models.promoting_models import PromotingSharedResidualLearner
from row.models.numpy_mlp import ScratchResidualMLP
from row.models.torch_oracle import LearnedOperator, OracleCompositor

__all__ = [
    "ContinuousBasisLearner",
    "GatedInnovationLearner",
    "DenseLearner",
    "DiscreteLibraryLearner",
    "HypernetworkLearner",
    "PresenceGatedDiscreteLibraryLearner",
    "PromotingSharedResidualLearner",
    "SharedParentResidualLearner",
    "VariationalSharedResidualLearner",
    "LearnedOperator",
    "OracleCompositor",
    "ScratchResidualMLP",
]
