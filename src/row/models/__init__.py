"""ROW learner implementations."""

from row.models.learned_models import (
    ContinuousBasisLearner,
    DenseLearner,
    DiscreteLibraryLearner,
    HypernetworkLearner,
    PresenceGatedDiscreteLibraryLearner,
    RotatedDiscreteLibraryLearner,
    SharedParentResidualLearner,
    VariationalSharedResidualLearner,
)
from row.models.gated_models import GatedInnovationLearner
from row.models.promoting_models import PromotingSharedResidualLearner
from row.models.lifecycle_models import AbstractionRecord, LifecycleLibraryLearner
from row.models.numpy_mlp import ScratchResidualMLP
from row.models.torch_oracle import (
    HouseholderOrthogonal,
    LearnedOperator,
    OracleCompositor,
    RotatedLearnedOperator,
)

__all__ = [
    "AbstractionRecord",
    "ContinuousBasisLearner",
    "LifecycleLibraryLearner",
    "GatedInnovationLearner",
    "DenseLearner",
    "DiscreteLibraryLearner",
    "HouseholderOrthogonal",
    "HypernetworkLearner",
    "PresenceGatedDiscreteLibraryLearner",
    "RotatedDiscreteLibraryLearner",
    "RotatedLearnedOperator",
    "PromotingSharedResidualLearner",
    "SharedParentResidualLearner",
    "VariationalSharedResidualLearner",
    "LearnedOperator",
    "OracleCompositor",
    "ScratchResidualMLP",
]
