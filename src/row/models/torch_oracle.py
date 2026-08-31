"""PyTorch operator modules and the true-route oracle compositor."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class LearnedOperator(nn.Module):
    """Learner-side instance of the teacher's residual bottleneck family."""

    def __init__(
        self,
        d: int,
        rank: int,
        alpha: float,
        seed: int,
        learnable_alpha: bool = False,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        V = torch.randn(rank, d, generator=generator)
        U = torch.randn(d, rank, generator=generator)
        V = V / torch.linalg.matrix_norm(V, ord=2)
        U = U / torch.linalg.matrix_norm(U, ord=2)
        self.V = nn.Parameter(V)
        self.U = nn.Parameter(U)
        self.b = nn.Parameter(torch.zeros(rank))
        if activation not in {"tanh", "gelu"}:
            raise ValueError("activation must be 'tanh' or 'gelu'")
        self.activation = activation
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.alpha = float(alpha)

    def forward(self, z: Tensor) -> Tensor:
        hidden = torch.nn.functional.linear(z, self.V, self.b)
        hidden = torch.tanh(hidden) if self.activation == "tanh" else torch.nn.functional.gelu(hidden)
        return torch.tanh(z + self.alpha * torch.nn.functional.linear(hidden, self.U))


class HouseholderOrthogonal(nn.Module):
    """Exact trainable orthogonal map, represented as Householder reflections.

    A product of ``d`` reflections spans one component of O(d); ``d - 1``
    reflections spans the other. Rotated library slots alternate components, so
    the learner can represent teacher rotations of either determinant without
    receiving a teacher primitive identity or rotation at initialization.
    """

    def __init__(self, d: int, reflections: int, seed: int) -> None:
        super().__init__()
        if reflections not in {d - 1, d}:
            raise ValueError("reflections must be d - 1 or d")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        vectors = torch.randn(reflections, d, generator=generator)
        vectors = vectors / torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        self.vectors = nn.Parameter(vectors)

    def forward(self, z: Tensor) -> Tensor:
        for vector in self.vectors:
            denominator = torch.sum(vector.square()).clamp_min(1e-12)
            projection = torch.sum(z * vector, dim=-1, keepdim=True) / denominator
            z = z - 2.0 * projection * vector
        return z

    def matrix(self) -> Tensor:
        """Return the represented matrix in the row-vector convention."""
        return self(torch.eye(self.vectors.shape[1], device=self.vectors.device,
                              dtype=self.vectors.dtype))


class RotatedLearnedOperator(nn.Module):
    """Learner-side instance of the rotated teacher operator family.

    The orthogonal map is learned independently in every slot. Its initialization
    depends only on the model seed, never on the world seed or hidden teacher
    library.
    """

    def __init__(
        self,
        d: int,
        rank: int,
        alpha: float,
        seed: int,
        learnable_alpha: bool = False,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        V = torch.randn(rank, d, generator=generator)
        U = torch.randn(d, rank, generator=generator)
        self.V = nn.Parameter(V / torch.linalg.matrix_norm(V, ord=2))
        self.U = nn.Parameter(U / torch.linalg.matrix_norm(U, ord=2))
        self.b = nn.Parameter(torch.zeros(rank))
        if activation not in {"tanh", "gelu"}:
            raise ValueError("activation must be 'tanh' or 'gelu'")
        self.activation = activation
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.alpha = float(alpha)
        # Consecutive slot seeds differ by the odd stride 997, hence alternate
        # between det(Q)=+1 and det(Q)=-1 when d is even (and still cover both
        # components when d is odd).
        reflections = d - (seed & 1)
        self.rotation = HouseholderOrthogonal(d, reflections, seed + 48611)

    def forward(self, z: Tensor) -> Tensor:
        hidden = torch.nn.functional.linear(z, self.V, self.b)
        hidden = torch.tanh(hidden) if self.activation == "tanh" else torch.nn.functional.gelu(hidden)
        residual = z + self.alpha * torch.nn.functional.linear(hidden, self.U)
        return self.rotation(residual)


class OracleCompositor(nn.Module):
    """Shared learned slots composed according to teacher-provided routes."""

    def __init__(
        self,
        d: int,
        rank: int,
        operators: int,
        alpha: float,
        seed: int,
        learnable_alpha: bool = False,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.operators = nn.ModuleList(
            LearnedOperator(
                d=d,
                rank=rank,
                alpha=alpha,
                seed=seed + 997 * index,
                learnable_alpha=learnable_alpha,
                activation=activation,
            )
            for index in range(operators)
        )

    def forward(self, x: Tensor, route: Sequence[int]) -> Tensor:
        z = x
        for operator_id in route:
            z = self.operators[int(operator_id)](z)
        return z

    def forward_routes(self, x: Tensor, routes: Sequence[Sequence[int]]) -> Tensor:
        if len(x) != len(routes):
            raise ValueError("each input must have one route")
        return torch.cat(
            [self.forward(sample.unsqueeze(0), route) for sample, route in zip(x, routes, strict=True)],
            dim=0,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
