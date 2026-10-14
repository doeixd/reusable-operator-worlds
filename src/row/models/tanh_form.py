"""Learner of the `tanh` operator form selected by G0F (decision 20 (a); `G1T_TANH_FORMATION_PLAN.md`).

G0F (`reports/g0f_form_census.json`) selected the teacher form `P(z) = c_g Q tanh(g R z + b)` at `g = 2`: no identity
skip, so a computed state is no longer close to an orthogonal map of the input. The program rule "the learner does
not change" requires the learner to take the same form, and this module is that learner. Each slot is

    P(z) = c Q tanh(A z + b)

with `A` a free `d x d` matrix (it absorbs the teacher's gain and rotation `g R`), `b` a free bias, `c` a learnable
output scale and `Q` the same Householder orthogonal map as `RotatedLearnedOperator` (`d - (seed & 1)` reflections,
seed `seed + 48611`, so consecutive slots alternate between the two components of O(d), as the teacher's Haar `Q`
does). The teacher form is therefore inside the slot family exactly.

Initialization depends only on the model seed, never on the world: `A` is a Haar-orthogonal matrix at gain 1 (NOT the
teacher's gain 2, as the rotated learner starts `alpha` at 0.2 against the teacher's 0.35), `b = 0`, and
`c = c_1 = 1 / sqrt(E[tanh(x)^2])`, so a fresh slot keeps unit variance. The scale is named `alpha` so that the
lifetime's optimizer puts it in the no-weight-decay group, exactly as it treats the rotated slots' `alpha`.

Everything except the slot form (task codes, temperature, hard routes, per-task depth from a plan) is inherited from
`PlannedDepthRotatedLearner` unchanged; its rotated slots are built by the parent constructor and then replaced.
`FrozenTanhLibrary` is `FrozenLibrary`'s search view for this form; its `candidates` is the same function the learner
calls, so a one-hot route gives the learner's own hard prediction bitwise (gate L1).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.form_world import tanh_gain_scale
from row.models.learned_models import batched_householder
from row.models.online_variable_depth import PlannedDepthRotatedLearner
from row.models.torch_oracle import HouseholderOrthogonal

INIT_GAIN = 1.0


def tanh_candidates(z: Tensor, A: Tensor, b: Tensor, c: Tensor, Q: Tensor) -> Tensor:
    """Every slot applied to every row: `(batch, d) -> (batch, slots, d)`, row-vector convention."""
    hidden = torch.tanh(torch.einsum('bd,sed->bse', z, A) + b)
    return c.view(1, -1, 1) * torch.einsum('bse,sef->bsf', hidden, Q)


class TanhFormOperator(nn.Module):
    def __init__(self, d: int, seed: int, learnable_scale: bool = True) -> None:
        super().__init__()
        generator = torch.Generator(device='cpu').manual_seed(seed)
        q, r = torch.linalg.qr(torch.randn(d, d, generator=generator, dtype=torch.float64))
        q = q * torch.sign(torch.diagonal(r))
        self.A = nn.Parameter((INIT_GAIN * q).to(torch.float32))
        self.b = nn.Parameter(torch.zeros(d))
        scale = tanh_gain_scale(INIT_GAIN)
        self.alpha = nn.Parameter(torch.tensor(float(scale))) if learnable_scale else float(scale)
        self.rotation = HouseholderOrthogonal(d, d - (seed & 1), seed + 48611)

    def scale(self) -> Tensor:
        return self.alpha if torch.is_tensor(self.alpha) else torch.tensor(float(self.alpha))

    def forward(self, z: Tensor) -> Tensor:
        return self.scale() * self.rotation(torch.tanh(torch.nn.functional.linear(z, self.A, self.b)))


def stacked(library) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    library = list(library)
    return (torch.stack([op.A for op in library]), torch.stack([op.b for op in library]),
            torch.stack([op.scale() for op in library]),
            batched_householder([op.rotation.vectors for op in library]))


class TanhFormLearner(PlannedDepthRotatedLearner):
    """`PlannedDepthRotatedLearner` whose slots are `TanhFormOperator`s; every task runs at its own depth."""

    implementation = 'tanh_form_planned_depth_v1'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d = self.library[0].rotation.vectors.shape[1]
        seed = kwargs['seed']
        learnable = kwargs.get('learnable_alpha', True)
        self.library = nn.ModuleList(TanhFormOperator(d, seed + 997 * index, learnable)
                                     for index in range(self.operator_slots))

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        A, b, c, Q = stacked(self.library)
        coefficients = self._coefficients(task_id)
        z = x
        for step in range(self.depth_of(task_id)):
            z = torch.sum(coefficients[step].view(1, -1, 1) * tanh_candidates(z, A, b, c, Q), dim=1)
        return z


class FrozenTanhLibrary(FrozenLibrary):
    """`FrozenLibrary`'s interface (`candidates`, `forward`, `hard`, `all_route_support_mse`) for the tanh form."""

    def __init__(self, model):
        with torch.no_grad():
            self.A, self.b, self.c, self.Q = (t.detach() for t in stacked(model.library))
        self.gelu = False
        self.slots = len(model.library)
        self.steps = model.task_steps

    def candidates(self, z):
        return tanh_candidates(z, self.A, self.b, self.c, self.Q)


def build_tanh_learner(config, plan=None, task_steps=None) -> TanhFormLearner:
    """The construction every arm uses: `configs/v1.yaml`'s discrete settings, the config's model seed."""
    sel = config.discrete_model
    return TanhFormLearner(
        d=config.world.state_dim, operator_slots=sel.operator_slots, operator_rank=sel.operator_rank,
        task_steps=sel.task_steps if task_steps is None else task_steps, alpha=sel.operator_alpha_init,
        initial_temperature=sel.initial_temperature, final_temperature=sel.final_temperature, seed=sel.seed,
        learnable_alpha=sel.learnable_alpha, activation=sel.operator_activation, depth_plan=plan)
