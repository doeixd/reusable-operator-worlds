"""A learner whose BRANCH tasks route each example by a learned gate on the state (SUCCESSOR_LADDER.md, B1-online).

Additive; no existing learner changes. `BranchGatedLearner` is `PlannedDepthRotatedLearner` plus, for each task
listed in `branch_plan`:
- a second route code `branch_codes[task]` (same shape as the task code; initialized N(0, 0.1) from a generator
  seeded by `branch_seed` and the task id, so no global RNG is consumed: symmetry breaking, B1g's finding that
  mid-program decisions need it);
- a linear gate `gate_w[task]`, `gate_b[task]` on the decision state (zero init, so the gate starts at 0.5).
A branch task's prediction is g * F(x; route 1) + (1 - g) * F(x; route 2), with g = sigmoid(w.s + b) in training
and the hard gate (w.s + b > 0) in evaluation; routes use the parent's coefficient rule (softmax(code / T) in
training, argmax one-hot in evaluation). Decision state s: the input x ('input') for every branch task here.

SWITCH (tested): with an empty `branch_plan` the class adds no parameters, consumes no RNG and every forward is the
parent's, so a lifetime with no branch tasks is BITWISE the straight-line learner's.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from row.models.learned_models import batched_householder
from row.models.online_variable_depth import PlannedDepthRotatedLearner


class BranchGatedLearner(PlannedDepthRotatedLearner):
    implementation = 'branch_gated_v1'

    def __init__(self, *args, branch_plan: dict[str, str] | None = None, branch_seed: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.branch_plan: dict[str, str] = dict(branch_plan or {})
        self.branch_seed = int(branch_seed)
        self.branch_codes = nn.ParameterDict()
        self.gate_w = nn.ParameterDict()
        self.gate_b = nn.ParameterDict()

    def begin_task(self, task_id: str, depth: int | None = None, **kwargs):
        code = super().begin_task(task_id, depth=depth, **kwargs)
        if task_id not in self.branch_plan:
            return code
        seed = np.random.SeedSequence([self.branch_seed, *task_id.encode()]).generate_state(1)[0]
        g = torch.Generator().manual_seed(int(seed))
        self.branch_codes[task_id] = nn.Parameter(0.1 * torch.randn(code.shape, generator=g))
        d = self.library[0].V.shape[1]
        self.gate_w[task_id] = nn.Parameter(torch.zeros(d))
        self.gate_b[task_id] = nn.Parameter(torch.zeros(1))
        return [code, self.branch_codes[task_id], self.gate_w[task_id], self.gate_b[task_id]]

    def branch_parameters(self, task_ids) -> list[nn.Parameter]:
        out = []
        for t in task_ids:
            if t in self.branch_codes:
                out += [self.branch_codes[t], self.gate_w[t], self.gate_b[t]]
        return out

    def _route_coefficients(self, logits: Tensor) -> Tensor:
        if self.training:
            return torch.softmax(logits / self.temperature, dim=-1)
        return torch.nn.functional.one_hot(torch.argmax(logits, dim=-1), self.operator_slots).to(logits.dtype)

    def _route(self, x: Tensor, coefficients: Tensor, depth: int) -> Tensor:
        library = list(self.library)
        V = torch.stack([op.V for op in library])
        U = torch.stack([op.U for op in library])
        b = torch.stack([op.b for op in library])
        alpha = torch.stack([op.alpha if torch.is_tensor(op.alpha) else torch.tensor(float(op.alpha))
                             for op in library])
        Q = batched_householder([op.rotation.vectors for op in library])
        gelu = library[0].activation == 'gelu'
        z = x
        for step in range(depth):
            hidden = torch.einsum('bd,srd->bsr', z, V) + b
            hidden = torch.nn.functional.gelu(hidden) if gelu else torch.tanh(hidden)
            residual = z.unsqueeze(1) + alpha.view(1, -1, 1) * torch.einsum('bsr,sdr->bsd', hidden, U)
            candidates = torch.einsum('bsd,sde->bse', residual, Q)
            z = torch.sum(coefficients[step].view(1, -1, 1) * candidates, dim=1)
        return z

    def gate(self, x: Tensor, task_id: str) -> Tensor:
        logit = x @ self.gate_w[task_id] + self.gate_b[task_id]
        return torch.sigmoid(logit) if self.training else (logit > 0).to(x.dtype)

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        if task_id not in self.branch_codes:
            return super().forward(x, task_id)
        depth = self.depth_of(task_id)
        f1 = self._route(x, self._route_coefficients(self.task_codes[task_id]), depth)
        f2 = self._route(x, self._route_coefficients(self.branch_codes[task_id]), depth)
        g = self.gate(x, task_id).unsqueeze(1)
        return g * f1 + (1 - g) * f2

    def branch_routes(self) -> dict[str, tuple[list[int], list[int]]]:
        return {t: (torch.argmax(self.task_codes[t].detach(), -1)[: self.depth_of(t)].tolist(),
                    torch.argmax(self.branch_codes[t].detach(), -1)[: self.depth_of(t)].tolist())
                for t in self.branch_codes}
