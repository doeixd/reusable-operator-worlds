"""A learner whose LOOP tasks iterate one routed operator while a learned predicate on the state holds
(SUCCESSOR_LADDER.md, B2-online). Additive; no existing learner changes.

`LoopLearner` is `BranchGatedLearner` (with whatever `branch_plan` it is given; empty here) plus, for each task in
`loop_plan`:
- the task's own route code (planned depth 1) is the loop BODY: one routed operator;
- a halting predicate `halt_w[task]`, `halt_b[task]` on the current state (zero init: every step starts at
  continue-probability 0.5); `state_halt=False` gives one learned bias PER STEP and no state weights (a
  state-independent count: in evaluation a learned fixed count L, i.e. the straight route [body]^L; the
  capacity control, same iteration machinery).
Iteration, at most `max_iterations` (K) body applications: z_0 = x; at step j the loop continues with probability
g_j = sigmoid(h.z_j + b) and otherwise stops with output z_j; after K applications it stops with z_K. Training
output: the expected stopping state, sum_j P(stop at j) z_j (differentiable in the body, the predicate and the
library). Evaluation: hard (continue iff h.z_j + b > 0) and the argmax body route.

SWITCH (tested): with an empty `loop_plan` the class adds no parameters, consumes no RNG and every forward is the
parent's, so a lifetime with no loop tasks is BITWISE the parent learner's.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from row.models.branch_gated import BranchGatedLearner


class LoopLearner(BranchGatedLearner):
    implementation = 'loop_gated_v1'

    def __init__(self, *args, loop_plan: dict[str, str] | None = None, max_iterations: int = 6,
                 state_halt: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loop_plan: dict[str, str] = dict(loop_plan or {})
        self.max_iterations = int(max_iterations)
        self.state_halt = bool(state_halt)
        self.halt_w = nn.ParameterDict()
        self.halt_b = nn.ParameterDict()

    def begin_task(self, task_id: str, depth: int | None = None, **kwargs):
        code = super().begin_task(task_id, depth=depth, **kwargs)
        if task_id not in self.loop_plan:
            return code
        if self.depth_of(task_id) != 1:
            raise ValueError(f'{task_id}: a loop body has planned depth 1')
        if not self.state_halt:
            self.halt_b[task_id] = nn.Parameter(torch.zeros(self.max_iterations))
            return [code, self.halt_b[task_id]]
        self.halt_b[task_id] = nn.Parameter(torch.zeros(1))
        d = self.library[0].V.shape[1]
        self.halt_w[task_id] = nn.Parameter(torch.zeros(d))
        return [code, self.halt_w[task_id], self.halt_b[task_id]]

    def loop_parameters(self, task_ids) -> list[nn.Parameter]:
        out = []
        for t in task_ids:
            if t in self.halt_b:
                out += [self.halt_b[t]] + ([self.halt_w[t]] if t in self.halt_w else [])
        return out

    def halt_logit(self, z: Tensor, task_id: str, step: int) -> Tensor:
        if task_id in self.halt_w:
            return z @ self.halt_w[task_id] + self.halt_b[task_id]
        return self.halt_b[task_id][step].expand(z.shape[0])

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        if task_id not in self.loop_plan:
            return super().forward(x, task_id)
        coefficients = self._route_coefficients(self.task_codes[task_id])
        z = x
        if self.training:
            alive = torch.ones(x.shape[0], 1, dtype=x.dtype)
            out = torch.zeros_like(x)
            for step in range(self.max_iterations):
                go = torch.sigmoid(self.halt_logit(z, task_id, step)).unsqueeze(1)
                out = out + alive * (1 - go) * z
                alive = alive * go
                z = self._route(z, coefficients, 1)
            return out + alive * z
        done = torch.zeros(x.shape[0], dtype=torch.bool)
        for step in range(self.max_iterations):
            done = done | ~(self.halt_logit(z, task_id, step) > 0)
            z = torch.where(done.unsqueeze(1), z, self._route(z, coefficients, 1))
        return z

    def iteration_counts(self, x: Tensor, task_id: str) -> Tensor:
        """Hard iteration count per input (evaluation semantics)."""
        coefficients = torch.nn.functional.one_hot(torch.argmax(self.task_codes[task_id], -1),
                                                   self.operator_slots).to(x.dtype)
        z, k = x, torch.zeros(x.shape[0], dtype=torch.long)
        done = torch.zeros(x.shape[0], dtype=torch.bool)
        with torch.no_grad():
            for step in range(self.max_iterations):
                done = done | ~(self.halt_logit(z, task_id, step) > 0)
                k = k + (~done).long()
                z = torch.where(done.unsqueeze(1), z, self._route(z, coefficients, 1))
        return k
