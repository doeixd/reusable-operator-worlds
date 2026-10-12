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

`detach_halt_input=True` (B2-online v2): the predicate reads the state DETACHED, so its gradient trains the halting
parameters but never reaches the library or the body through the states (`reports/b2o_localize_census.json`: that
path broke formation in sleep). Evaluation is unchanged; default False keeps v1 bitwise.

`step_bias=True` (B2-hard): with `state_halt=True` the predicate is h.z_j + b_j, one bias PER STEP, so the halting rule
NESTS the count form (h = 0 gives any fixed count L, in particular L = 1, the single route): a task the learner is
not told is a loop can still be a plain one-step task (`reports/b2h_data_test_census.json`: a linear state predicate
alone cannot express "apply once and stop"). `halt_decay` puts h in its own optimizer group with that weight decay (a
usage cost on reading the state, as B1-hard's gate decay). Defaults False / None keep the construction bitwise.

`halt_init=c` (B2-hard v2): per-step biases start at the ONE-STEP form `[+c, -c, ..., -c]` instead of zero (with
`state_halt=False`, or with `step_bias=True`), so a candidate begins as a plain single application rather than a
50/50 continue mixture whose dominant terms are the identity and repeated steps; at moderate c the gradients stay
alive. B2-hard's census (`reports/b2h_wake_census.json`) found that the zero start lets wake form ROOT slots
(Q^L ~ P). Default None keeps zeros, bitwise.

`body_depth=2` (B2-depth2): loop tasks may have planned depth up to `body_depth`, and the loop BODY is the task's own
route at its planned depth (one iteration = the whole route). Default 1 keeps the depth-1 check, and a depth-1 body
is computed exactly as before, bitwise.

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
                 state_halt: bool = True, detach_halt_input: bool = False, step_bias: bool = False,
                 halt_decay: float | None = None, halt_init: float | None = None, body_depth: int = 1,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loop_plan: dict[str, str] = dict(loop_plan or {})
        self.max_iterations = int(max_iterations)
        self.state_halt = bool(state_halt)
        self.detach_halt_input = bool(detach_halt_input)
        self.step_bias = bool(step_bias)
        self.halt_decay = halt_decay
        self.halt_init = halt_init
        self.body_depth = int(body_depth)
        self.halt_w = nn.ParameterDict()
        self.halt_b = nn.ParameterDict()

    def begin_task(self, task_id: str, depth: int | None = None, **kwargs):
        code = super().begin_task(task_id, depth=depth, **kwargs)
        if task_id not in self.loop_plan:
            return code
        if not 1 <= self.depth_of(task_id) <= self.body_depth:
            raise ValueError(f'{task_id}: a loop body has planned depth 1..{self.body_depth}')
        if not self.state_halt:
            self.halt_b[task_id] = nn.Parameter(self._initial_biases())
            return [code, self.halt_b[task_id]]
        self.halt_b[task_id] = nn.Parameter(self._initial_biases() if self.step_bias else torch.zeros(1))
        d = self.library[0].V.shape[1]
        self.halt_w[task_id] = nn.Parameter(torch.zeros(d))
        if self.halt_decay is None:
            return [code, self.halt_w[task_id], self.halt_b[task_id]]
        return [code, self.halt_b[task_id], {'params': [self.halt_w[task_id]], 'weight_decay': float(self.halt_decay)}]

    def _initial_biases(self) -> Tensor:
        if self.halt_init is None:
            return torch.zeros(self.max_iterations)
        c = float(self.halt_init)
        return torch.tensor([c] + [-c] * (self.max_iterations - 1))

    def loop_parameters(self, task_ids) -> list[nn.Parameter]:
        out = []
        for t in task_ids:
            if t in self.halt_b:
                out += [self.halt_b[t]] + ([self.halt_w[t]] if t in self.halt_w else [])
        return out

    def halt_logit(self, z: Tensor, task_id: str, step: int) -> Tensor:
        if task_id in self.halt_w:
            if self.detach_halt_input:
                z = z.detach()
            b = self.halt_b[task_id][step] if self.step_bias else self.halt_b[task_id]
            return z @ self.halt_w[task_id] + b
        return self.halt_b[task_id][step].expand(z.shape[0])

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        if task_id not in self.loop_plan:
            return super().forward(x, task_id)
        coefficients = self._route_coefficients(self.task_codes[task_id])
        depth = self.depth_of(task_id)
        z = x
        if self.training:
            alive = torch.ones(x.shape[0], 1, dtype=x.dtype)
            out = torch.zeros_like(x)
            for step in range(self.max_iterations):
                go = torch.sigmoid(self.halt_logit(z, task_id, step)).unsqueeze(1)
                out = out + alive * (1 - go) * z
                alive = alive * go
                z = self._route(z, coefficients, depth)
            return out + alive * z
        done = torch.zeros(x.shape[0], dtype=torch.bool)
        for step in range(self.max_iterations):
            done = done | ~(self.halt_logit(z, task_id, step) > 0)
            z = torch.where(done.unsqueeze(1), z, self._route(z, coefficients, depth))
        return z

    def iteration_counts(self, x: Tensor, task_id: str) -> Tensor:
        """Hard iteration count per input (evaluation semantics)."""
        coefficients = torch.nn.functional.one_hot(torch.argmax(self.task_codes[task_id], -1),
                                                   self.operator_slots).to(x.dtype)
        depth = self.depth_of(task_id)
        z, k = x, torch.zeros(x.shape[0], dtype=torch.long)
        done = torch.zeros(x.shape[0], dtype=torch.bool)
        with torch.no_grad():
            for step in range(self.max_iterations):
                done = done | ~(self.halt_logit(z, task_id, step) > 0)
                k = k + (~done).long()
                z = torch.where(done.unsqueeze(1), z, self._route(z, coefficients, depth))
        return k
