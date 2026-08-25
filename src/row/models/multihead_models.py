"""H53: several candidate organizations developing concurrently in one lifetime.

`H53_PARALLEL_FORMATION_PLAN.md` (Amendment 1). A HEAD is a candidate
organization made live during formation: its per-task route policy (the
`mask_group` mechanism that produced `L_4`) plus whatever state the sharing
level grants it. Every head predicts every task and is updated on the same
examples; what varies between the registered levels is which tensors the heads
share.

The learner IS its primary head (index 0) and additionally holds the followers,
so every existing runner call, diagnostic, promotion step and scorer sees an
ordinary `ParameterizedSlotLearner`. At `H = 1` there are no followers, nothing
is shared with anyone, and the object is the ordinary learner bitwise -- which
is what the plan's two equivalence controls check against `M_4` and `L_4`.

Sharing, per Amendment 1's table:

    object                       L1        L2        L3
    12 basis operators           shared    shared    shared
    argument matrices U_k        shared    HEAD      HEAD
    task route codes             shared    shared    HEAD
    task residuals               shared    shared    HEAD
    task slot arguments alpha    HEAD      HEAD      HEAD
    route policy / mask          HEAD      HEAD      HEAD
    promoted abstractions        shared    shared    shared
    retirement                   shared    shared    shared

Gradient rule (Amendment 1, stricter than a plain mean over head losses):
the backward pass accumulates `sum_h L_h`, then SHARED parameters' gradients
are scaled by `1/H` while HEAD-SPECIFIC parameters keep their own gradient at
full scale. A plain mean would also divide the head-specific gradients by `H`
and silently change each head's effective learning rate relative to `M_4` and
`L_4`. At `H = 1` both rules are the ordinary objective.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from row.models.pslot_models import ParameterizedSlotLearner

LEVELS = ("L1", "L2", "L3")
# what each level BRANCHES, beyond the always-branched alphas and masks
BRANCHED = {
    "L1": (),
    "L2": ("arguments",),
    "L3": ("arguments", "codes", "residuals"),
}


class MultiHeadPslotLearner(ParameterizedSlotLearner):
    def __init__(
        self,
        *args,
        head_names: tuple[str, ...] = ("SHAM",),
        sharing_level: str = "L1",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if sharing_level not in LEVELS:
            raise ValueError(f"sharing_level must be one of {LEVELS}")
        if not head_names:
            raise ValueError("at least one head is required")
        self.head_names = tuple(head_names)
        self.sharing_level = str(sharing_level)
        self._kwargs = dict(kwargs)
        self._args = args
        branch = BRANCHED[self.sharing_level]
        followers = []
        for _ in self.head_names[1:]:
            # Same construction and the same seed, so every head starts from an
            # identical state; only the sharing wiring and the mask differ.
            follower = ParameterizedSlotLearner(*args, **kwargs)
            follower.basis = self.basis
            follower.abstractions = self.abstractions
            follower.task_reference = self.task_reference
            follower.retired = self.retired
            if "arguments" not in branch:
                follower.argument_matrices = self.argument_matrices
                if self.pslot_count > 1:
                    follower.extra_argument_matrices = self.extra_argument_matrices
            if "codes" not in branch:
                follower.task_codes = self.task_codes
            if "residuals" not in branch:
                follower.task_residuals = self.task_residuals
            followers.append(follower)
        self.followers = nn.ModuleList(followers)
        # head name -> {task_id: slot position or None}; filled by the runner.
        self.head_assignment: dict[str, dict[str, int | None]] = {n: {} for n in self.head_names}

    def sync_heads(self) -> None:
        """Re-establish the shared bindings after the primary REBINDS one.

        Sharing is by object identity, but promotion's prune paths do
        `self.abstractions = nn.ParameterList(...)` rather than mutating in
        place, which silently leaves the followers holding the previous object
        while the SHARED `task_reference` already points at the new indices.
        Called before every forward and at every task boundary; assignment is a
        no-op when the object has not changed.
        """

        branch = BRANCHED[self.sharing_level]
        for follower in self.followers:
            if follower.basis is not self.basis:
                follower.basis = self.basis
            if follower.abstractions is not self.abstractions:
                follower.abstractions = self.abstractions
            follower.task_reference = self.task_reference
            follower.retired = self.retired
            if "arguments" not in branch and follower.argument_matrices is not self.argument_matrices:
                follower.argument_matrices = self.argument_matrices
                if self.pslot_count > 1:
                    follower.extra_argument_matrices = self.extra_argument_matrices
            if "codes" not in branch and follower.task_codes is not self.task_codes:
                follower.task_codes = self.task_codes
            if "residuals" not in branch and follower.task_residuals is not self.task_residuals:
                follower.task_residuals = self.task_residuals

    # ---- head access ---------------------------------------------------

    @property
    def heads(self) -> list[ParameterizedSlotLearner]:
        return [self] + list(self.followers)

    def head(self, name: str) -> ParameterizedSlotLearner:
        return self.heads[self.head_names.index(name)]

    @property
    def head_count(self) -> int:
        return len(self.head_names)

    @staticmethod
    def head_parameters(head) -> list[nn.Parameter]:
        """A head's OWN tensors, enumerated structurally.

        `head.parameters()` cannot be used for this: the followers are
        submodules of the primary, so every follower tensor would also appear
        in the primary's list and be misclassified as shared. That
        misclassification would have scaled the follower gradients by 1/H —
        precisely the effective-learning-rate error Amendment 1 exists to
        prevent — so this is enumerated explicitly instead.
        """

        out = list(head.basis.parameters())
        out.append(head.argument_matrices)
        if head.pslot_count > 1:
            out.append(head.extra_argument_matrices)
        out.extend(list(head.abstractions))
        out.extend(head.task_codes.values())
        out.extend(head.task_residuals.values())
        out.extend(head.task_alphas.values())
        return out

    def shared_parameter_ids(self) -> set[int]:
        """Ids of tensors reachable from more than one head."""
        if self.head_count == 1:
            return set()
        counts: dict[int, int] = {}
        for head in self.heads:
            for parameter in {id(p): p for p in self.head_parameters(head)}.values():
                counts[id(parameter)] = counts.get(id(parameter), 0) + 1
        return {i for i, c in counts.items() if c > 1}

    # ---- task state ----------------------------------------------------

    def begin_task(self, task_id: str, schema_index: int = 0):
        self.sync_heads()
        code, residual, alpha = super().begin_task(task_id)
        alphas = [alpha]
        for follower in self.followers:
            # Shared dicts already hold the task; begin_task on the follower
            # would overwrite the primary's fresh state, so create only what
            # this head owns.
            if follower.task_codes is not self.task_codes:
                follower.task_codes[task_id] = nn.Parameter(torch.zeros_like(code))
            if follower.task_residuals is not self.task_residuals:
                follower.task_residuals[task_id] = nn.Parameter(
                    self.initial_residual_state.detach().clone())
            shape = alpha.shape
            follower.task_alphas[task_id] = nn.Parameter(torch.zeros(shape))
            alphas.append(follower.task_alphas[task_id])
        return code, residual, alphas

    def forget_task(self, task_id: str) -> None:
        super().forget_task(task_id)
        for follower in self.followers:
            follower.task_alphas.pop(task_id, None)
            follower.task_mask.pop(task_id, None)
            if follower.task_codes is not self.task_codes:
                follower.task_codes.pop(task_id, None)
            if follower.task_residuals is not self.task_residuals:
                follower.task_residuals.pop(task_id, None)

    def extra_task_param_groups(self, task_id: str, task_lr: float, residual_lr: float):
        """Optimizer groups for follower state the 3-tuple cannot express."""
        groups = []
        for follower in self.followers:
            if follower.task_codes is not self.task_codes:
                groups.append({"params": [follower.task_codes[task_id]],
                               "lr": task_lr, "weight_decay": 0.0})
            if follower.task_residuals is not self.task_residuals:
                groups.append({"params": [follower.task_residuals[task_id]],
                               "lr": residual_lr, "weight_decay": 0.0})
        return groups

    def shared_parameters(self) -> list[nn.Parameter]:
        """Union over heads, deduped: a shared tensor is optimized once."""
        seen, out = set(), []
        groups = [ParameterizedSlotLearner.shared_parameters(self)]
        groups += [follower.shared_parameters() for follower in self.followers]
        for group in groups:
            for parameter in group:
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    out.append(parameter)
        return out

    # ---- policy --------------------------------------------------------

    def apply_head_policies(self, task_id: str) -> None:
        """Pin each head's parameterized-slot mass per its own assignment."""
        self.sync_heads()
        for name, head in zip(self.head_names, self.heads):
            position = self.head_assignment[name].get(task_id)
            if position is None:
                head.task_mask.pop(task_id, None)
            else:
                head.task_mask[task_id] = int(head.pslot_indices[position])

    # ---- objective -----------------------------------------------------

    def multihead_loss(self, x: Tensor, y: Tensor, task_ids) -> Tensor:
        """Sum of the heads' ordinary MSE objectives on the same batch."""
        self.sync_heads()
        total = None
        for head in self.heads:
            loss = torch.nn.functional.mse_loss(head.forward_tasks(x, task_ids), y)
            total = loss if total is None else total + loss
        return total

    @torch.no_grad()
    def finalize_gradients(self) -> None:
        """Amendment 1: shared parameters take the MEAN of the head gradients."""
        if self.head_count == 1:
            return
        scale = 1.0 / self.head_count
        shared = self.shared_parameter_ids()
        for head in self.heads:
            for parameter in self.head_parameters(head):
                if id(parameter) in shared and parameter.grad is not None:
                    parameter.grad.mul_(scale)
                    shared.discard(id(parameter))   # one tensor, scaled once

    @torch.no_grad()
    def head_predictions(self, x: Tensor, task_id: str) -> dict[str, np.ndarray]:
        self.sync_heads()
        return {name: head(x, task_id).cpu().numpy()
                for name, head in zip(self.head_names, self.heads)}

    # ---- diagnostics ---------------------------------------------------

    @torch.no_grad()
    def neutralized_divergence(self, probe: Tensor, task_ids) -> dict[str, object]:
        """Amendment 1 control 3: compare LEARNED state, not hard-coded masks.

        Every head's mask is cleared and its route temperature reset before the
        probe runs, so a head whose only difference is its externally supplied
        policy scores zero here.
        """
        saved = [(dict(head.task_mask), float(head.route_temperature)) for head in self.heads]
        try:
            for head in self.heads:
                head.task_mask.clear()
                head.set_route_temperature(1.0)
            outputs = []
            for head in self.heads:
                outputs.append(torch.stack([head(probe, t) for t in task_ids]))
            pairs = {}
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    scale = float(torch.sqrt(torch.mean(outputs[i] ** 2))) or 1.0
                    pairs[f"{self.head_names[i]}|{self.head_names[j]}"] = float(
                        torch.sqrt(torch.mean((outputs[i] - outputs[j]) ** 2)) / scale)
        finally:
            for head, (mask, temperature) in zip(self.heads, saved):
                head.task_mask.clear()
                head.task_mask.update(mask)
                head.set_route_temperature(temperature)
        values = list(pairs.values())
        return {"pairwise": pairs, "mean": float(np.mean(values)) if values else 0.0}

    @torch.no_grad()
    def multihead_diagnostics(self, probe: Tensor | None = None, task_ids=None) -> dict[str, object]:
        out = {
            "head_names": list(self.head_names),
            "sharing_level": self.sharing_level,
            "head_count": self.head_count,
            "shared_parameter_tensors": len(self.shared_parameter_ids()),
            "masked_tasks_by_head": {
                name: len(head.task_mask) for name, head in zip(self.head_names, self.heads)
            },
            "head_state_scalars": {
                name: int(sum(p.numel() for p in self._head_own_parameters(index)))
                for index, name in enumerate(self.head_names)
            },
            "shared_state_scalars": int(sum(
                p.numel() for p in {id(q): q for q in self.head_parameters(self)}.values()
                if self.head_count == 1 or id(p) in self.shared_parameter_ids()
            )),
        }
        if probe is not None and task_ids:
            out["neutralized_divergence"] = self.neutralized_divergence(probe, task_ids)
            out["masked_divergence_tautological"] = self._masked_divergence(probe, task_ids)
        return out

    def _head_own_parameters(self, index: int) -> list[nn.Parameter]:
        shared = self.shared_parameter_ids()
        head = self.heads[index]
        seen, out = set(), []
        for parameter in self.head_parameters(head):
            if id(parameter) in shared or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            out.append(parameter)
        return out

    @torch.no_grad()
    def _masked_divergence(self, probe: Tensor, task_ids) -> float:
        """Reported, labelled, and used for nothing (Amendment 1 control 3)."""
        outputs = [torch.stack([head(probe, t) for t in task_ids]) for head in self.heads]
        vals = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                scale = float(torch.sqrt(torch.mean(outputs[i] ** 2))) or 1.0
                vals.append(float(torch.sqrt(torch.mean((outputs[i] - outputs[j]) ** 2)) / scale))
        return float(np.mean(vals)) if vals else 0.0

    # ---- persistence ---------------------------------------------------

    def save_extras(self, output: Path) -> None:
        super().save_extras(output)
        (output / "multihead.json").write_text(
            json.dumps({
                "diagnostics": self.multihead_diagnostics(),
                "head_assignment": {k: dict(v) for k, v in self.head_assignment.items()},
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        for index, name in enumerate(self.head_names):
            if index == 0:
                continue
            torch.save({"model_state_dict": self.heads[index].state_dict()},
                       output / f"head_{index}_{name.replace('-', '_')}.pt")
