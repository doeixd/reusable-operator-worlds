"""Per-task depth chosen from a plan, for ONLINE lifetimes.

`learned_lifetime.run` calls `model.begin_task(task_id)` with no depth, so a
mixed-length online stream needs the learner to look each task's depth up
itself. This subclass does exactly that and nothing else: with an empty plan
every task takes the default depth and the model is the parent class, which at
uniform depth is the committed learner bitwise (see `variable_depth`).

Kept separate from `variable_depth` so N1, N1b and N1c's learner is untouched.
"""
from __future__ import annotations

from row.models.variable_depth import VariableDepthRotatedLearner


class PlannedDepthRotatedLearner(VariableDepthRotatedLearner):
    implementation = "planned_depth_rotation_v1"

    def __init__(self, *args, depth_plan: dict[str, int] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.depth_plan: dict[str, int] = dict(depth_plan or {})

    def begin_task(self, task_id: str, depth: int | None = None, **_ignored):
        if depth is None:
            depth = self.depth_plan.get(task_id)
        return super().begin_task(task_id, depth=depth)
