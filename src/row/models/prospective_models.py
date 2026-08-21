"""V6: a learner that is rewarded for making related future tasks cheap.

V5's closing result is that ordinary wake learning does not preserve the
higher-order structure whose economics V5 itself established. H29
localized the loss to representation formation, and the population-span
test showed no post-hoc refactor of the finished objects can recover it.
So the intervention has to happen while the representation is being
formed.

The mechanism is deliberately small. After a task is learned, the
learner is shown a HELD-OUT SIBLING it will never be trained on
normally, adapts its task-specific parameters to a few of the sibling's
examples, and measures the loss on the sibling's remaining examples.
That query loss is then charged back to the SHARED parameters:

    theta'  = current shared representation
    phi'    = adapt(theta', sibling support)      task-local only
    penalty = L(sibling query | theta', phi')

Only the shared parameters move. The sibling's own task code is
discarded afterwards, so nothing about the sibling is retained except
its effect on the shared representation — which is the whole point: the
question is whether the shared representation becomes a better substrate
for learning relatives, not whether the learner memorized a relative.

FIRST-ORDER. The inner adaptation is detached, so the outer gradient
treats the adapted task code as a constant rather than backpropagating
through the inner steps. This is the Reptile/FOMAML approximation and it
is stated rather than hidden: full unrolls are the registered follow-up
if the existence result holds (review 52, failure mode 12).

WHAT THIS DOES NOT DO. It does not see the teacher's operator, the
family's parameters, or the query set during adaptation. Oracle
knowledge enters in exactly one place — WHICH task is offered as a
sibling — and removing that is V6.2.
"""

from __future__ import annotations

import torch

from row.models.lifecycle_models import LifecycleLibraryLearner


class ProspectiveLifecycleLearner(LifecycleLibraryLearner):
    """Lifecycle learner plus a prospective adaptation penalty."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Diagnostics only; no decision reads these.
        self.prospective_log: list[dict[str, float]] = []

    def prospective_penalty(
        self,
        sibling_id: str,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        steps: int = 4,
        inner_lr: float = 0.05,
        inner_optimizer: str = "adam",
        sigma: float = 0.1,
    ) -> torch.Tensor:
        """Loss on a held-out sibling after adapting only its task code.

        Returns a differentiable scalar whose gradient flows into the
        SHARED parameters. The sibling's task code is created for this
        measurement and removed afterwards.
        """

        fresh = sibling_id not in self.task_codes
        if fresh:
            self.begin_task(sibling_id)
        code = self.task_codes[sibling_id]
        residual = self.task_residuals[sibling_id]
        original = (code.detach().clone(), residual.detach().clone())

        # Inner loop: task-local parameters only, detached from the
        # outer graph. The shared representation is what is being
        # judged, so it must not be updated here.
        #
        # ADAM, NOT SGD. Measured on a trained model: four SGD steps at
        # lr 0.05 reduce the support loss by 0.000% and move the task
        # code by 8e-4, because gradients here are ~1e-3. The penalty
        # was therefore the query loss of an UNADAPTED code -- i.e. it
        # applied "make siblings predictable with no adaptation", which
        # is the explicit-family-sharing objective, not the registered
        # prospective one. Adam at the task learning rate is what the
        # lifetime itself uses to fit a task code.
        inner = (
            torch.optim.Adam([code, residual], lr=inner_lr)
            if inner_optimizer == "adam"
            else torch.optim.SGD([code, residual], lr=inner_lr)
        )
        for _ in range(steps):
            inner.zero_grad()
            prediction = self(support_x, sibling_id)
            loss = torch.mean((prediction - support_y) ** 2)
            loss.backward(inputs=[code, residual])
            inner.step()

        # Outer measurement: query loss with the adapted task code held
        # fixed, so the gradient reaching the shared parameters answers
        # "would a better shared representation have made this cheaper?"
        with torch.no_grad():
            adapted = (code.detach().clone(), residual.detach().clone())
        code.data.copy_(adapted[0])
        residual.data.copy_(adapted[1])
        prediction = self(query_x, sibling_id)
        penalty = torch.mean((prediction - query_y) ** 2) / (2 * sigma * sigma)

        self.prospective_log.append({
            "sibling": sibling_id,
            "query_mse": float(torch.mean((prediction - query_y) ** 2).detach()),
        })

        # Restore: the sibling leaves no trace in task-local state.
        with torch.no_grad():
            code.data.copy_(original[0])
            residual.data.copy_(original[1])
        return penalty

    def forget_task(self, task_id: str) -> None:
        """Drop a task's local state entirely (used for probe siblings)."""

        # ParameterDict does not support pop(); delete by key.
        if task_id in self.task_codes:
            del self.task_codes[task_id]
        if task_id in self.task_residuals:
            del self.task_residuals[task_id]
        self.task_reference.pop(task_id, None)
        self.retired.discard(task_id)
