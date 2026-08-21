"""V6.1: the prospective adaptation penalty must be a clean instrument."""

from __future__ import annotations

import unittest

import torch

from row.models.prospective_models import ProspectiveLifecycleLearner


def _learner():
    return ProspectiveLifecycleLearner(
        d=8, operator_slots=4, operator_rank=4, residual_rank=1,
        task_steps=3, alpha=0.2, seed=0,
    )


def _batch(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(16, 8, generator=generator),
            torch.randn(16, 8, generator=generator))


class ProspectivePenaltyTest(unittest.TestCase):
    def test_gradient_reaches_the_shared_representation(self) -> None:
        # The whole point: the penalty must be able to change the shared
        # parameters. If it only moved task-local state it would be a
        # measurement, not a pressure.
        learner = _learner()
        support, query = _batch(0), _batch(1)
        learner.prospective_penalty("sib", *support, *query, steps=2).backward()
        shared = [p for name, p in learner.named_parameters()
                  if name.startswith("basis")]
        self.assertTrue(shared)
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0
                            for p in shared))

    def test_the_sibling_leaves_no_trace(self) -> None:
        # Oracle knowledge may pick WHICH task is offered. It may not
        # leave the learner holding a trained code for it.
        learner = _learner()
        support, query = _batch(2), _batch(3)
        learner.prospective_penalty("sib", *support, *query, steps=2)
        learner.forget_task("sib")
        self.assertNotIn("sib", learner.task_codes)
        self.assertNotIn("sib", learner.task_residuals)

    def test_existing_task_state_is_restored(self) -> None:
        learner = _learner()
        learner.begin_task("real")
        with torch.no_grad():
            learner.task_codes["real"].normal_()
        before = learner.task_codes["real"].detach().clone()
        support, query = _batch(4), _batch(5)
        learner.prospective_penalty("real", *support, *query, steps=3)
        self.assertTrue(torch.equal(before, learner.task_codes["real"].detach()))

    def test_penalty_is_finite_and_positive(self) -> None:
        learner = _learner()
        support, query = _batch(6), _batch(7)
        penalty = learner.prospective_penalty("sib", *support, *query, steps=2)
        self.assertTrue(torch.isfinite(penalty))
        self.assertGreater(float(penalty.detach()), 0.0)

    def test_adaptation_reduces_support_loss(self) -> None:
        # If the inner loop does not actually adapt, the penalty is just
        # a random-init loss and carries no information about fertility.
        learner = _learner()
        support, query = _batch(8), _batch(9)
        learner.begin_task("probe")
        with torch.no_grad():
            first = float(torch.mean((learner(support[0], "probe") - support[1]) ** 2))
        code = learner.task_codes["probe"]
        residual = learner.task_residuals["probe"]
        inner = torch.optim.SGD([code, residual], lr=0.05)
        for _ in range(8):
            inner.zero_grad()
            loss = torch.mean((learner(support[0], "probe") - support[1]) ** 2)
            loss.backward(inputs=[code, residual])
            inner.step()
        with torch.no_grad():
            last = float(torch.mean((learner(support[0], "probe") - support[1]) ** 2))
        self.assertLess(last, first)


if __name__ == "__main__":
    unittest.main()


class InnerAdaptationTest(unittest.TestCase):
    """The penalty must measure ADAPTATION, not zero-shot loss."""

    def test_inner_loop_materially_reduces_support_loss(self) -> None:
        # The original inner loop used SGD at lr 0.05 and moved the
        # support loss by 0.000% on a trained model, so the penalty was
        # the query loss of an UNADAPTED code -- the
        # explicit-family-sharing objective wearing the prospective
        # one's name. This test is what would have caught it.
        learner = _learner()
        generator = torch.Generator().manual_seed(11)
        support_x = torch.randn(16, 8, generator=generator)
        support_y = torch.randn(16, 8, generator=generator)
        learner.begin_task("probe")
        code = learner.task_codes["probe"]
        residual = learner.task_residuals["probe"]
        with torch.no_grad():
            first = float(torch.mean(
                (learner(support_x, "probe") - support_y) ** 2))
        inner = torch.optim.Adam([code, residual], lr=0.05)
        for _ in range(16):
            inner.zero_grad()
            loss = torch.mean((learner(support_x, "probe") - support_y) ** 2)
            loss.backward(inputs=[code, residual])
            inner.step()
        with torch.no_grad():
            last = float(torch.mean(
                (learner(support_x, "probe") - support_y) ** 2))
        self.assertLess(last, 0.9 * first,
                        "inner adaptation must move the support loss by more "
                        "than 10%, or the penalty is not an adaptation cost")

    def test_default_inner_optimizer_is_adam(self) -> None:
        import inspect

        signature = inspect.signature(
            ProspectiveLifecycleLearner.prospective_penalty)
        self.assertEqual(
            signature.parameters["inner_optimizer"].default, "adam")


class NestedObjectiveNonVacuityTest(unittest.TestCase):
    """A nested objective has TWO non-vacuity requirements, not one.

    The inner learner must actually learn, AND the outer gradient must
    actually depend on that learning. V6 verified the second and not the
    first, and shipped an arm whose inner loop moved the support loss by
    0.000% -- so the "adaptation cost" it charged was a zero-shot loss
    and the prospective arm was silently running the family-sharing
    objective (review 54).
    """

    def _batches(self, seed: int):
        generator = torch.Generator().manual_seed(seed)
        return (torch.randn(16, 8, generator=generator),
                torch.randn(16, 8, generator=generator),
                torch.randn(16, 8, generator=generator),
                torch.randn(16, 8, generator=generator))

    def test_zero_step_and_full_step_objectives_differ(self) -> None:
        # The cheapest possible check that the adaptation objective is
        # alive: if k = 0 and k = many give the same penalty to float
        # noise, the objective is dead however plausible the code looks.
        learner = _learner()
        support_x, support_y, query_x, query_y = self._batches(21)
        zero = float(learner.prospective_penalty(
            "s0", support_x, support_y, query_x, query_y, steps=0).detach())
        learner.forget_task("s0")
        full = float(learner.prospective_penalty(
            "s1", support_x, support_y, query_x, query_y, steps=32).detach())
        self.assertGreater(
            abs(zero - full), 1e-3 * max(abs(zero), 1.0),
            f"zero-step penalty {zero:.6f} and 32-step penalty {full:.6f} are "
            "indistinguishable: the adaptation objective is not doing anything",
        )

    def test_more_adaptation_lowers_the_charged_cost(self) -> None:
        # Direction matters too: adapting further should make the
        # sibling cheaper, not merely different.
        #
        # The targets must be a FUNCTION of the inputs. With independent
        # random targets, 32 steps on 16 support points simply overfits
        # noise and the query loss rises (measured 59.2 -> 57.0 -> 75.5),
        # which says nothing about the mechanism.
        learner = _learner()
        generator = torch.Generator().manual_seed(22)
        weight = torch.randn(8, 8, generator=generator) * 0.3
        support_x = torch.randn(64, 8, generator=generator)
        query_x = torch.randn(64, 8, generator=generator)
        support_y = torch.tanh(support_x @ weight)
        query_y = torch.tanh(query_x @ weight)
        costs = []
        for index, steps in enumerate((0, 8, 32)):
            name = f"probe{index}"
            costs.append(float(learner.prospective_penalty(
                name, support_x, support_y, query_x, query_y,
                steps=steps).detach()))
            learner.forget_task(name)
        self.assertLess(costs[-1], costs[0],
                        f"32 steps ({costs[-1]:.4f}) did not beat 0 steps "
                        f"({costs[0]:.4f})")
