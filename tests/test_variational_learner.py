"""Unit tests for the V3 variational wake learner."""

from __future__ import annotations

import copy
import math
import unittest

import torch

from row.models import SharedParentResidualLearner, VariationalSharedResidualLearner


def _build(
    prior_scale_init: float = 1e-2, prior_warmup_tasks: int = 0
) -> VariationalSharedResidualLearner:
    return VariationalSharedResidualLearner(
        d=4,
        operator_slots=3,
        operator_rank=2,
        residual_rank=2,
        task_steps=3,
        alpha=0.2,
        seed=11,
        prior_scale_init=prior_scale_init,
        prior_warmup_tasks=prior_warmup_tasks,
    )


class VariationalLearnerTest(unittest.TestCase):
    def test_initial_code_is_precision_not_content(self) -> None:
        model = _build()
        model.begin_task("t0")
        # The posterior starts precise and the prior starts wide, so the
        # initial code is the precision term log(s / sigma) and carries
        # essentially no information about the task. Routes start exactly on
        # the prior mean; U and V carry the frozen shared-residual init's
        # 1e-3 noise, which is a fraction of a bit for the whole task.
        mean_information = model.coordinate_mean_information("t0").detach()
        self.assertAlmostEqual(
            float(torch.sum(mean_information[: model.route_size])), 0.0, places=9
        )
        self.assertLess(float(torch.sum(mean_information)) / math.log(2.0), 1.0)
        per_coordinate = float(
            torch.mean(model.coordinate_kl("t0").detach())
        )
        # Dominated by the precision term; the small excess is the residual
        # init noise's mean-information contribution.
        expected = math.log(1e-2 / 1e-3) - 0.5 + 0.5 * (1e-3 / 1e-2) ** 2
        self.assertAlmostEqual(per_coordinate, expected, places=2)

    def test_relaxing_precision_to_the_prior_empties_the_code(self) -> None:
        model = _build()
        model.begin_task("t0")
        with torch.no_grad():
            # A coordinate the task does not need: the learner relaxes its
            # posterior onto the prior and the code goes to zero.
            model.task_code_log_sigma["t0"].fill_(
                float(model.prior_log_scales[0])
            )
        self.assertAlmostEqual(
            float(torch.sum(model.coordinate_kl("t0").detach()[: model.route_size])),
            0.0,
            places=6,
        )

    def test_closed_form_prior_tracks_the_population(self) -> None:
        model = _build()
        model.begin_task("t0")
        model.begin_task("t1")
        with torch.no_grad():
            model.task_codes["t0"].fill_(0.4)
            model.task_codes["t1"].fill_(0.4)
        model.update_prior_scales()
        route_scale = float(torch.exp(model.prior_log_scales[0]))
        # s^2 = mean(mu^2 + sigma^2) with mu = 0.4 and sigma = 1e-3.
        self.assertAlmostEqual(route_scale, 0.4, places=4)

    def test_prior_does_not_run_away_when_repeatedly_updated(self) -> None:
        model = _build()
        model.begin_task("t0")
        with torch.no_grad():
            model.task_codes["t0"].fill_(0.25)
        for _ in range(500):
            model.update_prior_scales()
        # Gradient-learned shared priors collapse under repeated updates; the
        # closed-form M step is a fixed point at the population spread.
        self.assertAlmostEqual(float(torch.exp(model.prior_log_scales[0])), 0.25, places=4)

    def test_prior_update_waits_for_a_population(self) -> None:
        model = _build(prior_warmup_tasks=8)
        before = float(torch.exp(model.prior_log_scales[0]))
        for index in range(7):
            model.begin_task(f"t{index}")
            with torch.no_grad():
                model.task_codes[f"t{index}"].fill_(0.3)
            model.update_prior_scales([f"t{i}" for i in range(index + 1)])
        # Fewer tasks than the warmup: the wide initial prior still stands, so
        # early task codes are not strangled by a prior estimated from
        # untrained state.
        self.assertEqual(float(torch.exp(model.prior_log_scales[0])), before)
        model.begin_task("t7")
        with torch.no_grad():
            model.task_codes["t7"].fill_(0.3)
        model.update_prior_scales([f"t{i}" for i in range(8)])
        self.assertAlmostEqual(float(torch.exp(model.prior_log_scales[0])), 0.3, places=4)

    def test_scoring_paths_are_deterministic_and_use_the_mean(self) -> None:
        model = _build()
        model.begin_task("t0")
        x = torch.randn(5, 4)
        model.eval()
        with torch.no_grad():
            first = model(x, "t0")
            second = model(x, "t0")
        self.assertTrue(torch.equal(first, second))
        # A no_grad forward in training mode must still use the mean: every
        # scoring path in the harness runs under no_grad.
        model.train()
        with torch.no_grad():
            third = model(x, "t0")
        self.assertTrue(torch.equal(first, third))

    def test_training_forward_samples(self) -> None:
        model = _build(prior_scale_init=0.5)
        model.begin_task("t0")
        x = torch.randn(5, 4)
        model.train()
        first = model(x, "t0")
        second = model(x, "t0")
        self.assertFalse(torch.equal(first, second))

    def test_sampling_does_not_touch_global_rng(self) -> None:
        model = _build(prior_scale_init=0.5)
        model.begin_task("t0")
        torch.manual_seed(1234)
        before = torch.randn(3)
        model.train()
        model(torch.randn(2, 4), "t0")
        torch.manual_seed(1234)
        after = torch.randn(3)
        self.assertTrue(torch.equal(before, after))

    def test_deepcopy_preserves_behavior(self) -> None:
        model = _build()
        model.begin_task("t0")
        clone = copy.deepcopy(model)
        x = torch.randn(4, 4)
        model.eval()
        clone.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x, "t0"), clone(x, "t0")))

    def test_retained_state_matches_shared_residual(self) -> None:
        variational = _build()
        baseline = SharedParentResidualLearner(
            d=4,
            operator_slots=3,
            operator_rank=2,
            residual_rank=2,
            task_steps=3,
            alpha=0.2,
            seed=11,
        )
        variational.begin_task("t0")
        baseline.begin_task("t0")
        # Posterior scales are training state, not retained state, so the
        # two-part comparison stays scalar-for-scalar matched.
        self.assertEqual(
            variational.task_state_scalar_count, baseline.task_state_scalar_count
        )
        self.assertEqual(
            variational.variational_training_state_scalar_count,
            baseline.task_state_scalar_count,
        )

    def test_prior_scales_are_shared_and_counted_in_shared_state(self) -> None:
        model = _build()
        model.begin_task("t0")
        model.begin_task("t1")
        names = {name for name, _ in model.named_parameters()}
        self.assertIn("prior_log_scales", names)
        self.assertEqual(model.prior_log_scales.numel(), 4)
        self.assertTrue(
            any(
                parameter is model.prior_log_scales
                for parameter in model.shared_parameters()
            )
        )

    def test_l1_surrogate_is_disabled(self) -> None:
        model = _build()
        model.begin_task("t0")
        self.assertEqual(float(model.storage_penalty(["t0"])), 0.0)

    def test_pruning_drops_uninformative_coordinates(self) -> None:
        model = _build()
        model.begin_task("t0")
        with torch.no_grad():
            # One coordinate whose mean is far from the prior mean; pruning
            # keys on mean information, not on posterior precision.
            model.task_codes["t0"][0] = 3.0
        report = model.apply_information_prune(0.5)
        self.assertEqual(report["retained_task_scalars"], 1)
        self.assertGreater(report["total_task_scalars"], 1)
        self.assertEqual(float(model.task_codes["t0"][1]), 0.0)
        self.assertEqual(float(model.task_codes["t0"][0]), 3.0)


if __name__ == "__main__":
    unittest.main()
