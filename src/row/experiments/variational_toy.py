"""Does a Gaussian task code make "no information" mean "no perturbation"?

A one-dimensional audit of the V3 wake parameterization, requested before
any further beta tuning. The world is y = x + delta_tau: half the tasks
truly need delta = 0, half need a known nonzero offset. The question is
whether a factorized Gaussian posterior against a shared Gaussian prior can
simultaneously achieve

    KL(q || p) ~ 0 for the unused tasks, and
    accurate adaptation for the used ones.

It cannot if the two requirements pull on the same scale: q = p is the
zero-INFORMATION state, but it is not the zero-PERTURBATION state, because
sampling from a prior wide enough to make a useful offset cheap injects
noise of that same width into every unused task's forward pass. The
optimizer must then buy quiet with precision (sigma << s) and pay
log(s / sigma) nats for a coordinate that carries nothing.

The gated variant codes presence instead: delta = g * v with a relaxed
Bernoulli g, so g = 0 is simultaneously zero information and exact
identity. Both are run here under the identical objective and budget.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

LN2 = math.log(2.0)
BITS_PER_SCALAR = 8


def _gaussian_kl(mu, log_sigma, prior_log_sigma):
    prior_variance = torch.exp(2.0 * prior_log_sigma)
    return (
        prior_log_sigma
        - log_sigma
        + (torch.exp(2.0 * log_sigma) + torch.square(mu)) / (2.0 * prior_variance)
        - 0.5
    )


class GaussianCode(nn.Module):
    """delta ~ N(mu, sigma^2) against a shared N(0, s^2)."""

    def __init__(self, tasks: int, dim: int, prior_scale: float, posterior_scale: float):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(tasks, dim))
        self.log_sigma = nn.Parameter(
            torch.full((tasks, dim), math.log(posterior_scale))
        )
        self.prior_log_scale = nn.Parameter(
            torch.tensor(math.log(prior_scale)), requires_grad=False
        )

    def sample(self, generator):
        noise = torch.randn(self.mu.shape, generator=generator)
        return self.mu + torch.exp(self.log_sigma) * noise

    def mean(self):
        return self.mu

    def kl(self):
        return _gaussian_kl(self.mu, self.log_sigma, self.prior_log_scale)

    def active_mask(self):
        # No null state exists, so activity has to be inferred with the same
        # mean-information criterion the ROW pruner uses.
        information_bits = (
            torch.square(self.mu)
            / (2.0 * torch.exp(2.0 * self.prior_log_scale))
        ) / LN2
        return information_bits >= 0.5

    def update_prior(self):
        with torch.no_grad():
            second = torch.mean(
                torch.square(self.mu) + torch.exp(2.0 * self.log_sigma)
            )
            self.prior_log_scale.copy_(torch.log(torch.sqrt(second).clamp_min(1e-6)))


class GatedCode(nn.Module):
    """delta = g * v with a relaxed Bernoulli gate; g = 0 is exact identity."""

    def __init__(self, tasks: int, dim: int, prior_scale: float, prior_presence: float):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(tasks, dim))
        self.gate_logit = nn.Parameter(torch.zeros(tasks, dim))
        self.prior_log_scale = nn.Parameter(
            torch.tensor(math.log(prior_scale)), requires_grad=False
        )
        self.prior_presence = float(prior_presence)
        self.temperature = 0.5

    def _gate(self, generator, hard: bool):
        probability = torch.sigmoid(self.gate_logit)
        if hard:
            return (probability > 0.5).to(probability.dtype)
        uniform = torch.rand(
            probability.shape, generator=generator
        ).clamp(1e-6, 1 - 1e-6)
        logistic = torch.log(uniform) - torch.log1p(-uniform)
        return torch.sigmoid((self.gate_logit + logistic) / self.temperature)

    def sample(self, generator):
        return self._gate(generator, hard=False) * self.value

    def mean(self):
        return self._gate(None, hard=True) * self.value

    def active_mask(self):
        return torch.sigmoid(self.gate_logit) > 0.5

    def kl(self):
        probability = torch.sigmoid(self.gate_logit)
        prior = self.prior_presence
        presence_kl = probability * torch.log(
            probability.clamp_min(1e-8) / prior
        ) + (1 - probability) * torch.log(
            (1 - probability).clamp_min(1e-8) / (1 - prior)
        )
        # The value is only transmitted when the gate is on.
        value_kl = probability * (
            torch.square(self.value)
            / (2.0 * torch.exp(2.0 * self.prior_log_scale))
        )
        return presence_kl + value_kl

    def update_prior(self):
        with torch.no_grad():
            probability = torch.sigmoid(self.gate_logit)
            weight = probability.sum().clamp_min(1e-6)
            second = (probability * torch.square(self.value)).sum() / weight
            self.prior_log_scale.copy_(
                torch.log(torch.sqrt(second).clamp_min(1e-6))
            )


def run(kind: str, tasks: int, dim: int, examples: int, beta: float, steps: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Half the tasks need nothing beyond the shared computation.
    used = np.zeros(tasks, dtype=bool)
    used[tasks // 2 :] = True
    true_delta = torch.zeros(tasks, dim)
    for index in range(tasks):
        if used[index]:
            true_delta[index] = torch.as_tensor(
                rng.normal(scale=1.0, size=dim), dtype=torch.float32
            )
    x = torch.as_tensor(rng.normal(size=(tasks, examples, dim)), dtype=torch.float32)
    y = x + true_delta.unsqueeze(1)

    model = (
        GaussianCode(tasks, dim, prior_scale=1.0, posterior_scale=1e-3)
        if kind == "gaussian"
        else GatedCode(tasks, dim, prior_scale=1.0, prior_presence=0.1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    sigma_y = 0.1
    # Same MDL conversion as the V3 wake learner: the KL is spread over the
    # N * d observations whose likelihood it accompanies.
    kl_scale = beta * 2.0 * sigma_y * sigma_y / (examples * dim)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        delta = model.sample(generator)
        prediction = x + delta.unsqueeze(1)
        loss = torch.nn.functional.mse_loss(prediction, y)
        loss = loss + kl_scale * torch.sum(model.kl(), dim=-1).mean()
        loss.backward()
        optimizer.step()
        if step % 50 == 49:
            model.update_prior()

    with torch.no_grad():
        delta = model.mean().detach()
        prediction = x + delta.unsqueeze(1)
        per_task_mse = torch.mean(torch.square(prediction - y), dim=(1, 2)).numpy()
        per_task_kl_bits = (torch.sum(model.kl(), dim=-1) / LN2).numpy()
        recovery = torch.mean(torch.abs(delta - true_delta), dim=-1).numpy()
        # LITERAL sparse code, charged identically for both models so the
        # comparison is not an artifact of KL being a divergence rather than
        # a code length: one presence bit per coordinate plus 8 bits per
        # active payload scalar. KL(q || p) = 0 means "no information beyond
        # the prior", which is not the same as "free to transmit", so the
        # gated model's zero-KL coordinates still pay their bitmap bit here.
        active = model.active_mask()
        literal_bits = (
            active.shape[1] + BITS_PER_SCALAR * active.sum(dim=-1).double()
        ).numpy()
    return {
        "model": kind,
        "beta": beta,
        "unused_mean_kl_bits": float(np.mean(per_task_kl_bits[~used])),
        "used_mean_kl_bits": float(np.mean(per_task_kl_bits[used])),
        "unused_mean_mse": float(np.mean(per_task_mse[~used])),
        "used_mean_mse": float(np.mean(per_task_mse[used])),
        "used_mean_absolute_recovery_error": float(np.mean(recovery[used])),
        "unused_mean_literal_sparse_bits": float(np.mean(literal_bits[~used])),
        "used_mean_literal_sparse_bits": float(np.mean(literal_bits[used])),
        "unused_active_fraction": float(np.mean(active.double().numpy()[~used])),
        "used_active_fraction": float(np.mean(active.double().numpy()[used])),
        "unused_mean_absolute_delta": float(np.mean(np.abs(delta.numpy()[~used]))),
        "prior_scale": float(torch.exp(model.prior_log_scale)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.1, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("reports/v3_variational_toy.json"))
    args = parser.parse_args()

    rows = []
    for kind in ("gaussian", "gated"):
        for beta in args.betas:
            per_seed = [
                run(kind, args.tasks, args.dim, args.examples, beta, args.steps, seed)
                for seed in args.seeds
            ]
            averaged = {
                key: float(np.mean([row[key] for row in per_seed]))
                for key in per_seed[0]
                if key != "model"
            }
            averaged["model"] = kind
            averaged["seeds"] = args.seeds
            rows.append(averaged)
            print(
                f"{kind:9s} beta={beta:<4} | unused: KL={averaged['unused_mean_kl_bits']:7.2f} "
                f"literal={averaged['unused_mean_literal_sparse_bits']:6.2f} "
                f"active={averaged['unused_active_fraction']:.2f} "
                f"|delta|={averaged['unused_mean_absolute_delta']:.4f} "
                f"| used: KL={averaged['used_mean_kl_bits']:7.2f} "
                f"literal={averaged['used_mean_literal_sparse_bits']:6.2f} "
                f"recovery_err={averaged['used_mean_absolute_recovery_error']:.4f}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
