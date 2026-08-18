"""H9a analysis: residual allocation versus per-primitive recurrence.

Frozen procedure (V2 spec, H9a measurement): each task-step residual is
attributed to the teacher primitive occupying that position of the task's
hidden program (post-hoc ground-truth diagnostic; well-defined regardless
of slot recovery). Per world: Spearman rank correlation between
per-primitive measured recurrence and per-primitive mean residual
functional ratio over the six primitives (descriptive). Inference: the
sign of the per-world correlation across the ten development worlds,
exact binomial sign test. Secondary: Hungarian slot matching with the
ambiguity-counts-against rule (reported, no inferential weight).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

from row.mixed_world import generate_mixed_world
from row.models import SharedParentResidualLearner
from row.world import WorldConfig


@torch.no_grad()
def per_step_residual_ratios(
    model: SharedParentResidualLearner, task_id: str, probe: torch.Tensor
) -> list[float]:
    route, residual_u, residual_v, residual_b = model._unpack(task_id)
    coefficients = torch.softmax(route, dim=-1)
    z = probe
    ratios = []
    for step in range(model.task_steps):
        candidates = torch.stack([op(z) for op in model.basis], dim=0)
        parent = torch.sum(
            coefficients[step].view(model.operator_slots, 1, 1) * candidates, dim=0
        )
        hidden = torch.tanh(
            torch.nn.functional.linear(z, residual_v[step], residual_b[step])
        )
        residual = torch.nn.functional.linear(hidden, residual_u[step])
        parent_rms = float(torch.sqrt(torch.mean(parent**2)))
        residual_rms = float(torch.sqrt(torch.mean(residual**2)))
        ratios.append(residual_rms / max(parent_rms, 1e-12))
        z = parent + residual
    return ratios


def spearman(x: list[float], y: list[float]) -> float:
    def rank(v):
        order = np.argsort(v)
        ranks = np.empty(len(v))
        ranks[order] = np.arange(len(v), dtype=float)
        return ranks

    rx, ry = rank(np.array(x)), rank(np.array(y))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze_world(run_dir: Path) -> dict[str, object]:
    raw = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    profile_info = json.loads(
        (run_dir / "rho_profile.json").read_text(encoding="utf-8")
    )
    profile = profile_info["rho_profile"]
    world_cfg = WorldConfig(**raw["world"])
    world = generate_mixed_world(world_cfg, profile)
    mc = raw["shared_residual_model"]
    model = SharedParentResidualLearner(
        d=world_cfg.state_dim,
        operator_slots=int(mc["operator_slots"]),
        operator_rank=int(mc["operator_rank"]),
        residual_rank=int(mc["residual_rank"]),
        task_steps=int(mc["task_steps"]),
        alpha=float(mc["operator_alpha_init"]),
        seed=int(mc["seed"]),
        learnable_alpha=bool(mc.get("learnable_alpha", True)),
        activation=str(mc.get("operator_activation", "tanh")),
    )
    for task in world.tasks:
        model.begin_task(task.task_id)
    try:
        ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    for key in ckpt["model_state_dict"]:
        for prefix in ("task_codes.", "task_residuals."):
            if key.startswith(prefix + "task_novel_composition"):
                name = key.removeprefix(prefix)
                if name not in model.task_codes:
                    model.begin_task(name)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    generator = np.random.default_rng(np.random.SeedSequence([world_cfg.seed, 97]))
    probe = torch.as_tensor(
        generator.normal(size=(1024, world_cfg.state_dim)), dtype=torch.float32
    )
    per_primitive: dict[int, list[float]] = {k: [] for k in range(len(profile))}
    for task in world.tasks:
        ratios = per_step_residual_ratios(model, task.task_id, probe)
        for position, ratio in enumerate(ratios):
            primitive = task.program.primitive_ids[position]
            per_primitive[primitive].append(ratio)
    recurrence = [
        row["measured_recurrence"]
        for row in profile_info["per_primitive_recurrence"]
    ]
    mean_ratio = [float(np.mean(per_primitive[k])) for k in range(len(profile))]
    rho_corr = spearman(recurrence, mean_ratio)
    return {
        "world_seed": world_cfg.seed,
        "profile": profile,
        "per_primitive_measured_recurrence": recurrence,
        "per_primitive_mean_residual_ratio": mean_ratio,
        "spearman_recurrence_vs_ratio": rho_corr,
        "prediction_holds": rho_corr < 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("artifacts/v2_mixed/canonical"))
    parser.add_argument("--output", type=Path, default=Path("reports/v2_mixed"))
    args = parser.parse_args()
    worlds = []
    for run_dir in sorted(args.runs.glob("world_*/shared_residual")):
        worlds.append(analyze_world(run_dir))
        w = worlds[-1]
        print(
            f"world {w['world_seed']}: spearman {w['spearman_recurrence_vs_ratio']:+.3f} "
            f"({'holds' if w['prediction_holds'] else 'fails'})"
        )
    negatives = sum(1 for w in worlds if w["prediction_holds"])
    n = len(worlds)
    from math import comb

    tail = sum(comb(n, k) for k in range(negatives, n + 1)) * 0.5**n
    p_two_sided = min(1.0, 2 * min(tail, 1 - tail + comb(n, negatives) * 0.5**n))
    report = {
        "scope": "H9a primary: allocation vs per-primitive recurrence; canonical mixed profile",
        "attribution": "task-step residual ratios attributed by true-route position (frozen procedure)",
        "worlds": worlds,
        "negative_correlations": negatives,
        "n_worlds": n,
        "sign_test_two_sided_p": p_two_sided,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "h9a-allocation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nH9a allocation: {negatives}/{n} worlds negative, p = {p_two_sided:.2e}")


if __name__ == "__main__":
    main()
