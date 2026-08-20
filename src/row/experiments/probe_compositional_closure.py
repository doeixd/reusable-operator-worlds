"""V5 checklist B7: do learned neural operators compose past training depth?

Review 41: the project has only composed d=16 operators 3-4 deep. V6-V8
assume depth 8+. This probe uses existing Continuous artifacts. Teacher
primitives are Hungarian-matched to learned slots; the same random
programs are then executed at depths 1, 2, 3, 4, 6, 8 through the
teacher and through the matched learned slots.

The quantity is NMSE of the learned composition against the teacher
composition on fresh inputs, plus a saturation check. Training depth is
3; the gate is whether depth-8 error is a small multiple of depth-3
error, not whether it is small in absolute terms.

Draft pass (notes/v5-sketch.txt): median depth-8 NMSE <= 5x depth-3
NMSE, and learned outputs are no more saturated than the teacher at
the same depth. A teacher that itself saturates at depth 8 is a
family-stability finding, reported separately, not a learner fail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from row.config import load_config
from row.experiments.learned_lifetime import _build_model, _tensor
from row.metrics import nmse
from row.world import World, WorldConfig

DEPTHS = (1, 2, 3, 4, 6, 8)
SATURATION = 0.95
PROGRAMS_PER_DEPTH = 64
PROBE_N = 256
PASS_RATIO = 5.0


def _load_continuous(artifact: Path):
    config = load_config(artifact / "config.yaml")
    model = _build_model(config, "continuous")
    try:
        checkpoint = torch.load(artifact / "model.pt", map_location="cpu", weights_only=True)
    except Exception:
        # Legacy local checkpoints embed NumPy scalars; trusted artifacts only.
        checkpoint = torch.load(artifact / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    world = World.generate(WorldConfig(
        seed=config.world.seed,
        state_dim=config.world.state_dim,
        teacher_rank=config.world.teacher_rank,
        teacher_primitives=config.world.teacher_primitives,
        program_length=config.world.program_length,
        tasks=config.world.tasks,
        examples_per_task=config.world.examples_per_task,
        evaluation_examples=config.world.evaluation_examples,
        reuse_rho=config.world.reuse_rho,
        alpha=config.world.alpha,
    ))
    return model, world, config


def match_slots(model, world, probe: np.ndarray) -> dict[int, int]:
    learned = [operator(_tensor(probe)).detach().cpu().numpy() for operator in model.basis]
    teacher = [primitive(probe) for primitive in world.library]
    distances = np.empty((len(teacher), len(learned)), dtype=np.float64)
    for i, teacher_out in enumerate(teacher):
        denom = float(np.var(teacher_out)) or 1.0
        for j, learned_out in enumerate(learned):
            distances[i, j] = float(np.mean(np.square(teacher_out - learned_out)) / denom)
    teacher_idx, learned_idx = linear_sum_assignment(distances)
    return {int(t): int(s) for t, s in zip(teacher_idx, learned_idx, strict=True)}


def compose_teacher(world, route: list[int], x: np.ndarray) -> np.ndarray:
    state = x
    for primitive_id in route:
        state = world.library[primitive_id](state)
    return state


def compose_learned(model, slot_of: dict[int, int], route: list[int], x: np.ndarray):
    state = _tensor(x)
    with torch.no_grad():
        for primitive_id in route:
            state = model.basis[slot_of[primitive_id]](state)
    return state.cpu().numpy()


def saturation_fraction(y: np.ndarray, threshold: float = SATURATION) -> float:
    return float(np.mean(np.abs(y) > threshold))


def probe_one(model, world, slot_of: dict[int, int], rng: np.random.Generator) -> dict:
    n_prim = len(world.library)
    d = world.library[0].U.shape[0]
    x = rng.normal(size=(PROBE_N, d))
    rows = []
    for depth in DEPTHS:
        nmses = []
        learned_sat = []
        teacher_sat = []
        teacher_var = []
        learned_var = []
        for _ in range(PROGRAMS_PER_DEPTH):
            route = rng.integers(0, n_prim, size=depth).tolist()
            teacher_y = compose_teacher(world, route, x)
            learned_y = compose_learned(model, slot_of, route, x)
            nmses.append(nmse(learned_y, teacher_y))
            learned_sat.append(saturation_fraction(learned_y))
            teacher_sat.append(saturation_fraction(teacher_y))
            teacher_var.append(float(np.var(teacher_y)))
            learned_var.append(float(np.var(learned_y)))
        rows.append({
            "depth": depth,
            "mean_nmse": float(np.mean(nmses)),
            "median_nmse": float(np.median(nmses)),
            "learned_saturation": float(np.mean(learned_sat)),
            "teacher_saturation": float(np.mean(teacher_sat)),
            "learned_variance": float(np.mean(learned_var)),
            "teacher_variance": float(np.mean(teacher_var)),
        })
    by_depth = {row["depth"]: row for row in rows}
    ratio = by_depth[8]["median_nmse"] / max(by_depth[3]["median_nmse"], 1e-12)
    teacher_exploded = by_depth[8]["teacher_saturation"] > 0.5
    learned_worse_sat = (
        by_depth[8]["learned_saturation"] > by_depth[8]["teacher_saturation"] + 0.1
    )
    passed = (ratio <= PASS_RATIO) and not learned_worse_sat
    return {
        "slot_match": slot_of,
        "depths": rows,
        "depth8_over_depth3_median_nmse": float(ratio),
        "teacher_saturated_at_8": teacher_exploded,
        "learned_more_saturated_than_teacher": learned_worse_sat,
        "pass": passed and not teacher_exploded,
        "pass_if_teacher_stable": passed,
        "pass_ratio_threshold": PASS_RATIO,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/checkpoints_development/world_0/continuous"),
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--artifact-template",
        default="artifacts/checkpoints_development/world_{world}/continuous",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/v5_compositional_closure.json"))
    args = parser.parse_args()

    print("COMPOSITIONAL CLOSURE PROBE  (V5 checklist, V6 gate)")
    print("  learned slots vs teacher primitives, random programs, depths 1-8\n")
    reports = []
    for world_seed in args.worlds:
        artifact = Path(args.artifact_template.format(world=world_seed))
        if not (artifact / "model.pt").exists():
            artifact = args.artifact
        model, world, config = _load_continuous(artifact)
        rng = np.random.default_rng(np.random.SeedSequence([config.world.seed, 41, 8]))
        probe = rng.normal(size=(512, config.world.state_dim))
        slot_of = match_slots(model, world, probe)
        result = probe_one(model, world, slot_of, rng)
        result["world"] = world_seed
        result["artifact"] = str(artifact)
        reports.append(result)
        print(f"  world {world_seed}  match {slot_of}")
        for row in result["depths"]:
            print(f"    d={row['depth']:<2}  median NMSE {row['median_nmse']:.4f}  "
                  f"learned sat {row['learned_saturation']:.3f}  "
                  f"teacher sat {row['teacher_saturation']:.3f}")
        verdict = "PASS" if result["pass"] else "FAIL"
        if result["teacher_saturated_at_8"]:
            verdict = "TEACHER SATURATED (family, not learner)"
        print(f"    depth8/depth3 = {result['depth8_over_depth3_median_nmse']:.2f}  {verdict}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
