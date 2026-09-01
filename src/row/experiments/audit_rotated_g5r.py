"""G5R: learnability with matched rotated teacher and learner families.

`ROTATED_SUBSTRATE_SPEC.md`, frozen at Amendment 4. G5 is retained as the
teacher/learner mismatch result. G5R changes only the learner operator family:
each slot now learns an independent exact orthogonal map, parameterized by
Householder reflections and initialized solely from the model seed.

The registered criterion is unchanged, in at least two of three worlds:

  (a) the trained library beats a matched from-scratch learner on held-out
      programs by >= 0.75 log units of query NMSE; and
  (b) final lifetime NMSE is within 2x the standard substrate's existing result.

The old G5 final NMSE is also carried as a floor, not as a new gate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_e0_export import git_commit, load_model
from row.experiments.audit_e1_export import ADAPT_STEPS, scratch_model
from row.experiments.audit_e8_length import adapt_cell
from row.experiments.audit_rotated_g5 import held_out_programs
from row.experiments.learned_lifetime import resolved_learned_config, run
from row.provenance import validate_artifact
from row.rotated_world import generate_rotated_world, rotated_library
from row.support_split_world import _build_tasks

WORLDS = (0, 1, 2)
HELD_OUT = 12
MARGIN = 0.75
COMPARABILITY = 2.0
ARTIFACTS = Path("artifacts/g5r_rotated")
STANDARD_ARTIFACTS = Path("artifacts/g5_rotated")
G5_REPORT = Path("reports/rotated_g5.json")
PROTOCOL = "G5R-Amendment4-householder-v1"


def _final_nmse(summary: dict) -> float:
    value = summary["final_nmse"]
    return float(value["median"] if isinstance(value, dict) else value)


def _comparability_label(learnability_worlds: int, comparability_worlds: int) -> str:
    """Keep the secondary clause from implying that failed G5R was learnable."""
    if learnability_worlds < 2:
        return (
            "not learnable despite comparability"
            if comparability_worlds >= 2
            else "not learnable; comparability also fails"
        )
    return "comparable" if comparability_worlds >= 2 else "learnable but harder"


def _validate_standard(config, world_seed: int) -> float:
    """Load G5's standard control only after its full provenance validates."""
    path = STANDARD_ARTIFACTS / f"world_{world_seed}_standard"
    stamp = {"tasks": config.world.tasks, "seed": world_seed,
             "slots": config.discrete_model.operator_slots, "arm": "standard"}
    stored = json.loads((path / "g5_stamp.json").read_text(encoding="utf-8"))
    if stored != stamp:
        raise SystemExit(f"FATAL: standard control {path} has stamp {stored}, not {stamp}")
    expected = resolved_learned_config(
        replace(config, output_directory=path), "discrete", "forward"
    )
    validate_artifact(path, expected, "discrete", backfill_missing_fingerprint=False)
    return _final_nmse(json.loads((path / "summary.json").read_text(encoding="utf-8")))


def _load_or_run(config, world) -> tuple[object, dict]:
    path = Path(config.output_directory)
    stamp = {
        "protocol": PROTOCOL,
        "tasks": config.world.tasks,
        "world_seed": config.world.seed,
        "model_seed": config.discrete_model.seed,
        "slots": config.discrete_model.operator_slots,
        "teacher_family": "rotated",
        "learner_family": "rotated_discrete",
        "rotation": "trainable Householder; d or d-1 reflections alternating by slot; model seed only",
    }
    stamp_path = path / "g5r_stamp.json"
    summary_path = path / "summary.json"
    if summary_path.exists():
        if not stamp_path.exists():
            raise SystemExit(f"FATAL: {path} has a summary but no G5R intervention stamp")
        stored = json.loads(stamp_path.read_text(encoding="utf-8"))
        if stored != stamp:
            raise SystemExit(f"FATAL: {path} was produced under {stored}, not {stamp}")
        expected = resolved_learned_config(config, "rotated_discrete", "forward")
        validate_artifact(
            path, expected, "rotated_discrete", backfill_missing_fingerprint=False
        )
    else:
        if stamp_path.exists():
            stored = json.loads(stamp_path.read_text(encoding="utf-8"))
            if stored != stamp:
                raise SystemExit(f"FATAL: partial artifact {path} has wrong stamp {stored}")
        path.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
        run(config, kind="rotated_discrete", world=world)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model = load_model(config, path, "rotated_discrete")[0]
    return model, summary


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/rotated_g5r.json"))
    parser.add_argument("--worlds", nargs="+", type=int, default=list(WORLDS))
    args = parser.parse_args()
    if tuple(args.worlds) != WORLDS:
        raise SystemExit(f"G5R is registered on exactly worlds {WORLDS}")
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("preregistration check failed")
    if not G5_REPORT.exists():
        raise SystemExit(f"missing retained G5 report: {G5_REPORT}")

    base = load_config(args.config)
    old_g5 = json.loads(G5_REPORT.read_text(encoding="utf-8"))
    out = {
        "frozen_spec": "ROTATED_SUBSTRATE_SPEC.md (Amendment 4)",
        "git_commit": git_commit(),
        "protocol": {
            "id": PROTOCOL,
            "held_out": HELD_OUT,
            "margin": MARGIN,
            "comparability": COMPARABILITY,
            "adapt_steps": ADAPT_STEPS,
            "learner_rotation": "trainable exact Householder map per slot",
            "initialization": "model seed only; alternating O(d) determinant components",
            "standard_source": str(STANDARD_ARTIFACTS),
            "mismatched_floor_source": str(G5_REPORT),
        },
        "worlds": {},
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    for world_seed in WORLDS:
        cfg = replace(base, world=replace(base.world, seed=world_seed))
        world = generate_rotated_world(cfg.world)
        path = ARTIFACTS / f"world_{world_seed}"
        model, summary = _load_or_run(replace(cfg, output_directory=path), world)
        matched_final = _final_nmse(summary)
        standard_final = _validate_standard(cfg, world_seed)
        mismatched_floor = float(old_g5["worlds"][str(world_seed)]["rotated_final_nmse"])

        rng = np.random.default_rng(np.random.SeedSequence([1500, world_seed]))
        programs = held_out_programs(cfg, world, rng, HELD_OUT)
        library = rotated_library(cfg.world)
        scratch = scratch_model(cfg, "rotated_discrete", 7717)
        trained_values, scratch_values = [], []
        for index, program in enumerate(programs):
            task = _build_tasks(
                cfg.world,
                library,
                [program],
                [f"g5r_{world_seed}_{index}"],
                index_offset=95000 + index,
            )[0]
            trained = adapt_cell(
                model,
                task,
                f"g5rL_{world_seed}_{index}",
                cfg.world.program_length,
                False,
                library,
                program,
                steps=ADAPT_STEPS,
            )
            scratch_result = adapt_cell(
                scratch,
                task,
                f"g5rS_{world_seed}_{index}",
                cfg.world.program_length,
                True,
                library,
                program,
                steps=ADAPT_STEPS,
            )
            trained_values.append(trained["query_nmse"])
            scratch_values.append(scratch_result["query_nmse"])

        geo = lambda values: float(
            np.exp(np.mean(np.log(np.maximum(values, 1e-12))))
        )
        trained_nmse = geo(trained_values)
        scratch_nmse = geo(scratch_values)
        margin = math.log(scratch_nmse) - math.log(trained_nmse)
        ratio = matched_final / max(standard_final, 1e-12)
        entry = {
            "matched_rotated_final_nmse": matched_final,
            "mismatched_g5_final_nmse": mismatched_floor,
            "beats_mismatched_floor": bool(matched_final < mismatched_floor),
            "standard_final_nmse": standard_final,
            "comparability_ratio": ratio,
            "held_out_trained_nmse": trained_nmse,
            "held_out_scratch_nmse": scratch_nmse,
            "learnability_margin": margin,
            "clause_a_passes": bool(margin >= MARGIN),
            "clause_b_passes": bool(ratio <= COMPARABILITY),
        }
        out["worlds"][str(world_seed)] = entry
        print(
            f"[w{world_seed}] matched final {matched_final:.5f} "
            f"(G5 floor {mismatched_floor:.5f}; standard {standard_final:.5f}, "
            f"x{ratio:.2f}) | held-out trained {trained_nmse:.5f} scratch "
            f"{scratch_nmse:.5f} -> margin {margin:+.2f} "
            f"{'PASSES' if entry['clause_a_passes'] else 'FAILS'}",
            flush=True,
        )
        write(out, args.output)

    a = sum(out["worlds"][str(w)]["clause_a_passes"] for w in WORLDS)
    b = sum(out["worlds"][str(w)]["clause_b_passes"] for w in WORLDS)
    floor = sum(out["worlds"][str(w)]["beats_mismatched_floor"] for w in WORLDS)
    out["verdict"] = {
        "clause_a_worlds": a,
        "clause_b_worlds": b,
        "beats_mismatched_floor_worlds": floor,
        "G5R": "PASSES" if a >= 2 else "FAILS",
        "comparability": _comparability_label(a, b),
    }
    write(out, args.output)
    print(
        f"\nG5R learnability {a}/3, comparability {b}/3, beats G5 floor "
        f"{floor}/3 -> {out['verdict']['G5R']} ({out['verdict']['comparability']})"
    )


if __name__ == "__main__":
    main()
