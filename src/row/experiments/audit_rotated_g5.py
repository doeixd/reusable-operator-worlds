"""G5: is the rotated substrate learnable at all?

`ROTATED_SUBSTRATE_SPEC.md`, frozen at Amendment 3. This is the ONE gate that
spends lifetimes; G1, G2 and G4 were teacher-side arithmetic and passed.

Amendment 3's registered criterion, 2 of 3 worlds:

  (a) LEARNABILITY, which decides G5. A library trained on the rotated world beats
      a FROM-SCRATCH learner on held-out programs by >= 0.75 log units of query
      NMSE under support-only route inference. That is E1's export margin and
      E5.1's eligibility margin, unchanged, so the rotated substrate is held to
      the bar the existing one already cleared.

  (b) COMPARABILITY, reported alongside. Final training NMSE within a factor of 2
      of the standard substrate's on its own world. A failure of (b) alone is
      "learnable but harder", not a failed gate.

The rotated lifetimes use the IDENTICAL config to the existing artifacts, so `Q`
is the only difference between the two substrates. No standard lifetime is re-run
and no existing artifact is touched.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_e0_export import git_commit, load_model
from row.experiments.audit_e1_export import ADAPT_STEPS, nmse, scratch_model
from row.experiments.audit_e8_length import adapt_cell
from row.experiments.learned_lifetime import run
from row.rotated_world import generate_rotated_world, rotated_library
from row.support_split_world import _build_tasks
from row.world import Program

WORLDS = (0, 1, 2)
HELD_OUT = 12
MARGIN = 0.75            # E1's export margin, unchanged
COMPARABILITY = 2.0
ARTIFACTS = Path("artifacts/g5_rotated")


def held_out_programs(config, world, rng, count: int):
    """Programs the lifetime never trained on, in the world's own index space."""
    trained = {tuple(t.program.primitive_ids) for t in world.tasks}
    out = []
    while len(out) < count:
        cand = tuple(int(v) for v in
                     rng.integers(0, config.world.teacher_primitives,
                                  config.world.program_length))
        if cand in trained or cand in out:
            continue
        out.append(cand)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/rotated_g5.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    base = load_config(args.config)
    out = {"frozen_spec": "ROTATED_SUBSTRATE_SPEC.md (Amendment 3)",
           "git_commit": git_commit(),
           "protocol": {"held_out": HELD_OUT, "margin": MARGIN,
                        "comparability": COMPARABILITY, "adapt_steps": ADAPT_STEPS,
                        "note": "identical config to the existing artifacts; Q is "
                                "the only difference between substrates"},
           "worlds": {}}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    for world_seed in WORLDS:
        cfg = replace(base, world=replace(base.world, seed=world_seed))
        rot_world = generate_rotated_world(cfg.world)

        # --- the two lifetimes: rotated, and standard on the SAME config -------
        # `run` persists through `_write_artifacts` to `config.output_directory`,
        # and the model is then loaded with `load_model` -- the same persistence
        # and loading path every other artifact in this project uses. `run`'s
        # returned "model" key is the kind STRING, not the object.
        trained, standard = {}, {}
        for name, world in (("rotated", rot_world), ("standard", None)):
            out_dir = ARTIFACTS / f"world_{world_seed}_{name}"
            run_cfg = replace(cfg, output_directory=out_dir)
            # A resume must validate its own intervention record, not merely
            # that a summary file exists -- the V6 allocation pool silently
            # reused cells trained under different settings. Refuse a mismatch.
            stamp = {"tasks": cfg.world.tasks, "seed": cfg.world.seed,
                     "slots": cfg.discrete_model.operator_slots, "arm": name}
            stamp_path = out_dir / "g5_stamp.json"
            if (out_dir / "summary.json").exists() and stamp_path.exists():
                stored = json.loads(stamp_path.read_text(encoding="utf-8"))
                if stored != stamp:
                    raise SystemExit(
                        f"FATAL: {out_dir} was produced under {stored}, not {stamp}. "
                        "Refusing to mix; delete it or use a fresh path.")
            else:
                run(run_cfg, kind="discrete", world=world)
                out_dir.mkdir(parents=True, exist_ok=True)
                stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
            summary_json = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            fn = summary_json["final_nmse"]
            slot = trained if name == "rotated" else standard
            slot["model"] = load_model(run_cfg, out_dir, "discrete")[0]
            slot["final_nmse"] = float(fn["median"] if isinstance(fn, dict) else fn)
            slot["prequential"] = float(
                summary_json["cumulative_prequential_gaussian_log_loss"])

        # --- clause (a): trained library vs scratch on held-out programs -------
        rng = np.random.default_rng(np.random.SeedSequence([1500, world_seed]))
        programs = held_out_programs(cfg, rot_world, rng, HELD_OUT)
        lib = rotated_library(cfg.world)
        # The SAME scratch construction E1 and E8 use -- `scratch_model(...)`, a
        # freshly built model -- not a short lifetime dressed as one. E5 shipped a
        # fine-tuning arm labelled "scratch" by getting this wrong.
        scratch = scratch_model(cfg, "discrete", 7717)

        r_vals, s_vals = [], []
        for index, program in enumerate(programs):
            task = _build_tasks(cfg.world, lib, [program],
                                [f"g5_{world_seed}_{index}"],
                                index_offset=95000 + index)[0]
            got_r = adapt_cell(trained["model"], task, f"g5R_{world_seed}_{index}",
                               cfg.world.program_length, False, lib, program,
                               steps=ADAPT_STEPS)
            got_s = adapt_cell(scratch, task, f"g5S_{world_seed}_{index}",
                               cfg.world.program_length, True, lib, program,
                               steps=ADAPT_STEPS)
            r_vals.append(got_r["query_nmse"])
            s_vals.append(got_s["query_nmse"])

        geo = lambda v: float(np.exp(np.mean(np.log(np.maximum(v, 1e-12)))))
        margin = math.log(geo(s_vals)) - math.log(geo(r_vals))
        ratio = trained["final_nmse"] / max(standard["final_nmse"], 1e-12)
        entry = {
            "rotated_final_nmse": trained["final_nmse"],
            "standard_final_nmse": standard["final_nmse"],
            "comparability_ratio": ratio,
            "held_out_trained_nmse": geo(r_vals),
            "held_out_scratch_nmse": geo(s_vals),
            "learnability_margin": margin,
            "clause_a_passes": bool(margin >= MARGIN),
            "clause_b_passes": bool(ratio <= COMPARABILITY)}
        out["worlds"][str(world_seed)] = entry
        print(f"[w{world_seed}] rotated final NMSE {trained['final_nmse']:.5f} vs "
              f"standard {standard['final_nmse']:.5f} (x{ratio:.2f}) | held-out "
              f"trained {geo(r_vals):.5f} scratch {geo(s_vals):.5f} -> margin "
              f"{margin:+.2f} "
              f"{'PASSES' if entry['clause_a_passes'] else 'FAILS'} (need {MARGIN})",
              flush=True)
        write(out, args.output)

    a = sum(1 for w in WORLDS if out["worlds"][str(w)]["clause_a_passes"])
    b = sum(1 for w in WORLDS if out["worlds"][str(w)]["clause_b_passes"])
    out["verdict"] = {"clause_a_worlds": a, "clause_b_worlds": b,
                      "G5": "PASSES" if a >= 2 else "FAILS",
                      "comparability": "comparable" if b >= 2 else "learnable but harder"}
    write(out, args.output)
    print(f"\nG5 learnability {a}/3, comparability {b}/3 -> {out['verdict']['G5']}"
          f" ({out['verdict']['comparability']})")


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
