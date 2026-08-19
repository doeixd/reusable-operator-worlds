"""V4R §1: the opportunity census.

Asks, with the V3 learner FROZEN and no operator implemented, which
structural edit has any oracle advantage at all in a given regime. V4
built three operators and found no opportunity for any of them; the
census exists so that never happens again.

Every edit is scored in both currencies at once -- nats of held-out
Gaussian loss paid against nats of description saved -- and the answer
for a regime is the edit with the largest positive net, or KEEP if none
is positive.

The ladder is ordered by structural ambition (V4R §0.1):

    KEEP  <  COMPRESS  <  FACTORIZE

COMPRESS is scored FIRST and FACTORIZE must beat it, not merely beat
KEEP. That ordering is the whole lesson of the V4.2 failure: a shared
parameterized family looked like a 1,192-1,637 nat win against full
precision atoms and lost 9/10 worlds to simply storing the same atoms at
coarser precision for the same bits.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_factorization import _load, _loss
from row.task_group_world import TaskGroupSpec

LN2 = math.log(2.0)
BITS = 8


def census(config, path: Path, world_seed: int, spec: TaskGroupSpec, slots: int,
           precisions=(1, 2, 3, 4, 6), ranks=(1, 2, 4)) -> dict:
    model, world = _load(config, path, world_seed, spec, slots)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    lifecycle = summary.get("lifecycle")
    if lifecycle is None or "task_reference" not in lifecycle:
        raise ValueError(f"{path} has no persisted reference table")
    references = {k: int(v) for k, v in lifecycle["task_reference"].items()}
    for task_id, reference in references.items():
        model.task_reference[task_id] = reference
        model.retired.add(task_id)
    tasks = {task.task_id: task for task in world.tasks}
    probe = _tensor_probe(world_seed, config.world.state_dim)

    original = [p.detach().clone() for p in model.abstractions]
    count = len(original)
    if count < 2:
        return {"world_seed": world_seed, "library_size": count,
                "winner": "KEEP", "reason": "library too small to restructure"}
    width = model.residual_u_size + model.residual_v_size + model.residual_b_size
    full_bits = BITS * width * count
    ceiling = _loss(model, tasks, references)

    def restore():
        with torch.no_grad():
            for index in range(count):
                model.abstractions[index].copy_(original[index])

    def score(vectors, bits_after) -> float:
        with torch.no_grad():
            for index in range(count):
                model.abstractions[index].copy_(vectors[index])
            value = _loss(model, tasks, references)
        restore()
        return LN2 * (full_bits - bits_after) - (value - ceiling)

    compress = []
    for bits in precisions:
        budget = bits * width * count
        compress.append({"per_scalar_bits": bits, "bits_after": budget,
                         "net_nats": score(model.quantize_to_budget(budget), budget)})
    best_compress = max(compress, key=lambda row: row["net_nats"])

    factorize = []
    for rank in ranks:
        report = model.factorize(probe, rank=rank)
        factorize.append({
            "rank": rank,
            "bits_after": report["bits_after"],
            "net_nats": score(report["rebuilt"], report["bits_after"]),
            # The matched-budget comparison: the SAME bits spent privately.
            "matched_compress_net_nats": score(
                model.quantize_to_budget(report["bits_after"]), report["bits_after"]
            ),
        })
    best_factorize = max(factorize, key=lambda row: row["net_nats"])

    beats_matched = best_factorize["net_nats"] > best_factorize["matched_compress_net_nats"]
    if best_factorize["net_nats"] > best_compress["net_nats"] and beats_matched:
        winner = "FACTORIZE"
    elif best_compress["net_nats"] > 0:
        winner = "COMPRESS"
    else:
        winner = "KEEP"
    return {
        "world_seed": world_seed,
        "library_size": count,
        "dependents": len(references),
        "compress": compress,
        "factorize": factorize,
        "best_compress_nats": best_compress["net_nats"],
        "best_compress_bits": best_compress["per_scalar_bits"],
        "best_factorize_nats": best_factorize["net_nats"],
        "factorize_beats_matched_budget": beats_matched,
        "winner": winner,
    }


def _tensor_probe(seed: int, dim: int) -> torch.Tensor:
    return torch.as_tensor(
        np.random.default_rng(seed + 4242).normal(size=(256, dim)), dtype=torch.float32
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v4r_census"))
    parser.add_argument("--families", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arm", default="lifecycle")
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("reports/v4r_census.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    results = []
    print("V4R OPPORTUNITY CENSUS   (learner frozen; no operator implemented)")
    print("  KEEP < COMPRESS < FACTORIZE; FACTORIZE must beat MATCHED-BUDGET compression")
    for families in args.families:
        spec = TaskGroupSpec(
            groups=families, eta=0.9, future_tasks=8, family_onset=16,
            new_primitive_families=True,
        )
        print(f"\n  F = {families} families")
        print("    %-6s %6s %14s %14s %10s" %
              ("world", "|L|", "COMPRESS net", "FACTORIZE net", "winner"))
        for world in args.worlds:
            row = census(config, args.root / f"F{families}" / f"world_{world}" / args.arm,
                         world, spec, args.slots)
            row["families"] = families
            results.append(row)
            if row.get("reason"):
                print("    %-6d %6d %s" % (world, row["library_size"], row["reason"]))
                continue
            print("    %-6d %6d %14.0f %14.0f %10s" %
                  (world, row["library_size"], row["best_compress_nats"],
                   row["best_factorize_nats"], row["winner"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tally = {}
    for row in results:
        key = (row["families"], row["winner"])
        tally[key] = tally.get(key, 0) + 1
    print("\n  regime -> oracle-optimal edit:")
    for (families, winner), n in sorted(tally.items()):
        print(f"    F={families}: {winner} in {n} world(s)")


if __name__ == "__main__":
    main()
