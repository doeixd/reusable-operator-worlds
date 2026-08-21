"""H29: where does the higher-order structure disappear?

C3 established that a schema pays over teacher family operators and
fails over the promoted library. That leaves two very different
explanations, and review 48 names the measurement that separates them:

    R_meta(P_0)  >>  R_meta(L_promoted)   promotion DESTROYS an
                                          available family structure

    R_meta(P_0)  ~=  R_meta(L) ~= 0       the wake learner never
                                          represented the meta-structure
                                          in a recoverable form

P_0 is the set of member residuals a promotion consumed, snapshotted at
the sleep that consumed them — state a finished artifact does not
otherwise contain, which is why this needed fresh runs.

The two populations are compared with the SAME instrument used on the
teacher operators in the H20 gates: leave-one-out functional
shared-subspace capture, fitted on a probe set and evaluated on the
held-out object, against an isotropic null of matched size. Using the
in-sample number would manufacture structure — at r_meta = 0 it reads
0.73 where the truth is none.

A caveat that belongs with the result rather than after it: P_0
residuals are private task states and the promoted objects are
consolidations of them, so the two populations differ in more than
"before and after promotion". If R_meta is low in both, that is
informative. If it is high in P_0 only, the causal reading is available
but still wants the matched-count control this module reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from row.experiments.audit_learned_schema import effect
from row.experiments.audit_meta_recurrence import fit_schema, loo_capture
from row.world import _rng


def capture_of(vectors: np.ndarray, rank: int) -> float:
    return loo_capture(vectors, rank)


def isotropic_null(count: int, width: int, rank: int, seed: int,
                   draws: int = 10) -> float:
    """95th percentile of LOO capture on matched random objects."""

    scores = []
    for draw in range(draws):
        generator = np.random.default_rng(seed * 1000 + draw)
        scores.append(loo_capture(generator.normal(size=(count, width)), rank))
    return float(np.nanpercentile(scores, 95))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/v5_h29"))
    parser.add_argument("--conditions", nargs="+", default=["r100"])
    parser.add_argument("--worlds", type=int, nargs="+",
                        default=[600, 601, 602, 603, 604, 605])
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--residual-rank", type=int, default=2)
    parser.add_argument("--task-steps", type=int, default=3)
    parser.add_argument("--schema-rank", type=int, default=2)
    parser.add_argument("--probe", type=int, default=96)
    parser.add_argument("--max-pre", type=int, default=24,
                        help="cap on the P_0 population per world; the LOO "
                             "null is quadratic in it")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v5_promotion_structure.json"))
    args = parser.parse_args()

    d, rank, steps = args.state_dim, args.residual_rank, args.task_steps
    rows = []
    for condition in args.conditions:
        for world in args.worlds:
            path = args.root / condition / f"world_{world}" / "lifecycle"
            snapshot_file = path / "promotion_snapshots.npz"
            if not snapshot_file.exists():
                continue
            data = np.load(snapshot_file)
            probe = _rng(world, 97).normal(size=(args.probe, d))

            born = [data[k] for k in sorted(data.files)
                    if k.startswith("born_")]
            members = [data[k] for k in sorted(data.files)
                       if k.startswith("members_")]
            if len(born) < args.schema_rank + 2:
                rows.append({"condition": condition, "world": world,
                             "reason": f"only {len(born)} promotions"})
                continue

            promoted = np.stack([
                effect(b.astype(np.float64), probe, d, rank, steps) for b in born
            ])
            # Subsample before computing effects. The full pre-sleep
            # population is ~70 objects per promotion and the effect
            # vectors are thousands of dimensions wide, so the LOO fit
            # and its null become thousands of SVDs -- the first version
            # of this scorer silently timed out and left a stale report
            # that read as "no structure before promotion".
            pool = [r for group in members if len(group) for r in group]
            if len(pool) > args.max_pre:
                pick = np.random.default_rng(world).choice(
                    len(pool), size=args.max_pre, replace=False)
                pool = [pool[i] for i in pick]
            pre = (np.stack([effect(r.astype(np.float64), probe, d, rank, steps)
                             for r in pool])
                   if pool else np.empty((0, promoted.shape[1])))

            row = {
                "condition": condition, "world": world,
                "promotions": len(born), "pre_objects": int(len(pre)),
                "r_meta_promoted": capture_of(promoted, args.schema_rank),
                "null_promoted": isotropic_null(
                    len(promoted), promoted.shape[1], args.schema_rank, world),
            }
            if len(pre) >= args.schema_rank + 2:
                row["r_meta_pre"] = capture_of(pre, args.schema_rank)
                row["null_pre"] = isotropic_null(
                    len(pre), pre.shape[1], args.schema_rank, world + 7)
                # Matched-count control: the populations differ in size,
                # and LOO capture depends on how many objects the fit
                # sees. Subsample P_0 to the promoted count before
                # comparing.
                index = np.random.default_rng(world).choice(
                    len(pre), size=len(promoted), replace=False)
                row["r_meta_pre_matched"] = capture_of(
                    pre[index], args.schema_rank)
            rows.append(row)

    scored = [r for r in rows if "reason" not in r]
    if not scored:
        print("no snapshots found — H29 needs runs made after the P_0 fix")
        for row in rows:
            print(f"  {row['condition']} world {row['world']}: {row['reason']}")
        return

    print("H29 — does promotion destroy the family structure, or was it never there?")
    print(f"  leave-one-out subspace capture, rank {args.schema_rank}, "
          f"probe {args.probe}\n")
    print(f"  {'world':>6} {'promos':>7} {'pre objs':>9} {'R_pre':>8} "
          f"{'R_pre(m)':>9} {'R_promoted':>11} {'null':>7}")
    for row in scored:
        print(f"  {row['world']:>6} {row['promotions']:>7} "
              f"{row['pre_objects']:>9} "
              f"{row.get('r_meta_pre', float('nan')):>8.3f} "
              f"{row.get('r_meta_pre_matched', float('nan')):>9.3f} "
              f"{row['r_meta_promoted']:>11.3f} {row['null_promoted']:>7.3f}")

    pre = float(np.nanmean([r.get("r_meta_pre", np.nan) for r in scored]))
    matched = float(np.nanmean(
        [r.get("r_meta_pre_matched", np.nan) for r in scored]))
    post = float(np.mean([r["r_meta_promoted"] for r in scored]))
    null = float(np.mean([r["null_promoted"] for r in scored]))
    print(f"\n  mean R_meta before promotion (all)     {pre:.3f}")
    print(f"  mean R_meta before promotion (matched) {matched:.3f}")
    print(f"  mean R_meta after promotion            {post:.3f}")
    print(f"  isotropic null                         {null:.3f}")

    print("\n  READING")
    if matched > null and matched > post + 0.15:
        print("    Structure was PRESENT before promotion and is degraded after.")
        print("    Promotion is causally destroying an available family structure,")
        print("    which makes REFACTOR a repair with something to repair.")
    elif matched <= null and post <= null:
        print("    Structure is ABSENT on both sides, at the isotropic null.")
        print("    The wake learner never represented the teacher's meta-structure")
        print("    in a recoverable form, so promotion destroys nothing — the")
        print("    question moves upstream, to what the wake learner encodes.")
    else:
        print("    Neither clean reading holds; report the numbers, not a story.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"rows": rows, "mean_pre": pre, "mean_pre_matched": matched,
         "mean_post": post, "null": null}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
