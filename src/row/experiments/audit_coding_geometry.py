"""H27: why are shared scalars individually cheaper than private ones?

V5.0 measured D*_shared ~ 3.9 bits/scalar against D*_private ~ 5.0.
Shared abstractions are not merely stored fewer times; each of their
scalars costs less. Four mechanisms were registered — noise
purification, effective-dimension reduction, a PROMOTE selection
effect, and representation restructuring — and H27 predicts the gap
tracks a SPECTRAL difference rather than size or usage.

Measured here, on frozen artifacts, with no new lifetimes:

    D* bits/scalar      rate needed so the object's FUNCTION survives
                        quantization, contribution-relative
    spectral ratio      sigma_2 / sigma_1 of the object's effect on a
                        probe set: how much of its available functional
                        dimension it actually uses

Both populations are compared at MATCHED PARTICIPANT COUNT, because an
uncontrolled comparison would confound the mechanism with a selection
effect — and distinguishing those is the point.

WHAT THIS FILE DOES NOT DO. H29's causal decomposition needs the same
cluster measured at three moments: P_0 the private residuals before
promotion, P_1 the fitted shared residual before further training, P_2
that residual after post-promotion SGD. A finished lifecycle artifact
holds P_2 and the surviving private residuals; it does NOT hold P_0,
because nothing checkpoints the residuals at the sleep that promoted
them. That is a provenance gap of the same kind as the missing
`task_reference` table that once voided a coding audit, and it is
reported rather than approximated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_learned_schema import effect, private_bits


def spectral_ratio(vector: np.ndarray, probe: np.ndarray, d: int, rank: int,
                   steps: int) -> float:
    """sigma_2 / sigma_1 of the object's effect, averaged over steps."""

    from row.experiments.audit_learned_schema import split

    u, v, b = split(vector, d, rank, steps)
    ratios = []
    for step in range(steps):
        hidden = np.tanh(probe @ v[step].T + b[step])
        singular = np.linalg.svd(hidden @ u[step].T, compute_uv=False)
        if singular[0] > 0 and len(singular) > 1:
            ratios.append(float(singular[1] / singular[0]))
    return float(np.mean(ratios)) if ratios else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+",
                        default=[Path("artifacts/v5_h20b")])
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--residual-rank", type=int, default=2)
    parser.add_argument("--task-steps", type=int, default=3)
    parser.add_argument("--probe", type=int, default=256)
    parser.add_argument("--distortion", type=float, default=1e-4)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v5_coding_geometry.json"))
    args = parser.parse_args()

    d, rank, steps = args.state_dim, args.residual_rank, args.task_steps
    rows = []
    for root in args.roots:
        for model_path in sorted(root.glob("**/model.pt")):
            state = torch.load(model_path, weights_only=True)["model_state_dict"]
            shared = [
                state[k].detach().cpu().numpy().astype(np.float64)
                for k in sorted(state) if k.startswith("abstractions.")
            ]
            private = [
                state[k].detach().cpu().numpy().astype(np.float64)
                for k in sorted(state) if k.startswith("task_residuals.")
            ]
            if not shared or not private:
                continue
            probe = np.random.default_rng(7).normal(size=(args.probe, d))
            # MATCHED PARTICIPANT COUNT: compare the shared objects
            # against an equally sized sample of private ones, so a
            # difference cannot be a population-size artifact.
            sample = np.random.default_rng(11).choice(
                len(private), size=min(len(shared), len(private)), replace=False)
            chosen = [private[i] for i in sample]

            def describe(objects):
                bits = [private_bits(o, probe, d, rank, steps, args.distortion)[1]
                        for o in objects]
                ratios = [spectral_ratio(o, probe, d, rank, steps) for o in objects]
                scale = [float(np.sqrt(np.mean(
                    effect(o, probe, d, rank, steps) ** 2))) for o in objects]
                return bits, ratios, scale

            shared_bits, shared_ratio, shared_scale = describe(shared)
            private_bits_, private_ratio, private_scale = describe(chosen)
            rows.append({
                "artifact": str(model_path.parent),
                "n_shared": len(shared), "n_private_sampled": len(chosen),
                "shared_bits": float(np.mean(shared_bits)),
                "private_bits": float(np.mean(private_bits_)),
                "shared_spectral_ratio": float(np.nanmean(shared_ratio)),
                "private_spectral_ratio": float(np.nanmean(private_ratio)),
                "shared_scale": float(np.mean(shared_scale)),
                "private_scale": float(np.mean(private_scale)),
            })

    if not rows:
        print("no artifacts with both shared and private objects")
        return

    print("H27 CODING GEOMETRY — shared vs private, matched participant count\n")
    print(f"  {'artifact':<46} {'D* shared':>10} {'D* priv':>9} "
          f"{'s2/s1 sh':>9} {'s2/s1 pr':>9}")
    for row in rows:
        print(f"  {row['artifact'][-46:]:<46} {row['shared_bits']:>10.2f} "
              f"{row['private_bits']:>9.2f} {row['shared_spectral_ratio']:>9.3f} "
              f"{row['private_spectral_ratio']:>9.3f}")

    bit_gap = float(np.mean([r["private_bits"] - r["shared_bits"] for r in rows]))
    ratio_gap = float(np.mean(
        [r["private_spectral_ratio"] - r["shared_spectral_ratio"] for r in rows]))
    gaps = [(r["private_bits"] - r["shared_bits"],
             r["private_spectral_ratio"] - r["shared_spectral_ratio"]) for r in rows]
    if len(gaps) > 2:
        x = np.array([g[0] for g in gaps])
        y = np.array([g[1] for g in gaps])
        order_x, order_y = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        correlation = float(np.corrcoef(order_x, order_y)[0, 1])
    else:
        correlation = float("nan")

    print(f"\n  mean D* gap (private - shared)      {bit_gap:+.3f} bits/scalar")
    print(f"  mean spectral gap                   {ratio_gap:+.3f}")
    print(f"  rank correlation of the two gaps    {correlation:+.3f}"
          f"   (H27 registers |r| >= 0.5)")
    print(f"  artifacts: {len(rows)}")
    print("\n  H29's P_0/P_1/P_2 decomposition is NOT computed: a finished")
    print("  artifact holds P_2 and the surviving private residuals but not")
    print("  P_0, the residuals as they stood BEFORE promotion. Nothing")
    print("  checkpoints them at the promoting sleep. Reported as a")
    print("  provenance gap, not approximated.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"rows": rows, "mean_bit_gap": bit_gap, "mean_spectral_gap": ratio_gap,
         "rank_correlation": correlation,
         "h29": "not computed; P_0 is not checkpointed"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
