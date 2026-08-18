"""Generate the seven paper figures from committed reports and artifacts."""

from __future__ import annotations

import collections
import json
import os
import statistics as st
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "savefig.bbox": "tight",
    }
)
CONT = "#1965B0"
DENSE = "#DC050C"
NEUTRAL = "#555555"


def load_confirmatory() -> dict[int, dict[float, tuple[float, float, float]]]:
    """world -> rho -> (paired effect, measured recurrence, novel effect)."""
    pair: dict[tuple[float, int], dict[str, dict]] = collections.defaultdict(dict)
    for f in glob("artifacts/rho_confirmatory/*/*/*/summary.json"):
        parts = os.path.normpath(f).split(os.sep)
        rho = float(parts[2].replace("rho_", "").replace("p", "."))
        world = int(parts[3].split("_")[1])
        model = parts[4]
        s = json.load(open(f))
        pair[(rho, world)][model] = s
    effects: dict[int, dict[float, tuple[float, float, float]]] = collections.defaultdict(dict)
    for (rho, world), models in pair.items():
        dense, cont = models["dense"], models["continuous"]
        effects[world][rho] = (
            dense["cumulative_prequential_gaussian_log_loss"]
            - cont["cumulative_prequential_gaussian_log_loss"],
            dense["world_functional_reuse"]["mean_pairwise_residual_correlation"],
            dense["novel_composition"]["nmse_by_support"]["32"]
            - cont["novel_composition"]["nmse_by_support"]["32"],
        )
    return effects


def fig1_regime_map(effects) -> None:
    rhos = sorted({rho for w in effects.values() for rho in w})
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for world, per in sorted(effects.items()):
        xs = sorted(per)
        ax.plot(xs, [per[r][0] for r in xs], color=NEUTRAL, alpha=0.18, lw=0.8, zorder=1)
    means = [st.mean(effects[w][r][0] for w in effects) for r in rhos]
    ax.plot(rhos, means, "o-", color=CONT, lw=2, ms=5, zorder=3, label="mean of 30 sealed worlds")
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between([0, 1], 0, ax.get_ylim()[1], color=CONT, alpha=0.05)
    ax.text(0.02, 0.95, "Continuous better", transform=ax.transAxes, color=CONT, va="top")
    ax.text(0.02, 0.06, "Dense better", transform=ax.transAxes, color=DENSE)
    crossings = []
    for w, per in effects.items():
        pts = sorted(per.items())
        for (x1, (y1, *_)), (x2, (y2, *_)) in zip(pts, pts[1:]):
            if y1 < 0 <= y2:
                crossings.append(x1 + (0 - y1) * (x2 - x1) / (y2 - y1))
    ax.axvline(st.mean(crossings), color=NEUTRAL, ls="--", lw=1)
    ax.annotate(
        f"crossover $\\rho^*$ = {st.mean(crossings):.3f} $\\pm$ {st.stdev(crossings):.3f}",
        (st.mean(crossings), ax.get_ylim()[0] * 0.7),
        textcoords="offset points",
        xytext=(-6, 0),
        ha="right",
        color=NEUTRAL,
    )
    ax.set_xlabel("configured reuse $\\rho$")
    ax.set_ylabel("Dense$-$Continuous lifetime loss (nats)\n(positive favors Continuous)")
    ax.set_title("Regime map on sealed worlds 100$-$129 (confirmatory)")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.28, 1.02))
    fig.savefig(OUT / "fig1_regime_map.png")
    plt.close(fig)


def fig2_dose_response(effects) -> None:
    xs, ys = [], []
    for per in effects.values():
        for rho, (eff, rec, _) in per.items():
            xs.append(rec)
            ys.append(eff)
    xs, ys = np.array(xs), np.array(ys)
    coef = np.polyfit(xs, ys, 1)
    yhat = np.polyval(coef, xs)
    r2 = 1 - ((ys - yhat) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.scatter(xs, ys, s=10, alpha=0.45, color=CONT, edgecolors="none")
    grid = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(grid, np.polyval(coef, grid), color=DENSE, lw=1.6,
            label=f"linear fit: {coef[0]:,.0f}$\\,r$ $-$ {abs(coef[1]):,.0f}   ($R^2$ = {r2:.3f})")
    ax.axhline(0, color="black", lw=0.8)
    root = -coef[1] / coef[0]
    ax.axvline(root, color=NEUTRAL, ls="--", lw=1)
    ax.annotate(f"$r^*\\approx$ {root:.2f}", (root, ax.get_ylim()[0] * 0.9),
                textcoords="offset points", xytext=(5, 0), color=NEUTRAL)
    ax.set_xlabel("measured functional recurrence $r$")
    ax.set_ylabel("Dense$-$Continuous lifetime loss (nats)")
    ax.set_title("Sharing effect is linear in measured recurrence (confirmatory)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "fig2_dose_response.png")
    plt.close(fig)


def fig3_checkpoint_divergence() -> None:
    s = json.load(open("artifacts/checkpoints_development/sweep.json"))
    by = collections.defaultdict(dict)
    for r in s["records"]:
        by[r["world_seed"]][r["model"]] = r["checkpoint_32_shot_nmse"]
    checkpoints = [8, 16, 32, 64]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for model, color in (("continuous", CONT), ("dense", DENSE)):
        per_world = [[by[w][model][str(c)] for c in checkpoints] for w in sorted(by)]
        arr = np.array(per_world)
        for row in arr:
            ax.plot(checkpoints, row, color=color, alpha=0.15, lw=0.8)
        ax.plot(checkpoints, arr.mean(axis=0), "o-", color=color, lw=2, ms=5,
                label=f"{model.capitalize()} (mean of 10 worlds)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(checkpoints, [str(c) for c in checkpoints])
    ax.set_xlabel("lifetime tasks completed before freezing shared parameters")
    ax.set_ylabel("32-shot novel-composition NMSE")
    ax.set_title("Learning-to-learn is acquired: equal at 8 tasks, ~2x apart at 64\n(development)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "fig3_checkpoint_divergence.png")
    plt.close(fig)


def fig4_recovery_onset() -> None:
    s = json.load(open("reports/rho_operator_recovery/operator-recovery.json"))
    rows = s["rho_summaries"]
    baseline = s["untrained_baseline"]["mean"]
    rhos = [r["configured_rho"] for r in rows]
    dist = [r["mean_one_to_one_distance"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(rhos, dist, "o-", color=CONT, lw=2, ms=5, label="trained continuous basis")
    ax.axhline(baseline, color=NEUTRAL, ls="--", lw=1.2,
               label=f"untrained baseline ({baseline:.4f})")
    ax.axvspan(0.8, 0.87, color=DENSE, alpha=0.10)
    ax.text(0.835, ax.get_ylim()[1] * 0.97, "performance\ncrossover", ha="center",
            va="top", color=DENSE, fontsize=8)
    ax.set_xlabel("configured reuse $\\rho$")
    ax.set_ylabel("one-to-one distance to hidden primitives")
    ax.set_title("Primitive recovery begins only at the performance crossover\n(development)")
    ax.legend(frameon=False, loc="center left")
    fig.savefig(OUT / "fig4_recovery_onset.png")
    plt.close(fig)


def fig5_resource_frontier() -> None:
    models = [
        ("Discrete\n(hardened)", 26208, 768),
        ("Continuous", 29248, 6528),
        ("Hypernetwork", 33928, 7296),
        ("Dense-24", 56448, 5376),
        ("Dense-C", 66688, 6144),
    ]
    lifetime_rank = {"Continuous": 1, "Hypernetwork": 2, "Dense-C": 3, "Dense-24": 3,
                     "Discrete\n(hardened)": 4}
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for name, bits, macs in models:
        rank = lifetime_rank[name]
        color = CONT if rank == 1 else (DENSE if rank >= 3 else NEUTRAL)
        ax.scatter(bits / 1000, macs, s=70, color=color, zorder=3)
        offset, ha = {
            "Dense-24": ((8, -14), "left"),
            "Hypernetwork": ((10, 2), "left"),
            "Continuous": ((-4, -28), "left"),
        }.get(name, ((8, 4), "left"))
        ax.annotate(f"{name}\n(online rank {rank})", (bits / 1000, macs),
                    textcoords="offset points", xytext=offset, fontsize=8, ha=ha)
    ax.set_xlabel("evaluated int8 retained description (kilobits)")
    ax.set_ylabel("inference multiply-adds per prediction")
    ax.set_xlim(20, 80)
    ax.set_ylim(0, 8600)
    ax.set_title("Resource frontier: storage, execution, and online-learning rank\ndisagree (development)")
    fig.savefig(OUT / "fig5_resource_frontier.png")
    plt.close(fig)


def fig6_robustness_forest() -> None:
    s = json.load(open("reports/robustness/robustness.json"))
    labels, means, lo, hi = [], [], [], []
    pretty = {"replay0": "no replay", "replay1": "canonical replay (1:1)",
              "replay4": "heavy replay (1:4)", "reverse": "reverse task order"}
    for c in s["condition_summaries"]:
        e = c["lifetime_loss_effect"]
        labels.append(f"{pretty.get(c['condition'], c['condition'])}  ({e['wins']}/{e['worlds']})")
        means.append(e["mean"])
        lo.append(e["mean"] - e["bootstrap_95_percent_ci_of_mean"][0])
        hi.append(e["bootstrap_95_percent_ci_of_mean"][1] - e["mean"])
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ax.errorbar(means, y, xerr=[lo, hi], fmt="o", color=CONT, capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(m + h for m, h in zip(means, hi)) * 1.15)
    ax.set_xlabel("Dense$-$Continuous lifetime loss (nats), mean and bootstrap 95% CI")
    ax.set_title("The advantage is invariant to order and replay (10 development worlds)")
    fig.savefig(OUT / "fig6_robustness_forest.png")
    plt.close(fig)


def fig7_shared_residual() -> None:
    s = json.load(open("reports/shared_residual/shared-residual.json"))
    jw = json.load(open("reports/shared_residual/j-weighted.json"))
    rhos = [r["configured_rho"] for r in s["rho_summaries"]]
    envelope_gain = [r["mean_best_fixed_minus_shared_gaussian_log_loss"] for r in s["rho_summaries"]]
    ratio = [r["mean_functional_ratio"] for r in s["rho_summaries"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax = axes[0]
    ax.plot(rhos, envelope_gain, "o-", color=CONT, lw=2, label="raw prequential (nats)")
    LN2 = np.log(2)
    two_part = []
    for rho in rhos:
        cells = [r for r in jw["rows"] if r["rho"] == rho]
        two_part.append(st.mean(
            min(c["gain_vs_continuous"] - c["extra_bits_vs_continuous"] * LN2,
                c["gain_vs_dense"] - c["extra_bits_vs_dense"] * LN2)
            for c in cells))
    ax.plot(rhos, two_part, "s--", color=DENSE, lw=1.6,
            label="two-part code ($\\lambda=\\ln 2$/bit)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("configured reuse $\\rho$")
    ax.set_ylabel("gain over best fixed architecture")
    ax.set_title("Envelope win in nats; loss in bits")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.plot(rhos, ratio, "o-", color=NEUTRAL, lw=2)
    ax.set_xlabel("configured reuse $\\rho$")
    ax.set_ylabel("mean residual functional ratio")
    ax.set_title("Allocation: specialization falls\nas recurrence rises")
    axes[0].set_title("Shared parent + rank-2 residuals:\nenvelope win in nats; loss in bits")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_shared_residual.png")
    plt.close(fig)


def main() -> None:
    effects = load_confirmatory()
    fig1_regime_map(effects)
    fig2_dose_response(effects)
    fig3_checkpoint_divergence()
    fig4_recovery_onset()
    fig5_resource_frontier()
    fig6_robustness_forest()
    fig7_shared_residual()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
