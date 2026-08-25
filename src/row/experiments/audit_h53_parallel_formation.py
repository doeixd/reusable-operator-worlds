"""H53 shared-parameter co-formation frontier (H53_PARALLEL_FORMATION_PLAN.md, Am. 1).

Six candidate organizations developed CONCURRENTLY in one lifetime at two
bracketing sharing levels (L1 most shared, L3 least). This scorer asks whether
the true one is retrospectively distinguishable using past data only, with the
H49 LOO instrument and H50/H51's margins UNCHANGED, and at what cost.

What it must not claim (Amendment 1): this is a coupled learner in which every
head contributes gradient to the parameters TRUE develops through. A failure is
a failure of THIS co-training rule at THIS sharing level, not proof that
computation cannot be amortized.

Fails closed; per-cell cache with a protocol fingerprint; atomic report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments import learned_lifetime
from row.experiments.audit_h49_discoverability import LR as H49_LR
from row.experiments.audit_h49_discoverability import STEPS as H49_STEPS
from row.experiments.audit_h49_discoverability import refit
from row.experiments.audit_h50_reorganization import MARGIN, SUBST_MARGIN, WORLDS, git_commit, score_loo
from row.experiments.score_h39b_pslot import factorized_fit, read_json
from row.meta_world import MetaFamilySpec, generate_meta_world
from row.models.multihead_models import MultiHeadPslotLearner

LEVELS = ("L1", "L3")
HEADS = ("SHAM", "TRUE", "WRONG-A", "WRONG-B", "RANDOM-1", "RANDOM-2")
WRONG = ("WRONG-A", "WRONG-B", "RANDOM-1", "RANDOM-2")
COLLAPSE_FRACTION = 0.05
L2_MARGIN_TRIGGER = 0.05
CACHE = Path("reports/h53_cache")
BASE = {"slot_args": 4, "freeze_args": False, "freeze_matrices": False, "pslot_count": 2}


def protocol_fingerprint() -> str:
    payload = {"heads": list(HEADS), "levels": list(LEVELS), "base": BASE,
               "refit_steps": H49_STEPS, "refit_lr": H49_LR,
               "margin": MARGIN, "subst_margin": SUBST_MARGIN}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cached(key: str, compute):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.json"
    fingerprint = protocol_fingerprint()
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("protocol") != fingerprint:
            raise SystemExit(f"cached cell {key} under protocol {stored.get('protocol')}, "
                             f"not {fingerprint}; delete reports/h53_cache to rescore")
        print(f"[cache] {key}", flush=True)
        return stored["value"]
    value = compute()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"protocol": fingerprint, "value": value}), encoding="utf-8")
    os.replace(tmp, path)
    return value


def build_multihead(config, world: int, level: str, heads=HEADS) -> MultiHeadPslotLearner:
    saved = learned_lifetime.MULTIHEAD_SETTINGS
    learned_lifetime.MULTIHEAD_SETTINGS = dict(BASE, head_names=tuple(heads), sharing_level=level)
    try:
        local = replace(config, shared_residual_model=replace(
            config.shared_residual_model, operator_slots=12))
        local = replace(local, world=replace(local.world, seed=int(world)))
        model = learned_lifetime._build_model(local, "multihead")
    finally:
        learned_lifetime.MULTIHEAD_SETTINGS = saved
    return model


def load_multihead(config, path: Path, world: int, level: str) -> MultiHeadPslotLearner:
    """Reconstruct ALL of the learner's state: primary, followers, library, policies."""
    extras = read_json(path / "multihead.json")
    heads = tuple(extras["diagnostics"]["head_names"])
    model = build_multihead(config, world, level, heads)
    state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
    count = sum(1 for k in state if k.startswith("abstractions."))
    for index in range(count):
        model.abstractions.append(torch.nn.Parameter(
            state[f"abstractions.{index}"].clone(), requires_grad=False))
    model.sync_heads()
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) - {"route_temperature"} or result.unexpected_keys:
        raise SystemExit(f"primary state mismatch at {path}: {result}")
    for index, name in enumerate(heads):
        if index == 0:
            continue
        follower_path = path / f"head_{index}_{name.replace('-', '_')}.pt"
        if not follower_path.exists():
            raise SystemExit(f"missing follower artifact {follower_path}")
        follower_state = torch.load(follower_path, weights_only=True)["model_state_dict"]
        out = model.heads[index].load_state_dict(follower_state, strict=False)
        if set(out.missing_keys) - {"route_temperature"} or out.unexpected_keys:
            raise SystemExit(f"follower {name} state mismatch at {path}: {out}")
    summary = read_json(path / "summary.json")
    table = summary.get("reference_table") or {}
    for task_id, reference in (table.get("task_reference") or {}).items():
        model.task_reference[task_id] = int(reference)
    for task_id in table.get("retired_task_ids") or []:
        model.retired.add(task_id)
    model.head_assignment = {k: dict(v) for k, v in extras["head_assignment"].items()}
    for task_id in model.task_codes:
        model.apply_head_policies(task_id)
    model.sync_heads()
    model.eval()
    return model


def head_view(config, world: int, model, index: int):
    """A standalone plain learner holding head `index`'s state.

    Scoring must not deep-copy the whole multi-head object once per re-fit: the
    primary head IS the container, so `refit` would clone all six heads 64 times
    per cell. The view carries exactly one head's tensors and is the same class
    every earlier rung was scored on, so the instrument is unchanged.
    """
    from row.experiments.score_h39b_pslot import load_pslot  # noqa: F401  (build path)
    from row.models.pslot_models import ParameterizedSlotLearner

    saved = learned_lifetime.PSLOT_SETTINGS
    learned_lifetime.PSLOT_SETTINGS = dict(BASE)
    try:
        local = replace(config, shared_residual_model=replace(
            config.shared_residual_model, operator_slots=12))
        local = replace(local, world=replace(local.world, seed=int(world)))
        view = learned_lifetime._build_model(local, "pslot")
    finally:
        learned_lifetime.PSLOT_SETTINGS = saved
    head = model.heads[index]
    state = {k: v for k, v in head.state_dict().items() if not k.startswith("followers.")}
    for _ in range(len(model.abstractions)):
        view.abstractions.append(torch.nn.Parameter(
            model.abstractions[_].detach().clone(), requires_grad=False))
    for key in list(state):
        if key.startswith("task_codes."):
            view.begin_task(key.split(".", 1)[1])
    result = view.load_state_dict(state, strict=False)
    if set(result.missing_keys) - {"route_temperature"} or result.unexpected_keys:
        raise SystemExit(f"head view mismatch for head {index}: {result}")
    view.task_reference.update(model.task_reference)
    view.retired.update(model.retired)
    view.task_mask.update(head.task_mask)
    view.eval()
    return view


def reference_divergence(config, world: int, probe, task_ids) -> float:
    """What INDEPENDENT development looks like: M_4 versus L_4 on the same probe."""
    from row.experiments.score_h39b_pslot import load_pslot
    record = dict(BASE, model="pslot", pslot_index=11)
    m = load_pslot(config, Path("artifacts/h39c/w_m4") / f"world_{world}" / "lifecycle", record,
                   world_seed=world)
    l = load_pslot(config, Path("artifacts/h39c/w_l4") / f"world_{world}" / "lifecycle", record,
                   world_seed=world)
    for model in (m, l):
        model.task_mask.clear()
        model.set_route_temperature(1.0)
    with torch.no_grad():
        a = torch.stack([m(probe, t) for t in task_ids])
        b = torch.stack([l(probe, t) for t in task_ids])
        scale = float(torch.sqrt(torch.mean(a ** 2))) or 1.0
        return float(torch.sqrt(torch.mean((a - b) ** 2)) / scale)


def training_device_seconds(config, world: int, level: str, steps: int = 20) -> dict:
    """A_train's numerator and denominator, measured as PROCESS CPU time.

    Wall-clock is an engineering number: concurrency can manufacture apparent
    amortization in it. Per-step device-seconds times identical step counts is
    the scientific ratio, so it is measured here on the real configuration.
    """
    out = {}
    x = torch.randn(8, config.world.state_dim)
    y = torch.randn(8, config.world.state_dim)
    for name, heads in (("H", HEADS), ("one", (HEADS[0],))):
        model = build_multihead(config, world, level, heads)
        model.begin_task("probe")
        params = [p for p in {id(p): p for p in model.parameters()}.values()]
        optimizer = torch.optim.Adam(params, lr=1e-3)
        model.train()
        model.multihead_loss(x, y, ["probe"] * 8).backward()   # warm up
        optimizer.zero_grad(set_to_none=True)
        start = time.process_time()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = model.multihead_loss(x, y, ["probe"] * 8)
            loss.backward()
            model.finalize_gradients()
            optimizer.step()
        out[name] = (time.process_time() - start) / steps
    return {"per_step_device_seconds_H": out["H"], "per_step_device_seconds_one": out["one"],
            "A_train": out["H"] / (len(HEADS) * out["one"])}


def score_level(config, spec, level: str, worlds=WORLDS) -> dict:
    per_world = {}
    for world in worlds:
        generated = generate_meta_world(replace(config.world, seed=world, tasks=spec.total_tasks), spec)
        family_tasks = [t for i, t in enumerate(generated.tasks) if spec.family_of(i) is not None]
        futures = list(generated.novel_family_tasks)
        path = Path(f"artifacts/h53/h53_{level.lower()}") / f"world_{world}" / "lifecycle"
        if not (path / "summary.json").exists():
            raise SystemExit(f"missing H53 artifact: {path}")
        model = load_multihead(config, path, world, level)
        probe = torch.tensor(np.random.default_rng(np.random.SeedSequence([53, world, 7])).normal(
            size=(128, config.world.state_dim)), dtype=torch.float32)
        probe_tasks = [t.task_id for t in family_tasks[:16]]
        divergence = model.neutralized_divergence(probe, probe_tasks)
        reference = cached(f"refdiv_w{world}",
                           lambda: reference_divergence(config, world, probe, probe_tasks))
        rows = {}
        for index, name in enumerate(model.head_names):
            head = head_view(config, world, model, index)
            assignment = {t.task_id: model.head_assignment[name].get(t.task_id) for t in family_tasks}
            s = cached(f"{level}_w{world}_{name}",
                       lambda head=head, assignment=assignment, name=name:
                       score_loo(head, family_tasks, assignment, f"h53_{level}_{name}_w{world}"))
            rows[name] = {"C_LOO": s["C_LOO"], "D_star_nats": s["D_star_nats"], "fits": s["fits"]}
            print(f"[{level}] world {world} {name:9s}: C_LOO {s['C_LOO']:.5f}", flush=True)
        best_wrong = min(WRONG, key=lambda a: rows[a]["C_LOO"])
        for name in ("TRUE", best_wrong):
            head = head_view(config, world, model, list(model.head_names).index(name))
            comp = {t.task_id: (None if model.head_assignment[name].get(t.task_id) is None
                                else 1 - model.head_assignment[name][t.task_id])
                    for t in family_tasks}
            own = rows[name]["fits"]
            rows[name]["S_subst"] = cached(
                f"{level}_w{world}_{name}_subst",
                lambda head=head, comp=comp, own=own, name=name: float(np.mean(
                    [np.log(refit(head, t, comp[t.task_id], f"h53sub_{level}_{name}_w{world}")["nmse"])
                     - np.log(w["nmse"]) for t, w in zip(family_tasks, own)])))
        for name in ("TRUE", best_wrong, "SHAM"):
            head = head_view(config, world, model, list(model.head_names).index(name))
            rows[name]["sibling_alpha_k128"] = cached(
                f"{level}_w{world}_{name}_sibling",
                lambda head=head, name=name: float(np.mean([
                    factorized_fit(head, task, 128, "alpha_only", "adam", 0.01, 2000,
                                   f"h53sib_{level}_{name}_w{world}_t{i}")["final_query_scaled"]
                    for i, task in enumerate(futures)])))
        diagnostics = read_json(path / "multihead.json")["diagnostics"]
        for row in rows.values():
            row.pop("fits", None)
        per_world[world] = {
            "heads": rows, "best_wrong": best_wrong,
            "neutralized_divergence": divergence,
            "reference_divergence_M4_vs_L4": reference,
            "collapsed": bool(divergence["mean"] < COLLAPSE_FRACTION * reference),
            "state": {"shared_scalars": diagnostics["shared_state_scalars"],
                      "head_scalars": diagnostics["head_state_scalars"]},
        }
    return per_world


def decide(per_world) -> dict:
    rows = {}
    for world, data in per_world.items():
        heads = data["heads"]
        true_c = heads["TRUE"]["C_LOO"]
        rows[world] = {
            "margin_vs_best_wrong": float(np.log(min(heads[a]["C_LOO"] for a in WRONG)) - np.log(true_c)),
            "margin_vs_sham": float(np.log(heads["SHAM"]["C_LOO"]) - np.log(true_c)),
        }
    worlds = list(per_world)
    sep = (sum(rows[w]["margin_vs_best_wrong"] >= MARGIN for w in worlds) >= 2
           and sum(rows[w]["margin_vs_sham"] >= MARGIN for w in worlds) >= 2)
    subst_ok = True
    for world, data in per_world.items():
        s_true = data["heads"]["TRUE"].get("S_subst")
        s_bw = data["heads"][data["best_wrong"]].get("S_subst")
        if s_true is None or s_bw is None or (s_true - s_bw) < SUBST_MARGIN:
            subst_ok = False
    return {"per_world": rows, "separation_cloo": bool(sep),
            "substitutability_corroborates": bool(subst_ok),
            "SEPARATION": bool(sep and subst_ok),
            "collapsed_worlds": [w for w, d in per_world.items() if d["collapsed"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--h49", type=Path, default=Path("reports/h49_discoverability.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/h53_parallel_formation.json"))
    parser.add_argument("--levels", nargs="+", default=list(LEVELS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    config = load_config(args.config)
    spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0, subspace_rank=2, schema_groups=2)
    h49 = read_json(args.h49)
    out = {"frozen_plan": "H53_PARALLEL_FORMATION_PLAN.md (Amendment 1)", "git_commit": git_commit(),
           "measures": "shared-parameter co-formation frontier; NOT amortization of computation in general",
           "protocol": {"heads": list(HEADS), "margin": MARGIN, "subst_margin": SUBST_MARGIN,
                        "collapse_fraction": COLLAPSE_FRACTION,
                        "instrument": "H49 refit and H50/H51 margins, unchanged"},
           "levels": {}}
    for level in args.levels:
        per_world = score_level(config, spec, level)
        decisions = decide(per_world)
        cost = {str(w): cached(f"cost_{level}_w{w}",
                               lambda w=w: training_device_seconds(config, w, level))
                for w in WORLDS}
        state_ratios = {}
        for world, data in per_world.items():
            shared = data["state"]["shared_scalars"]
            own = sum(data["state"]["head_scalars"].values())
            single = shared + own / len(HEADS)
            state_ratios[str(world)] = {"unique_live_state": shared + own,
                                        "A_state": (shared + own) / (len(HEADS) * single)}
        recovery = {}
        for world in per_world:
            ref = h49["worlds"][str(world)]["L4"]["margin_vs_best_wrong"]
            base = {0: 0.059, 1: -0.034, 2: -0.043}[world]
            recovery[str(world)] = ((decisions["per_world"][world]["margin_vs_best_wrong"] - base)
                                    / (ref - base))
        out["levels"][level] = {"worlds": {str(k): v for k, v in per_world.items()},
                                "decisions": decisions, "cost_train": cost,
                                "cost_state": state_ratios,
                                "recovery_fraction_cloo_margin": recovery}
        print(f"[{level}] SEPARATION={decisions['SEPARATION']} "
              f"margins {[round(decisions['per_world'][w]['margin_vs_best_wrong'], 3) for w in per_world]} "
              f"vs SHAM {[round(decisions['per_world'][w]['margin_vs_sham'], 3) for w in per_world]} "
              f"collapsed {decisions['collapsed_worlds']}", flush=True)
    out["outcome"], out["l2_trigger"] = verdict(out["levels"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(f"OUTCOME {out['outcome']}; L2 required: {out['l2_trigger']}")


def verdict(levels) -> tuple[str, bool]:
    if not all(l in levels for l in LEVELS):
        return "INCOMPLETE", False
    l1, l3 = levels["L1"]["decisions"], levels["L3"]["decisions"]
    a_train = float(np.mean([levels["L1"]["cost_train"][w]["A_train"] for w in levels["L1"]["cost_train"]]))
    deeper = [l3["per_world"][w]["margin_vs_best_wrong"] - l1["per_world"][w]["margin_vs_best_wrong"]
              for w in l3["per_world"]]
    depth_effect = sum(d >= L2_MARGIN_TRIGGER for d in deeper) >= 2
    trigger = ((l1["SEPARATION"] != l3["SEPARATION"])
               or (not l1["SEPARATION"] and not l3["SEPARATION"] and depth_effect))
    if l1["SEPARATION"]:
        return ("A-PARALLEL-FORMATION-WORKS-CHEAPLY" if a_train <= 0.5
                else "B-WORKS-AT-A-LIFETIME-PER-HYPOTHESIS"), trigger
    if l3["SEPARATION"]:
        return "C1-FRONTIER-BRACKETED", trigger
    if depth_effect:
        return "C2-DEPTH-MATTERS-FRONTIER-UNLOCALIZED", trigger
    return "D-NO-DISCRIMINATION-FRONTIER-DEEPER-THAN-L3", trigger


if __name__ == "__main__":
    main()
