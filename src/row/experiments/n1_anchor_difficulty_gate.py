"""N1 anchor-difficulty gate: are length-1 tasks actually the informed ones?

Answers Audit 2 of `N1_ANCHOR_SUPPLY_PLAN.md`. The plan's original non-vacuity
check compared "median single-step route margin" on length-1 against length-3
tasks. That is not a comparable statistic: a length-1 task chooses among 12
routes and a length-3 task among 12**3 = 1,728, and "single-step" is undefined
at length 3. It is the route-margin comparability failure in a new place.

The replacement measures the PREMISE directly and with a chance-corrected
statistic that is comparable across depths. The premise, from `AGENTS.md`, is
that "at length 1 tasks sharing an operation look alike, so routing is
CLUSTERING". Operationally: on an UNTRAINED library - the uninformed state at
which J1's first commitment is made - do tasks that share a teacher primitive
get assigned the same slot more often than chance?

ARI is chance-corrected and scale-free, so ARI at depth 1 and ARI at depth 3 are
the same quantity even though the route spaces differ by 144x. No training, no
lifetime: this is a property of the task stream against a random library.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_h47_baselines import ari_nmi
from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.audit_so1r_route_only import unflatten
from row.experiments.l0d_depth4_execution_gate import DepthLibrary

OUTPUT = Path('reports/n1_anchor_difficulty_gate.json')
MODEL_SEED = 5000
WORLDS = (0, 1, 2)


def selected_slots(library, tasks, depth):
    """Support-argmin route per task, on the given (untrained) library."""
    routes = []
    for task in tasks:
        x = torch.tensor(task.train_x, dtype=torch.float32)
        y = torch.tensor(task.train_y, dtype=torch.float32)
        with torch.no_grad():
            mse = library.all_route_support_mse_depth(x, y, depth)
        if not bool(torch.isfinite(mse).all()):
            raise ValueError('nonfinite support losses')
        routes.append(unflatten(int(torch.argmin(mse)), library.slots, depth))
    return routes


def measure_world(world_seed):
    # The random library is the stage-3 learner at initialisation: the state at
    # which J1's first, uninformed commitment is actually made.
    cfg3, world3, _, _ = stage_setup(world_seed, 3, MODEL_SEED)
    model = build_fast(cfg3)
    library = DepthLibrary(model)
    before = library_sha(model)

    cfg1, world1, _, _ = stage_setup(world_seed, 1, MODEL_SEED)

    # Teacher libraries must agree, or the two depths are not the same world.
    probe = torch.tensor(np.random.default_rng(np.random.SeedSequence(
        [7781, world_seed])).normal(size=(32, cfg3.world.state_dim)), dtype=torch.float32)
    same_teacher = all(
        np.allclose(world1.tasks[0].teacher_library[i](probe.numpy()),
                    world3.tasks[0].teacher_library[i](probe.numpy()))
        for i in range(cfg3.world.teacher_primitives))

    depth1 = selected_slots(library, world1.tasks, 1)
    depth3 = selected_slots(library, world3.tasks, 3)

    ari1, nmi1 = ari_nmi([t.program.primitive_ids[0] for t in world1.tasks],
                         [r[0] for r in depth1])
    per_position = []
    for position in range(3):
        a, n = ari_nmi([t.program.primitive_ids[position] for t in world3.tasks],
                       [r[position] for r in depth3])
        per_position.append({'position': position, 'ari': a, 'nmi': n})

    if library_sha(model) != before:
        raise ValueError('library changed during the gate')

    depth3_ari = float(np.mean([p['ari'] for p in per_position]))
    return {
        'world': world_seed,
        'library_sha256': before,
        'same_teacher_library': bool(same_teacher),
        'depth1': {'tasks': len(world1.tasks), 'routes': 12, 'ari': ari1, 'nmi': nmi1,
                   'distinct_slots_used': len({r[0] for r in depth1})},
        'depth3': {'tasks': len(world3.tasks), 'routes': 12 ** 3,
                   'per_position': per_position, 'mean_ari': depth3_ari,
                   'distinct_routes_used': len({tuple(r) for r in depth3})},
        'depth1_more_clustered': bool(ari1 > depth3_ari),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)

    rows = [measure_world(w) for w in WORLDS]
    passes = all(r['depth1_more_clustered'] for r in rows)
    report = {
        'id': 'n1-anchor-difficulty-gate-v1',
        'statistic': 'adjusted Rand index between teacher primitive labels and '
                     'support-argmin slot labels, on an UNTRAINED library',
        'why_ari': 'chance-corrected and scale-free, so depth 1 (12 routes) and '
                   'depth 3 (1,728 routes) are the same quantity',
        'rule': 'the premise holds when depth-1 ARI exceeds mean depth-3 ARI in '
                'EVERY world; otherwise N1 is unscoreable rather than negative',
        'worlds': rows,
        'passes': passes,
        'verdict': 'PREMISE_HOLDS' if passes else 'PREMISE_FAILS',
        'interpretation': 'non-vacuity gate for N1 only; no verdict about anchors',
    }
    args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    for r in rows:
        d1, d3 = r['depth1']['ari'], r['depth3']['mean_ari']
        print(f"world {r['world']}: depth-1 ARI {d1:+.4f} | depth-3 mean ARI {d3:+.4f} "
              f"| slots used {r['depth1']['distinct_slots_used']}/12 "
              f"| same teacher {r['same_teacher_library']}")
    print(f"\n{report['verdict']}  ->  {args.output}")
    return 0 if passes else 1


if __name__ == '__main__':
    raise SystemExit(main())
