"""B2-hard v2 sizing (Tier 1, EXPLORATORY, no verdict): does starting every candidate at the one-step form remove the
wake ROOT impostor without killing wake's loop discovery, and at which bias scale c?

B2-hard (`reports/b2_hard.json`) and its censuses (`reports/b2h_root_census.json`, `reports/b2h_wake_census.json`):
with zero-initialized halting on all 108 depth-1 candidates, a third of cells form a ROOT slot in wake (Q^L ~ P for
one to three operations), which marks exactly the formation failures. `LoopLearner(halt_init=c)` starts the per-step
biases at `[+c, -c, ...]`. This sizing runs B2-hard's ALLLOOP construction with `halt_init` in (None, 2, 4) on ONE
fresh development world (band 5, world 56, stream 0), with the root census's ONE_STEP end of stream (count forms
restricted to L in {0, 1}) and B2-hard's sleep. It records, per c: wake state-weight AUC and norms, wake canonical /
loop / anchor medians, rooted operations at the end of wake (teacher programs used only here), routes changed,
decisions, terminal medians. Its job is to pick c for a registered plan and to show the question is live; one cell
per c cannot measure a formation rate (B2-hard failed 1 cell in 3), so nothing here is a verdict.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import b1_online as b1o
from row.experiments import b2_hard as bh
from row.experiments import b2_online as b2
from row.experiments import census_b2h_root as cr
from row.experiments import census_b2h_wake as cw
from row.experiments import deep_reroute as dr
from row.experiments import learned_lifetime as ll
from row.experiments import loop_stream as ls
from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now
from row.models.loop_gated import LoopLearner

ROOT = Path('artifacts/b2h_init_sizing')
OUTPUT = Path('reports/b2h_init_sizing.json')
WORLD, STREAM = 56, 0
INITS = (None, 2.0, 4.0)


def run_init(c, w=WORLD, s=STREAM, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, mixed, st, plan, canonical, ltasks, meta = ls.stream(w, s, bh.N_LOOP)
    loop_ids = {t.task_id for t in ltasks}
    cands = [(i, t) for i, t in enumerate(st) if plan[t.task_id] == 1]
    cand_ids = {t.task_id for _, t in cands}
    anchors = [t for _, t in cands if t.task_id not in loop_ids]
    model = b1o.build(LoopLearner, cfg, plan, loop_plan={t: 'state' for t in cand_ids}, max_iterations=bh.K,
                      state_halt=True, step_bias=True, detach_halt_input=True, halt_decay=bh.HALT_DECAY, halt_init=c)
    work = ROOT / ('work' if scale == 1 else f'dry{scale}') / f'c{c}_w{w}_s{s}'
    summary = ll.run(dataclasses.replace(o2.run_cfg(cfg, scale), output_directory=work / 'lifetime'), o2.KIND,
                     world=mixed, model=model, return_model=True, replay_seed=o2.replay_seed_for(w, s))
    model = summary.pop('terminal_model')

    def med(ts):
        return score(model, SimpleNamespace(tasks=ts))['median']

    norms = {t: float(model.halt_w[t].detach().norm()) for t in cand_ids}
    rec = {'halt_init': c, 'world': w, 'stream': s, 'scale': scale,
           'wake_canonical_median': med(canonical), 'wake_loop_median': med(ltasks), 'wake_anchor_median': med(anchors),
           'wake_state_auc': bh.auc([norms[t] for t in loop_ids], [v for t, v in norms.items() if t not in loop_ids]),
           'wake_state_norm_median': {'loop': float(np.median([norms[t] for t in loop_ids])),
                                      'straight': float(np.median([v for t, v in norms.items() if t not in loop_ids]))},
           'wake_count_accuracy_median': float(np.median(b2.count_accuracy(model, ltasks, meta)))}
    pool = o2d.reservoir(st, w, s, bh.MEMORY)
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    lib = FrozenLibrary(model)
    per_op = collections.defaultdict(list)
    for t in anchors:
        X = np.stack([a for a, _ in by[t.task_id]])
        Y = np.stack([b for _, b in by[t.task_id]])
        per_op[int(t.program.primitive_ids[0])].append(cw.best_count(lib, X, Y)[2])
    rec['wake_anchor_best_L_by_op'] = {str(k): v for k, v in sorted(per_op.items())}
    rec['wake_rooted_ops'] = sorted(op for op, Ls in per_op.items() if all(L >= 2 for L in Ls))
    rec['routes_changed'] = dr.reroute(model, [t for t in st if t.task_id not in model.loop_plan], plan, pool)
    rec['reroute_canonical_median'] = med(canonical)
    before = library_sha(model)
    dec = cr.one_step_end_of_stream(model, cands, pool, w, 'ALLLOOP')
    rec['loops_kept'] = sum(dec[t]['loop'] for t in loop_ids)
    rec['straight_kept_count'] = sum(not v['loop'] for t, v in dec.items() if t not in loop_ids)
    bh.sleep_decay(cfg, model, pool, [t.task_id for t in st], [1941, w, s, bh.MEMORY])
    rec['library_changed_in_sleep'] = library_sha(model) != before
    rec.update({'canonical_median': med(canonical), 'loop_median': med(ltasks), 'anchor_median': med(anchors),
                'count_accuracy_median': float(np.median(b2.count_accuracy(model, ltasks, meta))),
                'seconds': time.perf_counter() - started})
    return rec


def main(scale=1):
    """Restartable: each finished cell is written atomically to `cells/`, stamped with the commit and scale; a
    relaunch reuses stamped cells and reruns the rest from initialization; a mismatched stamp fails closed."""
    out = OUTPUT if scale == 1 else ROOT / f'dry{scale}.json'
    if scale == 1:
        require_clean_code(OUTPUT)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    stamp = {'commit': commit, 'scale': scale, 'world': WORLD, 'stream': STREAM}
    cells = ROOT / ('cells' if scale == 1 else f'cells_dry{scale}')
    cells.mkdir(parents=True, exist_ok=True)
    done, todo = {}, []
    for c in INITS:
        f = cells / f'c{c}.json'
        if f.exists():
            saved = json.loads(f.read_text())
            if saved['stamp'] != stamp:
                raise ValueError(f'{f}: stamp {saved["stamp"]} != {stamp}')
            done[c] = saved['record']
        else:
            todo.append(c)
    print(now(), 'LAUNCH', json.dumps(stamp), 'reused', sorted(map(str, done)), flush=True)
    with ProcessPoolExecutor(max_workers=max(1, len(todo))) as ex:
        futs = {ex.submit(run_init, c, WORLD, STREAM, scale): c for c in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            atomic_json(cells / f"c{futs[fut]}.json", {'stamp': stamp, 'record': rec})
            done[futs[fut]] = rec
            print(now(), json.dumps({k: v for k, v in rec.items() if k != 'wake_anchor_best_L_by_op'}), flush=True)
    atomic_json(out, {'tier': 1, 'exploratory': True, 'stamp': stamp, 'cells': [done[c] for c in INITS],
                      'finished_utc': now()})
    print(now(), 'EXIT 0', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(*(int(a) for a in sys.argv[1:2])))
