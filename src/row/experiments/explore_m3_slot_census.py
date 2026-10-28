"""EXPLORATORY Tier 0 census (decision 24a, after M2 `PARTIAL`; no verdict): on the saved M2 wake terminals, are the
three failed streams' defects visible at the level of single slots (merged or misfitting anchor slots, which a slot
repair could target), or only in how slots compose?

Per (world, stream) of M2 (`artifacts/m2_motif_cycle/work/w*_s*/lifetime/model.pt`), behind a reload gate (library
sha256 equals the cell's `wake_library_sha256`, and the WAKE per-task scores reproduce exactly):
- every length-1 anchor is routed exhaustively over the 12 slots on its reservoir examples (D2's chooser);
- learner-computable: per-slot occupancy by anchors, anchor reservoir NMSE under its best slot, second-best gap;
- teacher-side DIAGNOSTICS (never usable by a learner): per primitive, the slots its anchors choose (purity), median
  anchor NMSE; the mapped route (true program through the majority anchor slot) on every length-2 anchor and shallow
  task, against the exhaustive best route of the same length (length 2: 144 routes; shallow: mapped only).
Output `reports/m3_slot_census.json`.
"""
from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import deep_reroute as dr
from row.experiments import m2_motif_cycle as m2
from row.experiments import motif_stream as ms
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import dn_stream as dn
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary

CELLS = Path('artifacts/m2_motif_cycle/cells')
WORK = Path('artifacts/m2_motif_cycle/work')
OUT = Path('reports/m3_slot_census.json')


def load_wake(w, s):
    cfg, world, st, plan, shallow, deep, motifs = ms.build_stream_with_deep(w, s, m2.N_DEEP)
    model = dn.planned_model(ms.DEEP_DEPTH, cfg, plan)
    for t in st:
        model.begin_task(t.task_id)
    state = torch.load(WORK / f'w{w}_s{s}' / 'lifetime' / 'model.pt', weights_only=True)['model_state_dict']
    for k in state:
        if k.startswith('task_codes.') and k not in model.state_dict():
            model.begin_task(k.split('.', 1)[1], ms.DEEP_DEPTH)
    model.load_state_dict(state, strict=True)
    return cfg, model, st, plan, shallow, deep, motifs


def nmse_t(pred, y):
    return float(torch.mean((pred - y) ** 2) / torch.mean((y - y.mean(0)) ** 2))


def census(w, s):
    rec = json.loads((CELLS / f'w{w}_s{s}.json').read_text())['record']
    cfg, model, st, plan, shallow, deep, motifs = load_wake(w, s)
    if library_sha(model) != rec['wake_library_sha256']:
        raise RuntimeError(f'w{w}_s{s}: library sha differs')
    model.eval()
    sh = score(model, SimpleNamespace(tasks=shallow))['per_task']
    if sh != rec['branches']['WAKE']['shallow']:
        raise RuntimeError(f'w{w}_s{s}: WAKE shallow scores do not reproduce')
    xy = m2.pool_xy(o2d.reservoir(st, w, s, o3.MEMORY))
    lib = FrozenLibrary(model)
    out = {'world': w, 'stream': s, 'cyc_solved': m2.solved_fraction(rec, 'CYC'),
           'cyc_shallow_median': float(np.median(list(rec['branches']['CYC']['shallow'].values())))}
    anchors, by_prim = [], {}
    with torch.no_grad():
        for t in st:
            if plan[t.task_id] != 1:
                continue
            x, y = xy[t.task_id]
            errs = sorted((nmse_t(m2.slot_apply(lib, x, k), y), k) for k in range(lib.slots))
            a = {'prim': t.program.primitive_ids[0], 'slot': errs[0][1], 'nmse': errs[0][0], 'second': errs[1][0]}
            anchors.append(a)
            by_prim.setdefault(a['prim'], []).append(a)
        slot_of = {p: Counter(a['slot'] for a in v).most_common(1)[0][0] for p, v in by_prim.items()}
        out['primitives'] = {str(p): {'slots': dict(Counter(a['slot'] for a in v)),
                                      'purity': Counter(a['slot'] for a in v).most_common(1)[0][1] / len(v),
                                      'median_nmse': statistics.median(a['nmse'] for a in v),
                                      'median_second': statistics.median(a['second'] for a in v)}
                             for p, v in sorted(by_prim.items())}
        occ = Counter(a['slot'] for a in anchors)
        out['slot_occupancy'] = {str(k): occ.get(k, 0) for k in range(lib.slots)}
        out['slot_misfit'] = {str(k): statistics.median(a['nmse'] for a in anchors if a['slot'] == k) for k in occ}
        out['merged_slots'] = sorted(k for k in occ if len({a['prim'] for a in anchors if a['slot'] == k}) > 1)
        out['distinct_anchor_slots'] = len(set(slot_of.values()))
        l2 = []
        for t in st:
            if plan[t.task_id] != 2:
                continue
            x, y = xy[t.task_id]
            mapped = [slot_of[p] for p in t.program.primitive_ids]
            best = dr.exhaustive_route(lib, x, y, 2)
            l2.append({'mapped': nmse_t(m2.run_route(lib, mapped, x), y),
                       'best': nmse_t(m2.run_route(lib, best, x), y), 'equal': best == mapped,
                       'prims': list(t.program.primitive_ids)})
        out['length2'] = {'mapped_median': statistics.median(r['mapped'] for r in l2),
                          'best_median': statistics.median(r['best'] for r in l2),
                          'best_equals_mapped': sum(r['equal'] for r in l2), 'n': len(l2)}
        bad = Counter()
        for r in l2:
            if r['mapped'] > 0.1:
                bad.update(set(r['prims']))
        out['length2_mapped_bad_by_prim'] = {str(p): bad[p] for p in sorted(bad)}
        shm = []
        for t in shallow:
            x, y = xy[t.task_id]
            shm.append(nmse_t(m2.run_route(lib, [slot_of[p] for p in t.program.primitive_ids], x), y))
        out['shallow_mapped_median'] = statistics.median(shm)
        out['shallow_mapped_below_0.1'] = sum(v < 0.1 for v in shm)
    return out


def main():
    torch.set_num_threads(1)
    res = []
    for w in m2.WORLDS:
        for s in m2.STREAMS:
            r = census(w, s)
            res.append(r)
            print(f"w{w}_s{s} cyc {r['cyc_solved']:.2f} distinct {r['distinct_anchor_slots']} merged {r['merged_slots']} "
                  f"l2 mapped {r['length2']['mapped_median']:.3f} best {r['length2']['best_median']:.3f} "
                  f"shallow mapped {r['shallow_mapped_median']:.3f} bad-by-prim {r['length2_mapped_bad_by_prim']}",
                  flush=True)
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(res, indent=1), encoding='utf-8')
    os.replace(tmp, OUT)


if __name__ == '__main__':
    main()
