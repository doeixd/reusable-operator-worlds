import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from row.experiments import g1t_tanh_formation as g, o3_online_sleep as o3, o2d_sleep_memory as o2d, o2c_consolidation as o2c, o2_online_reliability as o2
from row.experiments.audit_rotated_g5r_interference import score
from row.metrics import nmse
torch.set_num_threads(4)
g.use_form()
out = {}
for w in (70, 71, 72):
    for arm in ('SHUFFLED', 'REROUTE_WAKE'):
        term = g.work_of(g.ROOT, arm, w) / 'lifetime' / 'model.pt'
        cfg3, m, st, canonical = o3.load_shuffled_terminal(term, w, g.STREAM)
        _, _, _, plan, _ = o2.build_stream('SHUFFLED', w, g.STREAM)
        res = {}
        for phase in ('wake', 'slept'):
            if phase == 'slept':
                pool = o2d.reservoir(st, w, g.STREAM, o3.MEMORY)
                o2c.consolidate(cfg3, m, pool, [t.task_id for t in st], [1941, w, g.STREAM, o3.MEMORY], updates=o3.EXTRA_UPDATES)
            per = score(m, SimpleNamespace(tasks=st))['per_task']
            byd = {}
            for t in st:
                byd.setdefault(plan[t.task_id], []).append(per[t.task_id])
            row = {f'd{d}_median': float(np.median(v)) for d, v in sorted(byd.items())}
            row.update({f'd{d}_n': len(v) for d, v in sorted(byd.items())})
            # best single slot on depth-1 tasks
            lib = g.FrozenTanhLibrary(m)
            best = []
            for t in st:
                if plan[t.task_id] == 1:
                    with torch.no_grad():
                        c = lib.candidates(torch.tensor(t.eval_x, dtype=torch.float32)).numpy()
                    best.append(min(nmse(c[:, s], t.eval_y) for s in range(c.shape[1])))
            row['d1_best_slot_median'] = float(np.median(best))
            res[phase] = row
        out[f'{arm}_w{w}'] = res
        print(arm, w, json.dumps(res), flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
