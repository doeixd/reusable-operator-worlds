"""Read-only stream audit of O1 (2026-09-24), written after the run; changes no stamped report.

For each single-lifetime cell (SHUFFLED, MIXED_L1) it checks, from the saved lifetime:
  - the consumed task order equals the registered shuffle, every task got 128 examples;
  - every task has a hard route of its planned length (the runner's check is non-vacuous);
  - the terminal model reloads strictly and reproduces the recorded terminal median;
  - ANCHOR: the last stream task's terminal error equals its logged end-of-task error;
  - DEFECT: the recorded `end_of_task_median` is a median over ALL stream tasks, not the
    canonical 64 that the terminal median (and STAGED's end-of-task median) cover. The
    canonical-64 and per-depth end-of-task medians are recomputed here.
Output: stream_audit.json beside this file. Run from the repo root.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import o1_online_anchor as o1
from row.experiments.audit_rotated_g5r_interference import score

WORK = Path('artifacts/o1_online_anchor/work')
CELLS = Path('artifacts/o1_online_anchor/cells')
out = {}
for arm in ('SHUFFLED', 'MIXED_L1'):
    for w in o1.WORLDS:
        cfg3, _, stream, plan, canon = o1.build_stream(arm, w)
        d = WORK / f'{arm}_w{w}' / 'lifetime'
        rows = [json.loads(line) for line in open(d / 'metrics.jsonl', encoding='utf-8')]
        order, counts = [], {}
        for r in rows:
            if r.get('record_type') == 'prequential':
                if r['task_id'] not in counts:
                    order.append(r['task_id'])
                counts[r['task_id']] = counts.get(r['task_id'], 0) + 1
        summaries = [r for r in rows if r.get('record_type') == 'task_summary']
        final = {r['task_id']: r['final_nmse'] for r in summaries}
        routes = json.loads((d / 'hard_routes.json').read_text())
        routes = routes.get('routes', routes)
        state = torch.load(d / 'model.pt', weights_only=True)['model_state_dict']
        model = o1.planned_model(cfg3, plan)
        for t in stream:
            model.begin_task(t.task_id, plan[t.task_id])
        probes = [k.split('.', 1)[1] for k in state if k.startswith('task_codes.') and k not in model.state_dict()]
        for probe in probes:
            model.begin_task(probe, 3)
        model.load_state_dict(state, strict=True)
        last = summaries[-1]
        last_task = next(t for t in stream if t.task_id == last['task_id'])
        terminal_last = score(model, SimpleNamespace(tasks=[last_task]))['per_task'][last_task.task_id]
        terminal_canon = score(model, SimpleNamespace(tasks=canon))['median']
        record = json.loads((CELLS / f'{arm}_w{w}.json').read_text())['record']
        out[f'{arm}_w{w}'] = {
            'order_matches_registered_stream': order == [t.task_id for t in stream],
            'prequential_rows_per_task': sorted(set(counts.values())),
            'routes_present': sum(t in routes for t in plan), 'routes_expected': len(plan),
            'routes_wrong_length': sum(1 for t in plan if t in routes and len(routes[t]) != plan[t]),
            'extra_state_task_codes': probes,
            'terminal_median_recomputed': terminal_canon, 'terminal_median_recorded': record['terminal_median'],
            'anchor_last_task_depth': plan[last_task.task_id],
            'anchor_end_of_task': last['final_nmse'], 'anchor_terminal': terminal_last,
            'anchor_abs_error': abs(terminal_last - last['final_nmse']),
            'end_of_task_median_recorded_all_tasks': record['end_of_task_median'],
            'end_of_task_median_canonical64': float(np.median([final[t.task_id] for t in canon])),
            'end_of_task_median_by_depth': {str(k): float(np.median([v for t, v in final.items() if plan[t] == k]))
                                            for k in (1, 2, 3) if any(plan[t] == k for t in final)},
        }
Path(__file__).with_name('stream_audit.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps({k: (v['order_matches_registered_stream'], v['anchor_abs_error'],
                      v['terminal_median_recomputed'] == v['terminal_median_recorded'],
                      round(v['end_of_task_median_canonical64'], 4)) for k, v in out.items()}, indent=1))
