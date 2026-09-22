"""N1 performance pass: time REAL updates, do not guess the cell cost."""
import json, time
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_j1c_curriculum import stage_setup, train_stage, BATCH

torch.set_num_threads(1)
PROBE = 512          # real updates, timed
FULL = 65536         # the registered per-cell budget
CELLS = 12           # 4 arms x 3 worlds

rows = []
for stage, label in ((3, 'length-3 (64 tasks)'), (1, 'length-1 (60 tasks)')):
    cfg, world, updates, seq = stage_setup(0, stage, 5000)
    t0 = time.perf_counter()
    train_stage(cfg, world, PROBE, seq)
    seconds = time.perf_counter() - t0
    rows.append({'stage': stage, 'label': label, 'tasks': len(world.tasks),
                 'registered_stage_updates': updates,
                 'probe_updates': PROBE, 'probe_seconds': seconds,
                 'seconds_per_update': seconds / PROBE, 'batch': BATCH})
    print(f"{label:22} {PROBE} updates in {seconds:7.2f}s "
          f"= {seconds/PROBE*1000:.3f} ms/update   (stage budget {updates})")

# A J1c-shaped cell is 16,384 len-1 + 16,384 len-2 + 32,768 len-3 updates.
# Use the measured len-3 rate for the len-3 portion and the len-1 rate for the
# anchor portion; len-2 sits between, so the len-3 rate is the conservative
# choice for it.
r1 = next(r['seconds_per_update'] for r in rows if r['stage'] == 1)
r3 = next(r['seconds_per_update'] for r in rows if r['stage'] == 3)
staged_cell = 16384 * r1 + 16384 * r3 + 32768 * r3
flat_cell = FULL * r3

print()
print(f"staged-shaped cell  ~ {staged_cell/60:6.1f} min")
print(f"flat length-3 cell  ~ {flat_cell/60:6.1f} min")
serial = (3 * staged_cell + 9 * flat_cell) / 3600   # rough: 3 STAGED + 9 others
print(f"12 cells serial     ~ {serial:6.2f} h")
for pool in (3, 4):
    print(f"12 cells at pool {pool}  ~ {serial/pool:6.2f} h  (memory-bounded cap)")

out = Path(__file__).resolve().parent / 'performance_pass.json'
out.write_text(json.dumps({
    'probe_updates': PROBE, 'registered_cell_updates': FULL, 'cells': CELLS,
    'measured': rows,
    'estimate_seconds': {'staged_shaped_cell': staged_cell, 'flat_length3_cell': flat_cell,
                         'twelve_cells_serial': serial * 3600},
    'note': 'timed on this host, one thread, no other load; extrapolation is linear '
            'in updates and does not include per-cell scoring checkpoints beyond '
            'those train_stage already performs',
}, indent=1), encoding='utf-8')
print('\nwrote', out)
