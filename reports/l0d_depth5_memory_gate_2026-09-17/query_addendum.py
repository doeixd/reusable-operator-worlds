"""Post-hoc diagnostic: query NMSE of the depth-five gate's selected route.
Read-only. Does not touch the protocol-stamped report."""
import json, torch, numpy as np
from pathlib import Path
from row.experiments.l0d_depth5_memory_gate import task, NAME, WORLD
from row.experiments.l0d_depth4_execution_gate import DepthLibrary
from row.experiments.preflight_l0d import load_source
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.so1_storage import digest

rec = json.loads(Path('reports/l0d_depth5_memory_gate_v2.json').read_text())['cell']
cfg, world, model, _, _ = load_source(NAME, WORLD)
lib = DepthLibrary(model)
assert library_sha(model) == rec['library_sha256'], 'library mismatch'
t = task(cfg, world)
assert list(t['program']) == rec['program'], 'program mismatch'
qx = torch.tensor(t['eval_x'], dtype=torch.float32)
qy = torch.tensor(t['eval_y'], dtype=torch.float32)
with torch.no_grad():
    pred = lib.hard(qx, rec['selected_route'])
    mse = float(torch.mean((pred - qy) ** 2))
    nmse = mse / float(torch.mean(qy ** 2))
out = {
 'source_report_sha256': digest(Path('reports/l0d_depth5_memory_gate_v2.json')),
 'library_sha256': rec['library_sha256'],
 'program': rec['program'], 'selected_route': rec['selected_route'],
 'query_examples': int(qx.shape[0]),
 'selected_query_mse': mse, 'selected_query_nmse': nmse,
 'passes_0_05': bool(nmse <= 0.05),
 'note': 'post-hoc read-only diagnostic recovering the plan-required query NMSE; '
         'not part of the stamped gate record and not a PX7 result',
}
print(json.dumps(out, indent=2))
Path('reports/l0d_depth5_memory_gate_2026-09-17/query_addendum.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
