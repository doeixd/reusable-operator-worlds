"""Detached O8 launcher (untracked; archived with the run). Gates, then the full run.

PI 2026-10-03: "It's ok to be low on memory" -> reserve lowered 4.5 -> 2.0 GiB. MIN_FREE_GIB is not in
the protocol fingerprint; precondition.json records the value used. Runner code is unchanged."""
import json
import sys

import torch

from row.experiments import o8_instream_reroute as o8
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code

if __name__ == '__main__':
    torch.set_num_threads(1)
    o8.MIN_FREE_GIB = 2.0
    require_clean_code(o8.OUTPUT)
    gpath = o8.ROOT / 'gates' / 'gates.json'
    g = json.loads(gpath.read_text()) if gpath.exists() else None
    if not g or g.get('protocol_sha256') != o8.fingerprint(o8.protocol()):
        g = o8.gates()
    print(json.dumps(g, indent=1), flush=True)
    if not g['all_pass']:
        sys.exit(2)
    o8.run(o8.cells())
    print(f'O8 report: {o8.OUTPUT}', flush=True)
