"""Detached O11 orchestrator (O9 pattern; module aliased as o9) (untracked; archived with the run).

1. E5 dry run on development world 27: leg 1 stops after one cell (exit 3); the dead parent's orphan pool
   workers are killed; leg 2 must reuse that cell and finish (exit 0).
2. Gates E1/E2/E3/E4b (o9.gates), skipped only if a gates.json at this exact protocol already passed.
3. The sealed run (O11 main path: clean code, check_prereg/check_invalid/check_adequacy, run(cells())).
Every step logs to o11_launch.log. Any failure stops before the sealed band is touched further."""
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch

from row.experiments import o11_sealed_instream_sleep as o9
from row.experiments.so1_storage import log_line

LOG = o9.ROOT / 'o11_launch.log'


def leg(stop_after=None):
    cmd = [sys.executable, '-m', 'row.experiments.o11_sealed_instream_sleep', '--dry-run']
    if stop_after is not None:
        cmd += ['--stop-after', str(stop_after)]
    p = subprocess.Popen(cmd)
    code = p.wait()
    killed = 0
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if f'parent_pid={p.pid}' in ' '.join(proc.info['cmdline'] or []):
            proc.kill(); killed += 1
    log_line(LOG, f'dry leg stop_after={stop_after} exit={code} orphans_killed={killed}')
    return code


if __name__ == '__main__':
    torch.set_num_threads(1)
    log_line(LOG, 'START')
    if (o9.DRY_ROOT / 'exit.json').exists() and json.loads((o9.DRY_ROOT / 'exit.json').read_text()).get('exit_code') == 0:
        log_line(LOG, 'dry run already complete; skipping E5')
    else:
        if leg(1) != 3:
            log_line(LOG, 'ABORT: dry leg 1 did not stop as a restart test'); sys.exit(2)
        time.sleep(5)
        if leg(None) != 0:
            log_line(LOG, 'ABORT: dry leg 2 failed'); sys.exit(2)
        if 'reused validated cell' not in (o9.DRY_ROOT / 'run.log').read_text():
            log_line(LOG, 'ABORT: restart did not reuse the completed cell'); sys.exit(2)
        log_line(LOG, 'E5 dry run + restart PASSED')
    gpath = o9.ROOT / 'gates' / 'gates.json'
    g = json.loads(gpath.read_text()) if gpath.exists() else None
    if not g or g.get('protocol_sha256') != o9.fingerprint(o9.protocol()) or not g.get('all_pass'):
        g = o9.gates()
    log_line(LOG, f'GATES {json.dumps({k: v for k, v in g.items() if k != "protocol_sha256"})}')
    if not g['all_pass']:
        log_line(LOG, 'ABORT: gates failed'); sys.exit(2)
    o9.require_clean_code(o9.OUTPUT)
    for check in ('tools/check_prereg.py', 'tools/check_invalid.py', 'tools/check_adequacy.py'):
        subprocess.run([sys.executable, check], check=True)
    log_line(LOG, 'LAUNCH sealed run')
    o9.run(o9.cells())
    log_line(LOG, f'DONE report {o9.OUTPUT}')
