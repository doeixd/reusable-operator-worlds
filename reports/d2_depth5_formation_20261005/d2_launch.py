"""Detached D2 orchestrator (D1 pattern; module aliased as d1) (untracked; archived with the run). O9/O11 pattern without gates (D1 has none
registered beyond the dry run): dry run leg 1 stops after one cell (exit 3), orphans of the dead parent are killed,
leg 2 must reuse that cell and finish (exit 0); then clean-code check, the three integrity checkers, and the run."""
import subprocess
import sys
import time

import psutil
import torch

from row.experiments import d2_depth5_formation as d1
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.so1_storage import log_line

LOG = d1.ROOT / 'd2_launch.log'


def leg(stop_after=None):
    cmd = [sys.executable, '-m', 'row.experiments.d2_depth5_formation', '--dry-run']
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
    if leg(1) != 3:
        log_line(LOG, 'ABORT: dry leg 1 did not stop as a restart test'); sys.exit(2)
    time.sleep(5)
    if leg(None) != 0:
        log_line(LOG, 'ABORT: dry leg 2 failed'); sys.exit(2)
    if 'reused validated cell' not in (d1.DRY_ROOT / 'run.log').read_text():
        log_line(LOG, 'ABORT: restart did not reuse the completed cell'); sys.exit(2)
    log_line(LOG, 'dry run + restart PASSED')
    require_clean_code(d1.OUTPUT)
    for check in ('tools/check_prereg.py', 'tools/check_invalid.py', 'tools/check_adequacy.py'):
        subprocess.run([sys.executable, check], check=True)
    log_line(LOG, 'LAUNCH run')
    d1.run(d1.cells())
    log_line(LOG, f'DONE report {d1.OUTPUT}')
