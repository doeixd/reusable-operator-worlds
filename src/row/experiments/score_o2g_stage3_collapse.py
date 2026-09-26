"""Independent scorer for O2G (plan `O2G_STAGE3_COLLAPSE_PLAN.md`, frozen 9a60528)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'o2g_stage3_collapse.py'
COLLAPSE = 1.0
WORLDS, STREAMS, RERUNS = range(13, 20), (0, 1, 2), (1, 2)


def label(c, h):
    if c >= 5 and h <= 3:
        return 'SYSTEMATIC'
    return 'STOCHASTIC' if c <= 2 else 'MIXED'


def score(root=Path('artifacts/o2g_stage3_collapse'), output=Path('reports/o2g_stage3_collapse.json')):
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads(output.read_text())
    protocol, sha = manifest['protocol'], manifest['protocol_sha256']
    if fingerprint(protocol) != sha or report['protocol_sha256'] != sha:
        raise ValueError('protocol mismatch')
    for path, expected in protocol['input_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'input changed since the run: {path}')
    if digest(Path(__file__).parent / RUNNER) != protocol['implementation_sha256']:
        raise ValueError('runner changed since the run')
    problems = []
    g0 = report['gates']['G0']
    if not g0['passes'] or len(g0['by_cell']) != 3:
        problems.append(f'G0 did not pass on the three collapsed cells: {g0}')
    base = json.loads(Path('reports/o2_online_reliability.json').read_text())['cells']
    C = {(w, s) for w in WORLDS for s in STREAMS if base[f'STAGED_w{w}_s{s}']['terminal_median'] >= COLLAPSE}
    if len(C) != 3:
        problems.append(f'{len(C)} collapsed O2 cells, plan says 3')
    cc = hc = 0
    by = {}
    for w in WORLDS:
        for s in STREAMS:
            for k in RERUNS:
                key = f'w{w}_s{s}_k{k}'
                saved = json.loads((root / 'cells' / f'{key}.json').read_text())
                rec = saved['record']
                if (saved['stamp'] != {'protocol_sha256': sha} or not saved['complete']
                        or fingerprint(rec) != saved['record_sha256'] or report['cells'][key] != rec):
                    raise ValueError(f'cell integrity failure: {key}')
                v = sorted(rec['terminal_per_task'].values())
                if len(v) != 64 or not math.isclose((v[31] + v[32]) / 2, rec['terminal_median'], rel_tol=1e-12):
                    problems.append(f'{key}: median')
                col = not math.isfinite(rec['terminal_median']) or rec['terminal_median'] >= COLLAPSE
                if (w, s) in C:
                    cc += col
                else:
                    hc += col
                by[key] = [base[f'STAGED_w{w}_s{s}']['terminal_median'], rec['terminal_median'],
                           rec['first16_end_of_task_median']]
    lab = label(cc, hc)
    if report['summary']['label'] != lab:
        problems.append('runner label disagrees')
    return {'valid': not problems, 'problems': problems, 'tier': 1, 'exploratory': True,
            'C_reruns_collapsed_of_6': cc, 'H_reruns_collapsed_of_36': hc, 'label': lab, 'by_rerun': by}


if __name__ == '__main__':
    result = score()
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result['valid'] else 1)
