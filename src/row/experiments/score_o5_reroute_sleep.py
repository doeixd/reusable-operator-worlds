"""Independent scorer for O5 (plan `O5_REROUTE_SLEEP_PLAN.md`, frozen 7ec7a28)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'o5_reroute_sleep.py'
THRESHOLD, COLLAPSE = 0.05, 1.0
REF = {'O2': (range(13, 20), 'reports/o2_online_reliability.json', 'reports/o2d_sleep_memory.json', 'RES64'),
       'O3': (range(20, 27), 'reports/o3_online_sleep_v2.json', 'reports/o3_online_sleep_v2.json', 'SLEEP'),
       'O4': (range(900, 915), 'reports/o4_sealed_confirmation.json', 'reports/o4_sealed_confirmation.json', 'SLEEP')}


def label(r_c, b):
    if b >= 2:
        return 'HARMS'
    if r_c >= 6:
        return 'COLLAPSE_REPAIRED'
    return 'PARTIAL' if r_c >= 3 else 'NO_REPAIR'


def score(root=Path('artifacts/o5_reroute_sleep'), output=Path('reports/o5_reroute_sleep.json')):
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
    problems = [] if report['gates']['G0']['passes'] else ['G0 failed']
    ok = lambda m: math.isfinite(m) and m < THRESHOLD  # noqa: E731
    C, P, rows, per_band, g1 = [], [], {}, {}, 0
    for band, (worlds, trep, srep, skey) in REF.items():
        tc, sc = json.loads(Path(trep).read_text())['cells'], json.loads(Path(srep).read_text())['cells']
        n_sleep = n_new = 0
        for w in worlds:
            for s in (0, 1, 2):
                key = f'{band}_w{w}_s{s}'
                saved = json.loads((root / 'cells' / f'{key}.json').read_text())
                rec = saved['record']
                if (saved['stamp'] != {'protocol_sha256': sha} or not saved['complete']
                        or fingerprint(rec) != saved['record_sha256'] or report['cells'][key] != rec):
                    raise ValueError(f'cell integrity failure: {key}')
                term = tc[f'SHUFFLED_w{w}_s{s}']['terminal_median']
                slp = sc[f'{skey}_w{w}_s{s}']
                if rec['routes_changed'] == 0:
                    g1 += 1
                    if not (rec['library_sha256'] == slp['library_sha256']
                            and rec['terminal_per_task'] == slp['terminal_per_task']):
                        problems.append(f'G1: {key} unchanged routes but differs from SLEEP')
                if rec['pool_size'] != 188 * 64 or len(rec['terminal_per_task']) != 64:
                    problems.append(f'{key}: construction')
                rows[key] = [term, slp['terminal_median'], rec['terminal_median'], rec['routes_changed']]
                if term >= COLLAPSE:
                    C.append(key)
                if ok(slp['terminal_median']):
                    P.append(key)
                n_sleep += ok(slp['terminal_median'])
                n_new += ok(rec['terminal_median'])
        per_band[band] = {'cells': 3 * len(worlds), 'sleep_pass': n_sleep, 'reroute_sleep_pass': n_new}
    r_c = sum(ok(rows[k][2]) for k in C)
    b = sum(not ok(rows[k][2]) for k in P)
    lab = label(r_c, b)
    if report['summary']['label'] != lab:
        problems.append('runner label disagrees')
    return {'valid': not problems, 'problems': problems, 'tier': 1, 'exploratory': True, 'label': lab,
            'r_C': r_c, 'collapse_cells': len(C), 'b': b, 'sleep_passing': len(P), 'g1_cells_checked': g1,
            'per_band': per_band, 'collapse_rows': {k: rows[k] for k in C}}


if __name__ == '__main__':
    result = score()
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result['valid'] else 1)
