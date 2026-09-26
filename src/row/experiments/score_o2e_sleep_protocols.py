"""Independent scorer for O2E (plan `O2E_SLEEP_OTHER_PROTOCOLS_PLAN.md`, frozen d0e771d)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'o2e_sleep_protocols.py'
THRESHOLD = 0.05
WORLDS, STREAMS = range(13, 20), (0, 1, 2)
EXPECTED_TASKS = {'STAGED_SLEEP': 64, 'MIXED_SLEEP': 124}


def label(r, b):
    if r >= 4 and b <= 1:
        return 'RESCUES'
    return 'HARMS' if b >= 2 else 'NO_RESCUE'


def score(root=Path('artifacts/o2e_sleep_protocols'), output=Path('reports/o2e_sleep_protocols.json')):
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
    base = json.loads(Path('reports/o2_online_reliability.json').read_text())['cells']
    out = {}
    for arm, b_arm in protocol['arms'].items():
        m0, m1 = {}, {}
        for w in WORLDS:
            for s in STREAMS:
                key = f'{arm}_w{w}_s{s}'
                saved = json.loads((root / 'cells' / f'{key}.json').read_text())
                rec = saved['record']
                if (saved['stamp'] != {'protocol_sha256': sha} or not saved['complete']
                        or fingerprint(rec) != saved['record_sha256'] or report['cells'][key] != rec):
                    raise ValueError(f'cell integrity failure: {key}')
                v = sorted(rec['terminal_per_task'].values())
                if len(v) != 64 or not math.isclose((v[31] + v[32]) / 2, rec['terminal_median'], rel_tol=1e-12):
                    problems.append(f'{key}: median')
                if (rec['tasks'] != EXPECTED_TASKS[arm] or rec['pool_size'] != 64 * EXPECTED_TASKS[arm]
                        or rec['library_sha256'] == rec['library_sha256_before']):
                    problems.append(f'{key}: construction or non-vacuity')
                m0[(w, s)] = base[f'{b_arm}_w{w}_s{s}']['terminal_median']
                m1[(w, s)] = rec['terminal_median']
        ok = lambda x: math.isfinite(x) and x < THRESHOLD  # noqa: E731
        N = [k for k, v in m0.items() if THRESHOLD <= v < 0.2]
        P = [k for k, v in m0.items() if v < THRESHOLD]
        C = [k for k, v in m0.items() if v >= 1.0]
        r, b, c = sum(ok(m1[k]) for k in N), sum(not ok(m1[k]) for k in P), sum(ok(m1[k]) for k in C)
        out[arm] = {'near': len(N), 'passing': len(P), 'collapsed': len(C), 'r': r, 'b': b, 'c': c,
                    'label': label(r, b), 'k_of_21': sum(ok(x) for x in m1.values()),
                    'by_cell': {f'w{w}_s{s}': [m0[(w, s)], m1[(w, s)]] for (w, s) in sorted(m1)}}
        if report['summary'][arm]['label'] != out[arm]['label']:
            problems.append(f'{arm}: runner label disagrees')
    return {'valid': not problems, 'problems': problems, 'tier': 1, 'exploratory': True, 'arms': out}


if __name__ == '__main__':
    result = score()
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result['valid'] else 1)
