"""Independent scorer for O2D (plan `O2D_SLEEP_MEMORY_DOSE_PLAN.md`, frozen 7902346)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'o2d_sleep_memory.py'
THRESHOLD = 0.05
WORLDS, STREAMS = range(13, 20), (0, 1, 2)


def label(k, b):
    if k >= 18 and b <= 1:
        return 'SLEEP_MEMORY_SUFFICES'
    return 'HARMS' if b >= 2 else 'SHORT'


def score(root=Path('artifacts/o2d_sleep_memory'), output=Path('reports/o2d_sleep_memory.json')):
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads(output.read_text())
    protocol, sha = manifest['protocol'], manifest['protocol_sha256']
    if fingerprint(protocol) != sha or report['protocol_sha256'] != sha:
        raise ValueError('protocol mismatch')
    for path, expected in protocol['input_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'input changed since the run: {path}')
    here = Path(__file__).parent
    if digest(here / RUNNER) != protocol['implementation_sha256']:
        raise ValueError('runner changed since the run')
    problems = []
    if not report['gates']['G0']['passes']:
        problems.append('G0 failed')
    base = json.loads(Path('reports/o2_online_reliability.json').read_text())['cells']
    passing = [(w, s) for w in WORLDS for s in STREAMS if base[f'SHUFFLED_w{w}_s{s}']['terminal_median'] < THRESHOLD]
    if len(passing) != 9:
        problems.append(f'{len(passing)} O2-passing cells, plan says 9')
    arms, found = protocol['arms'], {}
    for arm, m_per in arms.items():
        m = {}
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
                if rec['pool_size'] != 188 * m_per or rec['library_sha256'] == rec['library_sha256_before']:
                    problems.append(f'{key}: pool or non-vacuity')
                m[(w, s)] = rec['terminal_median']
        k = sum(math.isfinite(x) and x < THRESHOLD for x in m.values())
        b = sum(not (math.isfinite(m[c]) and m[c] < THRESHOLD) for c in passing)
        found[arm] = {'memory_per_task': m_per, 'k_of_21': k, 'b_broken_of_9': b, 'label': label(k, b),
                      'by_cell': {f'w{w}_s{s}': m[(w, s)] for (w, s) in sorted(m)}}
        if report['summary']['arms'][arm]['label'] != found[arm]['label']:
            problems.append(f'{arm}: runner label disagrees')
    suff = [found[a]['memory_per_task'] for a in found if found[a]['label'] == 'SLEEP_MEMORY_SUFFICES']
    return {'valid': not problems, 'problems': problems, 'tier': 1, 'exploratory': True,
            'M_star': min(suff) if suff else None, 'arms': found}


if __name__ == '__main__':
    result = score()
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result['valid'] else 1)
