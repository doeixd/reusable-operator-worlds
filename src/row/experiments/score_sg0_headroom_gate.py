"""Independent scorer for SG0.

Recomputes every headline quantity from the per-cell records rather than
trusting the report's own summary, and refuses a report that is stale, that was
produced under a different protocol, or that presents a partial run as the
registered triage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'sg0_headroom_gate.py'
STAGED_CELLS_IN_FULL_GRID = 36


def score(root=Path('artifacts/sg0_headroom_gate'), output=Path('reports/sg0_headroom_gate.json')):
    root, output = Path(root), Path(output)
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads(output.read_text())
    protocol, sha = manifest['protocol'], manifest['protocol_sha256']

    if fingerprint(protocol) != sha or report['protocol'] != protocol or report['protocol_sha256'] != sha:
        raise ValueError('protocol mismatch')
    for path, expected in protocol['input_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'input changed since the run: {path}')
    if digest(Path(__file__).with_name(RUNNER)) != protocol['implementation_sha256']:
        raise ValueError('runner changed since the run')

    # Stale-report guard: the report must not predate any file it depends on.
    report_mtime = output.stat().st_mtime
    for path in list(protocol['input_sha256']) + [str(Path(__file__).with_name(RUNNER))]:
        if Path(path).stat().st_mtime > report_mtime:
            raise ValueError(f'report is older than its input: {path}')

    problems, cells = [], {}
    for cell in protocol['cells']:
        name, world, depth, support = cell
        key = f'{name}_w{world}_d{depth}_s{support}'
        saved = json.loads((root / 'cells' / f'{key}.json').read_text())
        if (not saved.get('complete') or saved['stamp'] != {'protocol_sha256': sha}
                or fingerprint(saved['record']) != saved['record_sha256']):
            raise ValueError(f'cell integrity failure: {key}')
        if report['cells'][key] != saved['record']:
            raise ValueError(f'report disagrees with the durable cell: {key}')
        cells[key] = saved['record']

    for key, record in cells.items():
        rows = record['rows']
        if len(rows) != protocol['tasks']:
            problems.append(f'{key}: task count')
        regrets = [r['regret'] for r in rows]
        floors = [r['null_floor'] for r in rows]
        if not math.isclose(record['median_regret'], float(np.median(regrets)), rel_tol=1e-12, abs_tol=1e-12):
            problems.append(f'{key}: median regret disagrees')
        if not math.isclose(record['median_floor'], float(np.median(floors)), rel_tol=1e-12, abs_tol=1e-12):
            problems.append(f'{key}: median floor disagrees')
        if record['cell_counts_toward_k'] != (float(np.median(regrets)) > float(np.median(floors))):
            problems.append(f'{key}: k-membership disagrees')
        for row in rows:
            if not math.isclose(row['regret'], row['nmse_hat_qb'] - row['nmse_star_qb'],
                                rel_tol=1e-12, abs_tol=1e-15):
                problems.append(f'{key} task {row["task"]}: regret is not the scored difference')
            if not row['routes_differ'] and row['regret'] != 0.0:
                problems.append(f'{key} task {row["task"]}: identical routes with nonzero regret')
            if row['near_tie_size'] < 1:
                problems.append(f'{key} task {row["task"]}: empty near-tie set')
            for field in ('nmse_hat_qb', 'nmse_star_qb', 'null_floor', 'identifiability'):
                if not math.isfinite(row[field]) or row[field] < 0:
                    problems.append(f'{key} task {row["task"]}: bad {field}')
            if depth == 3 and support == 128 and not (row.get('anchor_route_matches')
                                                      and row.get('anchor_nmse_matches')):
                problems.append(f'{key} task {row["task"]}: J2A ENUM anchor missing or failed')

    staged = [r for r in cells.values() if r['staged']]
    control = [r for r in cells.values() if not r['staged']]
    k = sum(r['cell_counts_toward_k'] for r in staged)
    if report['summary']['k'] != k:
        problems.append('reported k disagrees with recomputation')

    label = protocol['label']
    if label == 'full':
        if len(staged) != STAGED_CELLS_IN_FULL_GRID:
            problems.append(f'full grid must have {STAGED_CELLS_IN_FULL_GRID} staged cells, has {len(staged)}')
        triage = 'NO-HEADROOM' if k <= 2 else 'HEADROOM (ordering test decides which)'
    else:
        triage = 'NOT APPLICABLE'
        if report['summary'].get('triage') != 'NOT APPLICABLE':
            problems.append('a partial run must not report the registered triage')

    if json.loads((root / 'status.json').read_text())['state'] != 'complete':
        problems.append('status is not complete')
    if json.loads((root / 'exit.json').read_text())['exit_code'] != 0:
        problems.append('nonzero exit code')

    return {'valid': not problems, 'problems': problems, 'label': label,
            'staged_cells': len(staged), 'control_cells': len(control), 'k': k, 'triage': triage,
            'staged_median_regret': float(np.median([r['median_regret'] for r in staged])) if staged else None,
            'staged_median_floor': float(np.median([r['median_floor'] for r in staged])) if staged else None,
            'control_median_regret': float(np.median([r['median_regret'] for r in control])) if control else None,
            'routes_differ': sum(r['routes_differ_count'] for r in cells.values()),
            'route_comparisons': len(cells) * protocol['tasks']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/sg0_headroom_gate'))
    parser.add_argument('--report', type=Path, default=Path('reports/sg0_headroom_gate.json'))
    args = parser.parse_args()
    result = score(args.root, args.report)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
