"""Independent scorer for N1c: recomputes every outcome from durable cells."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'n1c_anchor_coverage.py'
THRESHOLD = 0.05
EXPECTED = {'COVER6_K6': (6, 6), 'COVER5_K6': (6, 5), 'COVER5_K18': (18, 5)}
PRIMARY = {(True, False): 'COVERAGE_MATTERS', (True, True): 'COVERAGE_NOT_NEEDED_AT_K6',
           (False, False): 'K6_INSUFFICIENT', (False, True): 'ANOMALOUS'}


def score(root=Path('artifacts/n1c_anchor_coverage'), output=Path('reports/n1c_anchor_coverage.json')):
    root, output = Path(root), Path(output)
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads(output.read_text())
    protocol, sha = manifest['protocol'], manifest['protocol_sha256']
    if fingerprint(protocol) != sha or report['protocol'] != protocol or report['protocol_sha256'] != sha:
        raise ValueError('protocol mismatch')
    for path, expected in protocol['input_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'input changed since the run: {path}')
    here = Path(__file__).parent
    if digest(here / RUNNER) != protocol['implementation_sha256']:
        raise ValueError('runner changed since the run')
    if digest(here / 'n1_anchor_supply.py') != protocol['n1_implementation_sha256']:
        raise ValueError('N1 module changed since the run')
    mtime = output.stat().st_mtime
    for path in list(protocol['input_sha256']) + [str(here / RUNNER)]:
        if Path(path).stat().st_mtime > mtime:
            raise ValueError(f'report is older than its input: {path}')

    problems, cells = [], {}
    for arm in protocol['arms']:
        for world in protocol['worlds']:
            key = f'{arm}_w{world}'
            saved = json.loads((root / 'cells' / f'{key}.json').read_text())
            if (not saved.get('complete') or saved['stamp'] != {'protocol_sha256': sha}
                    or fingerprint(saved['record']) != saved['record_sha256']):
                raise ValueError(f'cell integrity failure: {key}')
            if report['cells'][key] != saved['record']:
                raise ValueError(f'report disagrees with the durable cell: {key}')
            cells[key] = saved['record']

    for key, r in cells.items():
        count, coverage = EXPECTED[r['arm']]
        if not math.isfinite(r['terminal_median']) or r['terminal_median'] < 0:
            problems.append(f'{key}: bad terminal median')
        if r['pool_tasks'] != 188 or r['pool_examples'] != 24064 or r['scored_tasks'] != 64:
            problems.append(f'{key}: pool or scored set is not the registered one')
        if r['anchors'] != count or len(r['operations_covered']) != coverage:
            problems.append(f'{key}: {r["anchors"]} anchors covering {len(r["operations_covered"])}')
        if coverage == 5 and r['excluded_operation'] in r['operations_covered']:
            problems.append(f'{key}: covers its own excluded operation')
        # Recompute the registered split from per-task values rather than trusting it.
        split = r['split']
        if split['tasks_using_excluded'] + split['tasks_not_using_excluded'] != 64:
            problems.append(f'{key}: split does not partition the 64 tasks')

    worlds, arms = protocol['worlds'], protocol['arms']
    for w in worlds:
        if len({cells[f'{a}_w{w}']['excluded_operation'] for a in arms}) != 1:
            problems.append(f'w{w}: arms disagree on the excluded operation')
        base = cells[f'{arms[0]}_w{w}']['first_64_draws']
        if any(cells[f'{a}_w{w}']['first_64_draws'] != base for a in arms):
            problems.append(f'w{w}: arms drew different minibatch indices')

    suff = {a: sum(cells[f'{a}_w{w}']['terminal_median'] < THRESHOLD for w in worlds) >= 2
            for a in arms}
    primary = PRIMARY[(suff['COVER6_K6'], suff['COVER5_K6'])]
    secondary = 'COUNT_COMPENSATES' if suff['COVER5_K18'] else 'COUNT_DOES_NOT_COMPENSATE'
    descriptive = {}
    for arm in ('COVER5_K6', 'COVER5_K18'):
        for w in worlds:
            r = cells[f'{arm}_w{w}']
            if r['terminal_median'] >= THRESHOLD:
                free = r['split']['median_not_using_excluded']
                descriptive[f'{arm}_w{w}'] = 'LOCAL' if free is not None and free < THRESHOLD else 'GLOBAL'
    summary = report['summary']
    if summary.get('primary') != primary:
        problems.append(f"primary {summary.get('primary')} != recomputed {primary}")
    if summary.get('secondary') != secondary:
        problems.append(f"secondary {summary.get('secondary')} != recomputed {secondary}")
    if summary.get('descriptive_split') != descriptive:
        problems.append('descriptive split disagrees with recomputation')
    if json.loads((root / 'status.json').read_text())['state'] != 'complete':
        problems.append('status is not complete')
    if json.loads((root / 'exit.json').read_text())['exit_code'] != 0:
        problems.append('nonzero exit code')

    return {'valid': not problems, 'problems': problems, 'primary': primary, 'secondary': secondary,
            'descriptive_split': descriptive, 'suffices': suff,
            'by_arm': {a: {str(w): cells[f'{a}_w{w}']['terminal_median'] for w in worlds} for a in arms},
            'split': {k: r['split'] for k, r in cells.items() if r['arm'] != 'COVER6_K6'},
            'excluded_by_world': {str(w): cells[f'{arms[0]}_w{w}']['excluded_operation'] for w in worlds},
            'near_threshold_cells': sorted(k for k, r in cells.items() if r['near_threshold']),
            'interpretation': 'N1c development evidence on worlds 0-2; never confirmatory'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/n1c_anchor_coverage'))
    parser.add_argument('--report', type=Path, default=Path('reports/n1c_anchor_coverage.json'))
    args = parser.parse_args()
    result = score(args.root, args.report)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
