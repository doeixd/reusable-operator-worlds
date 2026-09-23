"""Independent scorer for O1 Tier 1. Recomputes the decision from durable cells."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'o1_online_anchor.py'
THRESHOLD = 0.05
EXPECTED_STREAM = {'STAGED': 188, 'SHUFFLED': 188, 'MIXED_L1': 124, 'PLAIN': 64}
EXPECTED_DEPTHS = {'SHUFFLED': {'1': 60, '2': 64, '3': 64}, 'MIXED_L1': {'1': 60, '2': 0, '3': 64}}


def score(root=Path('artifacts/o1_online_anchor'), output=Path('reports/o1_online_anchor.json')):
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
    if digest(here / 'audit_so2_online_gate.py') != protocol['so2_implementation_sha256']:
        raise ValueError('SO2 module changed since the run')
    if digest(here.parent / 'models' / 'online_variable_depth.py') != protocol['learner_sha256']:
        raise ValueError('learner changed since the run')
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
        if not math.isfinite(r['terminal_median']) or r['terminal_median'] < 0:
            problems.append(f'{key}: bad terminal median')
        if r['scored_tasks'] != 64 or len(r['terminal_per_task']) != 64:
            problems.append(f'{key}: not scored on exactly the canonical 64 tasks')
        if r['stream_tasks'] != EXPECTED_STREAM[r['arm']]:
            problems.append(f'{key}: stream has {r["stream_tasks"]} tasks')
        if r['arm'] in EXPECTED_DEPTHS and r['depth_histogram'] != EXPECTED_DEPTHS[r['arm']]:
            problems.append(f'{key}: depth histogram {r["depth_histogram"]}')
        if not r['route_lengths_match_plan']:
            problems.append(f'{key}: a route length disagrees with its planned depth')
        median = sorted(r['terminal_per_task'].values())
        recomputed = (median[31] + median[32]) / 2
        if not math.isclose(recomputed, r['terminal_median'], rel_tol=1e-12, abs_tol=1e-15):
            problems.append(f'{key}: terminal median is not the median of its per-task values')

    worlds, arms = protocol['worlds'], protocol['arms']
    # Single-lifetime arms must actually interleave: a stream whose first 20
    # tasks are all one depth would be an accidental curriculum.
    for arm in ('SHUFFLED', 'MIXED_L1'):
        for w in worlds:
            head = cells[f'{arm}_w{w}']['first_20_stream_depths']
            if head is not None and len(set(head)) < 2:
                problems.append(f'{arm}_w{w}: the first 20 tasks are all one depth')
    for w in worlds:
        if cells[f'PLAIN_w{w}']['terminal_median'] < THRESHOLD:
            problems.append(f'w{w}: PLAIN passed; the floor is not a floor')

    passes = {a: sum(cells[f'{a}_w{w}']['terminal_median'] < THRESHOLD for w in worlds) for a in arms}
    decision = 'LIVE' if passes['SHUFFLED'] >= 2 or passes['MIXED_L1'] >= 2 else 'NOT_LIVE'
    if report['summary'].get('decision') != decision:
        problems.append(f"decision {report['summary'].get('decision')} != recomputed {decision}")
    if json.loads((root / 'status.json').read_text())['state'] != 'complete':
        problems.append('status is not complete')
    if json.loads((root / 'exit.json').read_text())['exit_code'] != 0:
        problems.append('nonzero exit code')

    return {'valid': not problems, 'problems': problems, 'tier': 1, 'exploratory': True,
            'decision': decision, 'worlds_passing': passes,
            'by_arm': {a: {str(w): cells[f'{a}_w{w}']['terminal_median'] for w in worlds} for a in arms},
            'end_of_task_by_arm': {a: {str(w): cells[f'{a}_w{w}']['end_of_task_median'] for w in worlds}
                                   for a in arms},
            'note': 'Tier 1 indication; sizes a Tier 2, never a verdict'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/o1_online_anchor'))
    parser.add_argument('--report', type=Path, default=Path('reports/o1_online_anchor.json'))
    args = parser.parse_args()
    result = score(args.root, args.report)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
