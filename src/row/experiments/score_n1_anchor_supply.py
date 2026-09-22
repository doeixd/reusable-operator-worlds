"""Independent scorer for N1.

Recomputes the registered triage from the durable per-cell records rather than
trusting the report's summary, and enforces the construction invariants the
plan's amendments registered.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'n1_anchor_supply.py'
THRESHOLD = 0.05
RATIO = 5.0
SHAM_FLOOR = 0.45


def score(root=Path('artifacts/n1_anchor_supply'), output=Path('reports/n1_anchor_supply.json')):
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
    report_mtime = output.stat().st_mtime
    for path in list(protocol['input_sha256']) + [str(Path(__file__).with_name(RUNNER))]:
        if Path(path).stat().st_mtime > report_mtime:
            raise ValueError(f'report is older than its input: {path}')

    problems, cells = [], {}
    for world in protocol['worlds']:
        for arm in protocol['arms']:
            key = f'{arm}_w{world}'
            saved = json.loads((root / 'cells' / f'{key}.json').read_text())
            if (not saved.get('complete') or saved['stamp'] != {'protocol_sha256': sha}
                    or fingerprint(saved['record']) != saved['record_sha256']):
                raise ValueError(f'cell integrity failure: {key}')
            if report['cells'][key] != saved['record']:
                raise ValueError(f'report disagrees with the durable cell: {key}')
            cells[key] = saved['record']

    for key, record in cells.items():
        arm = record['arm']
        if not math.isfinite(record['terminal_median']) or record['terminal_median'] < 0:
            problems.append(f'{key}: bad terminal median')
        if record['scored_tasks'] != 64:
            problems.append(f'{key}: not scored on the canonical 64 tasks')
        # Amendment 1/3 construction invariants.
        if arm in ('INTERLEAVED', 'SHAM') and record['pool_tasks'] != 188:
            problems.append(f'{key}: pool is {record["pool_tasks"]}, must be 188')
        if arm == 'NONE' and record['pool_tasks'] != 64:
            problems.append(f'{key}: NONE must hold 64 tasks')
        if arm == 'SHAM' and record['depth_histogram'].get('3') != 188:
            problems.append(f'{key}: SHAM must be entirely length-3')
        if arm == 'INTERLEAVED' and record['depth_histogram'] != {'1': 60, '2': 64, '3': 64}:
            problems.append(f'{key}: INTERLEAVED depth histogram wrong')
        if arm == 'STAGED' and record.get('anchor_matches') is not True:
            problems.append(f'{key}: STAGED does not reproduce its committed J1c cell')

    # INTERLEAVED and SHAM must be pool-matched and share one sampling stream.
    for world in protocol['worlds']:
        a, b = cells.get(f'INTERLEAVED_w{world}'), cells.get(f'SHAM_w{world}')
        if a and b:
            if a['pool_examples'] != b['pool_examples']:
                problems.append(f'w{world}: INTERLEAVED and SHAM pools differ in examples')
            if a['first_64_draws'] != b['first_64_draws']:
                problems.append(f'w{world}: the two arms drew different minibatch indices')

    # Non-vacuity: the floor must fail.
    for world in protocol['worlds']:
        none = cells.get(f'NONE_w{world}')
        if none and none['terminal_median'] <= THRESHOLD:
            problems.append(f'w{world}: NONE passed the threshold; the floor is not a floor')

    m = {r['world']: r['terminal_median'] for r in cells.values() if r['arm'] == 'INTERLEAVED'}
    s = {r['world']: r['terminal_median'] for r in cells.values() if r['arm'] == 'SHAM'}
    suffice = sum(1 for v in m.values() if v < THRESHOLD)
    uninterpretable = sorted(w for w, v in s.items() if v < SHAM_FLOOR)
    ratio = float(np.median(list(s.values()))) / max(float(np.median(list(m.values()))), 1e-12)
    if suffice >= 2:
        verdict = 'ANCHORS_SUFFICE'
    elif uninterpretable:
        verdict = 'PARTIAL_UNINTERPRETABLE'
    elif ratio >= RATIO:
        verdict = 'ANCHORS_PARTIAL'
    else:
        verdict = 'ANCHORS_INSUFFICIENT'
    if report['summary'].get('verdict') != verdict:
        problems.append(f"reported verdict {report['summary'].get('verdict')} != recomputed {verdict}")

    if json.loads((root / 'status.json').read_text())['state'] != 'complete':
        problems.append('status is not complete')
    if json.loads((root / 'exit.json').read_text())['exit_code'] != 0:
        problems.append('nonzero exit code')

    return {'valid': not problems, 'problems': problems, 'verdict': verdict,
            'interleaved_by_world': m, 'sham_by_world': s,
            'staged_by_world': {r['world']: r['terminal_median'] for r in cells.values() if r['arm'] == 'STAGED'},
            'none_by_world': {r['world']: r['terminal_median'] for r in cells.values() if r['arm'] == 'NONE'},
            'worlds_under_threshold': suffice, 'sham_ratio': ratio,
            'uninterpretable_worlds': uninterpretable,
            'interpretation': 'N1 development evidence on worlds 0-2; never confirmatory'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/n1_anchor_supply'))
    parser.add_argument('--report', type=Path, default=Path('reports/n1_anchor_supply.json'))
    args = parser.parse_args()
    result = score(args.root, args.report)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
