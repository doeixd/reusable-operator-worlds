"""Independent scorer for O2 (plan `O2_ONLINE_ANCHOR_RELIABILITY_PLAN.md`, frozen b3c1c85).

Recomputes the registered decision from the durable cells, not from the runner's
summary: k per arm over ALL 21 cells (non-finite terminal = not passing), the
k-of-21 labels, the PLAIN floor clause, and the descriptive d_w. Independently of
the runner it also re-derives each cell's canonical-64 end-of-task median from the
raw lifetime `metrics.jsonl`, re-checks E4 (streams distinct) on the REAL cells,
E4b (interleaving), the last-task anchor, stream sizes and depth mixes, and that
the gates record passed at this protocol. Refuses a report older than its inputs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from row.experiments.so1_storage import digest, fingerprint

RUNNER = 'o2_online_reliability.py'
THRESHOLD = 0.05
ANCHOR_TOLERANCE = 1e-6
EXPECTED_STREAM = {'STAGED': 188, 'SHUFFLED': 188, 'MIXED_L1': 124, 'PLAIN': 64}
EXPECTED_DEPTHS = {'SHUFFLED': {'1': 60, '2': 64, '3': 64}, 'MIXED_L1': {'1': 60, '2': 0, '3': 64},
                   'STAGED': {'1': 60, '2': 64, '3': 64}, 'PLAIN': {'1': 0, '2': 0, '3': 64}}
THREE_STREAM = ('SHUFFLED', 'STAGED', 'MIXED_L1')


def label(k):
    if k >= 19:
        return 'RELIABLE'
    return 'INTERMEDIATE' if k >= 17 else 'UNRELIABLE'


def canonical_end_of_task(root: Path, key: str, arm: str, canonical_ids: set) -> float:
    work = root / 'work' / key
    path = work / ('lifetime' if arm in ('SHUFFLED', 'MIXED_L1') else 'stage3') / 'metrics.jsonl'
    final = {}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('record_type') == 'task_summary':
                final[row['task_id']] = float(row['final_nmse'])
    values = [final[t] for t in canonical_ids]
    return float(np.median(values))


def score(root=Path('artifacts/o2_online_reliability'), output=Path('reports/o2_online_reliability.json')):
    root, output = Path(root), Path(output)
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads(output.read_text())
    protocol, sha = manifest['protocol'], manifest['protocol_sha256']
    if fingerprint(protocol) != sha or report['protocol'] != protocol or report['protocol_sha256'] != sha:
        raise ValueError('protocol mismatch')
    if protocol.get('scale') != 1:
        raise ValueError('not a full-scale run')
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
    if digest(here / 'learned_lifetime.py') != protocol['lifetime_sha256']:
        raise ValueError('lifetime runner changed since the run')
    mtime = output.stat().st_mtime
    for path in list(protocol['input_sha256']) + [str(here / RUNNER)]:
        if Path(path).stat().st_mtime > mtime:
            raise ValueError(f'report is older than its input: {path}')

    problems = []
    gates = json.loads((root / 'gates' / 'gates.json').read_text())
    base_protocol = {k: v for k, v in protocol.items() if k != 'scale'}
    if not gates.get('all_pass'):
        problems.append('gates record did not pass')
    if gates.get('protocol_sha256') != fingerprint(base_protocol):
        problems.append('gates were run at a different protocol')

    worlds, arm_streams = protocol['worlds'], protocol['arm_streams']
    cells = {}
    for arm, streams in arm_streams.items():
        for w in worlds:
            for s in streams:
                key = f'{arm}_w{w}_s{s}'
                saved = json.loads((root / 'cells' / f'{key}.json').read_text())
                if (not saved.get('complete') or saved['stamp'] != {'protocol_sha256': sha}
                        or fingerprint(saved['record']) != saved['record_sha256']):
                    raise ValueError(f'cell integrity failure: {key}')
                if report['cells'][key] != saved['record']:
                    raise ValueError(f'report disagrees with the durable cell: {key}')
                cells[key] = saved['record']
    if len(cells) != len(report['cells']):
        problems.append('report carries cells outside the registered set')

    non_finite = []
    for key, r in cells.items():
        arm = r['arm']
        per_task = sorted(r['terminal_per_task'].values())
        if len(per_task) != 64 or r['scored_tasks'] != 64:
            problems.append(f'{key}: not scored on exactly the canonical 64 tasks')
            continue
        recomputed = (per_task[31] + per_task[32]) / 2
        if not math.isfinite(r['terminal_median']):
            non_finite.append(key)
        elif not math.isclose(recomputed, r['terminal_median'], rel_tol=1e-12, abs_tol=1e-15):
            problems.append(f'{key}: terminal median is not the median of its per-task values')
        if r['stream_tasks'] != EXPECTED_STREAM[arm] or r['trained_tasks'] != EXPECTED_STREAM[arm]:
            problems.append(f'{key}: stream {r["stream_tasks"]} / trained {r["trained_tasks"]}')
        if r['depth_histogram'] != EXPECTED_DEPTHS[arm]:
            problems.append(f'{key}: depth histogram {r["depth_histogram"]}')
        if not r['route_lengths_match_plan'] or r['routes_checked'] != EXPECTED_STREAM[arm]:
            problems.append(f'{key}: route check failed or vacuous')
        if not r['anchor_abs_error'] <= ANCHOR_TOLERANCE:
            problems.append(f'{key}: last-task anchor error {r["anchor_abs_error"]}')
        if arm in ('SHUFFLED', 'MIXED_L1') and len(set(r['first_20_stream_depths'])) < 2:
            problems.append(f'{key}: E4b - first 20 stream tasks are one depth')
        canonical_ids = set(r['terminal_per_task'])
        eot = canonical_end_of_task(root, key, arm, canonical_ids)
        if not math.isclose(eot, r['end_of_task_median'], rel_tol=1e-12, abs_tol=1e-15):
            problems.append(f'{key}: end-of-task median {r["end_of_task_median"]} != canonical-64 {eot}')

    for arm in THREE_STREAM:
        for w in worlds:
            shas = [cells[f'{arm}_w{w}_s{s}']['library_sha256'] for s in arm_streams[arm]]
            if len(set(shas)) != len(shas):
                problems.append(f'E4: {arm} world {w} streams share a terminal library')

    def passing(key):
        m = cells[key]['terminal_median']
        return math.isfinite(m) and m < THRESHOLD

    passes = {arm: sum(passing(f'{arm}_w{w}_s{s}') for w in worlds for s in arm_streams[arm])
              for arm in arm_streams}
    floor_failed = passes['PLAIN'] > 0
    labels = {a: ('FLOOR_FAILED' if floor_failed else label(passes[a])) for a in THREE_STREAM}
    per_world = {str(w): {a: sum(passing(f'{a}_w{w}_s{s}') for s in arm_streams[a]) for a in THREE_STREAM}
                 for w in worlds}
    d_w = {w: v['SHUFFLED'] - v['STAGED'] for w, v in per_world.items()}
    primary = 'FLOOR_FAILED' if floor_failed else 'ORDER_FREE_' + labels['SHUFFLED']
    summary = report.get('summary', {})
    if summary.get('primary') != primary or summary.get('passes') != passes:
        problems.append(f"runner summary {summary.get('primary')}/{summary.get('passes')} != "
                        f"recomputed {primary}/{passes}")
    if json.loads((root / 'status.json').read_text())['state'] != 'complete':
        problems.append('status is not complete')
    if json.loads((root / 'exit.json').read_text())['exit_code'] != 0:
        problems.append('nonzero exit code')

    return {'valid': not problems, 'problems': problems, 'tier': 2, 'primary': primary,
            'labels': labels, 'passes': passes, 'denominator': {a: len(worlds) * len(arm_streams[a])
                                                                for a in arm_streams},
            'non_finite_cells': non_finite, 'floor_failed': floor_failed,
            'per_world_passing_streams': per_world, 'd_w_shuffled_minus_staged': d_w,
            'd_w_sum': sum(d_w.values()),
            'terminal_by_cell': {k: v['terminal_median'] for k, v in sorted(cells.items())},
            'end_of_task_canonical_by_cell': {k: v['end_of_task_median'] for k, v in sorted(cells.items())},
            'gates': {k: v.get('passes') for k, v in gates.items() if isinstance(v, dict)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/o2_online_reliability'))
    parser.add_argument('--report', type=Path, default=Path('reports/o2_online_reliability.json'))
    args = parser.parse_args()
    result = score(args.root, args.report)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
