"""Validate the L0d preflight's saved observations; never emit a PX7 verdict."""
import argparse
import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint


def score(root, output):
    root, output = Path(root), Path(output)
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads(output.read_text())
    protocol = manifest['protocol']
    sha = fingerprint(protocol)
    if manifest['protocol_sha256'] != sha or report['protocol_sha256'] != sha or report['protocol'] != protocol:
        raise ValueError('protocol/report/manifest disagreement')
    if protocol['id'] != 'l0d-preflight-v1':
        raise ValueError('unknown protocol')
    families = ('STAGED5000', 'STAGED3001', 'NONSTAGED3001', 'RESET5000')
    jobs = [('STAGED5000', 0), ('STAGED5000', 1)] if protocol['dry_run'] else [(n, w) for n in families for w in (0, 1, 2)]
    count = 2 if protocol['dry_run'] else 16
    if protocol['jobs'] != [list(j) for j in jobs] or protocol['tasks'] != count or protocol['supports'] != [128, 32, 8]:
        raise ValueError('unexpected grid')
    expected = {f'{n}_w{w}' for n, w in jobs}
    if report.get('complete') is not True or set(report['cells']) != expected:
        raise ValueError('incomplete report')
    for path, sha256 in protocol['input_sha256'].items():
        if digest(Path(path)) != sha256:
            raise ValueError(f'input changed: {path}')
    if digest(Path(__file__).with_name('preflight_l0d.py')) != protocol['implementation_sha256']:
        raise ValueError('runner implementation changed')
    counts = {'libraries': len(jobs), 'tasks': len(jobs) * count, 'support_cells': len(jobs) * count * 3,
              'gap_monotone_tasks': 0, 'mapped_fails_enum_passes': 0, 'random_worse_than_enum_128': 0}
    for name, world in jobs:
        key = f'{name}_w{world}'
        stored = json.loads((root / 'cells' / key / 'result.json').read_text())
        record = stored['record']
        if (stored.get('complete') is not True or stored['stamp'] != {'protocol_sha256': sha, 'name': name, 'world': world}
                or fingerprint(record) != stored['record_sha256'] or record != report['cells'][key]):
            raise ValueError(f'cell corruption: {key}')
        if record['name'] != name or record['world'] != world or record['anchor_bitwise'] is not True or record['library_unchanged'] is not True:
            raise ValueError(f'cell identity or gates failed: {key}')
        rows = record['rows']
        if [r['task'] for r in rows] != list(range(count)):
            raise ValueError('missing/reordered tasks')
        for row in rows:
            supports = row['supports']
            if set(supports) != {'128', '32', '8'} or row['executor_bitwise'] is not True:
                raise ValueError('missing support/equivalence observations')
            for result in supports.values():
                for field in ('query_nmse', 'best_mse', 'gap_mse', 'relative_gap', 'seconds'):
                    if not math.isfinite(result[field]) or result[field] < 0:
                        raise ValueError(f'invalid {field}')
                if len(result['route']) != 3 or any(type(s) is not int or not 0 <= s < 12 for s in result['route']):
                    raise ValueError('invalid route')
            for field in ('mapped_nmse', 'j2a_random_nmse'):
                if not math.isfinite(row[field]) or row[field] < 0:
                    raise ValueError(f'invalid {field}')
            counts['gap_monotone_tasks'] += supports['8']['gap_mse'] <= supports['32']['gap_mse'] <= supports['128']['gap_mse']
            counts['mapped_fails_enum_passes'] += row['mapped_nmse'] > .05 and supports['128']['query_nmse'] <= .05
            counts['random_worse_than_enum_128'] += row['j2a_random_nmse'] > supports['128']['query_nmse']
    if any(report['summary'][k] != v for k, v in counts.items()):
        raise ValueError('summary counts disagree with raw observations')
    status = json.loads((root / 'status.json').read_text())
    exit_record = json.loads((root / 'exit.json').read_text())
    if status['state'] != 'complete' or status['cells_done'] != len(jobs) or exit_record['exit_code'] != 0 or exit_record['complete'] is not True:
        raise ValueError('operational completion failed')
    if 'EXIT 0 complete' not in (root / 'run.log').read_text():
        raise ValueError('missing completion log')
    return {'valid': True, **counts, 'interpretation': 'preflight only; no PX7 verdict'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/l0d_preflight'))
    parser.add_argument('--report', type=Path, default=Path('reports/l0d_preflight.json'))
    args = parser.parse_args()
    print(json.dumps(score(args.root, args.report), indent=2))


if __name__ == '__main__':
    main()
