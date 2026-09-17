"""Independent checker for the bounded sparse-evidence gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint


def score(root=Path('artifacts/l0d_ambiguity_gate'), output=Path('reports/l0d_ambiguity_gate.json')):
    root, output = Path(root), Path(output)
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads(output.read_text())
    protocol, sha = manifest['protocol'], manifest['protocol_sha256']
    if fingerprint(protocol) != sha or report['protocol'] != protocol or report['protocol_sha256'] != sha:
        raise ValueError('manifest/report protocol mismatch')
    if report.get('complete') is not True or set(protocol['supports']) != {128, 4, 2, 1}:
        raise ValueError('incomplete or unexpected report')
    for path, expected in protocol['input_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'input changed: {path}')
    if digest(Path(__file__).with_name('l0d_ambiguity_gate.py')) != protocol['implementation_sha256']:
        raise ValueError('runner implementation changed')
    cell = json.loads((root / 'cell.json').read_text())
    if cell.get('complete') is not True or cell['stamp'] != {'protocol_sha256': sha} or fingerprint(cell['record']) != cell['record_sha256']:
        raise ValueError('cell integrity failure')
    if cell['record'] != report['cell']:
        raise ValueError('report/cell mismatch')
    rows = cell['record']['rows']
    if len(rows) != 16 or [r['task'] for r in rows] != list(range(16)):
        raise ValueError('task grid incomplete')
    for row in rows:
        if set(row['supports']) != {'128', '4', '2', '1'}:
            raise ValueError('support grid incomplete')
        for support in ('128', '4', '2', '1'):
            result = row['supports'][support]
            for field in ('query_nmse', 'best_mse', 'gap_mse', 'relative_gap', 'seconds'):
                if not math.isfinite(result[field]) or result[field] < 0:
                    raise ValueError('nonfinite metric')
        for support in ('4', '2', '1'):
            result = row['supports'][support]
            if result['route_changed_from_128'] != (result['route'] != row['supports']['128']['route']):
                raise ValueError('route-change mismatch')
    status = json.loads((root / 'status.json').read_text())
    exit_record = json.loads((root / 'exit.json').read_text())
    if status['state'] != 'complete' or exit_record['exit_code'] != 0 or not exit_record['complete']:
        raise ValueError('operational completion failure')
    if report['summary']['interpretation'] != 'opportunity gate only; no PX7 verdict':
        raise ValueError('unexpected interpretation')
    return {'valid': True, **report['summary']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('artifacts/l0d_ambiguity_gate'))
    parser.add_argument('--report', type=Path, default=Path('reports/l0d_ambiguity_gate.json'))
    args = parser.parse_args()
    print(json.dumps(score(args.root, args.report), indent=2))


if __name__ == '__main__':
    main()
