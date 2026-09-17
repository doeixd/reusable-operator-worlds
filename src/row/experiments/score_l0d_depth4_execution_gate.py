"""Independent checker for the depth-four execution gate."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from row.experiments.so1_storage import digest, fingerprint

def score(root=Path('artifacts/l0d_depth4_execution_gate'), output=Path('reports/l0d_depth4_execution_gate.json')):
    root, output = Path(root), Path(output)
    manifest, report = json.loads((root/'manifest.json').read_text()), json.loads(output.read_text())
    protocol, sha = manifest['protocol'], manifest['protocol_sha256']
    if fingerprint(protocol) != sha or report['protocol'] != protocol or report['protocol_sha256'] != sha:
        raise ValueError('protocol mismatch')
    for path, expected in protocol['input_sha256'].items():
        if digest(Path(path)) != expected: raise ValueError(f'input changed: {path}')
    if digest(Path(__file__).with_name('l0d_depth4_execution_gate.py')) != protocol['implementation_sha256']:
        raise ValueError('runner changed')
    cell = json.loads((root/'cell.json').read_text())
    if not cell.get('complete') or cell['stamp'] != {'protocol_sha256': sha} or fingerprint(cell['record']) != cell['record_sha256']:
        raise ValueError('cell integrity failure')
    if cell['record'] != report['cell'] or len(cell['record']['rows']) != 16:
        raise ValueError('report/cell mismatch')
    for row in cell['record']['rows']:
        for field in ('enum_support_mse','enum_query_nmse','diagnostic_query_nmse','seconds'):
            if not math.isfinite(row[field]) or row[field] < 0: raise ValueError('invalid metric')
    if json.loads((root/'status.json').read_text())['state'] != 'complete' or json.loads((root/'exit.json').read_text())['exit_code'] != 0:
        raise ValueError('operational failure')
    return {'valid': True, **report['summary']}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path('artifacts/l0d_depth4_execution_gate')); parser.add_argument('--report',type=Path,default=Path('reports/l0d_depth4_execution_gate.json')); args=parser.parse_args(); print(json.dumps(score(args.root,args.report),indent=2))
if __name__ == '__main__': main()
