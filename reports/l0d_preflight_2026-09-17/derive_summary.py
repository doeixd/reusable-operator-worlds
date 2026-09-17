"""Reproduce descriptive counts from the validated L0d preflight report."""
import hashlib
import json
from pathlib import Path
from statistics import median


def derive(report):
    if report.get('complete') is not True or len(report['cells']) != 12:
        raise ValueError('expected complete twelve-library preflight')
    groups = {'staged': [], 'controls': []}
    cells = {}
    exceptions = []
    for key, cell in report['cells'].items():
        rows = cell['rows']
        if len(rows) != 16:
            raise ValueError('expected sixteen tasks per library')
        group = 'staged' if cell['name'] in ('STAGED5000', 'STAGED3001') else 'controls'
        groups[group].extend(rows)
        cells[key] = describe(rows)
        for row in rows:
            if row['j2a_random_nmse'] <= row['supports']['128']['query_nmse']:
                exceptions.append({'cell': key, 'task': row['task'], 'random_nmse': row['j2a_random_nmse'],
                                   'enum_nmse': row['supports']['128']['query_nmse']})
    return {'status': 'descriptive preflight; PX7 untested', 'launch_commit': report['protocol']['git_commit'],
            'groups': {name: describe(rows) for name, rows in groups.items()}, 'cells': cells,
            'random_not_worse_than_enum': exceptions,
            'terminal_tensor_bytes_at_support128': next(iter(report['cells'].values()))['terminal_tensor_bytes']}


def describe(rows):
    return {'tasks': len(rows), 'supports': {
        support: {
            'below_or_equal_nmse_0_05': sum(r['supports'][support]['query_nmse'] <= .05 for r in rows),
            'route_changes_from_128': sum(r['supports'][support]['route'] != r['supports']['128']['route'] for r in rows),
            'query_nmse_changes_from_128': sum(r['supports'][support]['query_nmse'] != r['supports']['128']['query_nmse'] for r in rows),
            'median_query_nmse': median(r['supports'][support]['query_nmse'] for r in rows),
            'median_enum_seconds': median(r['supports'][support]['seconds'] for r in rows),
        } for support in ('128', '32', '8')},
        'monotone_absolute_gap': sum(r['supports']['8']['gap_mse'] <= r['supports']['32']['gap_mse']
                                    <= r['supports']['128']['gap_mse'] for r in rows),
        'mapped_fails_enum_passes': sum(r['mapped_nmse'] > .05 and r['supports']['128']['query_nmse'] <= .05 for r in rows)}


if __name__ == '__main__':
    source = Path(__file__).resolve().parents[1] / 'l0d_preflight.json'
    raw = source.read_bytes()
    result = derive(json.loads(raw))
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    target = Path(__file__).with_name('descriptive_summary.json')
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result['groups'], indent=2))
