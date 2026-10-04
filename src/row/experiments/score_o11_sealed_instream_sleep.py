"""Independent scorer for O11 (plan `O11_SEALED_INSTREAM_SLEEP_PLAN.md`). Sealed worlds 945-959.

Re-derives the primary (RW_SLEEP passes, PLAIN floor) and the attribution (RW_SLEEP strictly below SLEEP) from the
per-cell records with its own rule code, checks constructions and E4 (distinct SHUFFLED libraries per world), and
refuses a report whose inputs moved since the run. Committed before the sealed band is opened.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from row.experiments.so1_storage import digest, fingerprint

REPORT = Path('reports/o11_sealed_instream_sleep.json')
WORLDS = list(range(945, 960))
STREAMS = [0, 1, 2]


def ok(m):
    return isinstance(m, (int, float)) and math.isfinite(m) and m < 0.05


def primary(k, plain_passes):
    return 'FLOOR_FAILED' if plain_passes else ('CONFIRMED' if k >= 41 else 'NOT_CONFIRMED')


def secondary(n_better):
    return 'ATTRIBUTED' if n_better >= 30 else 'NOT_ATTRIBUTED'


def score(report_path=REPORT):
    rep = json.loads(Path(report_path).read_text())
    problems = []
    proto = rep['protocol']
    if fingerprint(proto) != rep['protocol_sha256']:
        problems.append('protocol fingerprint mismatch')
    for p, sha in proto['input_sha256'].items():
        if not Path(p).exists() or digest(Path(p)) != sha:
            problems.append(f'input changed since the run: {p}')
    if not rep.get('complete') or proto.get('scale', 1) != 1 or not proto.get('sealed') or proto.get('worlds') != WORLDS:
        problems.append('not a complete full-scale sealed report on 945-959')
    c = rep['cells']

    def get(arm, w, s):
        r = c.get(f'{arm}_w{w}_s{s}')
        if r is None:
            problems.append(f'missing {arm}_w{w}_s{s}')
        elif r['arm'] != arm or r['world'] != w or r['stream'] != s or len(r['terminal_per_task']) != 64:
            problems.append(f'{arm}_w{w}_s{s}: identity or scoring')
        return r

    k = better = 0
    for w in WORLDS:
        libs = set()
        for s in STREAMS:
            rw, rws, sh, sl = get('REROUTE_WAKE', w, s), get('RW_SLEEP', w, s), get('SHUFFLED', w, s), get('SLEEP', w, s)
            if None in (rw, rws, sh, sl):
                continue
            if rw['reroute_passes'] != 187 or rw['extra_updates'] != 0 or rw['anchor_abs_error'] > 1e-6:
                problems.append(f'REROUTE_WAKE_w{w}_s{s}: construction')
            for name, r in (('RW_SLEEP', rws), ('SLEEP', sl)):
                if r['extra_updates'] != 8192 or r['pool_size'] != 188 * 64 or r['library_sha256'] == r['library_sha256_before']:
                    problems.append(f'{name}_w{w}_s{s}: construction')
            if rws['library_sha256_before'] != rw['library_sha256']:
                problems.append(f'RW_SLEEP_w{w}_s{s}: did not start from this run\'s REROUTE_WAKE terminal')
            if sl['library_sha256_before'] != sh['library_sha256']:
                problems.append(f'SLEEP_w{w}_s{s}: did not start from this run\'s SHUFFLED terminal')
            libs.add(sh['library_sha256'])
            a, b = rws['terminal_median'], sl['terminal_median']
            k += ok(a)
            better += isinstance(a, (int, float)) and math.isfinite(a) and a < b
        if len(libs) != 3:
            problems.append(f'w{w}: SHUFFLED libraries not distinct (E4)')
    plain = sum(ok(r['terminal_median']) for r in (get('PLAIN', w, 0) for w in WORLDS) if r is not None)
    out = {'valid': None, 'problems': problems, 'denominator': 45, 'k_RW_SLEEP': k, 'plain_passes': plain,
           'label': primary(k, plain), 'n_better': better, 'attribution': secondary(better)}
    runner = rep.get('summary', {})
    if runner and (runner.get('label') != out['label'] or runner.get('attribution') != out['attribution']):
        problems.append('runner labels disagree with the scorer')
    out['valid'] = not problems
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(REPORT))
    result = score(parser.parse_args().report)
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
