"""Independent scorer for G1V (`G1V_TANH_PROBE_PLAN.md`; Tier 1, exploratory).

Recomputes the label from durable per-cell records (the run's `cells/` or the archive in `reports/`), not from the
runner's `reading`: per-world FORMS at the registered 65,536 point (terminal < 0.05 and d1_ari >= 0.8), the count over
all five worlds, FORMS >= 4 / FAILS <= 1 / else MIXED, and the secondary counts. Re-checks stamps, record hashes, the
construction of every cell, the gates, and refuses a report whose plan, config or reference reports changed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PLAN = Path('G1V_TANH_PROBE_PLAN.md')
REPORT = Path('reports/g1v_tanh_probe.json')
CELLS = Path('artifacts/g1v_tanh_probe/cells')
WORLDS = (71, 72, 73, 74, 75)
REGISTERED = '65536'
THRESHOLD, LOOSE, IDENTITY_ARI, FORMS_K, FAILS_K = 0.05, 0.1, 0.8, 4, 1


def _fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def score(report=REPORT, cells=CELLS):
    rep = json.loads(Path(report).read_text())
    proto = rep['protocol']
    for path, digest in proto['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{path} changed since the run')
    if PLAN.as_posix() not in proto['input_sha256']:
        raise ValueError('report does not stamp the plan')
    if (proto['threshold'], proto['loose'], proto['identity_ari'], proto['forms_k'], proto['fails_k'],
            str(proto['registered']), tuple(proto['worlds']), proto['wake_updates'], tuple(proto['setting'])) != (
            THRESHOLD, LOOSE, IDENTITY_ARI, FORMS_K, FAILS_K, REGISTERED, WORLDS, 2, (0.003, 1.0)):
        raise ValueError('registered rule, worlds or setting differ from the plan')
    if not rep['gates']['all_pass']:
        raise ValueError('gates did not pass')
    records = {}
    for arm in ('WAKE', 'RR_SLEEP'):
        for w in WORLDS:
            k = f'{arm}_w{w}'
            saved = json.loads((Path(cells) / f'{k}.json').read_text())
            if saved['stamp'] != {'protocol_sha256': rep['protocol_sha256']} or not saved['complete']:
                raise ValueError(f'{k}: stamp')
            if _fingerprint(saved['record']) != saved['record_sha256']:
                raise ValueError(f'{k}: record hash')
            r = saved['record']
            if r != rep['cells'][k] or r['scale'] != 1 or r['world'] != w:
                raise ValueError(f'{k}: report, scale or world')
            if arm == 'WAKE' and (r['lifetime_lr'], r['lifetime_updates_per_example']) != (0.003, 2):
                raise ValueError(f'{k}: wake construction')
            if arm == 'RR_SLEEP' and ((r['lr'], r['sleep_lr'], r['init_gain'], r['pool_size'])
                                      != (0.003, 0.003, 1.0, 188 * 64) or r['learner_class'] != 'TanhFormLearner'):
                raise ValueError(f'{k}: sleep construction')
            records[k] = r
    final = {w: records[f'RR_SLEEP_w{w}']['curve'][REGISTERED] for w in WORLDS}
    for w, c in final.items():
        if len(c['terminal_per_task']) != 64 or not all(math.isfinite(v) for v in c['terminal_per_task'].values()):
            raise ValueError(f'w{w}: terminal')
    forms = {w: c['terminal_median'] < THRESHOLD and c['d1_ari'] >= IDENTITY_ARI for w, c in final.items()}
    k = sum(forms.values())
    label = 'FORMS' if k >= FORMS_K else ('FAILS' if k <= FAILS_K else 'MIXED')
    out = {'label': label, 'forming_worlds': k, 'denominator': len(WORLDS),
           'forms': {str(w): v for w, v in forms.items()},
           'below_loose': sum(c['terminal_median'] < LOOSE for c in final.values()),
           'terminal': {str(w): c['terminal_median'] for w, c in final.items()},
           'collapsed_tasks': {str(w): sum(v > 0.5 for v in c['terminal_per_task'].values()) for w, c in final.items()},
           'd1_ari': {str(w): c['d1_ari'] for w, c in final.items()},
           'formed_s1_median': {str(w): records[f'RR_SLEEP_w{w}']['formed_s1']['median'] for w in WORLDS},
           'seconds': {kk: r['seconds'] for kk, r in records.items()}}
    mine = rep.get('reading', {})
    out['runner_agrees'] = all(mine.get(x) == out[x] for x in ('label', 'forming_worlds', 'below_loose', 'terminal'))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(REPORT))
    parser.add_argument('--cells', default=str(CELLS))
    args = parser.parse_args()
    print(json.dumps(score(args.report, args.cells), indent=1))


if __name__ == '__main__':
    main()
