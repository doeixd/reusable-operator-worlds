"""Independent scorer for C-seal (`B2_SEALED_CONFIRMATION_PLAN.md`). Recomputes every registered clause from the
durable cell records (per-task terminal NMSE, decisions, wake state norms, rooted operations) and the committed
B2-hard v2 report, without importing the runner, and requires agreement with the runner's summary. Refuses a report
whose cells do not match their stamp, whose populations are wrong, or whose plan or development reference changed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

WORLDS = tuple(range(1100, 1115))
ONESTEP_STREAMS, PAIRED_STREAM = (0, 1), 0
ANCHOR_KEY = 'ONESTEP_w57_s0'
REPRO_FIELDS = ('library_sha256', 'canonical_per_task', 'loop_per_task', 'anchor_per_task', 'decisions',
                'wake_rooted_ops')
N_LOOP, N_CAND = 48, 108
THRESHOLD, LOOP_BAR = 0.05, 0.38


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _auc(pos, neg):
    """P(pos > neg), ties half (as `b2_hard.auc`)."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def score(root=Path('artifacts/b2_sealed'), report=Path('reports/b2_sealed.json'),
          plan=Path('B2_SEALED_CONFIRMATION_PLAN.md'), dev_report=Path('reports/b2_hard_v2.json')):
    root, report = Path(root), Path(report)
    rep = json.loads(report.read_text())
    sha = rep['protocol_sha256']
    hashes = rep['protocol']['input_sha256']
    if _sha(plan) != hashes[Path(plan).as_posix()]:
        raise ValueError('plan changed since the run')
    if _sha(dev_report) != hashes[Path(dev_report).as_posix()]:
        raise ValueError('development report changed since the run')
    dev_anchor = json.loads(Path(dev_report).read_text())['cells'][ANCHOR_KEY]

    def load(key):
        path = root / 'cells' / f'{key}.json'
        if not path.exists():
            if key in rep['cells']:
                raise ValueError(f'{key}: in the report but no cell record')
            return None
        saved = json.loads(path.read_text())
        if saved['stamp'] != {'protocol_sha256': sha} or not saved['complete']:
            raise ValueError(f'{key}: stamp')
        if rep['cells'].get(key) != saved['record']:
            raise ValueError(f'{key}: report and cell record differ')
        r = saved['record']
        if len(r['canonical_per_task']) != 64 or len(r['loop_per_task']) != N_LOOP or r['candidates'] != N_CAND:
            raise ValueError(f'{key}: population')
        if r['arm'] != 'GIVEN':
            if len(r['decisions']) != N_CAND or len(r['wake_state_norms']) != N_CAND:
                raise ValueError(f'{key}: decisions')
        return r

    def med(per_task):
        return float(np.median(list(per_task.values())))

    def ok(m, bar=THRESHOLD):
        return math.isfinite(m) and m < bar

    anchor = load(ANCHOR_KEY)
    anchor_ok = anchor is not None and all(anchor.get(f) == dev_anchor.get(f) for f in REPRO_FIELDS)
    one = [load(f'ONESTEP_w{w}_s{s}') for s in ONESTEP_STREAMS for w in WORLDS]
    zero = [load(f'ZERO_w{w}_s{PAIRED_STREAM}') for w in WORLDS]
    given = [load(f'GIVEN_w{w}_s{PAIRED_STREAM}') for w in WORLDS]
    done = [r for r in one if r is not None]
    k_formation = sum(ok(med(r['canonical_per_task'])) for r in done)
    k_given = sum(ok(med(r['canonical_per_task'])) for r in given if r is not None)
    gate_ok = anchor_ok and all(r is not None for r in given) and k_given >= 12
    aucs = []
    kept = straight_count = 0
    for r in done:
        loops = set(r['loop_task_ids'])
        norms = r['wake_state_norms']
        aucs.append(_auc([v for t, v in norms.items() if t in loops], [v for t, v in norms.items() if t not in loops]))
        kept += sum(bool(r['decisions'][t]['loop']) for t in loops)
        straight_count += sum(not v['loop'] for t, v in r['decisions'].items() if t not in loops)
    auc_median = float(np.median(aucs)) if aucs else float('nan')
    recall = kept / (N_LOOP * len(one))
    specificity = straight_count / ((N_CAND - N_LOOP) * len(one))
    n_terminal = sum(ok(med(r['loop_per_task']), LOOP_BAR) for r in done)
    if not gate_ok:
        lab = 'UNINTERPRETABLE'
    elif k_formation < 27:
        lab = 'FORMATION_BROKEN'
    elif auc_median >= 0.9 and recall >= 0.8 and specificity >= 0.95 and n_terminal >= 27:
        lab = 'CONFIRMED'
    else:
        lab = 'NOT_DISCOVERED'
    if lab == 'CONFIRMED' and len(done) != len(one):
        raise ValueError('CONFIRMED requires every ONESTEP cell')
    r_one = sum(bool(r['wake_rooted_ops']) for r in done)
    r_zero = sum(bool(r['wake_rooted_ops']) for r in zero if r is not None)
    roots = 'NO_ROOTS_IN_CONTROL' if r_zero <= 1 else ('ROOTS_PREVENTED' if r_one <= 1 else 'ROOTS_NOT_PREVENTED')
    out = {'anchor_reproduces': anchor_ok, 'given_formation': k_given, 'gate_ok': gate_ok,
           'k_formation': k_formation, 'wake_state_auc_median': auc_median, 'loop_recall': recall,
           'straight_specificity': specificity, 'n_terminal': n_terminal, 'label': lab,
           'rooted_cells': {'ONESTEP': r_one, 'ZERO': r_zero}, 'roots_label': roots}
    s = rep['summary']
    same = all(out[k] == s[k] for k in out if k not in ('wake_state_auc_median', 'loop_recall', 'straight_specificity'))
    close = all(math.isclose(out[k], s[k], rel_tol=0, abs_tol=1e-12) or (math.isnan(out[k]) and math.isnan(s[k]))
                for k in ('wake_state_auc_median', 'loop_recall', 'straight_specificity'))
    out['agrees_with_runner'] = same and close
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='artifacts/b2_sealed')
    ap.add_argument('--report', default='reports/b2_sealed.json')
    args = ap.parse_args()
    out = score(args.root, args.report)
    print(json.dumps(out, indent=1))
    return 0 if out['agrees_with_runner'] else 1


if __name__ == '__main__':
    sys.exit(main())
