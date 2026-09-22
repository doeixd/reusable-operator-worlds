"""SG6 opportunity gate: is there a graded vocabulary-quality axis to measure?

SG0 found staged libraries functionally separated (median identifiability 53.9)
and control libraries ambiguous (0.025). The obvious successor hypothesis is
that inference difficulty TRACKS vocabulary quality - that route ambiguity is a
symptom of immature abstraction rather than an independent problem.

This module is the opportunity gate for that hypothesis, applied BEFORE any plan
was frozen, per review 68 and the nine rungs that failed for absence of
opportunity. It asks the cheap question first: does the held evidence contain a
graded quality axis at all, or only two clusters?

It is read-only over the committed SG0 grid. Nothing is re-run.

Outcome on `reports/sg0_full_v2.json`: NOT MEASURABLE HERE. Quality is bimodal
(staged ~0.005, control ~1.26, nothing between), the pooled correlation is
carried entirely by that gap, and within either cluster the relation is null or
reverses sign. See `notes/learnings.txt` and RESEARCH_STATUS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from row.experiments.so1_storage import atomic_json

#: A pooled relation is admissible only if the independent variable is graded.
#: Ratio of the between-cluster gap to the wider cluster's own spread; above
#: this the axis is bimodal and a pooled correlation is a two-point comparison.
BIMODALITY_RATIO = 3.0

#: Within-cluster concordance required before a pooled relation may be read as
#: a relation rather than as cluster membership (the route-margin lesson).
MIN_WITHIN_RHO = 0.4

ANCHOR_DEPTH, ANCHOR_SUPPORT = 3, 128


def spearman(xs, ys):
    """Average-rank Spearman; nan on zero variance (ties handled, see AGENTS)."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    if len(xs) != len(ys) or len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float('nan')
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return numerator / denominator if denominator else float('nan')


def libraries(report):
    """One point per library: its usability and its route identifiability."""
    out = {}
    for cell in report['cells'].values():
        if cell['depth'] != ANCHOR_DEPTH or cell['support'] != ANCHOR_SUPPORT:
            continue
        out[f"{cell['name']}_w{cell['world']}"] = {
            'library': cell['name'], 'world': cell['world'], 'staged': cell['staged'],
            'quality_nmse': median(r['nmse_hat_qb'] for r in cell['rows']),
            'identifiability': median(r['identifiability'] for r in cell['rows']),
        }
    return out


def gate(report=Path('reports/sg0_full_v2.json')):
    report = Path(report)
    data = json.loads(report.read_text())
    if not data.get('complete') or data['protocol'].get('label') != 'full':
        raise ValueError('SG6 reads the complete full grid only')

    points = libraries(data)
    staged = [p for p in points.values() if p['staged']]
    control = [p for p in points.values() if not p['staged']]

    def spread(group):
        values = sorted(p['quality_nmse'] for p in group)
        return values[0], values[-1], values[-1] - values[0]

    s_lo, s_hi, s_span = spread(staged)
    c_lo, c_hi, c_span = spread(control)
    gap = c_lo - s_hi
    widest = max(s_span, c_span)
    ratio = gap / widest if widest else float('inf')
    bimodal = ratio > BIMODALITY_RATIO

    pooled = spearman([p['quality_nmse'] for p in points.values()],
                      [p['identifiability'] for p in points.values()])
    within = {
        'staged': spearman([p['quality_nmse'] for p in staged], [p['identifiability'] for p in staged]),
        'control': spearman([p['quality_nmse'] for p in control], [p['identifiability'] for p in control]),
    }
    # A relation must survive WITHIN a cluster, with the sign the hypothesis
    # predicts (worse quality -> lower identifiability, i.e. positive rho on
    # nmse vs identifiability would be the WRONG direction).
    survives = all(w == w and w <= -MIN_WITHIN_RHO for w in within.values())

    if bimodal and not survives:
        verdict = 'NOT-MEASURABLE-HERE'
        reason = ('quality is bimodal (gap/spread = %.1f) and the pooled relation does not '
                  'survive within either cluster, so it re-detects cluster membership, not a '
                  'relation; this is the route-margin failure prospectively avoided' % ratio)
    elif survives:
        verdict = 'AXIS-PRESENT'
        reason = 'the relation survives within clusters with the predicted sign'
    else:
        verdict = 'UNRESOLVED'
        reason = 'quality is graded but the within-cluster relation does not meet the bar'

    return {
        'source_report': report.as_posix(),
        'source_protocol_sha256': data['protocol_sha256'],
        'status': 'OPPORTUNITY GATE, run before any SG6 plan was written; not a scientific result',
        'verdict': verdict, 'reason': reason,
        'bimodal': bimodal, 'gap_over_spread': ratio,
        'staged_quality_range': [s_lo, s_hi], 'control_quality_range': [c_lo, c_hi],
        'pooled_rho': pooled, 'within_rho': within,
        'n_libraries': len(points),
        'libraries': sorted(points.values(), key=lambda p: p['quality_nmse']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('reports/sg0_full_v2.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/sg6_quality_gate.json'))
    args = parser.parse_args()
    result = gate(args.report)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
