"""Design adequacy: can this experiment's DATA answer the question it asks?

ROW already has an OPPORTUNITY gate (review 68): could the effect exist in the
generator at all? Ten rungs have failed it. SG6 (2026-09-22) failed something
different - the effect could exist, the question was sensible, and the DATA had
no power to distinguish the outcomes, because the independent variable was
bimodal and the effective n was 2. Those are different failures with different
fixes and they need different tests.

- **Opportunity:** is there anything to measure? Asked of the GENERATOR.
- **Discrimination:** could this measurement, on this data, come out either way?
  Asked of the INSTRUMENT AND SAMPLE.

This module implements the second. Every check returns a verdict plus the number
it turned on, so a plan can paste the figure rather than the adjective.

The checks are honest about their reach: each one names, in `catches`, the
historical ROW failure it would have caught, and the doc `DESIGN_ADEQUACY.md`
records the ones it would NOT have. A checker that claims too much is the error
it exists to prevent.
"""
from __future__ import annotations

from statistics import median
from typing import Callable, Sequence

#: A rule that fires this often under the null is not a test, it is a formality.
MAX_FALSE_FIRE = 0.05

#: A rule that fires this rarely under the effect it was written for cannot
#: detect it even when it is real.
MIN_DETECTION = 0.80

#: Gap-to-spread ratio above which an axis is clustered rather than graded.
MAX_GAP_OVER_SPREAD = 3.0


def _verdict(passes, name, detail, catches):
    return {'check': name, 'passes': bool(passes), 'detail': detail, 'catches': catches}


def discriminating_power(rule: Callable[[object], bool],
                         null_samples: Sequence,
                         effect_samples: Sequence,
                         max_false_fire=MAX_FALSE_FIRE,
                         min_detection=MIN_DETECTION):
    """The central check: run the REGISTERED RULE against simulated worlds.

    `rule` is the plan's decision rule, exactly as it will be applied. It is run
    over `null_samples` (worlds where the effect is absent) and `effect_samples`
    (worlds where it is present at the size the plan claims to detect).

    Two failure modes, and both are real history:
      - fires under the null (E6's `H* = L/(L-1)` = 1.5 uses, which nearly every
        pattern clears; E5.1's first-crossing statistic, which always names a
        depth because a wobbling series always crosses something);
      - never fires under the effect (a threshold registered so far from the
        baseline that nothing reachable satisfies it).

    A rule whose false-fire rate is ~1 is not a weak test. It is not a test.
    """
    if not null_samples or not effect_samples:
        raise ValueError('discriminating power needs both null and effect samples')
    false_fire = sum(1 for s in null_samples if rule(s)) / len(null_samples)
    detection = sum(1 for s in effect_samples if rule(s)) / len(effect_samples)
    passes = false_fire <= max_false_fire and detection >= min_detection
    reason = []
    if false_fire > max_false_fire:
        reason.append(f'fires on {false_fire:.0%} of null worlds (max {max_false_fire:.0%})')
    if detection < min_detection:
        reason.append(f'detects {detection:.0%} of effect worlds (min {min_detection:.0%})')
    return _verdict(passes, 'discriminating_power',
                    {'false_fire_rate': false_fire, 'detection_rate': detection,
                     'reason': '; '.join(reason) or 'the rule separates null from effect'},
                    'E6 H*=1.5 vacuous pass; E5.1 first-crossing that always fires')


def graded_axis(values: Sequence[float], labels: Sequence = None,
                max_gap_over_spread=MAX_GAP_OVER_SPREAD):
    """Is the independent variable graded, or two clusters pretending to be n?

    Splits at the largest gap and compares it to the wider side's own spread.
    A large ratio means the sample is a two-point comparison and the effective n
    is the number of clusters, not the number of rows.
    """
    if len(values) < 3:
        return _verdict(False, 'graded_axis', {'reason': 'fewer than three points'}, 'SG6')
    ordered = sorted(values)
    gaps = [(ordered[i + 1] - ordered[i], i) for i in range(len(ordered) - 1)]
    gap, cut = max(gaps)
    low, high = ordered[:cut + 1], ordered[cut + 1:]
    spread = max(low[-1] - low[0], high[-1] - high[0])
    ratio = gap / spread if spread else float('inf')
    passes = ratio <= max_gap_over_spread
    return _verdict(passes, 'graded_axis',
                    {'gap_over_spread': ratio, 'largest_gap': gap, 'wider_cluster_spread': spread,
                     'effective_n': len(values) if passes else 2,
                     'clusters': [len(low), len(high)],
                     'reason': ('graded' if passes else
                                f'bimodal: the gap is {ratio:.1f}x the wider cluster own spread, '
                                f'so the effective n is 2, not {len(values)}')},
                    'SG6 vocabulary-quality census; the route-margin pooled correlation')


def survives_within_cluster(values, statistic, cluster_of, rho, min_abs_rho=0.4, expected_sign=-1):
    """A pooled relation must survive WITHIN a cluster, with the predicted sign.

    The route-margin candidate passed pooled (-0.514) and died once made
    comparable. SG6 repeated it exactly: pooled -0.692, within-cluster +0.086
    and +0.429.
    """
    clusters = {}
    for value, stat, key in zip(values, statistic, cluster_of):
        clusters.setdefault(key, []).append((value, stat))
    withins = {k: rho([a for a, _ in v], [b for _, b in v]) for k, v in clusters.items() if len(v) >= 3}
    ok = bool(withins) and all(w == w and expected_sign * w >= min_abs_rho for w in withins.values())
    return _verdict(ok, 'survives_within_cluster',
                    {'within': withins, 'min_abs_rho': min_abs_rho, 'expected_sign': expected_sign,
                     'reason': 'survives' if ok else
                               'the pooled relation does not survive within cluster with the '
                               'predicted sign, so it re-detects cluster membership'},
                    'route-margin normalization census; SG6')


def partitions_outcomes(intervals: Sequence[tuple], universe: tuple):
    """Every possible value of the count must land in exactly one outcome.

    Sealed C2 read 79%/73% counting one way and 37%/63% the other, and the
    clause resolved to neither pass nor fail.
    """
    lo, hi = universe
    uncovered, overlapping = [], []
    for value in range(lo, hi + 1):
        hits = [name for name, (a, b) in intervals if a <= value <= b]
        if not hits:
            uncovered.append(value)
        elif len(hits) > 1:
            overlapping.append((value, hits))
    passes = not uncovered and not overlapping
    return _verdict(passes, 'partitions_outcomes',
                    {'uncovered': uncovered[:10], 'overlapping': overlapping[:10],
                     'reason': 'partitions' if passes else 'the triage can resolve to neither '
                                                           'pass nor fail, or to both'},
                    'sealed C2 denominator; SG0 triage')


def baseline_registered(threshold, baseline, direction='greater'):
    """A threshold is only meaningful against its own measured baseline.

    S0 registered `p_reuse >= 0.5` against an unmodified world that scored 0.25.
    This check requires the baseline to EXIST and reports the margin; it does
    NOT judge whether the margin is reachable, which needs a pilot and is what
    `discriminating_power` is for.
    """
    if baseline is None:
        return _verdict(False, 'baseline_registered',
                        {'reason': 'no measured baseline; compute it before registering'}, 'S0')
    already = (baseline >= threshold) if direction == 'greater' else (baseline <= threshold)
    return _verdict(not already, 'baseline_registered',
                    {'threshold': threshold, 'baseline': baseline, 'margin': threshold - baseline,
                     'reason': 'the control already satisfies the threshold, so a pass is vacuous'
                               if already else 'the control does not already satisfy it'},
                    'S0 p_reuse >= 0.5 against a 0.25 control')


def floor_matches_statistic(floor_of: str, statistic: str):
    """A null floor must be built for the statistic it is compared against.

    SG0 Revision 3 borrowed the regret floor - a support-resampling bound - as
    the scale for near-tie query spread, a different quantity. Caught by the
    double-check and recorded in the amendment rather than silently used.
    """
    passes = floor_of == statistic
    return _verdict(passes, 'floor_matches_statistic',
                    {'floor_of': floor_of, 'statistic': statistic,
                     'reason': 'matched' if passes else
                               f'the floor is derived for {floor_of!r} and compared against '
                               f'{statistic!r}; derive it for the statistic actually used'},
                    'SG0 Revision 3 borrowed floor')


def audit(checks):
    """Collect verdicts. Adequate only if every check passes."""
    failed = [c['check'] for c in checks if not c['passes']]
    return {'adequate': not failed, 'failed': failed, 'checks': list(checks)}
