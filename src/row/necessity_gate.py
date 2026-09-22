"""Necessity: will the task ELICIT the behaviour we want to study?

The third gate, and the one ROW has failed most expensively. The other two ask
whether an effect could exist and whether the data could distinguish outcomes.
This one asks something prior to both: **is the target behaviour the cheapest
way for the learner to succeed at this task?** If it is not, the learner will
not produce it, and the experiment measures the learner's good judgement rather
than its incapacity.

Review 68 states the principle - truth of a latent decomposition is not utility
of representing it; abstraction tracks computational necessity, not ground-truth
labels - and the repository has been calling this the "opportunity gate" while
also using that name for "is there anything to measure at all". They are
different questions:

- **NECESSITY:** does the task REQUIRE the behaviour? Asked of the TASK and the
  cheapest alternative that refuses it.
- **OPPORTUNITY:** could the effect exist in the generator?
- **DISCRIMINATION:** could this data distinguish the outcomes?

"Hard enough" is TWO-SIDED, and both ends produce no signal. A task the learner
solves without the behaviour elicits nothing; a task beyond the reachable range
returns uninterpretable, not negative (the count-horizon rule, E5's eligibility
gate). Every necessity claim therefore registers a BAND, never a floor.

Checks are in `DESIGN_ADEQUACY.md` with what they do not catch.
"""
from __future__ import annotations

from math import log
from typing import Sequence

#: A refusal that costs less than this fraction of the oracle's own advantage
#: is free, and the behaviour will not be elicited.
MIN_REFUSAL_COST = 0.10

#: The target must beat the best cheaper impostor by this margin, in the same
#: currency, or the task elicits the impostor.
MIN_IMPOSTOR_MARGIN = 0.10

#: The incumbent's SCALING EXPONENT against the axis being grown:
#: `log(cost growth) / log(axis growth)`. Linear is 1.0; E5.1's route search
#: was 0.068, i.e. logarithmic, which is why growing the space sold nothing.
MIN_SCALING_EXPONENT = 0.30


def _verdict(passes, name, detail, catches):
    return {'check': name, 'passes': bool(passes), 'detail': detail, 'catches': catches}


def refusal_cost(oracle_loss, denied_loss, scale=None, min_fraction=MIN_REFUSAL_COST):
    """What does it cost to REFUSE the target structure?

    `oracle_loss` is an arm told the structure; `denied_loss` is the matched
    arm that ignores it. `scale` is the quantity the fraction is taken against -
    state it explicitly, because a tolerance normalized against total output
    variance instead of the contribution it licenses is V4.1's error.

    A NEGATIVE cost is the loudest possible failure: the learner that ignores
    the structure is BETTER, so representing it is not merely unnecessary but
    harmful. H48b measured exactly this - the label-free learner beat the
    told-membership oracle by ~500 nats on present cost.
    """
    cost = denied_loss - oracle_loss
    reference = abs(scale) if scale else abs(oracle_loss)
    fraction = cost / reference if reference else float('inf')
    if cost < 0:
        reason = ('refusing the structure is BETTER by %.4g: representing it is harmful, '
                  'not merely unnecessary' % (-cost))
    elif fraction < min_fraction:
        reason = ('refusing costs %.4g (%.1f%% of the reference), which is free at this '
                  'scale; the behaviour will not be elicited' % (cost, 100 * fraction))
    else:
        reason = 'refusing costs %.4g (%.1f%% of the reference)' % (cost, 100 * fraction)
    return _verdict(cost >= 0 and fraction >= min_fraction, 'refusal_cost',
                    {'cost': cost, 'fraction_of_reference': fraction, 'reference': reference,
                     'reason': reason},
                    'review 68 / H48b: the learner that ignored true groups beat the oracle')


def no_cheaper_impostor(target_score, impostors: dict, lower_is_better=True,
                        min_margin=MIN_IMPOSTOR_MARGIN):
    """Does a SIMPLER construct reach the same score?

    If it does, the task elicits the impostor, not the target. ROW's ordered
    edit vocabulary - KEEP < COMPRESS < SHARE/FACTORIZE < CREATE/FORK - exists
    because every structural edit lost to local compression, and a
    trace-compressing MACRO is the expected impostor for a LOOP.

    Scores must be in one currency and at matched budget; "shared beats
    unshared at full precision" and "at equal bits" are different claims.
    """
    if not impostors:
        return _verdict(False, 'no_cheaper_impostor',
                        {'reason': 'no impostor was evaluated; name the cheapest simpler '
                                   'construct and score it'}, 'E6 macro-as-loop; V4.2 COMPRESS')
    name, best = min(impostors.items(), key=lambda kv: kv[1]) if lower_is_better \
        else max(impostors.items(), key=lambda kv: kv[1])
    margin = (best - target_score) if lower_is_better else (target_score - best)
    reference = abs(best) or 1.0
    fraction = margin / reference
    passes = fraction >= min_margin
    return _verdict(passes, 'no_cheaper_impostor',
                    {'best_impostor': name, 'impostor_score': best, 'target_score': target_score,
                     'margin': margin, 'fraction': fraction,
                     'reason': 'target beats %s by %.1f%%' % (name, 100 * fraction) if passes else
                               'the cheaper construct %r matches the target (%.1f%% margin); '
                               'the task elicits it instead' % (name, 100 * fraction)},
                    'E6 macro-as-loop; V4.2 factorization losing to COMPRESS')


def incumbent_degrades(costs: Sequence[float], axis: Sequence[float],
                       min_exponent=MIN_SCALING_EXPONENT):
    """Does the dumb baseline get worse *at the rate the axis grows*?

    The absolute growth is not the question, and reading it as such was this
    check's own first defect. E5.1: program space grew 3.58e7-fold while
    route-optimization seconds grew 3.30-fold. A 3.3x slowdown sounds like
    degradation until it is put against a 36-million-fold axis: the scaling
    exponent is `log(3.30)/log(3.58e7) = 0.068`, essentially flat. Growing the
    space bought no difficulty, so no learned proposer could sell anything.

    Measure how the incumbent scales in the dimension you plan to stress BEFORE
    building the thing meant to beat it, and report the EXPONENT, not the ratio.
    """
    if len(costs) < 2 or len(axis) < 2 or len(costs) != len(axis):
        raise ValueError('need matched cost and axis series of length >= 2')
    order = sorted(range(len(axis)), key=lambda i: axis[i])
    lo_cost, hi_cost = costs[order[0]], costs[order[-1]]
    lo_axis, hi_axis = axis[order[0]], axis[order[-1]]
    if lo_cost <= 0 or lo_axis <= 0 or hi_axis <= lo_axis:
        raise ValueError('costs must be positive and the axis must increase')
    axis_growth = hi_axis / lo_axis
    cost_growth = hi_cost / lo_cost
    exponent = log(cost_growth) / log(axis_growth) if cost_growth > 0 else float('-inf')
    passes = exponent >= min_exponent
    return _verdict(passes, 'incumbent_degrades',
                    {'axis_growth': axis_growth, 'cost_growth': cost_growth,
                     'scaling_exponent': exponent,
                     'reason': 'cost grows %.2fx against a %.3gx axis: exponent %.3f'
                               % (cost_growth, axis_growth, exponent) +
                               ('' if passes else ' - effectively flat, so growing this axis '
                                                  'buys no difficulty and nothing can be sold '
                                                  'against the incumbent')},
                    'E5/E5.1: search cost logarithmic in program-space size')


def difficulty_band(scores: Sequence[float], band, higher_is_harder=True):
    """Is the task hard enough - and not too hard to read?

    Two-sided by construction, because both ends produce no signal. Inside the
    band the task can elicit the behaviour. Past the easy end it is solved
    without it. Past the hard end the arm is degraded and its result is
    UNINTERPRETABLE, not negative - the count-horizon rule, and E5's registered
    eligibility gate, which is the only reason its depth-6 forecast was safe.

    `band` is `(lo, hi)` in the SAME UNITS as `scores`. `higher_is_harder` says
    which end is which, because half the project's difficulty statistics run the
    other way: route identifiability is higher-is-EASIER, and getting that
    backwards was this check's second defect.
    """
    lo, hi = band
    if lo >= hi:
        raise ValueError('band is empty: lo must be < hi')
    values = sorted(scores)
    middle = values[len(values) // 2]
    too_easy = middle < lo if higher_is_harder else middle > hi
    unreachable = middle > hi if higher_is_harder else middle < lo
    if too_easy:
        state, passes = 'TOO_EASY', False
        reason = ('median %.4g is past the easy end of [%.4g, %.4g]: the task is solved without '
                  'the target behaviour, so it cannot elicit it' % (middle, lo, hi))
    elif unreachable:
        state, passes = 'UNINTERPRETABLE', False
        reason = ('median %.4g is past the hard end of [%.4g, %.4g]: the arm is degraded, so a '
                  'negative result here is uninterpretable rather than evidence'
                  % (middle, lo, hi))
    else:
        state, passes = 'IN_BAND', True
        reason = 'median %.4g lies inside [%.4g, %.4g]' % (middle, lo, hi)
    return _verdict(passes, 'difficulty_band',
                    {'state': state, 'median': middle, 'band': [lo, hi],
                     'higher_is_harder': higher_is_harder, 'reason': reason},
                    'L0d/SG0 TOO_EASY; count horizons and E5 eligibility UNINTERPRETABLE')


def capacity_forces_structure(monolithic_score, structured_score, lower_is_better=True,
                              min_margin=MIN_IMPOSTOR_MARGIN):
    """Does one undifferentiated channel absorb the structure you want discovered?

    Review 68: a single 64-direction channel absorbed the union of two rank-2
    subspaces, and the label-free learner was ~500 nats better than the oracle
    told the true grouping. Schema count and within-schema width trade off
    under one `D*`, so "how many abstractions" is a WIDEN-versus-SPLIT decision
    and capacity per slot is the knob that can make discrete identity necessary.
    Run the monolithic arm at matched total capacity before predicting that a
    learner will split.
    """
    margin = (monolithic_score - structured_score) if lower_is_better \
        else (structured_score - monolithic_score)
    reference = abs(monolithic_score) or 1.0
    fraction = margin / reference
    passes = fraction >= min_margin
    return _verdict(passes, 'capacity_forces_structure',
                    {'monolithic': monolithic_score, 'structured': structured_score,
                     'margin': margin, 'fraction': fraction,
                     'reason': 'structure wins by %.1f%%' % (100 * fraction) if passes else
                               'one undifferentiated channel matches or beats the structured arm '
                               '(%.1f%% margin); splitting is not necessary at this capacity'
                               % (100 * fraction)},
                    'review 68: 1x64 absorbs 2x32')


def audit(checks):
    failed = [c['check'] for c in checks if not c['passes']]
    return {'elicits': not failed, 'failed': failed, 'checks': list(checks)}
