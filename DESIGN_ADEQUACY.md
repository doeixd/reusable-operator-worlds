# Design adequacy: can the data answer the question?

Standing protocol, adopted 2026-09-22 (PI directive). Not an experiment; it
constrains every experiment written from here on. Implemented in
`src/row/design_adequacy.py`, tested in `tests/test_design_adequacy.py`,
enforced by `tools/check_adequacy.py`.

# Two different gates, previously sharing one name

ROW already had an OPPORTUNITY gate (review 68). Ten rungs have failed it. SG6
then failed something else, and calling both "no opportunity" hides the fact
that they need different fixes.

| gate | asks | of what | ROW failures |
|---|---|---|---|
| **OPPORTUNITY** | could the effect EXIST? | the GENERATOR | loop census, E6.2, E7, H47, H48b, H49, E9, L0d x4, SG0 |
| **DISCRIMINATION** | could this measurement come out EITHER WAY? | the INSTRUMENT and the SAMPLE | SG6, E5.1, E6's `H*`, sealed C2, S0 |

An experiment can pass the first and fail the second: SG6's question was
sensible and the effect could exist, but the twelve held libraries had no
graded quality axis, so no statistic computed on them could have answered it.
The reverse also happens: L0d's instruments all worked; there was nothing to
measure.

**Both gates are now required before any plan is frozen**, and each plan states
the number that turned each one, not the adjective.

# The checks

Each is a function in `row.design_adequacy` returning a verdict plus the figure
it turned on, so a plan pastes evidence rather than a claim.

## `discriminating_power(rule, null_samples, effect_samples)`

The central one. Runs the plan's REGISTERED DECISION RULE, exactly as it will be
applied, against simulated worlds where the effect is absent and where it is
present at the size the plan claims to detect. Reports the false-fire rate and
the detection rate; requires `<= 5%` and `>= 80%`.

This is the check that would have caught the largest number of past defects,
because most of them reduce to a rule that cannot lose or cannot win:

- **E5.1's first crossing.** `min{D : X_D > tau}` always names a depth, because
  a wobbling series always crosses something. On pure noise the reconstruction
  fires on 42.5% of worlds against a 5% bound. The rule printed
  `SEARCH BINDS FIRST` off a single cell.
- **E6's macro threshold.** `H* = L/(L-1)` is 1.5 uses at `L = 3`, which nearly
  every pattern clears - false-fire above 50%. The alphabet-tax threshold that
  replaced it (5-14 uses) passes the same check, which is why it was adopted.

Write the null sampler from the generator's own structure, not from intuition
about what noise looks like.

## `graded_axis(values)`

Splits at the largest gap and compares it to the wider cluster's own spread.
Above `3.0` the axis is clustered, the sample is a two-point comparison, and the
effective n is the number of clusters.

- **SG6**, on the committed numbers: staged 0.00465-0.00725, control
  1.26012-1.30157, ratio **30.2**, effective n **2**, not 12.

## `survives_within_cluster(values, statistic, cluster_of, rho)`

A pooled relation must survive within a cluster with the predicted sign.

- **The route-margin candidate** passed pooled (-0.514) and died once made
  comparable. **SG6** repeated it: pooled -0.692, within-cluster +0.086 and
  +0.429 - null, and wrong-signed.

## `partitions_outcomes(intervals, universe)`

Every possible value of the count lands in exactly one outcome: no hole, no
overlap.

- **Sealed C2** read 79%/73% counting one way and 37%/63% the other, and its
  clause resolved to neither pass nor fail.

## `baseline_registered(threshold, baseline)`

Requires a MEASURED baseline to exist, and fails a threshold the control already
satisfies.

## `floor_matches_statistic(floor_of, statistic)`

A null floor must be derived for the statistic it is compared against.

- **SG0 Revision 3** borrowed the regret floor - a support-resampling bound - as
  the scale for near-tie query spread, a different quantity. Caught by the
  double-check and disclosed in the amendment rather than used silently.

# What these checks do NOT catch

Stated because a checker that claims too much is the error it exists to prevent,
and because every entry here is a place a human still has to think.

- **S0's actual defect.** S0 registered `p_reuse >= 0.5` against a control at
  0.25. `baseline_registered` passes that - the control does not already satisfy
  the bound. What was wrong is that nothing reachable was shown to satisfy it
  either, and only a pilot plus `discriminating_power` can say so. The check
  catches a MISSING baseline, not an UNREACHABLE threshold.
- **Wrong constructs.** V4.1's tolerance normalized against total output
  variance, V6's inner loop that never learned, E9's perturbation built in
  teacher coordinates, the whole-library test whose target belonged to one arm
  by construction. These are defects in what is being measured, not in whether
  the rule can fire. The opportunity gate and the "could it come out any other
  way?" question remain the tools for them.
- **Comparability.** Whether two quantities live in the same coordinate system
  (slot versus primitive indices, per-task probes, parameter mean versus
  function) is not mechanizable here.
- **Registration order.** Whether a rule was written before its data is a fact
  about the repository's history, not about the rule.

# Required section in every plan

From 2026-09-22, every plan that registers a threshold, a triage or a decision
rule carries a section headed `# Discriminating power`, containing:

1. the decision rule as it will be applied, in one line;
2. the null and effect samplers used, and where they come from;
3. the measured false-fire and detection rates;
4. the graded-axis / partition / baseline / floor verdicts that apply;
5. anything the checks do not cover for this design, named explicitly.

`tools/check_adequacy.py` fails closed if a plan in its scope lacks the section.
It checks presence and structure, not correctness: it cannot tell whether the
numbers are right, only whether the work was done and written down.

**Its scope is empty today**, and it prints `INACTIVE` rather than a reassuring
`OK`, because a guard that parses nothing must say so - the `check_invalid.py`
defect that printed a clean pass over a manifest listing six paths. Historical
plans are deliberately not retrofitted: they were written under different rules
and rewriting them would falsify the record. Each new plan is added to
`IN_SCOPE` at freeze time, and `tests/test_check_adequacy.py` exercises the
evaluator meanwhile, including a test that fails if `IN_SCOPE` grows without
the test being updated.

# Why this is worth the ceremony

Of the rungs ROW has run, the expensive failures were not wrong answers. They
were plausible numbers produced by designs that could only have given one
answer, and they were each found weeks later by a human re-reading a document.
The double-check-after-writing directive (2026-09-22) made that re-reading
mandatory; this makes the most mechanical part of it executable, and puts the
historical failures in a test suite so the checks themselves cannot quietly stop
working.
