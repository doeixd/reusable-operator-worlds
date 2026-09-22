# Design adequacy: can the data answer the question?

Standing protocol, adopted 2026-09-22 (PI directive). Not an experiment; it
constrains every experiment written from here on. Implemented in
`src/row/design_adequacy.py`, tested in `tests/test_design_adequacy.py`,
enforced by `tools/check_adequacy.py`.

# Three different gates, previously sharing one name

ROW had one gate called OPPORTUNITY (review 68) doing three jobs. Separating
them matters because they have different fixes.

| gate | asks | of what | fix when it fails | ROW failures |
|---|---|---|---|---|
| **NECESSITY** | does the task REQUIRE the behaviour? is it hard enough? | the TASK, against the cheapest refusal | change the TASK | loop census, E6/E6.2, E7, V4, H47, H48b, H49, L0d x4, SG0 |
| **OPPORTUNITY** | could the effect EXIST at all? | the GENERATOR | change the GENERATOR | E9, the 2026-08-31 census premise |
| **DISCRIMINATION** | could this measurement come out EITHER WAY? | the INSTRUMENT and the SAMPLE | change the SAMPLE or the STATISTIC | SG6, E5.1, E6's `H*`, sealed C2, S0 |

An experiment can pass one and fail another. SG6's question was sensible and the
effect could exist, but the twelve held libraries had no graded quality axis, so
no statistic computed on them could have answered it - discrimination, not
necessity. L0d's instruments all worked and its sample was fine; the task simply
did not require the behaviour - necessity, not discrimination.

**NECESSITY IS THE ONE THAT HAS COST THE MOST.** Most rungs on the list above
failed it, and it is the hardest to see from inside a plan, because a task that
does not require a behaviour still looks like a task that studies it. Review 68
states the principle: truth of a latent decomposition is not utility of
representing it; abstraction tracks computational necessity, not ground-truth
labels. A learner that ignores a structure it does not need is behaving
correctly, and an experiment that reads this as a learning failure has measured
its own design.

**"Hard enough" is TWO-SIDED.** Below the band the task is solved without the
behaviour and elicits nothing; above it the arm is degraded and its result is
UNINTERPRETABLE, not negative. Register a BAND, never a floor.

**All three gates are required before any plan is frozen**, and each plan states
the number that turned each one, not the adjective.

# The necessity checks (`row.necessity_gate`)

## `refusal_cost(oracle_loss, denied_loss, scale)`

The core. What does it cost to REFUSE the structure - an arm told it against a
matched arm that ignores it? If the gap is ~0 the behaviour will not be
elicited, and a NEGATIVE gap is louder still.

- **H48b / review 68:** on a world with two orthogonal family subspaces and
  teacher-level classification of 1.000, the label-free learner ignored the
  groups and was **~500 nats BETTER** than the told-membership oracle.
- `scale` must be the contribution the tolerance licenses, not the total.
  **V4.1** divided by total output variance while the object contributed ~0.2%,
  and every abstraction then substituted for every other.

## `no_cheaper_impostor(target_score, impostors)`

Does a SIMPLER construct reach the same score? If so the task elicits the
impostor. The edit vocabulary is ordered KEEP < COMPRESS < SHARE/FACTORIZE <
CREATE/FORK precisely because structural edits kept losing to local
compression, and a trace-compressing MACRO is the expected impostor for a LOOP
(the E6 line). Scores must be one currency at matched budget.

## `incumbent_degrades(costs, axis)`

Does the dumb baseline get worse *at the rate the axis grows*? Reports the
SCALING EXPONENT, `log(cost growth)/log(axis growth)`, not the ratio.

- **E5.1:** space grew 3.58e7-fold, route-optimization seconds 3.30-fold -
  exponent **0.068**, effectively flat. Growing the space bought no difficulty,
  so no learned proposer could sell anything. Reading the 3.3x as degradation
  was this check's own first defect, caught by its test.

## `difficulty_band(scores, band, higher_is_harder)`

The two-sided "hard enough" check. `higher_is_harder` is not cosmetic: route
identifiability is higher-is-EASIER, and getting it backwards inverts the
reading - this check's second defect, also caught by its test.

- **SG0:** staged median identifiability 53.9 against a band of (0.001, 5.0)
  reads `TOO_EASY`; the control arm at 0.025 is `IN_BAND`.

## `capacity_forces_structure(monolithic_score, structured_score)`

Does one undifferentiated channel absorb the structure you want discovered?
Review 68: a single 64-direction channel absorbed the union of two rank-2
subspaces. Schema count and within-schema width trade off under one `D*`, so
"how many abstractions" is a WIDEN-versus-SPLIT decision, and capacity per slot
is the knob that can make discrete identity necessary. Run the monolithic arm at
matched total capacity before predicting that a learner will split.

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
- **Whether the oracle arm is a real ceiling.** `refusal_cost` compares two
  numbers you supply. An oracle arm bounds performance only UNDER ITS OWN
  ASSIGNMENT: J1c's world 0 failed at 0.624 with oracle-pinned routes and passed
  at 0.0047 under the curriculum, so the oracle was the handicap. Choosing a
  defensible oracle remains a human judgement.
- **Whether the impostor list is complete.** `no_cheaper_impostor` scores the
  impostors named. Nothing mechanical suggests the one nobody thought of, and
  the macro-as-loop impostor was found by argument, not by search.

# Required section in every plan

From 2026-09-22, every plan carries a section headed `# Necessity`, containing:

1. the target behaviour, in one line;
2. the cheapest arm that REFUSES it, described as a construction;
3. the measured refusal cost and the scale it is taken against;
4. the impostors scored, and by how much the target beats the best of them;
5. the difficulty band, its direction, and where the arm's median falls;
6. the incumbent's scaling exponent, where an incumbent exists.

And every plan that registers a threshold, a triage or a decision rule also
carries a section headed `# Discriminating power`, containing:

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
