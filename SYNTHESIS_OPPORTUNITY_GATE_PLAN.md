# SG0: the synthesis opportunity gate

Status: DRAFT, 2026-09-22, REVISION 1 the same day after the first application
of the double-check-after-writing rule (PI directive, 2026-09-22) found five
gaps in the original draft. They are listed in "Revision 1" at the end, because
a plan that hides its own corrections is the thing this project refuses to do.
Tier 0, existing artifacts only, no new world, no learner, no lifetime.
Requires PI approval before implementation.

This plan asks ONE question, one level above any rung:

> On this substrate, is there any headroom for a learned program proposer at
> all - and if so, where?

It exists because eight rungs have now failed for absence of opportunity rather
than learner failure, and because the opportunity-gate rule (review 68) has been
applied after building each rung instead of before. SG0 applies it to the GOAL.

# Why this is the right question now

Four consecutive L0d gates found no route ambiguity on a usable frozen
vocabulary: support 128/32/8 (preflight), support 4/2/1 (sparse-evidence gate),
depth four and depth five. Independently, E5 found a learned writer that cost
more than not having one (7.6-36.7 s/task against search at 0.30 s) and failed
on quality (+0.32 / +0.31 against a registered <= 0.15), and E5.1 found search
cost logarithmic in program count - the space grew 3.58e7-fold while route
optimization's seconds grew 3.30-fold, at oracle parity.

Working reading, to be tested here and not assumed: these are the SAME
observation. Support data in this substrate overdetermines the route, so the
incumbent (enumerate, or optimize the route code) is already at the ceiling, and
an amortized proposer has nothing to sell. If that is true, every writer,
proposer, discovery and uncertainty rung on this substrate is dead on arrival,
and SG0 says so in hours instead of one rung at a time.

# What the L0d gates measured, and what they did not

All four asked whether the support-selected route CHANGES under an
intervention. None asked the question that decides headroom:

- **is the support-optimal route also the query-optimal route, and if not, what
  does choosing it cost?**

That quantity - commitment regret - is the headroom. A proposer, a posterior, a
beam or a learned prior can only ever recover some of it. If it is ~0, no
inference mechanism can pay, however clever, and PX7(a) has nothing to measure.
The four negative gates are consistent with regret 0 and with regret large;
they do not distinguish these, which is why they did not settle anything.

# Estimands

Operational definitions, tightened in Revision 2 (2026-09-22) before any code was
written, because the Revision 1 text left the null floor and the tie rule
underdetermined and an estimand that is not pinned down cannot be diffed against
its implementation.

For one task: `S` is its canonical support set (the first `m` of the 128 J2A
support examples, `m` the cell's support size); `Q` is its canonical 256
evaluation examples, split by a registered seed into disjoint halves `Q_a` and
`Q_b` of 128 each. Every route `r` in the `12**depth` enumeration is evaluated.

- `r_hat(S)` = argmin over r of support MSE on S - what any committed learner
  picks. **Ties break to the lowest flattened route index**, matching the
  existing L0d gates.
- `r_star(Q_a)` = argmin over r of MSE on `Q_a` - hindsight best, an ORACLE,
  selected on `Q_a` and never scored there.
- **`regret = NMSE_Qb(r_hat(S)) - NMSE_Qb(r_star(Q_a))`**, the PRIMARY estimand,
  scored entirely on `Q_b`.
- **`identifiability`** = `(L2 - L1) / max(L1, 1e-12)`, the relative support-loss
  gap between the best and second-best route on S. This is the preflight's
  `relative_gap`, reused deliberately so the two instruments are comparable.
- **`near_tie_set`** = routes with support MSE `<= (1 + eps) * L1`, at registered
  `eps = 0.01`; `near_tie_size` is its cardinality.
- **`near_tie_disagreement`** = `max - min` of `NMSE_Qb` over the near-tie set.
  This is the mechanism: ties that agree on query carry no headroom, ties that
  disagree do. Reported raw and normalized by the cell median `NMSE_Qb(r_hat)`.
- `C_enum` seconds, reported separately from quality, per the E5 rule.

Ranking is done on torch MSE; every REPORTED NMSE is recomputed with the
canonical `row.metrics.nmse` through `j2a.query_error`, so SG0 numbers are
comparable to every other number in the repository rather than to a private
metric.

# Selection-noise guard (mandatory)

Two distinct guards, both required.

**Hindsight guard.** `r_star` selected and scored on the same query set overfits
it: with 1,728 to 248,832 routes against 256 examples, some route wins by noise.
Never fit and score a selection on the same objects (V5 lesson). `r_star` is
selected on `Q_a` and regret is scored on `Q_b`.

**Null-calibration floor, operationalized (Revision 2).** The floor asks how far
`NMSE_Qb(r_hat)` moves under pure support-resampling noise, with no hindsight
involved:

> Draw `B = 200` bootstrap resamples `S'` of size `m` with replacement from the
> cell's support set. The floor is the **95th percentile of
> `|NMSE_Qb(r_hat(S')) - NMSE_Qb(r_hat(S))|`** over those draws.

Chosen over the Revision 1 wording ("a random half and a disjoint half") because
disjoint halves are undefined at `m = 2` and unavailable at `m = 128`, whereas a
bootstrap is defined at every support size the plan uses. If the support
overdetermines the route then `r_hat(S') == r_hat(S)` in every draw and the floor
is exactly 0, which is the expected reading and is itself informative.

A cell counts toward `k` only when its 16-program **median regret exceeds its own
floor**. A regret not clearly above that floor is ZERO for this plan's purposes
and must be reported as such.

**Expected sign, registered.** Under the null in which `r_hat` is genuinely the
best route, `r_star(Q_a)` differs from it only by `Q_a` noise and is therefore
WORSE on `Q_b` on average, so regret is `<= 0` in expectation. Positive median
regret is consequently already evidence, and the floor is a conservative extra
barrier rather than the only one.

**Implementation note that makes this affordable.** Per-route per-example squared
error is computed ONCE per task as a `(routes, examples)` matrix; every support
size, every bootstrap draw and both query halves are then means over subsets of
its columns. No route is re-executed. Depth 3 is 1,728 x 128 and depth 4 is
20,736 x 128, so the matrices are small, and the 200 bootstrap draws cost
reductions rather than forward passes.

# Cells

Existing frozen J2A staged libraries and their controls, as used by the L0d
preflight. No new world, model or training.

The twelve frozen libraries are the L0d preflight set: `STAGED5000` and
`STAGED3001` (usable/staged), `NONSTAGED3001` and `RESET5000` (controls), each
at worlds 0, 1 and 2, each with the same 16 held-out programs.

| axis | values | count |
|---|---|---|
| staged library kind | STAGED5000, STAGED3001 | 2 |
| control library kind | NONSTAGED3001, RESET5000 | 2 |
| world | 0, 1, 2 | 3 |
| depth | 3, 4 (5 only if 3-4 show nonzero regret) | 2 |
| support | 128, 8, 2 | 3 |
| programs per cell | the 16 held-out programs | 16 |

**A CELL is one (library kind, world, depth, support) combination, scored over
its 16 programs.** That gives **36 staged cells** and **36 control cells**
excluding depth 5, each summarised by a 16-program median. Every fraction below
is stated against one of those two denominators, never against a pooled 72.

Depth 3 is 1,728 routes and depth 4 is 20,736; both enumerate fully with query
losses as well as support losses. Depth 5 is 248,832 and reuses the committed
chunked evaluator (`MEMORY_SAFE_SEARCH`, c72b02b); it is included only if the
cheaper depths find headroom, per the scoring-economy rule, and would add 18
staged and 18 control cells costed separately.

Order cells cheapest- and most-decisive-first: depth 3, support 128, staged,
world 0 is the single cell most likely to make the outcome impossible, and is
run first.

# Registered triage

Computed before any interpretation, with the thresholds and denominators fixed
here. Let **k** = the number of the **36 staged cells** whose 16-program median
regret exceeds the null-calibration floor of that same cell. The three outcomes
partition every possible k, so the triage cannot resolve to neither pass nor
fail (the sealed-C2 denominator lesson).

- **`k <= 2` of 36 -> NO-HEADROOM.** Two cells are allowed so that a single
  noisy cell cannot manufacture a branch, on the same reasoning that a bare
  first crossing is not a horizon. Consequence: the amortized-proposer branch is
  CLOSED on this substrate. PX7 is retired as unmeasurable here, not as refuted;
  the full L0d census is withdrawn rather than left unfrozen; and any synthesis
  claim requires a different generator (`notes/identifiability-sketch.txt`).
- **`k >= 3` and the ordering test PASSES -> HEADROOM-LOCALIZED.** The ordering
  test is registered in advance and has two required clauses: across the 36
  staged cells, the normalized near-tie-disagreement statistic has Spearman rho
  `>= 0.4` with median regret at permutation p `<= 0.01` over 10,000
  permutations, AND the relation is concordant within world in `>= 2` of 3
  worlds so it is not a cross-world scale artifact. Both are required, because
  the route-margin candidate passed pooled and died once made comparable.
  Consequence: that statistic is the MEASURED ambiguity source the L0d census
  has been missing, and the census may then be frozen around it, with
  identifiability rather than support size as its independent variable.
- **`k >= 3` and the ordering test FAILS -> HEADROOM-UNEXPLAINED.** Consequence:
  report it, and freeze no mechanism census. This is the route-margin failure
  repeating, and that hunt produced three withdrawn candidates.

The `rho >= 0.4` and `p <= 0.01` values are PROVISIONAL and must be replaced
before this plan is frozen, by running the ordering test on the CONTROL cells
and on a shuffled-label null first and registering thresholds relative to what
those produce. A threshold not checked against its own baseline is the S0
`p_reuse >= 0.5` error, where the unmodified control already violated the bound.

# NO-HEADROOM sub-triage (Revision 3): which successor a negative licenses

Computed ONLY when the triage above returns `NO-HEADROOM`, from quantities the
runner already records, at no additional compute: the `(routes, examples)`
squared-error matrix is already built, and this is a further reduction over its
columns.

**Status: SECONDARY AND DIAGNOSTIC.** It produces no verdict, supports no
scientific claim, and is reported separately from the primary triage. Its one
job is to make ladder decision 6 a DECISION rather than a deliberation held
after the cheap answer has already arrived.

**Why it is needed.** `NO-HEADROOM` closes the amortized-proposer branch and
hands the program two successors - pursue the identifiability generator (SG3),
or bank the economics and negative-results papers - with, as written, no
criterion for choosing between them. But SG3's own first prerequisite gate is
"at least some routes are evidence-equivalent and query-different", and THIS
plan's cells measure exactly that. A `NO-HEADROOM` result that does not read its
own near-tie numbers throws away the evidence that decides its own successor.

**The discriminator.** Regret is zero either because there is nothing to be
wrong about, or because the support evidence is doing the work of being right.
Those are different substrates and they license different successors:

- **`NO-HEADROOM-SATURATED`** - the near-tie sets are trivial, or the routes in
  them agree on query. The route space is functionally separated: no restriction
  of the SUPPORT input distribution can create commitment regret, because the
  evidence is not what is preventing it. SG3's prerequisite (a) fails on this
  substrate, the identifiability generator is not licensed by these artifacts,
  and the recommended successor is to bank and stop opening rungs here.
- **`NO-HEADROOM-EVIDENTIAL`** - near-tie sets are non-trivial AND their query
  spread exceeds the cell's own floor, yet `r_hat` still selects a query-good
  route. Support evidence is carrying the selection, so restricting it is a LIVE
  intervention on a mechanism shown to be present rather than a hoped-for one.
  SG3's prerequisite (a) is satisfied here and the identifiability generator
  gate is warranted.

**Denominator and rule.** Over the same **36 staged cells** as the primary
triage, let **j** = the number whose near-tie set at the registered tolerance
contains at least one route functionally distinct from `r_hat` AND whose
16-program median within-near-tie query-NMSE spread exceeds that cell's own
bootstrap floor. The two outcomes partition every possible j, so the sub-triage
cannot resolve to neither:

- `j <= 2` of 36 -> `NO-HEADROOM-SATURATED`;
- `j >= 3` of 36 -> `NO-HEADROOM-EVIDENTIAL`.

The allowance of two matches the primary triage for the same reason: one noisy
cell must not manufacture a branch.

**Thresholds are PROVISIONAL, exactly as the ordering test's are.** The
near-tie tolerance and the `j` cut must be set before freezing by running this
same statistic on the 36 CONTROL cells and on a shuffled-label null, and
registered relative to what those produce. The control libraries change routes
constantly under support reduction, so they should read `EVIDENTIAL` strongly;
if they do not, the statistic is broken and the sub-triage is withdrawn rather
than read. A threshold not checked against its own baseline is the S0
`p_reuse >= 0.5` error.

**Two gaps this sub-triage inherits, disclosed rather than glossed.** Both must
be closed before the plan is frozen, and neither is closed by this revision.
(1) The bootstrap floor was derived for REGRET - how far `NMSE_Qb(r_hat)` moves
under support resampling - and the sub-triage reuses it as the scale for a
different quantity, the spread of query NMSE ACROSS near-tie members. That reuse
is provisional and not calibrated; the spread needs its own floor, derived the
same way but over the near-tie set, before `j` can be counted. (2) The near-tie
TOLERANCE has been referred to as "registered" since Revision 1 but no value is
written anywhere in this plan. It must be set against the control cells with the
other provisional thresholds, and the sub-triage is unreadable until it is.

**Could it come out any other way?** Yes, and that is checked rather than
asserted. Nothing in the construction of the J2A libraries fixes whether their
near-ties agree or disagree on query: the libraries were formed by a staged
curriculum with no term that separates or merges routes in query-relevant
directions, and the four L0d gates measured only whether the SELECTED route
moves, never the spread among unselected near-ties. Both outcomes are
attainable. This is not the defining-invariant error.

**What it does not do.** It does not measure regret under a restricted support
distribution - that is SG3's experiment, on a generator that does not exist yet.
It establishes only whether SG3's first prerequisite is satisfied on artifacts
already held, which is the cheapest thing that can distinguish the two
successors.

# Non-vacuity checks

- The NONSTAGED control libraries must show LARGE regret. A poor library whose
  routes change constantly under support reduction (65/96 at support 32, 93/96
  at support 8) must not read as zero headroom; if it does, the instrument is
  broken, not the substrate.
- Regret must be non-negative by construction on Q_a and may be negative on Q_b
  only within the noise floor; systematic negatives mean the split leaked.
- Reducing support must not reduce regret. If it does, the ordering is inverted
  and the metric is misread.
- The identifiability statistic must be COMPARABLE across cells before it is
  correlated with anything (the route-margin normalization lesson): register it
  as a library-intrinsic, within-cell-normalized quantity, and report the
  un-normalized version only as a diagnostic.

# Quantities the runner must record

Listed so the runner's output dict can be DIFFED against this list before
launch, per the depth-five lesson: that runner built evaluation arrays and never
used them, losing the plan's required query NMSE, and never captured peak RSS at
all. Per cell, per program:

- support loss and query NMSE for `r_hat(S)`, for `r_star(Q_a)` scored on Q_b,
  and for the registered null-calibration selection;
- `regret`, and the null-calibration floor it is compared against;
- `identifiability` and `near_tie_disagreement`, normalized and raw;
- the near-tie set size at the registered tolerance, the number of its
  members functionally distinct from `r_hat`, and the within-near-tie query
  NMSE spread, all per cell and per program - these are the sub-triage
  inputs and must be present in the report even when the primary triage is
  not `NO-HEADROOM`, so that a later reading needs no rerun;
- `C_enum` seconds and, where computed, `C_opt` seconds, reported separately
  from quality per the E5 rule;
- library hash, program, seeds, split assignment, and the support and query
  index sets actually used.

Per run: `run.log` (append-only, timestamped, one line per cell start and
finish), `status.json` (rewritten atomically after each cell), `run.pid`,
`exit.json`, and a durable error record on any exception. The
restartable-and-checkable directive has no Tier 0 exemption, and a short job is
exactly the kind that gets relaunched carelessly.

# Stale-report and performance rules

The report path is fixed, so the runner stamps every report with its protocol
fingerprint and the scorer REFUSES a report older than its inputs or computed
under a different fingerprint. A scorer that dies partway must not leave the
previous answer on disk readable as the current one (the H29 lesson, which was
read as a result twice).

A performance pass is required before launch even at Tier 0, per the
before-every-launch directive: time a few real cells rather than guessing, and
apply only wins that leave results bitwise identical, verified on a short run.
Depth-3 enumeration with query losses is milliseconds; depth 4 was 10.04 s for
16 programs on support alone, and adding query losses roughly doubles the
per-route work. Record measured before and after timings.

# Cost

Enumeration with query losses at depth 3 is milliseconds per cell; depth 4 took
10.04 s for 16 programs on support alone, so the full grid excluding depth 5 is
minutes. Depth 5 is about 4 minutes per program and is costed separately, only
if the cheaper depths find headroom. Tier 0 throughout: no lifetime, no pool,
and no host-memory precondition beyond the committed chunked evaluator.

# What this plan does NOT do

It does not test any inference MECHANISM (beam, posterior, commit-late,
amortized proposal). It measures the ceiling those mechanisms would compete for.
It is not a PX7 verdict under any outcome, and a NO-HEADROOM result is a
statement about this substrate, not about program synthesis.

# Anchors (free correctness checks, required)

At depth 3 the cells reuse the CANONICAL J2A held-out tasks, so two anchors are
available at no cost and are required to pass:

- at support 128, `r_hat(S)` must equal the J2A stored `enum_route` and its query
  NMSE must equal the stored `enum` value exactly, as the L0d preflight already
  requires;
- the `(routes, examples)` matrix reduced over all 128 support examples must
  reproduce `all_route_support_mse` bitwise, and the selected route's query NMSE
  must equal `hard()` evaluated directly.

Both are written as TESTS before launch, per the depth-five lesson. A subset-mean
implementation that changes reduction order is admissible only if these anchors
hold.

# Acceptance

Independent scorer recomputing the triage from the raw report; protocol
fingerprint over plan, config, library hashes, seeds, splits and implementation;
restart reuses completed cells bitwise; `check_prereg.py` and `check_invalid.py`
pass; operational records archived to `reports/` and committed with the result.
The equivalence check named above is written as a TEST before launch, per the
depth-five lesson.

# Revision 1 (2026-09-22): what the double-check found

First application of the double-check-after-writing rule, run against
`AGENTS.md` "Implementation learnings" item by item, on a plan written about two
hours earlier. Five gaps, all real, all fixed above:

1. **No denominators.** "Every staged cell" and "some cells" had no counts, and
   the triage could resolve to neither pass nor fail - precisely the sealed-C2
   error that read 79%/73% counting one way and 37%/63% counting the other.
   Fixed by defining a cell, counting 36 staged and 36 control, and partitioning
   every possible k.
2. **Thresholds not checked against their own baseline.** The ordering test had
   no numbers at all; the numbers now written in are explicitly provisional
   pending a control-cell and shuffled-label null, per the S0 rule.
3. **No list of quantities to record.** This breaks a lesson committed the SAME
   MORNING, from the depth-five runner that built evaluation arrays and never
   used them. Now listed explicitly so the runner output can be diffed against
   it before launch.
4. **No operational-record contract.** `run.log`, `status.json`, `run.pid`,
   `exit.json` and a durable error record were omitted because the job is short.
   The restartable-and-checkable directive has no Tier 0 exemption.
5. **No stale-report guard and no performance pass.** The report path is fixed,
   so a partial scorer run could leave the previous answer readable as current
   (H29); and the performance-pass directive says EVERY launch, not every long
   one.

Item 3 is the one worth dwelling on: the rule it breaks was written, committed
and pushed by me that morning, and I broke it in the next document I wrote. That
is the argument for the new directive in one line - the failure mode is not
ignorance of a rule, it is not re-reading against rules already known.

# Revision 2 (2026-09-22): estimands pinned before implementation

PI approved implementation (ladder decision 8). Applying the
double-check-after-writing rule to the plan a second time, this time against the
code that was about to be written, found three things underdetermined:

1. **The null floor was not implementable as written.** "A random half and a
   disjoint half" is undefined at `m = 2` and unavailable at `m = 128`. Replaced
   with a bootstrap floor defined at every support size, with the reasoning kept
   in the plan rather than silently changed in code.
2. **No tie rule.** `argmin` over 1,728 to 248,832 routes needs a stated
   tie-break or two implementations can disagree; fixed to the lowest flattened
   index, matching the existing L0d gates.
3. **No anchor.** Depth 3 reuses the canonical J2A held-out tasks, so the stored
   `enum_route` and `enum` error give a free bitwise correctness check that the
   original draft did not claim. Now required, and the reason the cheap
   subset-mean implementation is safe to use.

Also registered the expected SIGN of regret under the null, which the first two
revisions left implicit: it is `<= 0` in expectation if `r_hat` is truly best, so
a positive median is already evidence before the floor is applied.

# Revision 3 (2026-09-22): a negative must name its own successor

Third application of the double-check-after-writing rule, prompted by reading
the plan against the PROGRAM as a whole rather than against its own internals.
Revisions 1 and 2 checked whether SG0 measures its estimand correctly; this one
asks what SG0's most likely outcome leaves the program able to do.

**What it found.** On the accumulated evidence - four L0d negatives, search
logarithmic in program-space size at oracle parity, support overdetermining the
route at every size from 128 down to 2 - `NO-HEADROOM` is the most likely
outcome of this plan. Yet `NO-HEADROOM` as written discharged into ladder
decision 6, an open either/or with no criterion. The plan would therefore have
bought a cheap, decisive answer and then stalled on what to do with it, which is
the deferral pattern the plan's own consequence clause was written to prevent.

**What changed.** A registered sub-triage, above, computed only under
`NO-HEADROOM`, from quantities already recorded, partitioning that outcome into
the one that licenses the identifiability generator and the one that does not.
It is explicitly secondary and diagnostic: it decides a research-program
question, not a scientific one.

**What did not change.** No estimand, threshold, cell, denominator, anchor or
acceptance condition of the primary triage. The sub-triage cannot alter the
primary outcome, only what follows it.

**Checked against the rules, item by item.** Denominator stated (36 staged
cells) and the two outcomes partition every j; thresholds flagged PROVISIONAL
pending a control-cell and shuffled-label baseline, per the S0 rule; the
opportunity-gate question ("could it come out any other way?") answered from the
construction rather than asserted; the new quantities added to the record list
in the same pass, per the depth-five lesson; a non-vacuity condition attached
(controls must read `EVIDENTIAL`, or the statistic is withdrawn); and the whole
thing labelled diagnostic so it cannot be read as a verdict.

**Two gaps the check FOUND and did not close**, both written into the sub-triage
above rather than left implicit. The bootstrap floor is defined for regret and is
reused for a different quantity (near-tie query spread) without being
recalibrated for it; and the near-tie tolerance has been called "registered"
since Revision 1 while no value appears anywhere in this plan. Both block
freezing. Disclosing them is the point of the rule: a check that finds nothing
is not being run, and these are exactly the class - a threshold with no baseline,
and a statistic compared against a floor built for something else - that this
project has paid for repeatedly (the S0 `p_reuse` bound, the route-margin
scale-incomparability).
