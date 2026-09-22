# SG0 sub-triage, Amendment 1 (2026-09-22)

Amends the NO-HEADROOM sub-triage registered in
`SYNTHESIS_OPPORTUNITY_GATE_PLAN.md` Revision 3. It is a SEPARATE FILE, and
that is deliberate: the plan is hashed into the completed SG0 run's protocol
(`input_sha256` over `SYNTHESIS_OPPORTUNITY_GATE_PLAN.md`), and
`score_sg0_headroom_gate.score()` raises `input changed since the run` if that
digest moves. Editing the plan to fix the sub-triage would permanently destroy
the re-scorability of the SG0 verdict it is meant to interpret. Verified before
writing: all three input digests currently MATCH the `sg0_full_v2` manifest.

Scored by `row.experiments.sg0_subtriage`, read-only over the committed
`reports/sg0_full_v2.json`. Nothing is re-run and no artifact is regenerated.

# Registration order, stated first

Revision 3 was written on 2026-09-22 while the first full grid was already
running. **The sub-triage is therefore NOT preregistered for this run.** It is a
post-hoc DIAGNOSTIC whose only job is to decide a research-program question -
ladder decision 6, whether to build the identifiability generator - and it
supports no scientific claim. This amendment is later still. Both are reported
as diagnostics and neither is cited as a registered result anywhere.

That makes the admissibility argument below mandatory rather than optional: an
amendment written after seeing the data is only admissible if it cannot change
the answer, and that has to be demonstrated, not asserted.

# What Revision 3 got wrong

Three defects, all found on first contact with the scored grid.

1. **Aggregation was ambiguous and, on the median reading, self-defeating.** The
   rule counted a cell by its "16-program median" near-tie behaviour. Controls
   have 142 non-trivial near-ties spread over 576 programs, so their median
   near-tie size is 1 and the median reading gives the CONTROL arm `j = 0` -
   tripping the sub-triage's own withdrawal clause on an arm that is manifestly
   ambiguous. A median over a quantity that is 1 for most programs cannot detect
   a minority of ties. This is the wrong-denominator failure in a new costume.
2. **`eps = 0.01` was never calibrated.** Revision 3 called the near-tie
   tolerance "registered" while no value appeared in the plan; the runner
   carried `NEAR_TIE_EPS = 0.01` as a bare constant, never checked against its
   own baseline. That is the S0 `p_reuse >= 0.5` error.
3. **The floor was borrowed.** The bootstrap floor is defined for REGRET, under
   support resampling, and Revision 3 reused it as the scale for a different
   quantity, the query-NMSE spread across near-tie members.

# The amendment

**A. Aggregation is per program, with both denominators stated.** A staged cell
counts toward `j` when **at least one of its 16 programs** has a near-tie set
containing a rival (`near_tie_size >= 2`) whose query-NMSE spread
(`near_tie_disagreement`) exceeds that program's floor. Denominators: **36
staged cells** and **576 staged programs**; likewise 36 and 576 for controls.
Both are reported. The outcomes still partition every `j`:
`j <= 2 -> SATURATED`, `j >= 3 -> EVIDENTIAL`.

**B. `eps` becomes a sensitivity curve, not a constant.** A program has a rival
within tolerance `eps` exactly when its recorded `identifiability <= eps`, so
the curve is computable from the committed report with no re-execution. It is
evaluated at `eps in {0.01, 0.05, 0.10, 0.25, 0.50, 1.00}`. Because `j` requires
the tie to EXIST, the count of cells with any such program is an **upper bound
on `j(eps)`**, and an upper bound inside the `SATURATED` band certifies the
outcome without the disagreement clause - which is recorded only at the runner's
`eps = 0.01`. This is stated as a bound, not as a measurement of `j` at wider
tolerances.

**C. The spread floor is deferred, not dropped.** Where staged near-tie sets are
EMPTY the spread has nothing to measure and no floor is required: the first
conjunct fails on its own. The floor remains genuinely undefined for a future
run with non-empty staged near-ties, and the correct derivation - a bootstrap
over `Q_b` examples, matching how the regret floor is built over support
resamples - is recorded here so the gap is carried forward rather than forgotten.
**This sub-triage may not be used on a run with non-empty staged near-tie sets
until that floor exists.**

**D. Non-vacuity, unchanged in force and now evaluable.** The CONTROL arm must
read `EVIDENTIAL`. If it does not, the statistic is broken and the sub-triage is
withdrawn rather than read.

# Admissibility: the amendment cannot change the staged answer

Required because the amendment is post-hoc. Measured on
`reports/sg0_full_v2.json`:

- **All 576 of 576 staged programs have `near_tie_size == 1`.** Not one staged
  program has any route within 1% of its winner, at either depth or any support
  size. The first conjunct fails universally, so `j = 0` under EVERY aggregation
  rule - median, per-program, or any count threshold - and independently of any
  floor, since there is no spread to compare.
- **Staged median identifiability is 53.9**; the best rival is on average ~54x
  worse on support. The control median is **0.025**, a separation of roughly
  2,000x. The tightest staged case is `min identifiability = 0.0128`: the single
  closest rival anywhere in the staged arm is 1.28% worse on support, against
  the `eps = 0.01` tolerance. That margin is 0.28 percentage points, which is
  why the `eps` curve rather than the point value carries the reading - and the
  curve holds the band to `eps = 0.10`, where the count is exactly 2 of 36, at
  the edge of `j <= 2` rather than comfortably inside it.
- The `eps` bound on staged `j`: **0/36 cells at 0.01, 2/36 at 0.05, 2/36 at
  0.10**, i.e. inside the `j <= 2` SATURATED band across a tenfold widening of
  the tolerance. It reaches 4/36 at `eps = 0.25` and 15/36 at `eps = 1.00`,
  where a "rival" is allowed to be twice as bad on support and the word near-tie
  no longer means anything.

Every one of those quantities was recorded by the registered runner, and none of
them is a threshold this amendment chose. Amendment A was calibrated on the
CONTROL arm alone, which is the ordinary "check a threshold against its own
baseline" rule, and it is the only part that could move a count.

# Result

- **Staged `j = 0` of 36 cells, 0 of 576 programs -> `NO-HEADROOM-SATURATED`.**
- **Control `j = 10` of 36 cells -> `EVIDENTIAL`.** Non-vacuity passes; the
  instrument detects ambiguity where ambiguity exists. Stated precisely because
  the two counts are different quantities and only the first is `j`: 34 of 36
  control cells CARRY a non-trivial near-tie (the first conjunct alone), and 10
  of those also clear the second conjunct, the program's own floor. 142 of 576
  control programs carry ties, with disagreement to 0.389. The scorer computes
  `j = 10`; an earlier draft of this section quoted 34, which is the
  first-conjunct count, not `j`.
- **Scope of validity: `eps <= 0.10`.** Beyond that the bound alone no longer
  certifies the band, and the reading is not claimed there.

**Consequence, as registered in Revision 3 and SG5.** SG3's first prerequisite
gate - "at least some routes are evidence-equivalent and query-different" -
FAILS on these artifacts. The identifiability generator is NOT licensed by this
evidence, and the recommendation on ladder decision 6 is to bank the economics
and negative-results papers rather than build it. The PI decides.

**What this does not say.** It does not say no substrate has ambiguity; the
control libraries in this very grid have plenty. It says the usable staged
vocabulary has a functionally SEPARATED route space at depths 3-4, so
restricting the support input distribution has no evidence-equivalence to
expose. A substrate built to have one is a different construction, and SG3's
remaining prerequisite gates would still have to pass on it.

# Double-check against the rules

Run item by item against `AGENTS.md` "Implementation learnings" before this file
was committed.

- **Denominators:** both stated (36 cells, 576 programs), outcomes partition `j`.
- **Threshold against its own baseline:** `eps` is now a curve with the control
  arm as its baseline, replacing a bare constant.
- **Could it come out any other way?** Yes - the control arm, same code path,
  same tolerance, comes out `EVIDENTIAL`. This is not a defining invariant.
- **Fit and score on the same objects:** the aggregation rule is calibrated on
  controls and applied to staged cells; the staged answer is invariant to it
  regardless.
- **Non-vacuity that can fail:** the control clause, and it was close to firing -
  under Revision 3's median reading it DID fire.
- **Stale reports:** the scorer refuses a report whose protocol digest or
  completeness does not check out, and stamps its source's `protocol_sha256`
  into its own output.
- **What the check found:** that editing the plan would have destroyed the SG0
  verdict's re-scorability. The amendment moved to its own file because of it,
  and the digests were verified before anything was written.
