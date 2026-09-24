# Queued follow-up work, for another agent to execute

**Status 2026-09-24:** A1, A2 and P1 are DONE.
- A1: `reports/arm_provenance_audit.json`. Only the known E5 `S` arm is (c). No
  retrofit, because modules digest-checked by committed scorers must stay as
  run.
- A2: `reports/claims_audit_a2_2026-09-24.md`. 13 wordings scoped;
  `PREDICTIONS.md` correction appended.
- P1: `notes/identifiability-sketch.txt`.

A3 is optional and not done. P2 is still blocked on PI decision 10.

Written 2026-09-22 after the three gates were separated
(`DESIGN_ADEQUACY.md`: NECESSITY, OPPORTUNITY, DISCRIMINATION) and applied
retrospectively to what the repository already holds. Nothing here is an
experiment: no world, no learner, no lifetime, no new artifact. All five items
are Tier 0 over committed code and reports.

**These are IMPLEMENTATION AUDITS, not experiments.** They register no estimand
about the world, so `DESIGN_ADEQUACY.md`'s `# Necessity` and
`# Discriminating power` sections do not apply and are deliberately absent. That
is stated rather than skipped, per the rule that an implementation check should
be labelled as one instead of dressed as a result.

**Standing requirement for all five.** An audit that finds nothing must report
WHAT IT CHECKED, item by item, not merely that it passed. A check that never
finds anything is not being run.

**Do not re-run any experiment.** The retrospective pass found no wrong numbers.
Every item below is about labels, unverified constructions, and plans not yet
written - which is why none of them needs compute.

---

# A1. Arm-provenance retrofit (HIGHEST PRIORITY - the only item that could surface a real defect)

## Why

`src/row/arm_provenance.py` was built after E5's defect, where three modules in
one branch had an arm called `S` and the label hid the fact that E5 built it
with `copy.deepcopy(model)` - the TRAINED library - making a fine-tuning arm
mislabelled as scratch, while E1 and E8 built the same-named arm with
`scratch_model(config, "discrete", 7717)`. Measured impact 0.23-0.44 log units;
no E5 verdict flipped.

Every such arm is a REFUSAL ARM in the new vocabulary, and `refusal_cost` is
only as good as the construction of the arm that refuses. The harness exists,
its regression test encodes the E5 bug itself, and it was **never retrofitted**:

- **26** modules under `src/row/experiments/` build a scratch, fresh, or
  deep-copied arm.
- **2** of them use `arm_provenance` (`audit_e6_corpus.py`,
  `audit_e6b_search_savings.py`; `audit_e9_partial_coverage.py` also imports it).
- **24** do not.

## What to do

For each of the 24 modules listed below, classify every site into exactly one of:

- **(a) LEGITIMATE COPY.** A deep copy made so a probe cannot mutate lifetime
  training. This is REQUIRED practice ("checkpoint probes must operate on
  deep-copied models") and is not a defect. Expect most `deepcopy` hits here.
- **(b) CORRECTLY BUILT REFUSAL ARM.** A genuine from-scratch or
  structure-denied arm. Add `describe_arm` / `assert_arm` so the construction is
  checked rather than trusted.
- **(c) MISLABELLED ARM** - an arm named scratch/fresh/baseline that is in fact
  initialized from trained state, or two same-named arms built differently
  across modules. This is the E5 defect.

**For any (c) found, do NOT silently fix the code.** Decide by whether a
committed report depends on it:

- A committed result depends on it -> the code stays as it was RUN. Correct the
  LABEL: rename the arm in reporting, append a correction to `PREDICTIONS.md`,
  and state the measured impact if it is recoverable. Rewriting code that
  produced a committed number destroys that number's provenance.
- No committed result depends on it -> fix the construction, and add the
  `assert_arm` that would have caught it.

`audit_e5_synthesizer.py:254` still contains `scratch_lib = copy.deepcopy(model)`
and this is CORRECT to leave in place - the E5 report was produced by that code,
and the correction already lives in `AGENTS.md`. Verify the module's reporting
does not still call it scratch; that is the part worth fixing.

## The 24 modules

```
audit_e0_residual.py                  audit_implicit_promotion.py
audit_e1_export.py                    audit_promotion_oracle.py
audit_e1r_recurrence.py               audit_rank2_oracle.py
audit_e2_composition.py               audit_rotated_g5.py
audit_e3_program_economy.py           audit_rotated_g5r.py
audit_e5_1_search_scaling.py          audit_so2_online_gate.py
audit_e5_synthesizer.py               audit_so4_b2_retest.py
audit_e8_length.py                    audit_substitutability.py
audit_future_block.py                 learned_lifetime.py
audit_h39_confirmation_followups.py   score_export_confirmation.py
audit_h50_reorganization.py           score_variational.py
audit_h51_reorganizability.py         scratch_difficulty.py
```

Reproduce the list with a grep for `scratch_model|deepcopy\\(model|deepcopy\\(learner|from_scratch|fresh_model` against `arm_provenance|assert_arm|describe_arm|arms_differ`.

## Acceptance

A committed report at `reports/arm_provenance_audit.json` giving, per module and
per site: the classification (a/b/c), the arm name as reported, the actual init
source, and the disposition. A table of counts by class. Every (c) carries its
disposition and, where applicable, its `PREDICTIONS.md` correction. Tests pass.
If zero (c) are found, the report still lists all sites and their evidence.

## Expected outcome

Most sites are (a). The value is in the small number that are not, and in
leaving the 24 modules checked rather than assumed.

---

# A2. Claims audit: necessity failure versus genuine refutation

## Why

A necessity failure means "this TASK did not require the behaviour". It does not
mean "the behaviour does not help". Ten-plus rungs failed necessity, and the
negative-results paper is becoming a primary output, so the distinction is now
load-bearing rather than pedantic. Several documents already word this correctly
(V4's falsified premise, the 2026-08-31 loop-census correction, SG0's scoped
reading); the point is that it has never been checked end to end.

## What to do

Read `PREDICTIONS.md`, `paper/draft.md`, `README.md` and `RESEARCH_STATUS.md`
and classify every negative claim as:

- **NECESSITY FAILURE** - the task did not require the behaviour, so the result
  is about the construction. Must not be worded as a claim about the behaviour.
- **DISCRIMINATION FAILURE** - the data could not distinguish, so the result is
  UNMEASURABLE HERE, not refuted (SG6 is the model wording).
- **OPPORTUNITY FAILURE** - the generator could not produce the effect.
- **GENUINE REFUTATION** - the behaviour was available, required, measurable,
  and did not pay. These are the real negatives and should be stated strongly.

Reword only what is misclassified. `PREDICTIONS.md` is APPEND-ONLY: corrections
are appended, never edited in place. `paper/draft.md` and `README.md` may be
edited directly.

## Acceptance

A committed table (in the audit's own file, not in `PREDICTIONS.md`) of every
negative claim, its class, and whether its wording was changed. Any reclassified
claim gets an appended `PREDICTIONS.md` correction naming the earlier wording.

---

# A3. H39's two-slot claim against `capacity_forces_structure` (OPTIONAL)

## Why

`AGENTS.md` already warns that the confirmed H39 result must not be described in
cluster language - its two slots form ONE distributed argument channel and the
confirmed world has a single family subspace. Review 68 later found that a
single 64-direction channel absorbs the union of two rank-2 subspaces. The
interpretation on record looks right; what was never scored is the matched-
capacity MONOLITHIC arm as such.

## What to do

On held H39 artifacts only, compare the two-slot `K = 32` arm against a
single-slot arm at matched total capacity (`K = 64`), using
`row.necessity_gate.capacity_forces_structure`. Report the margin.

## Acceptance

Either the structured arm wins by a margin worth stating - in which case say so
and cite it - or it does not, in which case the existing "one distributed
channel" wording is confirmed by measurement rather than by argument. Both
outcomes are useful; neither changes a sealed verdict.

## Expected outcome

Least likely of the three to change anything. Run it last, or not at all.

---

# P1. Record SG3's necessity problem in `notes/identifiability-sketch.txt`

The identifiability generator would restrict the SUPPORT input distribution
specifically so that evidence-equivalent routes diverge on query. That
MANUFACTURES the difficulty it then measures, which is close to the
defining-invariant failure: could the comparison come out any other way, given
how the world was constructed?

Before SG3 could be registered it would have to pass its own NECESSITY gate:
does any TASK in that world REQUIRE program inference, or has the world merely
stipulated that inference is hard? A world engineered to be ambiguous can
demonstrate that a proposer helps under ambiguity while saying nothing about
whether ambiguity arises.

Record this as a risk in the sketch **regardless of how PI decision 6 goes**. It
is an argument for banking, but it belongs on the record either way, and SG0
returning SATURATED already means the prerequisite fails on held artifacts.

---

# P2. Re-specify the ladder's Tier 0 gates L0a-L0c before anyone runs them

They were written when "opportunity gate" meant all three things, and the
`DESIGN_ADEQUACY.md` sections did not exist.

- **L0a** already has a genuine necessity clause - "no fixed straight-line
  program of matched length reaches the task loss" - and an achievability
  clause. It needs the section written around them, plus a DIFFICULTY BAND.
- **L0b and L0c** have no necessity clause and no band. Both need one.
- All three need `# Necessity` and, where they register a threshold,
  `# Discriminating power`, and must add themselves to `IN_SCOPE` in
  `tools/check_adequacy.py` at freeze time.

**This is MOOT if PI decision 7 closes L1-L8**, since the Tier 0 gates exist to
feed the learner rungs. Do it only after decision 7 goes the other way.

---

# Ordering and cost

| item | priority | cost | blocked on |
|---|---|---|---|
| A1 arm-provenance retrofit | highest | hours, no compute | nothing |
| A2 claims audit | high | hours, read-only | nothing |
| P1 SG3 necessity risk | high | minutes | nothing |
| A3 H39 capacity re-read | optional | minutes on held artifacts | nothing |
| P2 L0a-L0c re-spec | deferred | hours | PI decision 7 |

A1 and A2 are independent and may run in either order or in parallel by
different agents, provided only one writes to `PREDICTIONS.md` at a time. None
of them may start a run, open a world, or touch a sealed seed band.
