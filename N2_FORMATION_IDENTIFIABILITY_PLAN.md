# N2: is route identifiability graded ALONG FORMATION?

Status: DRAFT, 2026-09-22. Not frozen. Tier 0 preflight, then Tier 0 census on
held artifacts. Requires PI approval, and the same world-budget ruling as
`N1_ANCHOR_SUPPLY_PLAN.md`.

Written under `DESIGN_ADEQUACY.md`; carries both required sections.

# The question, and why it is not SG6

SG6 asked whether inference difficulty tracks vocabulary quality and was closed
before opening because its axis was BIMODAL: the twelve held libraries are
either good (0.00465–0.00725) or broken (1.26012–1.30157), with nothing between
and a gap 30.2× the wider cluster's own spread. Effective n was 2, not 12.

That is a property of holding only ENDPOINTS. A formation run passes through the
whole range **by construction** — it starts at a random library and ends at a
usable one — so the TRAJECTORY is the graded axis SG6 lacked. This plan asks the
same question on the axis that can answer it.

The stake is SG0's own reading. SG0 found the staged vocabulary functionally
separated (median identifiability 53.9; 0 of 576 held-out programs with any
route within 1% of the winner) and returned `NO-HEADROOM-SATURATED`, closing the
amortized-proposer branch. The working hypothesis on record is that inference is
hard only on IMMATURE libraries. If identifiability is graded along formation,
that hypothesis becomes testable with the instrument SG0 already validated, on
artifacts already held. If it is still bimodal — if libraries jump from useless
to separated with nothing in between — the hypothesis is unmeasurable here too,
and it closes for the same reason SG6 did, at the same cost.

# Preflight (step 0, and it has already half-run)

**Verified: the checkpoints exist.** `artifacts/j1c_curriculum/cells/STAGED_w{0,1,2}/`
each contain `stage1/`, `stage2/` and `stage3/` with `model.pt`, `config.yaml`
and `model_state.json`, and the same holds for `j1cr_replication` at seed 3001.
That is up to 18 trajectory points across two initializations.

**Verified: the naive load FAILS, and why.** `restore_model(dir/"stageN", cfg,
world, build_fast)` succeeds for `stage3` and raises for `stage1` and `stage2`:

    RuntimeError: Error(s) in loading state_dict for
    FastRotatedDiscreteLibraryLearner: Missing key(s) in state_dict:
    "task_codes.task_e2..."

The stage-1 and stage-2 models carry the TASK CODES of their own stages, and the
stage-3 config expects stage-3's. The strict loader is right to refuse.

**Therefore step 0 is: build a LIBRARY-ONLY loader, and justify the partial
reconstruction.** The project's standing rule is that reconstructing a learner
means reconstructing ALL of its state (`load_learner` once restored promoted
references but not retirement, rebuilding a model that never existed). A
partial reconstruction is admissible here only because of what is measured:
identifiability and support-selected route quality on HELD-OUT programs are
computed by `FrozenLibrary` + exhaustive route search over support data alone,
and never read a task code. The loader must therefore load the library tensors
and REFUSE to expose anything else, so the restriction is enforced rather than
observed. Any quantity that touches task codes is outside this plan.

**Kill condition for step 0.** If the library-only load cannot be shown to
reproduce `stage3`'s identifiability bitwise against the full loader, the
preflight has failed and the census does not run.

# Estimands

Per (initialization, world, stage), on the canonical J2A held-out length-3
programs, with the library frozen:

- **`quality`** — median query NMSE of the support-selected route. The
  library's usability.
- **`identifiability`** — SG0's own definition, unchanged: the normalized
  support-loss gap between `r_hat(S)` and the best route functionally distinct
  from it. Reusing the definition is deliberate, so the trajectory points are
  comparable to SG0's endpoint numbers.

**Comparability warning, registered because it has already bitten.** A quick
exploratory calculation during the preflight produced "identifiability 2.163" on
stage 3 of world 0 using a DIFFERENT normalization (second-best over best,
support MSE, four tasks). That number is **not** comparable to SG0's 53.9 and
must not be reported beside it. Whichever definition is used, it is used
everywhere, and the instrument is anchored against SG0's committed stage-3
values before any trajectory point is read. This is the same coordinate-system
failure as slot-versus-primitive indices and per-task probes.

# Necessity

**Target behaviour.** None — this is an OBSERVATIONAL CENSUS over frozen
artifacts. No learner is trained, no task is solved, and there is no behaviour to
elicit. The necessity gate does not apply, and that is stated rather than
skipped, per the rule that an implementation or observational check is labelled
as one instead of dressed as a result.

What replaces it is the DISCRIMINATION gate below, which is the gate SG6 failed
and the only one that can close this plan.

# Discriminating power

**The decision rule, as it will be applied.** Pool the trajectory points (up to
18) and run `design_adequacy.graded_axis` on `quality`:

- `GRADED` — gap-over-spread ≤ 3.0. The axis supports a relation, and the census
  proceeds to `survives_within_cluster` on (quality, identifiability) with the
  world as the cluster key and a registered sign.
- `BIMODAL` — gap-over-spread > 3.0. The trajectory jumps, the axis is no better
  than SG6's, and the plan CLOSES with the hypothesis recorded as unmeasurable
  here.

The two outcomes partition every value. Denominators: up to 18 trajectory points
over 2 initializations × 3 worlds × 3 stages, stated as both counts.

**Null and effect samplers.** The null is SG6's own committed numbers: two
clusters at 0.00465–0.00725 and 1.26012–1.30157, which score 30.2 and must read
`BIMODAL`. The effect is a geometric progression across three stages spanning
the same endpoints, which must read `GRADED`. Both are run against the real
`graded_axis` implementation before freezing, and the measured rates are written
into this section — **they are not yet measured, and this plan is not frozen
until they are.**

**The honest limitation, stated before the result.** Three stages per cell is
COARSE. Three points spanning two orders of magnitude may read as bimodal purely
because the sampling is sparse, which would be a fact about the checkpointing
and not about formation. If step 0's census returns `BIMODAL` at 3 points, the
plan does NOT close: it escalates to a Tier 1 re-run of the J1c protocol with
finer checkpointing (every N task-blocks), which is an already-validated
protocol on already-used worlds and is cheap. The plan closes only if a
finely-sampled trajectory is still bimodal. **This distinction is registered in
advance so that a coarse negative cannot be read as a substantive one.**

**What the checks do not cover here.** Whether a trajectory point is a fair
sample of "a library of that quality" — a mid-formation library is not the same
object as a finished library of equal quality, and a relation found along
trajectories does not automatically transfer to endpoints. That is a
construct-validity question no mechanical check addresses, and any claim from
this census is scoped to formation trajectories.

# Non-vacuity checks that can fail

- The library-only loader must reproduce `stage3` identifiability bitwise
  against the full loader in all six staged cells (step 0's kill condition).
- Stage 1 must be materially worse than stage 3 in `quality` in every cell. If
  it is not, the checkpoints are not what they are believed to be.
- SG0's committed stage-3 endpoint values must be reproduced by this plan's
  instrument before any trajectory point is read.

# Acceptance

Frozen plan hashed in `check_prereg.py`; independent scorer; protocol
fingerprint; restartable with per-cell durable records; `check_prereg`,
`check_invalid` and `check_adequacy` pass; operational records archived. Report
at a stamped path with a stale-report guard, per the H29 lesson.

# Cost

Step 0 is minutes. The census is minutes to an hour on held artifacts — depth-3
enumeration with query losses is milliseconds per task. The Tier 1 escalation,
if triggered, is six J1c-protocol runs with added checkpointing.
