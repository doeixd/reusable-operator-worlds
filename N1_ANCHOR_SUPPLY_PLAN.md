# N1: is it the ORDER, or the supply of anchors?

Status: DRAFT, 2026-09-22, **NOT FREEZE-READY** - see "Audit 1" at the end,
which supersedes the earlier freeze-ready claim. Offline, development worlds
0-2, Tier 1. The world-budget ruling is GIVEN (see "World budget"); what blocks
freezing now is a construct inconsistency in the arms, not the ruling.

First plan written under `DESIGN_ADEQUACY.md`, so it carries `# Necessity` and
`# Discriminating power` sections with measured numbers rather than assertions,
and adds itself to `IN_SCOPE` in `tools/check_adequacy.py` at freeze time.

# The question

J1c stages tasks by program length (60 length-1, 64 length-2, then the canonical
length-3 world) and takes the rotated substrate from 0.92–0.97 to 0.0047–0.0072
in 3/3 worlds at two initializations. That intervention confounds two things
that have never been separated:

- **ORDER** — the monotone progression short → long, so that every commitment is
  made on a library shaped by strictly simpler tasks;
- **ANCHOR SUPPLY** — the mere PRESENCE of length-1 tasks, on which routing is
  clustering rather than search, because tasks sharing an operation look alike.

The mechanism recorded in `AGENTS.md` names the second ("the first commitment
must be made where it is INFORMED: at length 1 tasks sharing an operation look
alike, so routing is clustering") but the experiment only ever tested the first.

**Why the answer matters beyond bookkeeping.** If anchors suffice without
ordering, the online problem changes shape. A curriculum needs program length,
which an online learner does not have; an anchor SUPPLY needs only that easy
tasks be present, which is implementable online. It is also a different fix to
the deadlock that blocks review 85's confidence-gated proposal (C1): a gate on a
random library defers everything, whereas anchors give the library something it
can commit to informedly from the first round. **Inject, do not defer.**

# Arms, described as constructions

All arms share the world, the task contents, the budget in example-gradients,
the replay policy, and the learner seed. Only the composition and presentation
of the task stream differ.

**SUPERSEDED BY AUDIT 1.** This paragraph previously said "Total task count and
total budget are matched exactly; anchors REPLACE length-3 tasks rather than
being added". That is arithmetically impossible and inconsistent with a verbatim
`STAGED` arm; see Audit 1. The matching rule must be restated before freezing,
and the arms below are written against the superseded rule.

| arm | construction |
|---|---|
| `STAGED` | the J1c curriculum verbatim: all anchors first, in length order, library carried across stages. The reference, and a bitwise-reproduction check against the committed J1c cell. |
| `INTERLEAVED` | the same anchor tasks, drawn uniformly at random positions throughout a single undifferentiated stream. No stages, no ordering, no length revealed. |
| `SHAM` | matched anchor COUNT and matched positions, but the anchor slots are filled with length-3 tasks drawn from the same pool. Same task count, same budget, same stream positions; the only difference is whether the anchor is easy. |
| `NONE` | the matched non-staged control from J1c-R: length-3 only, same total budget. |

`SHAM` is the arm that makes this a test of ANCHORS rather than of stream
statistics: it perturbs the stream identically and supplies no easy tasks.
`STAGED` is the ceiling and `NONE` the floor, both with published values.

Every arm is described with `row.arm_provenance.describe_arm` and checked with
`assert_arm`, per the E5 lesson that an arm is a construction and not a name.

# Necessity

**Target behaviour.** Informed early commitment: the learner forms a library and
its own routes together, rather than fitting whatever arbitrary assignment its
first uninformed commitment produced (J1's failure, 0% route change after round
3–5).

**The arm that refuses it.** `NONE` — length-3 only at matched budget. It is a
genuine refusal rather than a weakened version: no task in it admits clustering-
style routing, because every task is a three-step composition.

**Measured refusal cost, and its scale.** Published J1c/J1c-R values: refusing
costs the difference between 0.0047–0.0072 and 0.92–0.97 terminal median NMSE, a
factor of roughly 150–200×, against a 0.05 usability threshold that the refusing
arm misses by more than an order of magnitude. The scale is the threshold the
claim is about, not total output variance (the V4.1 error). This is among the
largest refusal costs in the project and is measured, not projected.

**Impostors scored.** The cheapest simpler construct that could produce the same
result is a stream perturbation with no easy tasks — that is `SHAM`, and it is
an arm rather than an afterthought. The second impostor is "more compute": ruled
out by matched budget, and already ruled out once by J1c's library-reset control
landing at ~1.0 on the same compute.

**Difficulty band, and its direction.** Terminal median NMSE, higher-is-harder.
Band `(0.0005, 0.50)`. Below 0.0005 the substrate would be solved without
anchors and the comparison elicits nothing; above 0.50 every arm including the
ceiling is degraded and a null is UNINTERPRETABLE rather than evidence. The
published arms sit at 0.0047–0.0072 (ceiling) and 0.92–0.97 (floor), so the
FLOOR is deliberately outside the band on the hard side — which is correct and
expected for a floor, and is why the band is registered on the ARM UNDER TEST
(`INTERLEAVED`) and on `STAGED`, not on `NONE`.

**Incumbent scaling exponent.** Not applicable: there is no search incumbent
here. The comparison is between task-stream constructions at matched budget.

# Discriminating power

**The decision rule, as it will be applied.** Let `m` be `INTERLEAVED`'s terminal
median NMSE per world across worlds 0–2, and `s` the same for `SHAM`.

- `ANCHORS_SUFFICE` — `m < 0.05` in at least **2 of 3** worlds;
- `ANCHORS_PARTIAL` — not that, but median(`s`) / median(`m`) ≥ **5.0**;
- `ANCHORS_INSUFFICIENT` — otherwise.

Denominator: 3 worlds. The three outcomes partition every case; verified with
`design_adequacy.partitions_outcomes` over `(0, 3)`.

**Null and effect samplers.** Both are built from PUBLISHED values, not from
intuition about noise. Null: anchors do nothing, so `INTERLEAVED` behaves like
the matched non-staged control, `U(0.92, 0.97)` per world (J1c-R measured
0.92–0.97). Full effect: anchors reproduce staged formation, `U(0.0047, 0.0072)`
(J1c measured 0.0047–0.0072). Half effect: anchors help but do not reach
usability, `U(0.05, 0.09)`. `SHAM` is drawn at the null in every case. 2,000
draws each, seed 11.

**Measured rates.** With the two-way rule the plan originally carried:

| rule | false-fire (null) | detection (full) | detection (half) |
|---|---|---|---|
| `m < 0.05` in ≥2/3 only | 0.0000 | 1.0000 | **0.0000** |

That is the check doing its job on this plan: a 0.05 cliff cannot see a partial
effect at all, so a real-but-insufficient anchor effect would have been reported
as a flat negative. The three-way triage was added because of it:

| rule | false-fire (null) | detection (full) | detection (half) |
|---|---|---|---|
| `SUFFICE` | 0.0000 | 1.0000 | 0.0000 |
| `PARTIAL` or better | 0.0000 | 1.0000 | **1.0000** |

Classification by regime: null → `INSUFFICIENT` 2000/2000; full → `SUFFICE`
2000/2000; half → `PARTIAL` 2000/2000. Both bounds are met with room
(`false-fire ≤ 0.05`, `detection ≥ 0.80`).

**Graded axis.** Not applicable: the independent variable is a four-level
construction, not a continuous quantity to be correlated. No pooled correlation
is computed anywhere in this plan, so the SG6 failure mode cannot arise.

**What the checks do not cover here.** Three things, named because a check that
claims too much is the error it exists to prevent. (1) Whether `SHAM`'s stream
perturbation is genuinely matched — the replay buffer's storage and sampling
share one RNG, so changing what is stored reshuffles the replay stream, and the
arms must be verified to draw identical replay sequences or the difference is a
stream confound rather than an anchor effect. This is a MANDATORY pre-launch
check, not a registered estimand. (2) Whether `STAGED` reproduces its committed
J1c value; if it does not, the harness has drifted and nothing else is readable.
(3) Whether three worlds is enough to call a 2-of-3 result anything but
development evidence — it is not, and this rung produces no verdict.

# Non-vacuity checks that can fail

- `STAGED` must reproduce the committed J1c terminal medians for worlds 0–2
  within the published range; otherwise the run is void.
- `NONE` must fail, at 0.92–0.97. A floor that suddenly passes means the
  substrate or budget changed.
- `SHAM` must differ from `INTERLEAVED` in anchor CONTENT only: identical task
  count, identical stream positions, identical replay draws, verified before
  launch.
- Anchor tasks must actually be easier: median single-step route margin on
  length-1 tasks must exceed that on length-3 tasks in every world, or the
  premise of the manipulation is false and the rung is unscoreable rather than
  negative.

# World budget, and the ruling this plan needs

Development worlds 0–9 are recorded as SPENT for the ONLINE staged protocol
(SO2/SO3/SO4). This rung is OFFLINE and asks a different question, and worlds
0–2 are the same worlds J1c itself used, so reusing them is reusing a
development band for a development question. **But "spent for protocol X" and
"spent for all purposes" are not the same thing, and getting that wrong
contaminates a band.**

**RULING (2026-09-22, Claude under delegated judgement; the PI may reverse).
Worlds 0-2 MAY be reused for N1 and N2, under three binding conditions.**

1. **DEVELOPMENT ONLY, permanently.** Nothing measured on worlds 0-2 by N1 or
   N2 may be reported as confirmatory, now or later. A confirmatory version of
   either question needs a fresh band, and the fact that these worlds carry a
   known J1c outcome is precisely why they cannot be upgraded afterwards.
2. **The reason the ruling is "yes".** `AGENTS.md` defines seeds 0-9 as the
   DEVELOPMENT partition, "used for architecture, tuning, testbed design" - that
   is repeated use by construction. "Spent" was recorded about the ONLINE staged
   protocol, where each world's pass or fail is now known, so a further online
   run there is not independent evidence. N1 and N2 are offline, ask a different
   estimand, and their new arms (`INTERLEAVED`, `SHAM`, `NONE`) have never been
   run on any world. Knowing J1c's `STAGED` outcome does not reveal theirs.
3. **`STAGED` is not blind, and is therefore an ANCHOR, not evidence.** Its
   published value is known, so it is registered as a reproduction check whose
   failure voids the run. The registered estimand is the CONTRAST between arms,
   never `STAGED`'s level.

If the PI reverses this, both rungs need a new band and are blocked on
decision 5.

# Acceptance

Frozen plan hashed in `tools/check_prereg.py`; committed independent scorer
recomputing the triage from per-cell records; protocol fingerprint over plan,
config, seeds, stream construction and launch commit; restart reuses completed
cells bitwise, tested by interrupting the dry run; `check_prereg`,
`check_invalid` and `check_adequacy` pass; operational records archived to
`reports/` and committed with the result. Structural dry run (a few updates, two
tasks per arm) proves every path executes and the `STAGED` bitwise anchor holds
before any full cell runs.

# Cost

Four arms × 3 worlds = 12 lifetimes at the J1c stage-3 budget. J1c's own cells
are the reference for sizing. Memory-bounded pool at 3–4 concurrent for
`slots=12`, longest cells first. Performance pass before launch, timing a few
real updates rather than guessing.

# Audit 1 (2026-09-22): the arms are arithmetically inconsistent

Independent audit of this plan against the committed J1c protocol, run under the
double-check-after-writing rule before approving it for freezing. It finds one
BLOCKING defect and one consequence. The plan is **not freeze-ready**; the
earlier status line claiming otherwise is corrected above.

**The committed J1c construction** (`reports/j1c_curriculum.json`, protocol
`stages`): 60 length-1 tasks at 16,384 updates, 64 length-2 at 16,384, 64
length-3 at 32,768. **124 anchor tasks, 64 final-stage tasks, 188 tasks total,
65,536 updates total.** The published floor is length-3 only at the same UPDATE
budget, i.e. budget-matched, not task-count-matched.

**Defect: task-count matching and a verbatim `STAGED` arm cannot both hold.**

- The plan says anchors "REPLACE length-3 tasks rather than being added" and
  that total task count is "matched exactly". Replacing **124 anchors inside a
  64-task stream is impossible** - there are not enough slots. The replacement
  rule is unsatisfiable as written.
- The only self-consistent alternative is to match every arm at **188 tasks**.
  That keeps `STAGED` verbatim and preserves its bitwise J1c anchor, but it
  breaks two other things the plan relies on:
  - **`NONE` at 188 length-3 tasks is a NEW construction.** Its floor value is
    NOT the published 0.92-0.97, which was measured on the 64-task length-3
    baseline at matched budget. The non-vacuity check "`NONE` must fail, at
    0.92-0.97" therefore has no published referent, and the null sampler built
    from `U(0.92, 0.97)` is borrowed from a different construction.
  - **`SHAM` collapses toward `NONE`.** If the 124 anchor slots are filled with
    length-3 tasks and the remaining 64 are also length-3, then `SHAM` is 188
    length-3 tasks and differs from `NONE` only in which length-3 tasks occupy
    which positions. The arm that the plan calls "the arm that makes this a test
    of ANCHORS rather than of stream statistics" would then be a near-duplicate
    of the floor, and the design loses its key control.

**What must be decided before freezing**, and registered explicitly:

1. the matching invariant - task count, update budget, or example-gradients -
   stated once and applied to all four arms, since they are not simultaneously
   satisfiable;
2. `NONE`'s exact construction, and whether its non-vacuity referent is the
   published 64-task floor or a newly measured one;
3. how `SHAM` is kept distinct from `NONE` under whichever matching rule is
   chosen - the pool has up to `6**3 = 216` distinct length-3 programs, so
   distinctness is achievable, but it has to be constructed rather than assumed;
4. the samplers re-derived from whatever `NONE` and `SHAM` actually become, since
   the measured false-fire and detection rates were computed against published
   values for constructions the plan may no longer use.

**Consequence for the discriminating-power section.** Its measured rates
(false-fire 0.0000, detection 1.0000 and 1.0000) are not wrong arithmetic - they
are correct for the samplers as stated - but those samplers describe arms whose
construction is now in question. The rates must be recomputed once the arms are
fixed. This does not touch the NECESSITY section, whose 150-200x refusal cost is
measured from published J1c/J1c-R values and stands.

**Not a criticism of the check that was run.** The plan's own discrimination
check caught its two-way rule missing the half-effect entirely, which is the
check working. What it could not catch is an inconsistency between the arms and
the committed protocol they are matched against, because that lives in the
construction rather than in the statistic - the same class as E5's `S` arm,
where the label hid how the baseline was BUILT.

# Audit 2 (2026-09-22): a comparability question in the anchor-difficulty check

Not blocking, but register an answer before freezing. The non-vacuity check
requires "median single-step route margin on length-1 tasks must exceed that on
length-3 tasks in every world". A length-1 task chooses among `12` routes and a
length-3 task among `12**3 = 1,728`; a margin over 12 competitors and a margin
over 1,728 are not the same quantity, and "single-step" is undefined for a
length-3 task. This is the comparability failure that killed `route_margin` as a
cross-world predictor, in a new place. Either define the statistic so the two
depths are comparable, or replace it with a direct measure of the premise - for
example that length-1 tasks reach the usability threshold in materially fewer
updates than length-3 tasks - which is what "easier" is supposed to mean here.
