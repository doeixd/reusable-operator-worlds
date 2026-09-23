# N1b: how many anchors, and of which length?

Status: DRAFT, 2026-09-22. Offline, development worlds 0-2, Tier 1. Successor to
N1 (`ANCHORS_SUFFICE`, 3/3, `63e63f6`). Development evidence only, never
upgradable to confirmatory, under the same world-budget ruling as N1.

# The question

N1 showed that the PRESENCE of 124 easy tasks - 60 length-1 and 64 length-2 -
is sufficient for formation of the rotated substrate, and that ordering is not
required. It did not say which anchors do the work or how many are needed. Both
matter for the online successor: an online stream cannot guarantee 124 anchors,
and "easy" has to be operational.

Two sub-questions, one grid:

- **LENGTH** - do length-1 anchors alone suffice? Length-2 alone?
- **DOSE** - how few anchors, drawn from the full mixed supply, suffice?

# Tier 0 prediction, measured before this plan

The N1 anchor-difficulty gate measured clustering (ARI between teacher primitive
and support-argmin slot on an UNTRAINED library) at depths 1 and 3. Extended to
depth 2 before writing this plan:

| world | depth 1 | depth 2 (mean of 2 positions) | depth 3 |
|---|---|---|---|
| 0 | +0.7677 | +0.1163 | +0.0055 |
| 1 | +0.5850 | +0.1082 | +0.0240 |
| 2 | +0.4595 | +0.0665 | +0.0216 |

Clustering degrades smoothly with depth. Registered expectation, stated as a
working hypothesis (NB1-NB3 below): length-1 anchors carry the effect; length-2
anchors are the informative cell, since their clustering is 5-10x above chance
but 5-10x below length 1.

# Matching invariant, and arms as constructions

Identical to N1: **65,536 updates**, one pooled stream, **188 task slots**, 24,064
examples, the same minibatch index stream (seed sequence `[1911, world]`), scored
on the canonical 64 length-3 tasks only. Every arm replaces some or all of the
124 non-canonical slots with anchors and fills the rest with the SAME distinct
length-3 filler programs N1's `SHAM` used.

**Pool order is registered: `anchors(k) + canonical(64) + fillers(124 - k)`.** At
`k = 124` this is exactly N1's `INTERLEAVED` pool and at `k = 0` exactly N1's
`SHAM` pool, element for element - which is what lets the two endpoints be READ
from committed N1 cells rather than re-run (see Anchors).

| arm | anchors | fillers | varies |
|---|---|---|---|
| `L1_ONLY` | all 60 length-1 | first 64 of N1's filler list | length |
| `L2_ONLY` | all 64 length-2 | first 60 of N1's filler list | length |
| `DOSE_8` | 8 of the 124 mixed anchors | first 116 fillers | dose |
| `DOSE_32` | 32 of the 124 mixed anchors | first 92 fillers | dose |
| endpoint `k=0` | none | all 124 | = N1 `SHAM`, committed |
| endpoint `k=124` | all 124 | none | = N1 `INTERLEAVED`, committed |

Dose subsets are drawn by a registered permutation (seed `[1913, world]`) of the
124 anchors in N1 order; the first `k` are taken and re-sorted to N1 order, so
the length-1:length-2 mix is random rather than chosen. Every arm is described as
a construction in the runner and checked by `validate_cell`.

# Necessity

**Target behaviour.** Informed early commitment supplied by anchor tasks.

**The arm that refuses it.** `k = 0`, which is N1's committed `SHAM`: the same
188-task pool, the same 24,064 examples and the same minibatch draws, with every
non-canonical slot a length-3 filler and no anchor at all.

**Measured refusal cost, and its scale.** Refusing costs the difference between
0.0093 / 0.0101 / 0.0066 (`k = 124`) and 1.037 / 1.062 / 1.087 (`k = 0`), a
factor of ~110. The scale is the 0.05 usability threshold the claim is about -
the refusing arm misses it by a factor of ~20 and the full-supply arm clears it
by a factor of ~5 - not total output variance (the V4.1 error).

**Impostors scored.** Pool size, example count and draw sequence are held fixed
by construction, so data volume cannot produce the effect; N1 already showed the
volume impostor goes the WRONG way (hard fillers are worse than the 64-task
floor). The other impostor, stream order, is ruled out because N1 showed order
is not required and every arm here is pooled.

**Difficulty band, and its direction.** Terminal median NMSE, higher-is-harder,
band `(0.0005, 0.50)`, applied to the arms under test. Both endpoints are
committed and sit on either side of the band's interior, so an arm cannot fall
outside the band without the harness having changed.

# Discriminating power

**The decision rule, as it will be applied.** A condition SUFFICES when its
terminal median is `< 0.05` in at least **2 of 3** worlds. Denominator: 3 worlds
per condition; the two outcomes partition 0..3 (verified with
`design_adequacy.partitions_outcomes`).

**Null and effect samplers, built from committed values rather than intuition.**
Null (anchors at this dose do nothing): each world drawn from `U(1.037, 1.087)`,
N1's `SHAM`. Full effect: `U(0.0066, 0.0101)`, N1's `INTERLEAVED`. Intermediate
effect: `U(0.05, 0.09)`. 2,000 draws each, seed 11, computed before this plan was
frozen.

**Measured rates** for the rule exactly as written:

| regime | rate the rule fires |
|---|---|
| null | **false-fire 0.0000** |
| full effect | **detection 1.0000** |
| intermediate | 0.0000 |

Both bounds are met (`false-fire <= 0.05`, `detection >= 0.80`; the
`design_adequacy.discriminating_power` check passes). The intermediate regime
correctly reads as NOT sufficient, which is what a "suffices" claim should say.

**What the checks do not cover.** A condition landing NEAR the threshold. One
stream per cell makes a pass or fail inside `[0.03, 0.08]` fragile (the SO1
lesson), and the samplers above never put mass there. Any world value in that
band is flagged `NEAR_THRESHOLD` and the dose reading then reports the PATTERN
across conditions rather than any single cell. The checks also do not cover
whether the dose subsets happen to be unrepresentative of the anchor mix; that is
why the subsets are drawn by a registered random permutation rather than chosen.

# Registered triage

**LENGTH**, over `L1_ONLY` and `L2_ONLY`, four outcomes partitioning every case:
`BOTH_SUFFICE`, `L1_SUFFICES_ONLY`, `L2_SUFFICES_ONLY`, `NEITHER`.

**DOSE**, over the tested doses `{0, 8, 32, 124}` with both endpoints committed:
`k*` = the smallest tested dose that passes AND every larger tested dose also
passes (persistence, not first crossing - the E5.1 rule that a bare first
crossing manufactures thresholds from noise). Outcomes `k* = 8`, `k* = 32`,
`k* = 124`. If a smaller dose passes while a larger one fails, report
`NON_MONOTONE` beside `k*`.

# Working hypotheses (not preregistered verdicts; recorded in PREDICTIONS)

- **NB1** `L1_ONLY` suffices (2 of 3 worlds under 0.05): 0.75.
- **NB2** `L2_ONLY` suffices: 0.35.
- **NB3** `k* <= 32`: 0.5.

# Anchors (free correctness checks, required)

- **Endpoint identity, by construction.** A test asserts the pool built at
  `k = 124` equals N1's `INTERLEAVED` pool and at `k = 0` equals N1's `SHAM` pool:
  identical task-id sequence and identical training arrays. Training is a
  deterministic function of pool, seed sequence and code (N1's three `STAGED`
  cells reproduced J1c bitwise), so the committed N1 values ARE the endpoints.
- **Prefix determinism.** A test runs 32 updates on the `k = 124` construction
  and on N1's own `INTERLEAVED` construction and requires bitwise-equal library
  parameters.
- The N1 module is imported, never edited: its hash is in N1's protocol and
  editing it would break N1's re-scorability.

# Cost and execution

12 cells (4 arms x 3 worlds), ~25 min each measured in N1. Run as a bounded pool
of 3 workers (the `slots=12` cap), with the PARENT as the single writer of every
cell record. ~1.7 h. Launch precondition: 8 GiB free for a pool of 3, failing
closed. Most-decisive arms first: `L2_ONLY`, `DOSE_8`, then `L1_ONLY`,
`DOSE_32`.

# Acceptance

Runner and independent scorer committed together before launch; protocol
fingerprint over this plan, config, seeds and implementation; restart reuses
completed cells; `status.json` written AT LAUNCH as well as after each cell (the
N1 defect, fixed here); `check_prereg`, `check_invalid`, `check_adequacy` pass;
records archived to `reports/` and committed with the result.
