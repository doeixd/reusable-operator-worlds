# Generality program: are ROW's results about learning, or about this world?

Status: DRAFT program document, 2026-10-13. Direction approved by the PI (decision 19: Claude recommended it, the PI
answered "Ok write all this down. New plan"). It registers NO experiment: every rung below gets its own frozen plan,
with `# Necessity` and `# Discriminating power` sections, before any cell runs. It sits beside `SUCCESSOR_LADDER.md`
and takes priority over that ladder's remaining rungs.

# Why: the open threat over everything

Every result in this project, confirmed or exploratory, comes from ONE world family: six rotated-residual operators,
state dimension 16, 12 learner slots, depth 3-5. That leaves one threat standing over all of it: these may be facts
about this world, not about learning. The recent record says the threat is live, because the testbed has started
answering "no opportunity" more often than it answers questions:
- **B1-mid (2026-10-10):** each operator is a near-identity residual plus a rotation, so a hyperplane on a state
  after 1-8 steps is 97-98% a hyperplane on the input; "where a decision happens" cannot be necessary here.
- **SG0 (2026-09-22):** on a formed library, route inference is trivial (runner-up ~54x worse on support, 576/576
  programs unambiguous); inference difficulty exists only on unformed libraries.
- **B3 (2026-10-12):** no repeated structure recurs enough to repay a loop abstraction, and a fixed-count REPEAT is
  definitionally the tied unrolled route.
- **E6/E7, L0d, H47-H49:** macro compilation, learned inference and structure discovery failed necessity or
  opportunity on this generator.
- **SC0 / SC0b (2026-10-12/13):** the adaptive-curriculum track has no terminal opportunity against a
  uniform-over-lengths sampler (`LENGTHS_SUFFICE`).

More rungs on this generator now mostly cost a census and return "not here". A harder generator is the one move that
changes the weight of every existing result at once.

# What the project would gain

- **If the main results survive a materially different world,** they upgrade from "on this testbed" to general
  claims. The candidates to carry, in order of importance:
  1. stale routes, not a broken library, are the residual failure of online formation, and re-deriving
     assignments + consolidation repairs it (O5-O11, sealed);
  2. a quota of single-operation tasks is what makes a vocabulary form, presence not order (N1-N1c, SC0, SC0b);
  3. control flow forms online when every candidate starts at the plain form and must earn its iteration (B2-hard
     v2, B2-depth2; sealed by C-seal, 2026-10-13);
  4. a well-formed vocabulary makes program inference easy (SG0).
- **If they break,** we learn which results were artefacts of a near-linear, easy-inference world, before building
  further on them. Either outcome is a result; neither is a dead end.
- **Rungs this world closed may reopen:** B1-mid (mid-state decisions), B3 (control-flow compression), E6.2-style
  compilation and learned inference all failed because this world does not require the behaviour.

# Two tracks, run concurrently

They do not compete: the sealed run is machine time, the generator work is mostly design and minutes-scale censuses.

## Track C-seal: sealed confirmation of online control flow

The control-flow line's positives (B1-online, B2-online v2, B2-hard v2, B2-depth2) are all Tier 1 on development
worlds; it is the most novel unconfirmed claim in the project.
- **Construction:** B2-hard v2 (`B2_HARD_V2_PLAN.md`): every candidate starts at the one-step form
  (`halt_init = 4`), halting reads the detached state, the one-step data test at the end of the stream; arms and
  registered clauses carried from that plan, thresholds re-checked against its development baselines.
- **Band:** sealed 1100-1129 (allocated below); no world in it is generated before the plan is frozen and hashed in
  `tools/check_prereg.py`.
- **Size:** set in the plan from B2-hard v2's measured per-cell cost and the discrimination simulation; early-stop
  rules registered; must fit one overnight batch or go remote.
- **Scope stated in advance:** a confirmation on THIS generator. It does not address the generality threat; that is
  Track G.
- **Result (2026-10-13): `CONFIRMED` with `ROOTS_PREVENTED`** (plan `B2_SEALED_CONFIRMATION_PLAN.md`, frozen `3efb101`;
  `reports/b2_sealed.json`). Anchor reproduced bitwise, `GIVEN` 15/15; `ONESTEP` formation 30/30, wake AUC 1.000,
  recall 0.995, specificity 1.000, loops below 0.38 in 30/30; rooted 0/30 against `ZERO` 3/15. The control-flow
  construction joins the list of results for G2 to carry (item 3 below is now sealed on this generator).

## Track G: a harder generator

### G0 generator design and Tier 0 census (hours; no learner)

Build 2-3 candidate world generators in a new versioned module, each with a switch that recovers the current
generator BITWISE (verified on a real world), so every difference is attributable to the change. Candidate
directions, chosen by the census, not in advance:
- **less linear operators:** larger residual scale or a second nonlinearity per operator, so states move far from
  any orthogonal map of the input;
- **more primitives:** 8-12 instead of 6, so the library and the route space grow and depth-1 clustering is harder;
- **wider state:** d = 32, only if timing allows (cost per lifetime measured, not assumed).

Measured on teacher worlds and an untrained/oracle learner, each against a band registered before measuring:

| property | why it matters | current world |
|---|---|---|
| linear predictability of the state after k steps from the input (B1-mid statistic) | mid-state decisions can only be necessary if this is well below 1 | 0.97-0.98 |
| route identifiability on an oracle library (SG0 statistic) | inference must be neither trivial nor hopeless (two-sided band) | ~54 (too easy) |
| depth-1 clustering on an untrained library (N1 ARI) | the anchor mechanism must still be possible, or formation is not testable | 0.46-0.77 |
| teacher output variance and saturation | tasks must be learnable and comparable in scale | 0.21-0.23, none |
| seconds and resident memory per lifetime on this host | decides whether Tier 1/2 fit locally | ~8-12 min, ~340 MB |

A candidate passes G0 only if it moves at least one of the first two rows out of the saturated region while keeping
the third and fourth in a usable band. If none does, that is itself recorded: the rotated family cannot be made
harder along these axes without losing formability.

### G1 formation on the new generator (Tier 1)

The confirmed online pipeline (anchor supply + wake + end-of-stream exhaustive re-route + sleep, O6 construction) on
the chosen generator, with 60 single-operation anchors, 3 development worlds x 3 streams, against the same pipeline
with no anchors. Registered question: does the vocabulary form? Failure is a headline, not a nuisance: the formation
story would then be specific to the rotated world, and the next rung diagnoses which component broke (failure split
by mode, as O2-O6 did).

### G2 carry the confirmed mechanisms (Tier 1, then sealed)

Only if G1 forms: in order of the list above, the stale-route mechanism (re-route + sleep against sleep alone), the
anchor quota (a dose check), and the control-flow construction. Each survivor goes to a sealed block.

### G3 reopened rungs

Only after G1-G2: re-run the necessity censuses that closed B1-mid, B3 and the inference rungs, on the new generator.
A rung reopens only if its census now passes.

### G0 result (2026-10-13): `NO_CANDIDATE`

`G0_GENERATOR_CENSUS_PLAN.md` (frozen `56e0d78`), `reports/g0_generator_census.json`. 39 variants of the rotated
generator (`row.general_world` g1: alpha 0.35-1.5, hidden gain 1-4, K 6-12, shared rotation `m` 0-16):
- linear predictability of a computed state from the input never drops below 0.958 (bar 0.90; base 0.978): the
  near-linearity is a property of the operator family (identity skip + bounded residual + orthogonal map), which the
  learner shares;
- route inference becomes non-trivial only through shared rotations (runner-up NMSE 0.20-0.27 at `m = 4`, base 1.65),
  and the same similarity removes depth-1 clustering on an untrained library (ARI 0.03-0.04 at `m = 4`, bar 0.30).

The two saturations therefore need different moves, and the first conflicts with "it does not change the learner"
below. Decision 20 (RESEARCH_STATUS.md) is pending: Claude recommends an exploratory Tier 1 formation probe at
`m = 4` first (does formation need depth-1 clustering?), and a separate program decision on a new operator family.

### G1 formation probe (2026-10-13, Tier 1, exploratory): `FORMS`

`G1_FORMATION_PROBE_PLAN.md` (frozen `c51b967`), `reports/g1_formation_probe.json`. The O11 pipeline passes on `m = 4`
worlds 70-72 (0.010-0.030) with operation identity recovered (formed ARI 0.97-0.98 against untrained 0.03-0.04), so
G0's depth-1 clustering bar was too strict and `m = 4` is the inference-axis candidate. On the formed vocabulary the
best wrong program comes within 2-5% NMSE (current generator 74-153%) but exhaustive search still finds programs that
generalize. Next on this axis: a Tier 0 inference-necessity census on the formed `m = 4` libraries, then reliability
across streams before any Tier 2. The mid-state-decision axis still needs decision 20 (a).

### G3a inference census (2026-10-13, Tier 0): `NO_HEADROOM`

`G3A_INFERENCE_CENSUS_PLAN.md` (frozen `92e1b68`), `reports/g3a_inference_census.json`. On G1's formed `m = 4`
vocabularies, choosing the route from 2 support examples is as reliable as from 128 (1 of 192 programs binds at either;
identifiability flat). Noiseless targets make any non-tied wrong route excludable from a few inputs, so `m` cannot
create data-limited inference. Decision 21 (RESEARCH_STATUS.md): Claude recommends observation noise as the next knob,
starting with a support-noise census on the existing vocabularies.

### G3b noise census (2026-10-14, Tier 0): `HIGH_NOISE_ONLY`; the inference axis closes in the rotated family

`G3B_NOISE_CENSUS_PLAN.md` (frozen `3fd6a92`), `reports/g3b_noise_census.json`. Support noise makes route choice on the
formed `m = 4` vocabularies bind only with 2 examples and noise >= 30% of the signal. With G0 (the mid-state axis is
closed by near-linearity), G3a and G3b, both necessity axes Track G set out to open are now closed INSIDE the rotated
family. The remaining move is decision 20 (a): a new operator family for teacher and learner.

### G0F operator-form census (2026-10-13, Tier 0): `SELECTED` `tanh2`

`G0F_FORM_CENSUS_PLAN.md` (frozen `ff33942`), `reports/g0f_form_census.json`. Under decision 20 (a) (the PI answered
"Sure" to the survey), three non-rotated forms were measured. The `tanh` map without identity skip and the
multiplicative gate break near-linearity (S1 0.80-0.89 at one step, 0.68-0.81 at three) while keeping the depth-1
clustering premise and single-operator learnability; piecewise-linear folds do not (S1 >= 0.915). Chosen: `tanh2`.
Adopting it changes "the learner does not change": the next step, if the PI agrees, is a same-form learner and the
G1 formation probe on it.

### G1T tanh-form formation probe (2026-10-14, Tier 1): `FAILS`

`G1T_TANH_FORMATION_PLAN.md` (frozen `536be74`), `reports/g1t_tanh_formation.json`. The PI answered "sure" to
adopting `tanh2`. With a same-form learner at the rotated form's hyperparameters, nothing formed in 3 worlds under any
arm, the length curriculum included. The vocabulary's identity forms (depth-1 ARI ~1) but no slot fits its operation
(depth-1 error 0.13-0.16 after sleep), so the failure is fit, not formation. Next rung, pre-stated: learner-side
sizing (learning rate, initial gain) on one world, then the probe again on the remaining worlds if a setting fits.

# Track S, Track M (unchanged, lower priority)

- **Track S (self-curriculum): paused.** SC0 `GRADED`, SC0b `LENGTHS_SUFFICE` (quota 16-24 anchors; uniform over
  lengths clears it through depth 5), so the terminal endpoint has no opportunity; a cumulative-cost endpoint would
  need its own opportunity check.
- **Track M (procedural memory):** PM-stale (stable or versioned handles against stale routes) is the rung closest to
  the strongest mechanism; it is the natural next rung after Track G's G1, and could run on either generator.

# Seed bands (allocated with decision 19)

- **Development band 6: seeds 70-99** for Track G (G0 censuses, G1, G2 development). Verified unused before
  allocation: no runner, report or artifact uses 70-99 as a world seed.
- **Sealed band 1100-1129** for Track C-seal. Verified unused as world seeds; 1100-1102 occur only as SeedSequence
  components inside `audit_e7_census.py` (probe and noise streams, not worlds), disclosed as for 951.
- **960-989 stays reserved** for the deferred sealed depth test (decision 18), as already recommended there.

# Order of work

1. Record SC0b (done: `LENGTHS_SUFFICE`).
2. Freeze and launch the C-seal plan (done: `CONFIRMED` with `ROOTS_PREVENTED`, 2026-10-13).
3. While it runs: write the G0 plan, build the generator module with its bitwise switch, run the G0 census. (done 2026-10-13: `NO_CANDIDATE`, below.)
4. G1 when G0 picks a candidate.

# What this program does not do

- It does not change the learner. Every G rung uses the confirmed pipeline unchanged, so a failure localizes to the
  world.
- It does not reopen closed rungs by argument: only a passing census reopens one.
- It does not make the current results weaker by default: until a G rung says otherwise, they stand, scoped to the
  rotated world, as they always were.
