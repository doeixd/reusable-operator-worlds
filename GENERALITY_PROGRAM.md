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
     v2, B2-depth2; exploratory);
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
2. Freeze and launch the C-seal plan (Tier 2, sealed 1100-1129) after its dry run and restart test.
3. While it runs: write the G0 plan, build the generator module with its bitwise switch, run the G0 census.
4. G1 when G0 picks a candidate.

# What this program does not do

- It does not change the learner. Every G rung uses the confirmed pipeline unchanged, so a failure localizes to the
  world.
- It does not reopen closed rungs by argument: only a passing census reopens one.
- It does not make the current results weaker by default: until a G rung says otherwise, they stand, scoped to the
  rotated world, as they always were.
