# Successor ladder: from a confirmed online vocabulary to control flow, curriculum and procedural memory

Status: DRAFT program document (2026-10-05). It registers no experiment; each
rung below gets its own frozen plan before any cell runs. It implements issue
#2 section 14: a NEW ladder whose baseline is the confirmed learner, instead of
reopening the closed L1-L8 ladder (closed under its own registered O2 rule).
Sources: issues #2 (roadmap), #3 (control-flow ladder), #4 (self-curriculum),
#5 (procedural memory).

# Baseline learner (what every rung starts from)

The online-formation line established, under named assumptions (single-
operation anchors in the stream, 64 retained examples per task, one substrate
family, one learner):
- **Depth 3, sealed:** wake + end-of-stream exhaustive re-route + sleep, 45/45
  (O6); wake + in-stream re-route + sleep, 45/45 and beats sleep alone 45/45
  paired (O11).
- **Depth 4, development:** both protocols 21/21; sleep alone 0/21 (D1, census).
- **Depth 5, development:** end-of-stream protocol 21/21; sleep alone 0/21 (D2).

The baseline for every rung is therefore: **anchor supply + wake + re-routing
(end-of-stream, or in-stream where the rung needs it online) + consolidation**,
on the rotated substrate, at the depth the rung needs. Exhaustive re-routing is
exact and affordable through depth 5 (`deep_reroute`); O7's gradient re-router
is only needed from about depth 6.

# Phase A remainder (issue #2): what scaling still needs

- **A1 route adequacy (#2 section 3).** How good must a route be for sleep to
  finish the repair? Controlled-quality routes on the same terminal, identical
  sleep. Tier 0 on the surviving D1/D2 terminals. Decides whether approximate
  re-routing suffices beyond depth 5.
- **A2 sparse in-stream schedules (#2 section 2).** Re-route every k arrivals,
  or adaptively on a drift signal, at matched search compute. Needed only if a
  later rung needs the fully online property at depth >= 5.
- **A3 sealed depth test.** Deferred (decision 18): confirms a known
  development result; needs a new sealed band and ~1.5 GB of disk.

# Track C: control flow (issue #3), first

**Terminology (PI question, 2026-10-05: "Routing is control flow?").** No. In
this project a task's route is a fixed slot sequence chosen once per task from
support data and applied identically to every input: a straight-line program.
Choosing it is program SELECTION. Control flow starts when the next operation
depends on data available during execution (the intermediate state). Issue #3
section 4 draws the same line. The rungs below are split accordingly.

**Straight-line rungs (preconditions, not control flow):**
- **C0 repeated-circuit execution (Tier 0, done 2026-10-05):** a learned
  operator applied n times with the correct route drifts about linearly in n,
  clean to about 8 applications. A loop that iterates further inherits this
  drift.
- **C1 positional-recurrence sweep:** reuse economics WITHIN a program (tied vs
  untied computation across positions), the depth-wise analogue of the V1 reuse
  law. Not control flow.
- **C2 route-inference length extrapolation** and **C3 variable-length
  routing:** still program selection, with the length as part of the program.

**Control-flow rungs:**
- **B0 branch necessity census (Tier 0, next).** Tasks `IF(p(state), A, B)`
  with a hidden predicate. Measures, on a formed library: the best any single
  straight-line route can do (the refusal cost of the current learner, which
  must use one route per task), the oracle-branch ceiling, and whether the
  predicate can be recovered from support data alone. Issue #2 section 4's 2x2
  (branch structure x predicate, oracle or learned) is the frame.
- **B1 state-conditional routing:** a learner whose router chooses the next
  operator (or STOP) from the current state; the first registered control-flow
  rung, only if B0 shows necessity and opportunity.
- **B2 data-dependent iteration (`while p(state)`),** reading C0's drift as a
  measured baseline.
- **B3 emergent control-flow compression** (issue #3 section 5), only after B1-B2.

# Track S: self-curriculum (issue #4), second

- **SC0 read-only census:** which task classes carried the most marginal
  learning value at each stage of known lifetimes. Most saved lifetimes from
  the O-line were deleted on 2026-10-04; SC0 must use the surviving D1/D2
  artifacts or regenerate a small set.
- **SC1 adaptive sampler:** a bandit over task classes (length, novelty,
  repeated operator), rewards difficulty / immediate progress / future-transfer
  proxy, against uniform and the hand-designed anchor supply.
- **SC2 learned proposer:** only if SC1 beats both.

# Track M: procedural memory (issue #5), third

- **PM0 feasibility:** explicit callable memory entries replacing a subset of
  slots, bitwise-equivalent execution (switch-recovers-baseline gate).
- **PM1 frozen-core acquisition:** freeze the core; acquire a new recurring
  family by memory writes only.
- **PM-stale (#5 section 11):** do stable or versioned handles reduce the
  stale-route burden the O5-O11 line repairs? The most direct connection to the
  current baseline.
- **PM2 dynamic allocation:** only after PM1.

# Gates every rung must pass (DESIGN_ADEQUACY.md)

- **NECESSITY:** does the task REQUIRE the behaviour? A loop rung on a world
  with no iteration measures absence, not value (the 2026-08-31 loop census
  error). C0 is the necessity probe for Track C.
- **OPPORTUNITY:** could the effect exist in this generator?
- **DISCRIMINATION:** run the registered rule on null and effect samplers;
  false-fire <= 5%, detection >= 80%.

# Bounds from the record (what a rung must not silently repeat)

- **Post-hoc extraction is closed** (H47-H53, the H39 lesson "made, not
  mined"). #2 section 9's population refactoring must state why it is formed
  during learning, not mined afterwards.
- **Macros pay on this testbed; compilation does not** (E6). C5 and #5's
  macros must beat COMPRESS at matched bits (the V4 constitutional rule).
- **Lifecycle edits do not pay at this scale** (V4/V4R: COMPRESS dominates,
  FACTORIZE/FORK do not). #5's split/merge/retire must first show an
  opportunity census.
- **Depth drift is measured** (E5.1, sealed export block: per-step drift
  b = 0.581, continuation q = 0.785). C0 extends it to a repeated operator; a
  degradation's SHAPE is not whether it BINDS (review 80).
- **Development rates are optimistic** (O8 20/21 against O9 37/45 sealed).
  Exploratory rungs size; sealed rungs decide.

# Next action

**C0 (2026-10-05): READ_SHAPE.** A learned operator applied n times with the
correct route drifts about linearly in n: median NMSE 0.030 at n = 8, 0.070 at
16, 0.19 at 32 (generic mixed routes 0.115 at 32). Execution is clean to about
8 applications.

**A1 (2026-10-05):** sleep finishes the repair when about half of all tasks are
one position wrong, or a quarter are wrong everywhere, but not on stale routes,
which are coherent error. Approximate re-routing suffices if its errors are
incoherent.

**B0 (2026-10-05): NECESSITY holds, OPPORTUNITY missed at 128 support
examples.** No single straight-line route solves any of 96 branch tasks
(median 1.0); the oracle branch reaches 0.006-0.011. The learned predicate
(branch structure given) reaches 0.11-0.12 at 128 support examples with
perfectly identified labels; B0b (post hoc) shows 0.038-0.047 at 512 and
0.023-0.037 at 2,048. The decision is learnable; it needs about 512 examples.

**B1 (2026-10-05): LEARNABLE, 178 of 192.** On frozen formed vocabularies the
branch structure is free (192/192 given the split) and the decision is learned
from support (INPUT 95/96; MID 83/96, the residual). Joint learning loses
nothing against predicate-only learning.

**B1g (2026-10-05): GRADIENT_FINDS, 173 of 192** (search 178). Gradient on soft
codes and a linear gate finds branches on a frozen vocabulary; INPUT decisions
need no restarts, MID decisions need them (first restart fails on 10 of 96).

**B1-online (2026-10-06): ONLINE_BRANCHES, 9 of 9.** A learner with a second
route and a linear gate per branch task forms the straight-line vocabulary
(canonical 0.0111, same as without branches) and learns branches online (0.23
against 1.02 under refusal). Forcing branch tasks into one route triples the
straight-line error; gating protects formation. Branch tasks do not reach 0.05
at 128 examples (expected; the predicate needs ~512).

**Next, in order:**
1. **Gate-accuracy check (post hoc):** a deterministic re-run of one GATED cell
   measuring orientation-free gate accuracy (the registered field was
   orientation-dependent and uninterpretable).
2. **B1-online at 512 examples per branch task** (or branch tasks sampled more
   often), to see whether the online branch error reaches the threshold when the
   predicate is not under-sampled.
3. **Mid-program decisions online** (B1's MID variant), the harder case.
4. **B2, data-dependent iteration:** `while p(state)` as a branch whose decision
   is continue or stop, reading C0's execution drift as a baseline.
