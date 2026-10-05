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

Order, each rung gated by the one before:
1. **C0 repeated-circuit execution (Tier 0).** With the correct route given,
   does one learned operator keep its meaning applied 1..32 times? Separates
   execution from route inference (#3 section 2.1). If execution degrades fast,
   the bottleneck is the vocabulary and no loop rung can be read.
2. **C1 positional-recurrence sweep (#3 section 1).** A `rho_depth` knob from
   independent per-position computation to one operator reused across
   positions; tied vs untied learner at matched budget. Prediction: a
   crossover, the depth-wise analogue of the V1 reuse law.
3. **C2 route-inference length extrapolation (#3 section 2.2).** Can the router
   infer an unseen repeat count from support data?
4. **C3 variable-length routing (#3 section 3).** Learned route length; the
   current depth-4/5 streams already mix lengths, so this starts from a known
   construction.
5. **C4 state-conditional routing (#3 section 4).** The router chooses the next
   operator from the intermediate state, with STOP. The first rung that is
   control flow rather than composition.
6. **C5 emergent loop compression (#3 section 5).** Only after C4.

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

**First registered rung: C1, the positional-recurrence sweep, at program
lengths <= 8.** Reasons: Track C is first in this ladder; C0 shows execution is
clean in that range, so a tied-versus-untied comparison is not confounded by
drift; it is the depth-wise analogue of the V1 reuse law and has a registered
crossover prediction; it needs a new generator knob (`rho_depth`) and an untied
learner, so its plan must pass the NECESSITY and OPPORTUNITY gates with a Tier 0
opportunity census before any lifetime. Scaling (A2, sparse in-stream schedules)
is lower priority now that A1 shows approximate re-routing is enough.
