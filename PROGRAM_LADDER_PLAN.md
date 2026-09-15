# Program ladder: compositional operators -> control flow -> higher-order programs

Status: RESEARCH PROGRAM PLAN, drafted 2026-09-15 from the PI's proposal. This is
not a frozen experiment. Every rung below becomes runnable only through its OWN
frozen plan (hashed in `tools/check_prereg.py` if Tier 2), with a committed
scorer, a dry run and a restart test, as for any rung. The hypotheses are PX1-PX9
in `PREDICTIONS.md` (2026-09-15); they are working hypotheses, not registered
predictions.

This plan extends, and does not replace, Tracks C and D of
`POST_E6_RESEARCH_PROGRAM.md`. Nothing here revises a sealed verdict, a
development status, or a stop rule.

# Where the program stands (what the ladder is built on)

- **Straight-line composition is established.** Frozen learned objects execute
  unseen straight-line compositions: the sealed export block, and J2A offline on
  the rotated substrate.
- **Abstraction must be formed while learning.** Useful abstraction is
  path-dependent, formed during learning rather than extracted afterwards (H39,
  H50, H51), and parameterized as shared structure plus cheap arguments plus
  private innovation, `A(alpha) + eps`.
- **Control flow needs a vocabulary where it matters.** The ordinary substrate
  failed the iteration-necessity gate (`ITERATION_WORLD_SPEC.md`, withdrawn).
  The rotated family makes both iteration and branching necessary
  (`ROTATED_SUBSTRATE_SPEC.md`, Amendments 1-2).
- **That vocabulary is not yet learnable online.** Offline staged formation
  acquires and exports the rotated substrate (J1c, J1c-R, J2A). Online, SO2
  failed its terminal clause (`SO2_FAILS`), the loss is post-acquisition
  consolidation in the final stage (census), and SO3 is running.
- **Stop rule 2 binds:** "Online SO2 fails: do not run branching or iteration,
  even if offline cells are excellent." Track C's C2 "requires SO2".
- **Macros pay but are not loops.** Macros pay but could not be timed or
  compiled (E6 line), so a trace-compressing macro is the expected impostor for
  a loop.
- **Search is a strong opponent.** Search cost scales with program length, not
  program-space size, and exhaustive enumeration is the opponent to beat
  wherever feasible (E5, E5.1).
- **Loop economics has already been measured once, and misread.** The 2026-08-31
  loop opportunity census found that on straight-line E6 routes a `(LOOP, op,
  count)` construct is NET NEGATIVE at depth 6 (-84 bits after the alphabet tax)
  and pays only at depths 8-10, where routes have collapsed. Its same-day
  CORRECTION records that this measured the absence of iteration in a generator
  with no iteration, not the value of loops. The necessity gate therefore moves
  into generator design.
  - L0a-L0b are that gate for PX1 and PX2.
  - L3 must beat the alphabet tax and the macro alternative on a world where
    iteration is necessary.
  - Branching had no census at all, because no task in that generator depended
    on its input.
- **Parameterized macros were already tested against their own null.** The E7
  census refuted the parameterized-macro rationale: against grams the same
  learner wrote at unplanted sites, the apparent family structure largely
  vanished. Any promoted-graph proposal in L3 or L4 must use the same harder
  null (the learner's own unplanted structure), not random structure.

# Cross-cutting acceptance rule (applies to every rung)

A claimed program abstraction must pass all three tests at once.

1. **ECONOMICS.** It lowers lifetime prediction, search, or storage cost against
   the cheapest simpler alternative. The alternatives always include COMPRESS
   (coarser private representation at equal bits) and a matched-budget
   non-sharing arm. Prediction, search and storage are reported in separate
   currencies unless an exchange rate is frozen.
2. **CAUSAL SEMANTICS.** Changing exactly one purported argument (count,
   predicate, branch body, function argument, binding) produces the registered
   local effect, and unrelated behaviour stays within a registered tolerance.
3. **EXTRAPOLATION.** It executes outside the finite structures trained: unseen
   counts, novel branch triples, new function arguments, deeper nesting.

These project rules are inherited by every rung plan:
- **Opportunity gate first:** the counterfactual cost of refusing the structure
  must be non-zero.
- **"Could it come out any other way?"** Never use an object's defining
  invariant as evidence.
- **Current utility, future fertility and representational cost** are kept
  separate.
- **Sham arm** for any propose-reorganize-score loop.
- **Balance gates** on any relatedness knob.
- **Measured-coordinate fits;** a threshold is checked against its baseline and
  its arithmetic under the null before freezing.
- **Oracle arms bound only performance under their own assignment.**

# Gate structure

```
                        SO3 (running) or successor clears online learnability
                                            |
  Tier 0 now (no learner control flow)      |      learner rungs (need the gate)
  -------------------------------------     |      ------------------------------
  L0a  PX1 world census (teacher-side)  ----+----> L1  PX1 r_op x r_ctrl economics
  L0b  PX2 count-world gates            ----+----> L2  PX2 ITERATE, supplied schema
  L0c  oracle-executor count horizon    ----+          |
  L0d  PX7 inference census on existing              L3  PX4 promoted ITERATE
       straight-line libraries (runs now)             |
                                                  L4  PX3 IF, supplied -> promoted
                                                      |
                                        L5 PX6 bindings (straight-line part can run
                                           earlier, Track A dependent)
                                                      |
                                                  L6  PX5 function-valued arguments
                                                      |
                                        L7  PX8 formation-time editability (parallel,
                                            gated on its matching gates)
                                                      |
                                                  L8  PX9 endogenous DSL growth
```

Allowed before the online-learnability gate clears:
- teacher-side world construction and its necessity, achievability and balance
  gates (arithmetic on generators, no learner);
- oracle-executor horizons (the true library, no learning);
- inference comparisons on EXISTING frozen straight-line libraries (J2A's
  staged libraries).

None of these is "running branching or iteration" as stop rule 2 means it: no
learner is trained to branch or iterate. **PI confirmation of this reading is
requested before L0a-L0c are frozen.**

If SO3 fails, learner rungs L1-L6 can still be considered under the OFFLINE
staged protocol. That needs an explicit, PI-approved amendment to stop rule 2,
and every claim is then scoped "offline, supplied curriculum". This plan does
not assume that amendment.

# Rungs

## L0 - Tier 0 censuses (minutes to an hour; can run now)

**L0a, PX1 world census.** Construct a rotated-family world with two independent
knobs:
- `r_op`, operator recurrence (the existing correlated-teacher mechanism);
- `r_ctrl`, control-skeleton recurrence: the fraction of tasks sharing a
  skeleton such as `REPEAT(., k)`, `IF(p, ., .)` or fixed-shape nestings.

Before any learner, measure:
- **Necessity:** no fixed straight-line program of matched length reaches the
  task loss.
- **Achievability:** the true skeleton plus the true operators reaches it.
- **Balance gates across the `r_ctrl` sweep:** per-task difficulty, per-skeleton
  description length, and operator-usage marginals, each within 10%.
- **Linear-coordinate check:** measured skeleton recurrence against the
  configured knob.

Kill: if varying `r_ctrl` moves per-task difficulty beyond the balance gates at
every tried construction, PX1 is unscoreable (the V5.1 rule) and the rung is
redesigned before any learner.

Note: straight-line skeleton recurrence without necessity is route-pattern
recurrence, which the E6 macro line already covered. `r_ctrl` must be
recurrence of NECESSARY control.

**L0b, PX2 count-world gates.** Tasks are `f^n(x)` over rotated bodies.
- **Necessity:** no fixed depth approximates the count-varying task (C2's REPEAT
  gate).
- **Balance:** body difficulty is independent of n.
- **Holdout construction:** (f, n>2) pairs are withheld and bodies appear at
  n <= 2 elsewhere.
- **Degeneracy check:** unseen counts 5, 7, 10 and 20 must not be reachable by
  any trained-count program. A rotation with an order dividing n would make
  `f^n` equal a lower power; this must be excluded by construction.

**L0c, oracle-executor count horizon.** Execute `f^n` with the ORACLE-matched
learned library (or the teacher library plus the learner's measured per-step
error model) and measure where per-step drift pushes query NMSE above the
threshold. This sets the largest n at which a learner failure could mean
"no loop" rather than "numeric compounding", the E5 Amendment 1 forecast-gate
pattern. Extrapolation claims are only read below this horizon.

**L0d, PX7 inference census, straight-line and runnable now.** Freeze J2A's
staged libraries. On support-only program inference at depths 3-8 (E5.1 scale)
with controlled ambiguity (near-duplicate slots, reduced support), compare:
- exhaustive MAP;
- beam at several widths;
- posterior averaging;
- the learner's continuous codes;
- hard-commit-early versus commit-late hybrids.

Report search seconds, operator applications and query NMSE separately. This is
Tier 0 or Tier 1 on existing artifacts. It is the first place PX7(a-c) can be
measured, and it touches no control flow.

## L1 - PX1: the economics of control flow (first learner rung)

- **Prerequisites:** the online-learnability gate (or approved offline scope);
  L0a passes necessity, achievability and balance.
- **Arms**, matched on shared parameters, task-code budget and example-gradients:
  - TASK-SPECIFIC;
  - OPS-ONLY (reusable operators, private control);
  - CTRL-ONLY (reusable skeletons, private operators);
  - BOTH;
  - a COMPRESS arm (each private component quantized to the shared arms' bits);
  - a SHAM arm (the same structure-sharing machinery with skeleton assignments
    permuted).
- **Grid:** a coarse `r_op x r_ctrl` grid in measured coordinates, and at least
  3 development worlds not used by earlier rungs.
- **Estimands:** paired lifetime cost differences per arm pair; `a`, `b`, `c` and
  the crossing surface fitted in measured coordinates with world-level bootstrap;
  retained bits; search cost.
- **Registered in the rung plan, sized from L0a:** sign-consistency of `b` and
  interval bounds; whether `c` is distinguishable from 0; the fit-quality clause.
- **Triple test:** economics is primary. Causal semantics: swapping a task's
  skeleton assignment for a wrong skeleton must degrade it locally.
  Extrapolation: skeleton reuse on held-out operator sets.
- **Kill:** CTRL-ONLY never beats TASK-SPECIFIC at the highest `r_ctrl` in 2/3
  worlds, so control reuse has no amortization curve at this scale.

## L2 - PX2: ITERATE with the count as an argument (second learner rung)

- **Prerequisites:** the online gate; L0b and L0c.
- **Supplied schema first** (Track C's C2 scope): `REPEAT(count, body)` with a
  learned body library and inferred count.
- **Arms:**
  - ITERATE-schema learner;
  - fixed-depth compositional learner at matched parameters and budget;
  - an unrolled trace learner (a private straight-line program per task);
  - a wrong-count control (schema executed at count +/- 1);
  - a wrong-body control.
- **Holdouts:** unseen counts {5, 7, 10, 20}, read only up to the L0c horizon;
  (body, count>2) pairs; bodies never iterated in training.
- **Estimands:**
  - query NMSE by count;
  - a count-only intervention (the same inferred body, n changed): the
    predicted-versus-observed error ratio against the oracle `f^n`;
  - retained bits against the unrolled learner;
  - count-inference accuracy from support.
- **"Could it come out otherwise?":** a supplied schema executing its expansion
  is true by construction. Evidence must come from a LEARNED body substituting
  under an INFERRED count at an UNSEEN count.
- **Kill:** no extrapolation beyond trained counts below the oracle horizon in
  2/3 worlds.

## L3 - PX4: ITERATE promoted, not supplied (third learner rung, headline)

- **Prerequisites:** L2 passes (a loop is representable and usable when supplied)
  and the PROMOTE machinery is extended.
- **Construction:**
  - The lifetime begins with no ITERATE symbol; tasks are solved as private
    straight-line traces of the matching length.
  - The consolidation operator may propose a parameterized computational graph:
    a callable body slot plus a repeat operation with a count argument, charged
    its description length and a search cost.
  - Its candidates compete with MACRO proposals (E6-style compressed traces)
    and with COMPRESS.
- **Arms:**
  - PROMOTE-GRAPH;
  - PROMOTE-MACRO-ONLY;
  - no consolidation;
  - SHAM (graph proposals over permuted traces);
  - an oracle-supplied ITERATE ceiling (the L2 learner, labelled as a ceiling).
- **Discriminator:** execution at a count never seen as a trace. A macro passes
  trained counts and fails the unseen count; a loop passes both.
- **Estimands:**
  - whether and when promotion fires, against the amortization prediction
    `H* = lambda D*(A) / s_bar` computed from independently measured `D*` and
    `s_bar` (the V5 retention precedent: no fitted threshold);
  - unseen-count execution;
  - new-body substitution;
  - lifetime cost against both promotion arms and COMPRESS.
- **Headline claim licensed by a pass:** "experience made the learner create a
  control-flow abstraction when its repeated use economically justified it",
  scoped to the substrate, protocol and cost.
- **Kill:** graph promotion never fires, or fires and fails the unseen count
  (the macro impostor), in 2/3 worlds.

## L4 - PX3: IF with factored predicate and branches

- **Prerequisites:** the online gate; C2's IF necessity gate re-run on the exact
  substrate artifact.
- **Supplied schema first,** then promotion (the L3 pattern).
- **Holdouts:** whole (p, f, g) triples, and each filler held out from some
  roles.
- **Decisive interventions**, each with a registered expected effect computed
  from the oracle program:
  - swap f and g;
  - replace p;
  - dense evaluation near the boundary;
  - clamp the predicate output.
- **Controls:** wrong predicate, swapped branches, flat depth-matched programs,
  and a schema-free learner at matched budget.
- **Kill:** interventions produce global rather than local effects in 2/3
  worlds while task loss is competitive. That is recorded as "useful opaque
  function", the PX3(c) outcome.

## L5 - PX6: roles and bindings

- **Straight-line part:** register and handle renamings on the ordinary or
  rotated straight-line substrate. It can run after Track A's RF1, independently
  of control flow.
- **Control-flow part:** after L2 or L4.
- **Tests:** consistent renaming with the structure transformed accordingly;
  inconsistent renaming as the negative control; function-handle permutations;
  unseen positions.
- **Arms:** unstructured learner; role-factorized architecture; role-permuted
  control; collapsed-role control.
- **Kill:** equivariance fails under held-out renamings even with role
  factorization.

## L6 - PX5: function-valued arguments

- **Prerequisites:** L2 or L4 pass; L5's straight-line part passes.
- **World:** `MAP(f, sequence)`, `FOLD(f, init, sequence)`, `COMPOSE(f, g)`,
  `ITERATE(f, n)` over learned operators.
- **Decisive holdout:** a new function argument. Train MAP with A-D, learn E in a
  different context, query MAP(E, .) with the combinator frozen.
- **Controls:** retraining the combinator on E (a ceiling, charged); a wrong
  argument; a combinator-free learner at matched budget.
- **Estimand:** transfer against E's functional distance from A-D in the
  learner's own functional coordinates, never teacher identity.
- **Kill:** MAP(E) fails without combinator retraining in 2/3 worlds.

## L7 - PX8: formation-time editability

- **Prerequisites:** a successful earlier rung supplying a later
  "discovery cost" endpoint (L2, L3 or L4), and its own matching gates.
- **Construction:** generic edit probes during wake (substitute, call from a new
  context, parameterize, duplicate, nest, locally refit), with no future task
  labels.
- **Arms:** probe pressure; sham probes (same compute, edits that are never
  scored); no probes. All must pass matched present loss, `D*`, parameter and
  migration-variable gates first. Reorganizability bought with capacity is not
  reorganizability.
- **Endpoint:** the later control-flow acquisition cost, i.e. `C_restructure` in
  the H50 and H51 sense, measured with the downstream scorer held fixed.
- **Kill:** matched gates cannot be passed (the PX8(a) failure), or no 2x
  reduction against sham.

## L8 - PX9: endogenous DSL growth (north star)

Only after at least one promoted control abstraction (L3 or L4) passes the triple
test.
- **Substrate:** impoverished but universal, with no supplied LOOP, IF or MAP.
- **Lifetime:** a supplied structural curriculum (always labelled as supplied),
  or a discovered curriculum as a separate rung.
- **Measure per level:** creation times against the amortization inequality;
  the triple test; later-learning cost.
- **Scoring:** teacher-DSL agreement is diagnostic only.

# Recommended order

The PI's scientific order is preserved; prerequisites decide timing.

| step | what | tier | can run now? |
|---|---|---|---|
| 1 | L0d inference census on J2A libraries (PX7) | 0-1 | yes, no control flow |
| 2 | L0a PX1 world census | 0 | after PI confirms the stop-rule reading |
| 3 | L0b + L0c count-world gates and oracle horizon | 0 | same |
| 4 | SO3 outcome | 2 | running |
| 5 | L1 PX1 economics | 2 | after 2 and 4 |
| 6 | L2 supplied ITERATE | 2 | after 3 and 4 |
| 7 | L3 promoted ITERATE | 2 | after 6 |
| 8 | L4, L5, L6, L7, L8 | 1-2 | per prerequisites |

Steps 1-3 cost minutes to an hour on this host. L1-L3 each need their compute
sized at freeze. Several are likely multi-world, multi-stream lifetimes that
belong on remote workers if they exceed one overnight local batch.

# Known risks, stated before any design commitment

- **PX1's `r_ctrl` knob** is at high risk of moving per-task difficulty (V5.1).
  L0a exists to find out before any learner runs.
- **Count extrapolation on the rotated substrate** can fail from compounding
  numeric drift even for a true loop. L0c bounds the readable counts, and claims
  beyond them are uninterpretable, not negative.
- **The macro impostor** (E6) is the most likely false positive for L3. The
  unseen-count test is mandatory, not a diagnostic.
- **SO3 may fail.** Learner rungs then need an explicit PI decision on offline
  scope; this plan does not pre-authorize it.
- **Search baselines may be strong.** Exhaustive search scaled gracefully in E5
  and E5.1. PX7 and any amortized-inference claim must beat it where it is
  feasible, and state feasibility.

# Decisions requested from the PI

1. Stop rule 2's scope: may teacher-side and oracle-executor gates (L0a-L0c) run
   before online learnability is established? This plan reads "yes".
2. Whether L0d, the inference census on existing straight-line libraries, should
   go first, as proposed here, or wait for PX1's census.
3. If SO3 fails: close the learner rungs, or authorize an offline-scoped variant
   by amendment.
4. Whether L1-L3 should target remote workers from the start.
