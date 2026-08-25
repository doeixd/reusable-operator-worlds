# E0 / Phase 0: does anything here look like a program primitive?

Status: DRAFT (freeze commit in `tools/check_prereg.py` before any code).
Governed by `EXPORT_BRANCH_PROGRAM.md` (Amendment 1) — this rung may not change
that document's decision tree or thresholds. **No new lifetimes.** Existing
artifacts only. The H53 formation line continues untouched and none of its
results may enter here.

# What Phase 0 decides

Nothing about export or composition. It decides three things that gate the rest
of the branch:

1. **Which substrates are eligible for E1** (via the E1.0 gate), so that a later
   STOP is about export rather than about mixture routing.
2. **What the objects may be CALLED** (via E0.1), under the terminology
   contract.
3. **Whether E2 is constructible at all** (via E2-feas), before any world is
   designed around it.

# Artifacts (fixed here, not chosen after seeing results)

Two substrate families, per Amendment 1 section 1:

- **MIX — strongest-loss substrate.** `artifacts/rho_development/rho_1/world_*/continuous`
  (exact-reuse continuous basis, the family that wins lifetime economics), and
  `artifacts/h39c/w_m4/world_*/lifecycle` as the modern parameterized-slot
  member of the same family.
- **DISC — route-expressible substrate.** `artifacts/discrete/seed_0`
  (per-task-annealed hard discrete library; 92.2% exact route recovery, and a
  learned hard route that converges with the matched-slot teacher route).

DISC currently exists for one world only. Registered consequence: DISC results
in Phase 0 are n = 1 and are reported as such; if DISC passes E1.0, E1 requires
two further DISC lifetimes on worlds 1 and 2 from committed code before any E1
verdict is read. That re-run is a lifetime cost, not an audit, and is out of
Phase 0 scope.

# E2-feas — is the compositional test constructible? (run first, costs seconds)

Pure combinatorics on the teacher space, no model involved. For `K = 6, D = 3`
(216 programs) and a lifetime of 64 tasks, enumerate whether a training set
exists that satisfies the E2 coverage constraints while leaving usable held-out
strata:

- every primitive appears; every primitive appears in every position;
- every primitive appears in at least `c_min = 3` distinct surrounding contexts;
- program frequencies balanced (no primitive's count exceeds twice another's);
- **H1** = held-out triples all of whose adjacent pairs were seen;
- **H2** = held-out triples with at least one adjacent pair never seen.

Report `max |H1|` and `max |H2|` achievable, and the training-set size needed.
Registered thresholds: E2 is CONSTRUCTIBLE as specified iff `|H1| >= 16` and
`|H2| >= 16` are simultaneously achievable with a training set of at most 64
programs. If not, E2 must change its generator before it is designed
(more primitives, or depth 4) — and that plan must state that changing `K`
breaks comparability with every existing artifact.

# E1.0 — the oracle-route non-vacuity gate (Amendment 1 section 2)

Freeze the library, supply the TEACHER route, evaluate on programs the lifetime
actually trained on. Per artifact:

    intact NMSE          the model as trained, its own route
    oracle-route NMSE    same library, teacher route through the matched slots
    ratio                oracle-route / intact

Matching from learned object to teacher primitive is FUNCTIONAL (E0.1's
Hungarian assignment), never parameter identity. Registered gate: a substrate is
eligible for E1 iff `ratio <= 2.0`. Rationale is in the branch amendment: the
two known substrates sit at ~1.06 and ~4.9, whose geometric midpoint is 2.28.
A substrate failing E1.0 has its later E1 rows reported as UNINTERPRETABLE
rather than as export failures.

Non-vacuity of the gate itself: a random-assignment route and a shuffled-library
route are computed alongside, and the gate is only read if the teacher-route
ratio is better than both.

# E0.1 — contextual functional substitutability

For learned object `A_i` and teacher primitive `P_j`,

    d(A_i, P_j) = E_{c, x} [ || c[A_i](x) - c[P_j](x) ||^2 ]  / E_{c,x} || c[P_j](x) ||^2

where `c[.]` is a CONTEXT: a position in {1, 2, 3} and a surrounding pair of
teacher operations drawn from a frozen probe set (8 contexts per position,
seeded `SeedSequence([760, world, position])`), and `x` is a fixed probe batch.
Normalization by the teacher operation's own scale is required — the V4.1
lesson: a tolerance divided by total output scale hides everything.

Hungarian assignment over the normalized distance matrix. Reported:

- best assignment and its mean distance;
- **assignment margin**: mean distance of the best assignment against the
  second-best assignment (via the best forbidden-edge alternative);
- per-position distance, and the spread across contexts within a position
  (stability);
- controls: a random permutation assignment, and a shuffled-library assignment.

E0.1 governs TERMINOLOGY only. Registered readings: if the best assignment beats
both controls by a factor of 2 and its per-position distances agree within 50%,
the licensed sentence is "learned objects are functionally substitutable for
stable teacher operations across multiple program contexts." Otherwise the
objects keep their neutral names and E1 still runs — weak teacher alignment does
not falsify export, because the learner may hold a different composable basis.

# E0.2 / E7 — residual load-bearing

On trained tasks of each artifact, four conditions:

    L_full           intact
    L_no_residual    private residual zeroed (library and route intact)
    L_no_library     library contribution removed, residual retained
    L_refit          residual re-fit under the frozen library

    R_residual = (L_no_residual - L_full) / (L_no_library - L_full)

Reported with all four ABSOLUTE losses, always. The ratio is read only when
`L_no_library - L_full > 0` by a factor of at least 2 over `L_full`; otherwise
the denominator is degenerate and only absolutes are reported. Registered
reading: `R_residual` near 0 means task identity lives in the reusable objects;
near or above 1 means the library is a prior and the task program still lives in
private state — a warning, never a stop, since E1 is the direct test.

# Non-vacuity and honesty requirements

- Every artifact is reconstructed COMPLETELY: promoted abstractions, task
  references, and retirement state (the review-55 loader error must not recur).
- Every comparison is made on ONE shared probe and one shared context set per
  world; no per-object coordinate systems (the V5 coordinate error).
- Absolute losses reported beside every ratio.
- Random and shuffled controls for every structural statistic.
- DISC's n = 1 is stated wherever a DISC number appears.

# Registered predictions

Ours: E2-feas is the one we genuinely cannot guess and we say so — the coverage
constraints and 64-program budget may make `|H2| >= 16` impossible, and if it
is, that is a design finding worth the seconds it costs. E1.0: DISC passes
(expected ratio near 1.1), MIX fails (expected ratio 3-5), reproducing the V1
matched-slot measurement on today's loader. E0.1: partial alignment on MIX with
a weak margin; stronger and more stable alignment on DISC. E0.2/E7: this is the
rung we are least able to predict — H39 found the residual channel carries only
about 2% of FAMILY computation, but says nothing about ordinary task identity,
so `R_residual` could land anywhere; we register the uncertainty rather than a
number.

Review 75's: partial functional identity rather than clean one-to-one recovery;
residuals more load-bearing than we would like.

# Cost

Seconds for E2-feas; minutes for E1.0 and E0.1 (forward passes only); E0.2's
re-fit condition is the only one that trains anything, and it trains task-local
state only, on existing artifacts. Phase 0 runs beside the H53 scoring without
competing for memory.


# Amendment 1 (2026-08-25, before any E2-feas verdict is recorded): the training set must be the LIFETIME

The first E2-feas implementation satisfied the coverage constraints with 9-13
programs and reported CONSTRUCTIBLE with `|H1| = 19`, `|H2| = 203`. That reading
is void, and the run is preserved as an instrument dry run rather than a result.

The plan said "a training set of at most 64 programs". That is the wrong
quantifier: E2's lifetime TRAINS ON 64 TASKS, so the training set size is
fixed at the lifetime length, not bounded by it. A 13-program construction leaves
most of the 216-program space unseen for the trivial reason that the learner
never ran; the question the gate exists to ask is whether the held-out strata
survive a REAL lifetime, in which 64 programs cover most of the adjacent pairs.

Amended, before any verdict:

1. **`train_size` must EQUAL the lifetime task count** (64 by default, reported
   explicitly). The search fills to exactly that many distinct programs.
2. The filling objective is registered: after coverage is met, each further
   program is chosen to **minimise newly covered adjacent pairs** (subject to the
   balance constraint), because the adversarial question is whether a lifetime
   CAN be built that preserves an unseen-pair stratum — not whether a random one
   does. A random-fill arm is reported alongside as the neutral reference.
3. Both arms report `|H1|` and `|H2|` at the true train size; the registered
   thresholds (`|H1| >= 16`, `|H2| >= 16`) are unchanged and are read only at
   that size.
4. If the pair-minimising fill reaches the thresholds but the random fill does
   not, the finding is recorded as **E2 is constructible only with a designed
   program schedule**, and E2's plan must then freeze that schedule explicitly
   as part of its generator — a lifetime whose task order was chosen to preserve
   a test stratum is a designed instrument and must be declared as one.
