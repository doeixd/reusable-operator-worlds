# Neural Library Learning
## V2 Experimental Specification: The Economics of Abstraction

**Status: PROVISIONAL DRAFT — not authoritative until the V1 confirmation
gate closes.** The V1 spec (`neural_library_learning_v1_experimental_spec.md`)
remains frozen and governs everything up to and including the seeds 100–129
confirmation run. This document distills the post-V1 consensus from
`notes/` and `reviews/` as of 2026-08-17 (through reviewer-feedback-08) into
one place, so V2 work does not have to be reconstructed from a conversation
thread. Sections marked **[contingent]** depend on V1 results that are not
yet final and must be revisited when the gate closes.

**Date:** August 17, 2026
**Project:** Neural Library Learning / Prospective Neural Compression

---

# 0. What changed since V1 was written

V1 asked: *does a reusable computational substrate reduce lifetime learning
cost?* The development-stage answer is yes, with structure that reframes the
question:

1. **A replicated crossover.** On the paired development sweep (worlds 1-9,
   plus world 0 from the earlier sparse sweep), Continuous wins the paired
   prequential comparison in 0/9 worlds at every configured `rho <= 0.75`
   (mean Dense-minus-Continuous: -2,106 at 0.0; -2,153 at 0.25; -2,234 at
   0.5; -1,306 at 0.75) and 9/9 worlds at `rho = 0.9` (+1,167) and
   `rho = 1.0` (+3,615). Interpolated per-world crossings on worlds 0-2 sit
   at configured `rho` 0.811-0.852. The interesting object is no longer a
   winner but a *boundary*. [artifacts/rho_development/sweep.json]
2. **The advantage is acquired, not innate.** Across all ten development
   worlds, mean 32-shot novel-composition NMSE after 8 lifetime tasks is
   statistically indistinguishable (Continuous 0.0228 vs Dense-C 0.0230;
   Continuous better in only 4/10 worlds). After 64 tasks Continuous leads
   10/10 worlds (0.00343 vs 0.00645; ratio 1.88x). Learning-to-learn
   emerges over the lifetime from an equal start.
   [artifacts/checkpoints_development/sweep.json]
3. **Prior quality is a cost term.** Under the current configurations on
   world 0, the Continuous-over-Dense-C prequential advantage is +4,446
   with decoupled learnable-alpha tanh operators; +3,296 with teacher rank
   raised to 16 against learner rank 8 (reduced but substantial); and +441
   with GELU operators (a ~10x collapse). Representational alignment with
   the world's computational family matters economically; alpha leakage and
   rank alignment matter far less. [artifacts/family_controls/*/summary.json]
4. **A two-level decomposition of "reusable."** The generic hypernetwork
   beats Dense-C but loses to Continuous in every development world (world
   0: Continuous over Hypernetwork +2,820; Hypernetwork over Dense-32
   +1,626): continuous operator structure helps, and an explicit reusable
   basis helps more. Dense-24 vs Dense-32 differs by only +46, closing the
   task-code-dimension concern.
   [reports/structural_controls/structural-controls.json]
5. **Representation != inference.** Hard Discrete learns a near-correct
   library and routes (92.2% exact explained-route recovery on seed 0) but
   pays a large online route-inference tax. Per-task temperature annealing
   improves its world-0 prequential loss from -137,321 to -146,146 but
   still trails Continuous (-170,967) by ~25k nats — the tax is reduced,
   not closed. An explicit identity basis element does not materially help
   Continuous (-170,404 vs -170,967 baseline; slightly worse).
   [artifacts/high_priority_controls/*/summary.json]
6. **A falsified secondary hypothesis.** "Transfer improves before it
   amortizes" did not replicate. The parsimonious picture is one crossover:
   below sufficient recurrence, sharing is harmful bias; above it, useful
   inductive bias.
7. **A resource frontier.** Retained bits (Discrete < Continuous <
   Hypernetwork < Dense-24 < Dense-C), online learning cost, and inference
   compute order the architectures differently. There is no single best
   representation; there is a weighted objective.
8. **Statistical reuse != structural reuse.** At `rho = 0.9`, Continuous
   wins lifetime prequential cost in every paired world yet does NOT
   reliably win frozen-library novel-composition transfer; the transfer
   advantage appears only at `rho = 1`. Lower lifetime cost and clean
   compositional transfer are distinct phenomena: useful parameter sharing
   emerges at partial recurrence, while a cleanly recomposable library
   appears to require exact recurrence. (Note this is NOT the falsified
   early-transfer claim — the ordering is reversed: statistical sharing
   pays before, not after, the library becomes portable.)
   [notes/crossover.txt, six-world and nine-world sweeps]

V2's core question is therefore no longer "does reuse help?" but:

> **Can a learner discover — and progressively revise — the form, granularity,
> and extent of its own reusable computational representation, using the
> economics of lifetime learning cost as the criterion?**

---

# 1. Hypotheses

## H5 — Amortization hypothesis

The crossover is an amortization threshold, not a static bias preference.
If a shared representation costs `C` to acquire and saves `s(r)` per task at
recurrence `r`, reuse pays when `N * s(r) > C`. Predictions:

- **H5a:** the crossover `rho*(N)` declines as lifetime length `N` grows.
- **H5b:** the crossover expressed in *measured functional recurrence*
  aligns more tightly across worlds than in configured `rho`.

**Falsifier:** `rho*(N)` does not move with lifetime length. Then the
crossover is a representational-bias phenomenon and the economic framing is
wrong; the theory section of any paper changes accordingly.

**STATUS (bridge analyses run 2026-08-17, reports/rho_bridge/):**
- **H5a: PARTIALLY SUPPORTED.** Mean-curve crossover: rho* = 0.869 at 16
  tasks -> 0.822 at 32 -> 0.826 at 64. It moves in the predicted direction
  early, then saturates; 8/10 worlds decline from 16 to 64 tasks, net mean
  change -0.043, but the strict monotonic prediction fails. Read:
  amortization is real but confined to early lifetime; beyond ~32 tasks
  the boundary behaves like a stationary representational-bias sign flip.
  The paper's theory section presents amortization as the early-lifetime
  component, not the whole mechanism, and the "confirmed economic law"
  headline is NOT available in its strong form.
- **H5b: HALF SUPPORTED, with an upgrade.** Measured recurrence does NOT
  tighten cross-world crossing dispersion at 64 tasks — but it LINEARIZES
  the dose-response dramatically: the mean effect fits
  `Delta L ~= 5,906 * r - 2,477` with R^2 = 0.97 in measured-recurrence
  coordinates versus R^2 = 0.65 in configured rho. The apparent elbow was
  a coordinate artifact (the elbow test's first branch). Zero-crossing at
  r* ~= 0.42. This linearity is arguably a better headline than the
  threshold story: the sharing effect is a smooth linear function of
  functional recurrence with a sign flip, not a phase-like transition.
  Report both coordinates and the raw world curves regardless.

## H6 — Prior-cost hypothesis

Family mismatch raises the effective acquisition cost `C` of the shared
representation. Prediction: under a mismatched learner family (GELU
operators), the crossover shifts toward higher recurrence rather than the
advantage vanishing outright.

**Falsifier:** no crossover exists at any `rho` for the mismatched learner.
Then prior alignment is a gate, not a cost term, and H5/H6 do not unify.

**STATUS (runs 2026-08-18, artifacts/v2_gelu_crossover/): SUPPORTED IN
DIRECTION, WITH THE CROSSING PUSHED TO THE BOUNDARY.** Paired
Dense-minus-GELU-Continuous effects (tanh in parentheses):
rho 0.75: -3,659/-3,364/-3,873, 0/3 (tanh -1,597 mean);
rho 0.9: -1,142/-1,944/-1,662, 0/3 (tanh +1,298 mean, 3/3);
rho 1.0: +441/-26/-14, 1/3, mean +134 (tanh +3,492 mean, 3/3).
Reading: the return-on-recurrence slope SURVIVES mismatch (the GELU
effect climbs ~3,766 nats over the measured range vs ~5,089 for tanh),
but the line is pushed down by a penalty of 2,035 / 2,881 / 3,359 mean
nats at rho 0.75 / 0.9 / 1.0 — mostly additive with a mild growth in
rho (equivalently ~26% slope attenuation plus intercept drop). The zero
crossing is displaced to r ~= 1.0: at exact reuse the mismatched learner
reaches PARITY with Dense-C, not advantage. The gate-not-cost
alternative is rejected (the effect neither vanishes nor stops rising
with recurrence), but the strong form — a clearly interior displaced
crossing with a GELU win — is only marginally attained (1/3 at rho = 1).
Claim wording for the paper: under family mismatch the economics keep
their shape but the sharing penalty grows to consume the entire
attainable return in these worlds; alignment is a large, approximately
additive cost term, not a gate.

**Strategic weight:** H6 carries more than its share of the paper's
defensibility. The strongest external objection to the whole project is
circularity — "the worlds are generated by residual-tanh compositions and
the winning learner is built from residual-tanh operators." A confirmed H6
converts that objection into a *parameter of the theory* (prior alignment
is a measured cost term with a predictable effect on the crossover) instead
of a caveat. A refuted H6 means the circularity objection stands and every
claim must be stated as conditional on structural prior quality. Either
way, H6's outcome dictates how the paper's central claim is worded; treat
it as co-primary with H5.

## H7 — Premature-commitment hypothesis

The discrete learner's residual online deficit is caused by committing to a
single route hypothesis too early, not by discreteness itself. Prediction:
an exact Bayesian posterior over routes (enumerable: `12^3 = 1728`), with
Bayesian-model-average prediction, closes most of the discrete-vs-continuous
online gap when run over the frozen learned library.

**Falsifier:** exact posterior inference over a frozen, demonstrably correct
library still substantially trails Continuous online. Then discrete
representations carry an inherent online cost in this regime and the
consolidation direction (H8) weakens.

**Fairness asymmetry, stated up front:** the post-hoc posterior is
deliberately ADVANTAGED — it receives the final trained library, whereas
the online learners trained theirs concurrently. This is intentional: the
posterior is an upper bound on discrete online performance. If even the
advantaged bound trails Continuous, H7 is cleanly refuted. If the bound
wins, that alone does NOT establish that an online discrete learner can
reach it; the online (EM-like) variant is then required before any claim
stronger than "the ceiling is high enough." Never present 7a-post-hoc
numbers in the same table as online lifetime numbers without this caveat.

**STATUS (run 2026-08-18, reports/v2_route_posterior/): H7 STRONGLY
SUPPORTED at the advantaged bound, world 0.** The exact posterior over
the frozen per-task-annealed library scores -174,844 prequential
(BMA-point convention; -173,423 MAP-point; -173,913 proper mixture) —
better than online Continuous (-170,967) by ~3,900 nats and better than
the online discrete learner (-146,146) by ~28,700. The discrete deficit
is therefore inference cost in its entirety at this bound; the
representation supports better-than-continuous prediction when inference
is exact. Supporting diagnostics: median 28.5 examples to posterior
entropy < 0.1 nat; entropy drops from the 7.45-nat uniform prior to 5.61
after one example. The premature-commitment reading is reinforced by a
non-identifiability observation: the posterior's MAP route matches the
online learner's hard route on only 25/64 tasks (67% per position) while
massively outperforming — several routes are behaviorally
near-equivalent, and averaging over that ambiguity is precisely what
hard routing cannot do. Caveats stand: advantaged bound (final library),
one world; extend to worlds 1-2 libraries when per-task-anneal artifacts
exist for them. Consequence for H8: consolidation's compilation gate has
its instrument (the posterior) validated, and 28.5 examples is a first
empirical anchor for how much evidence "sleep" should demand.

## H8 — Consolidation hypothesis **[contingent on H7 direction]**

A learner that infers softly and consolidates ("wake/sleep": maintain a
continuous/posterior task representation online; compile high-confidence
programs into hard routes at checkpoints) retains Continuous-level online
learning while approaching Discrete-level storage and inference cost.
Predictions:

- **H8a:** post-consolidation tasks are learned no more expensively than in
  the never-consolidated continuous run, while retained bits and inference
  compute drop toward the discrete point on the frontier.
- **H8b (the critical falsifier):** at `rho = 0`, a correct consolidation
  criterion *refuses to compile* (or compiling measurably hurts). A
  criterion that fires at `rho = 0` is broken.

**Decision branches, written before the first run:** if consolidation wins
prequentially at high `rho` and refuses at low `rho`, it is the headline.
If it merely matches online while winning storage/inference, the honest
claim is compression, not learning speed (V1 spec §41's own branch). If it
damages subsequent learning even at `rho = 1`, report it as a bound on when
compilation is safe.

## H9 — Selective-sharing hypothesis

A learner with a shared parent plus penalized task residuals,
`P_{tau,k} = P_k + Delta_{tau,k}`, chooses its own degree of sharing. Two
predictions:

- **H9a (allocation, mixed worlds / Benchmark D):** residual capacity is
  allocated selectively — `||Delta_{.,k}||` small for high-recurrence
  primitives, large for low-recurrence primitives — and the model beats
  both fixed architectures on lifetime cost in mixed worlds.
- **H9b (lower envelope, homogeneous sweep):** across the existing
  homogeneous `rho` grid, the same model tracks the lower envelope of
  Dense-C and Continuous — at or below the better of the two at every
  `rho`, with residual magnitude decreasing in `rho`. This is the first
  concrete demonstration that a learner can DISCOVER how much
  computational structure deserves to be shared, rather than having the
  answer fixed architecturally; graphically, the hybrid curve sits under
  the Dense/Continuous crossing.

**Falsifiers:** residual magnitude does not track per-primitive recurrence;
or the model fails to beat both Dense-C and Continuous in mixed worlds; or
the residual becomes an escape hatch (see the Model 5 escape-hatch guard
in §4) and the win is
Dense-C in disguise.

**STATUS (homogeneous-rho control run 2026-08-17,
reports/shared_residual/): H9b STRONGLY SUPPORTED on worlds 0-2.** The
rank-2 shared-residual model beats the best fixed architecture (the
envelope) by +9,168 / +7,458 / +3,745 mean nats at rho 0.5 / 0.75 / 0.9,
3/3 worlds on both loss and novel transfer, degrading to parity (-246) at
rho = 1.0. The allocation signature holds: mean residual functional ratio
declines monotonically 0.284 -> 0.245 -> 0.160 -> 0.026 with rising
recurrence, under the escape-hatch guard (max task ratio 0.47 vs cap 1.0).
**J-WEIGHTED UPDATE (reports/shared_residual/j-weighted.json): the caveat
bit.** Under the two-part MDL code (lambda = ln 2 nats/bit), the shared-
residual model loses to BOTH fixed architectures in 12/12 world-rho cells:
its 130,624 retained bits (~101k more than Continuous) cost ~70k nats,
swamping the 3.7k-9.2k nat prequential gains. Break-even lambda* is only
0.037-0.143 nats/bit. Revised H9b status: SUPPORTED on prequential loss
and on the allocation signature; NOT supported under description-length
accounting at the current residual budget — the envelope win is partly
purchased per-task capacity. Honest claim wording: "a prediction-cost win
with the predicted allocation behavior, not (yet) a code-length win."
Constructive paths: rank-1/stronger-penalty residuals, and sparsity-aware
retention coding (near-zero residuals at high rho are charged full 8-bit
width under the current proxy — an entropy code would price them near
their information content and could rescue the high-rho cells). Any public
figure reports J at multiple lambda. H9a (per-primitive allocation in
mixed worlds, Benchmark D) remains untested.

**H9a measurement procedure, frozen before any mixed-world run** (added
after external review; the identifiability problem is real and V1's own
recovery result predicts it — below the crossover, slot-to-primitive
correspondence is exactly what fails):

- **Primary attribution is by true-route position, not slot matching.**
  Each task-step residual R_{tau,l} is attributed to the teacher
  primitive occupying position l of that task's hidden program. This is
  well-defined regardless of whether the learner's slots recover the
  primitives, uses ground truth only as a post-hoc diagnostic (exactly
  like operator recovery), and is therefore the H9a statistic.
- **Secondary (descriptive only): Hungarian slot matching** on functional
  distance over fixed probes, reported per world with an ambiguity flag:
  a slot's assignment is AMBIGUOUS when its best-match distance exceeds
  half its second-best (margin < 2x). Ambiguous assignments are reported
  and count AGAINST the secondary signature (never excluded); they do not
  affect the primary statistic.
**H9a STATUS (canonical mixed-profile grid run 2026-08-18,
reports/v2_mixed/): SUPPORTED, 10/10 on both measures.** Allocation: the
per-world Spearman between per-primitive measured recurrence and
per-primitive mean residual fraction (frozen true-route-position
attribution) is negative in 10/10 development worlds (range -0.43 to
-1.00; sign test p = 1.95e-3). Envelope: shared-residual beats
min(L_Dense, L_Continuous) in 10/10 mixed worlds, mean +7,153 nats; the
two-part code reverses it 0/10, exactly as the homogeneous development
result predicted — the prediction-cost-not-code-length pattern
reproduces in the heterogeneous setting. Per the pre-registered decision
rule in V2_CONFIRMATION_PLAN.md, Component B of the sealed confirmation
TRIGGERS. The learner allocates specialization inversely to ground-truth
recurrence, primitive by primitive, in worlds where no fixed
architecture is correct.

- **Aggregation, pre-specified:** per world, the Spearman rank
  correlation between per-primitive recurrence and per-primitive mean
  residual fraction is computed over the 6 primitives (descriptive; six
  points is underpowered alone). The INFERENTIAL unit is the sign of the
  per-world correlation across the 10 development worlds, tested with the
  exact binomial sign test — the same discipline as every other claim in
  this program. Pooled correlations are reported but carry no inferential
  weight.

## H10 — Within-lifetime amortized inference **[contingent on H7]**

A compiler network trained only on this lifetime's own solved tasks
(hindsight wake/sleep: expensive inference during wake produces
`(D_tau, pi_tau*)` pairs; sleep trains `C_phi(D, P) ~ pi*` behaviorally)
makes later tasks in the same lifetime cheaper than code-only gradient
descent from scratch.

**Falsifier:** compiler-initialized late tasks are no cheaper than the
gradient-descent baseline. 64 tasks may simply be too few; report the
negative result and defer amortization to cross-world scale (Phase III).

**STATUS (runs 2026-08-18, artifacts/v2_compiler/): NOT SUPPORTED at
64-task scale; the pre-registered fallback applies.** The warm-started
compiler LOSES to plain gradient descent 0/3 worlds at both endpoints
(mean -2,981 nats at rho = 1; -1,052 at rho = 0). Dream augmentation
halves the damage but never reaches positive (mean -1,426 at rho = 1).
Decisively, the protected dream falsifier PASSES: the dream benefit is
+1,554 at rho = 1 versus +512 at rho = 0 — three times larger where the
library matches the world — so the compiler demonstrably learns the
world's program distribution; within-lifetime data is simply
insufficient for one-shot amortization to beat 128 steps of direct
optimization. Verdict wording: within-lifetime amortized inference is
mechanism-sound but underpowered at this lifetime length; deferred to
cross-world scale (Phase III) as pre-registered.

Explicitly **not** V2 hypotheses: dynamic fork/merge under nonstationarity,
macros/loops, programmable perception, learned optimizers, cross-world
meta-training. See §7.

---

# 2. Bridge analyses (zero new training; may run before confirmation)

These are analyses of existing artifacts and logs. They inform V2 design and
none of them touches sealed worlds.

1. **Crossover in measured-recurrence coordinates** (tests H5b). Interpolate
   each development world's loss crossover against measured functional
   recurrence instead of configured `rho`. Report both coordinates always.
   Include the **elbow test**: the configured-`rho` effect curve is flat
   from 0 to 0.5 and moves sharply after — if the measured-recurrence
   version becomes approximately smooth, the apparent thresholdiness was a
   coordinate artifact of how parameter correlation maps to functional
   similarity; if a sharp elbow survives in measured coordinates, that is
   a deeper phenomenon and must be reported as such.
2. **`rho*(N)` from truncated lifetimes** (tests H5a cheaply). Prequential
   loss is logged per example; compute the Dense-minus-Continuous cumulative
   difference at `t = 2048, 4096, 8192` from existing runs and interpolate
   the crossover at each truncation. A truncated reading is a valid "lifetime
   that ended at N," not an approximation. Confirmatory-grade versions with
   genuinely shorter/longer lifetimes (32 and 128 tasks at two bracketing
   `rho` values) come later only if the free version shows movement.
3. **Route-posterior entropy curves** (instrument for H7). With the frozen
   final discrete library, replay task streams and record `H(R | D_t)` per
   example — the information each demonstration provides about the program.
   Deliverable figure regardless of H7's outcome.
4. **Checkpoint divergence figure.** Paired Dense-minus-Continuous 32-shot
   novel NMSE vs tasks completed, all ten development worlds: the
   equal-at-8, diverged-at-64 shape is the acquired-advantage result and
   replaces all gain-ratio quotations.
5. **Operator recovery across `rho`** (explains §0 item 8; near-free). The
   sweep summaries omit functional recovery (fast-tuning mode), but every
   run saves `model.pt`, so primitive-matching quality can be computed
   post-hoc from saved models — no retraining. Plot recovery quality
   against `rho` alongside the lifetime and novel-transfer effects.
   Prediction: mediocre recovery at `rho = 0.9` despite the lifetime win;
   sharp recovery at `rho = 1.0`. If confirmed, the interpretation is that
   partial recurrence favors a shared low-dimensional MANIFOLD while exact
   recurrence crystallizes IDENTIFIABLE PRIMITIVES — which also predicts
   the hypernetwork (pure manifold, no slots) should close more of its gap
   to Continuous at `rho = 0.9` than at `rho = 1.0`, a cheap corollary
   check worth running with it.
   **STATUS (run 2026-08-17, reports/rho_operator_recovery/): CONFIRMED,
   with a sharper shape than predicted.** Against an untrained-basis
   baseline of 0.0087, trained operators sit FARTHER from the shared
   primitives than untrained ones at `rho <= 0.5` (0.0115-0.0121), match
   baseline at 0.75 (0.0083), first drop below it at 0.9 (0.0048), and
   crystallize at 1.0 (0.0017). Primitive-recovery ONSET coincides with
   the performance crossover; partial recovery at 0.9 quantifies the
   statistical-vs-structural dissociation (§0 item 8). The untrained
   baseline is mandatory context for any recovery number quoted anywhere —
   absolute distances without it are uninterpretable (the family is
   already ~0.0087-close by construction).
   **Hypernetwork corollary (002b) STATUS (run 2026-08-18,
   artifacts/v2_hyper_rho09/): CONFIRMED, stronger than predicted.** The
   Continuous-over-Hypernetwork gap is smaller at rho = 0.9 than at 1.0
   in 3/3 worlds — and in 2/3 worlds the hypernetwork BEATS Continuous at
   0.9 (-1,496 and -943 world 0/2; +917 world 1) while losing in all
   worlds at 1.0 (+2,820 / +1,120 / +2,102). There is a second,
   representation-level crossover: the slotless operator manifold is the
   better substrate at partial recurrence, and explicit slots pay only
   where identifiable primitives exist. This is an independent-instrument
   confirmation of the manifold-vs-crystallized-primitives account and
   should appear alongside the recovery-onset result wherever the
   dissociation is presented.

---

# 3. Benchmarks

## Benchmark D — mixed-recurrence worlds (the V2 workhorse)

Per-primitive reuse levels within one world:

    rho_k = (1.0, 0.95, 0.8, 0.5, 0.2, 0.0)   [canonical profile]

plus a uniform-high and uniform-low profile as anchors. Neither fixed
architecture is correctly matched: a fully shared basis wrongly shares the
low-`rho` primitives; a dense learner under-exploits the high-`rho` ones.
The learner must discover the *factorization structure*. Implementation is a
per-primitive `rho` vector in the existing generator — no new statistical
machinery, and all V1 validity gates carry over, with these specifics:

- **Generation:** `_task_library` takes `rho_k` per primitive; the
  spectral renormalization after mixing is unchanged. Seed scheme: mixed
  worlds use the existing development seeds 0-9 with the profile encoded
  as an explicit SeedSequence component (never Python `hash`), so paired
  models and paired profiles share underlying draws where intended and
  differ only where the profile differs.
- **Validity gates before any model comparison:** (a) per-primitive
  measured functional recurrence must reproduce the profile ordering, with
  the endpoints validating as in V1 (correlation ~0 at `rho_k = 0`, ~1 at
  `rho_k = 1`); (b) scratch difficulty flat in task index; (c) output
  scale/saturation checks unchanged; (d) confirm program sampling does not
  correlate primitive usage with task index (uniform-usage check), since a
  usage/recurrence confound would mimic selective sharing.
- **Anchors:** uniform-high (`rho_k = 0.9` all k) and uniform-low
  (`rho_k = 0.25` all k) profiles run alongside the canonical mixed
  profile, so mixed-world effects are interpreted against matched
  homogeneous baselines rather than against V1 runs with different seeds.

Benchmark D **replaces Benchmark C in priority**. C (prospective macro
regimes under a hidden Markov process) is parked: it requires nonstationarity
machinery and new validity controls that mixed-`rho` worlds do not, and H9
motivates selective sharing without it.

## Benchmark E — hierarchical recurrence **[after D]**

Operators generated as `P_{tau,i} = P_global + P_family(i) + Delta_{tau,i}`
with a two-family structure. The optimal code is hierarchical; the learner
must discover how many levels of shared structure exist. Only after
Benchmark D produces its H9 verdict.

**E's second job (pre-registered post-hoc, zero extra compute): the
promotion testbed.** In mixed worlds the low-recurrence primitives carry
nothing shared, so correct behavior is allocation (H9a). In hierarchical
worlds the FAMILY components are structure that is neither global nor
task-specific — exactly the stratum V3's residual-promotion mechanism
exists to capture. On the E artifacts: compute functional similarity of
task-step residuals ACROSS tasks and test whether residuals cluster by
family (within-family residual similarity > cross-family, per world).
If they cluster, promotable structure is measurably present in the
residuals and the draft V3 hypothesis ("promotion wins both currencies
in hierarchical worlds") is scoped before it is built. If they do not,
that is an early warning that rank-2 residuals are too idiosyncratic a
vessel for shared structure, and the V3 spec must address the vessel
before the mechanism. Both outcomes are reported.

## Parked benchmarks

- **Benchmark C** (macro regimes / nonstationarity): third in line, after D
  and E, inheriting whatever consolidation and descriptor machinery exists.
- **ROW-P** (programmable perception over object sets): parked behind its
  own spec-with-falsifiers per `notes/perception.txt` rev 2, and behind the
  V2 gate. Its bandwidth-limited-read idea is not needed for H5–H10.
- **Benchmark L** (iteration / loops): parked for Phase III, but the design
  constraints from `notes/on-loops.txt` are binding on whoever builds it
  and are recorded here so they survive:
  1. **The fixed-point trap.** The current bounded-tanh teacher converges
     under repeated application, so `P^10 ~= P^20` and loop count becomes
     unidentifiable — a learner would "extrapolate" trivially because
     nothing happens after a few iterations. Loop worlds need operations
     whose repeated application stays behaviorally distinguishable
     (rotations, permutation-like maps, near-identity residuals), with the
     validity gate `d(P^n, P^{n+k})` remaining substantial over the tested
     range. Without this gate the benchmark is fake.
  2. **Loops must be economical, not just expressible.** If every task
     uses the same iteration count, a fused specialized operator `Q_n` is
     the RIGHT abstraction and no loop should emerge. Loop discovery only
     becomes the shortest description when iteration counts vary across
     tasks — the depth analogue of the amortization argument, with its own
     predicted crossover `n*` (unroll/fuse below, loop above).
  3. **The hard-to-fake test is count extrapolation:** train on
     `P^2..P^6`, test `P^10, P^20, P^50`. Description length of a true
     `Repeat(P, n)` abstraction grows with `log n`, not `n`; route
     memorization breaks. (Subject to gate 1, or the extrapolation is an
     artifact.)
  4. **Interpret vs compile is the deep question.** For reducible
     operations (e.g. rotations: `R_theta^n = R_{n*theta}`) compiling a
     shortcut beats looping; for irreducible-ish ones repeated execution
     is the best available strategy. Generate both families and sweep
     `lambda/mu` in `J = L + lambda*D + mu*C`: the learner should cross
     between interpreter-like and compiled representations as the
     memory/compute price ratio changes. The question is not "can it
     loop?" but "does its computational organization adapt rationally to
     resource economics?"
  5. **The novelty bar is high and known:** ACT, PonderNet, and Universal
     Transformers already do variable computation with halting; depth-
     recurrent Transformers already reuse one block with stabilization
     tricks. The contribution cannot be "call a block repeatedly until
     HALT." It must be control abstractions DISCOVERED from execution
     traces (run-length / repeated-subsequence / anti-unification
     compression during sleep is a sufficient v1 compiler), shown to make
     later programs shorter and later learning cheaper.
  6. **Closure under repeated execution** is a real prerequisite: a
     primitive learned on states `z ~ D` sees out-of-distribution states
     after several self-applications, so the workspace interface must be
     stable under composition (the "neural ABI" concern) or the learner
     acquires `P7-after-P3-at-depth-4`, not a callable `P7`.
  The taxonomy worth keeping from the same note: spatial reuse (share
  across tasks — V1/V2's subject), temporal reuse (loops — Benchmark L),
  structural reuse (macros/subroutines — Benchmark C's successor). A
  self-organizing computational language eventually needs all three; V2
  deliberately tests only the first.

## The universality ladder (why expressivity is deliberately closed off)

The benchmark ladder doubles as an expressivity ladder, and the spec's
position on it should be explicit:

- **Universality is table stakes, not a goal.** By Bohm-Jacopini,
  sequence + branching + unbounded iteration over readable/writable state
  suffice for any computable function; single-instruction computers, SKI
  combinators, and NAND show that tiny instruction sets are already
  universal. Six learned operators is a lavish instruction set. The moment
  the ladder reaches input-dependent HALT plus growable memory (the
  ROW-P-era "learned VM"), the substrate is Turing-complete in the same
  formal sense RNNs are — and that fact will be the LEAST interesting
  thing about it.
- **V1/V2 remove universality on purpose.** Fixed program length 3 makes
  the function class finite, so the economics of sharing can be measured
  without expressivity confounds. Each later rung (fixed REPEAT ->
  input-dependent HALT -> memory/branching) reopens a controlled amount of
  expressivity and re-asks whether the code-length criterion still selects
  the right structure. Universality arrives last because it is the only
  property acquired for free.
- **Neural universality has caveats symbolic universality does not,**
  and they are already visible in the V1 data: (a) finite precision makes
  any bounded-state system formally a finite automaton — unbounded memory
  must be an explicit growable store, never packed into the workspace
  vector; (b) approximate operators compound error under composition, so
  long programs require learned stability around used trajectories (the
  Benchmark L closure gate); (c) the current contractive tanh-residual
  family destroys input information under repetition (the fixed-point
  trap) — universal computation needs information-preserving primitive
  families (rotations, permutations, near-identity residuals), which is an
  instruction-set-family constraint, not a tuning detail.
- **Expressible != learnable is the project's actual subject.** Turing-
  completeness and universal approximation say nothing about whether
  gradient descent plus MDL pressure will FIND a program; V1's discrete
  result (full expressive machinery, near-perfect library, inference as
  the bottleneck) is the small-scale proof of that gap. Given a universal
  substrate, the open question is which programs experience makes cheap to
  infer and encode — that is the question every benchmark on this ladder
  is built to isolate.

---

# 4. Models

## Model 7 — posterior-route discrete (exact, then amortized)

- **7a (exact):** enumerate all routes; maintain `p(r | D_t)` online with the
  fixed Gaussian likelihood; predict by Bayesian model average; report MAP
  as an ablation. First run post-hoc over the frozen final library (clean);
  an online EM-like variant (library trains under posterior-weighted
  updates) only if 7a-post-hoc is promising.
- **7b (amortized, tests H10):** Set-Transformer-style encoder from the
  demonstration set to a route distribution / continuous code, trained
  behaviorally (query-set loss, weights favoring small support counts, an
  information bottleneck `beta * D(c)` on the task code so the code cannot
  smuggle the training set). Within-lifetime training data only (§6.1).
- **7c (hybrid):** amortized initialization plus 1–3 refinement steps.
- **7b-dream (dream-augmented compiler):** augment 7b's training set with
  FANTASY tasks: sample random routes over the learner's OWN current
  library, execute them to generate synthetic `(x, y)` demonstration sets,
  and train the compiler on real solved tasks plus fantasies (DreamCoder's
  dream phase, ported to neural operators). This is legal under §6.1 —
  fantasies are generated within one lifetime from the learner's own
  library, touching no external worlds and no sealed data — and it
  directly attacks 7b's data scarcity (64 real tasks is a thin compiler
  training set; 64 plus unlimited fantasies is not).
  Prediction with its own falsifier: the dreaming BENEFIT should track
  recurrence — fantasies from a well-matched library at `rho = 1` resemble
  real tasks and should help; fantasies at `rho = 0` are structured noise
  and should not (and may hurt). A dream benefit that fails to shrink as
  `rho` falls indicates the compiler is learning something other than the
  world's program distribution — investigate before trusting it.
  Controls: match total compiler training compute between 7b and 7b-dream
  (fantasy data is free; training on it is not), and report the
  fantasy:real ratio — an unbounded ratio is a hidden-complexity violation
  of §6.2 in spirit, since compiler quality bought with unlimited dream
  compute must be charged to C.

## Model 8 — consolidating learner (tests H8)

Continuous learner plus a sleep step at the existing checkpoints
(8/16/32/64 tasks). The procedure must be fully specified before the first
run; the following is the v1 procedure, amendable only before any result
is inspected:

1. **Library identification.** The discrete library IS the continuous basis
   (the 8 operator slots), unchanged. No separate distillation network in
   v1: compilation means replacing a task's soft mixture with hard calls
   into the same operators. (A merged/deduplicated distilled library is a
   later variant, gated on descriptor tooling.)
2. **Route posterior.** For each completed task, enumerate all `8^3 = 512`
   hard routes over the basis, score them on that task's stored replay
   examples plus the task's training set under the fixed Gaussian
   likelihood, and form `p(r | D_tau)` (Model 7a machinery, smaller
   enumeration).
3. **Compilation gate.** Compile task tau iff
   `H(R | D_tau) < H_threshold` AND the MAP route's evaluation NMSE is
   within a factor `kappa` of the soft mixture's NMSE on held-out
   evaluation data. Both `H_threshold` and `kappa` are frozen from
   development worlds 0-2 before any `rho`-sweep consolidation run.
   Uncompiled tasks keep their soft codes — partial consolidation is the
   expected state, not a failure.
4. **Accounting.** A compiled task's retained state is `3 * log2(8) = 9`
   bits (categorical route) instead of 24 int8-quantized logits
   (24 * 8 = 192 bits); report
   the mixed retained-description total. Sleep-step compute (enumeration +
   any replay fine-tuning) is reported as peak search cost, per §6.2.
5. **Continue the lifetime** from the partially compiled state; shared
   operators keep training as before.

Measured against the never-consolidated continuous run on paired worlds at
`rho in {0, 1}` first (H8a at 1, H8b at 0 — at `rho = 0` the gate should
fire rarely or not at all, and forcing compilation there should hurt;
report the gate's firing rate at both endpoints).

**GATE V1 STATUS (grid run 2026-08-18, artifacts/v2_consolidation/):
MIS-CALIBRATED WITH INVERTED SHAPE; one documented re-derivation
invoked.** Firing rates by rho: 43% / 48% / 48% / 30% / 7% / 0% at
0.0-1.0 — monotonically DECREASING, the opposite of the pre-registered
prediction. H8b fails as frozen (fires where no discrete structure
exists), H8a untestable (never fires at exact reuse; the consolidated
run is identical to continuous there). Mechanism: kappa is a RELATIVE
criterion, cleared most easily where the soft baseline is worst; and at
exact reuse a single hard route through the learner's own basis cannot
match its compensatory mixtures (consistent with V1's
mixture-beats-forced-route finding). The forced-compile control confirms
the harm the gate should prevent is real (+1,208 nats at rho = 0). The
lifetime damage of wrong compilation under the gate was nonetheless
small (mean deltas within ~650 nats; the rho = 0 world-0 cell is
excluded as stale-baseline-suspect: its legacy reference of -137,503 is
off-scale versus siblings at -142k). RE-DERIVATION (once, documented):
the relative kappa test is replaced by the ABSOLUTE sufficiency
criterion already frozen in the MDL-gating study — compile only if the
MAP route's held-out evaluation NMSE <= 0.02 — with H_threshold = 0.1
nat unchanged. Grid v2 runs under artifacts/v2_consolidation_gate2/.

**GATE V2 / FINAL MODEL 8 STATUS (grid run 2026-08-18,
artifacts/v2_consolidation_gate2/): H8b PASSES, H8a WEAK-TO-NEGATIVE;
Model 8 recorded as a characterized negative for this gate family.**
Firing rates 0/0/0/0/1/7 percent across rho 0-1 — the pre-registered
monotone shape holds and the gate refuses all compilation below
rho = 0.9 (H8b). But at exact reuse only 2-8 of 64 tasks compile (mostly
at the final sleep), storage gains are trivial, and firing cells cost
0-404 nats (H8a not achieved). Mechanism, closing a loop with H7: the
overcomplete basis contains behaviorally near-equivalent routes, so
posterior entropy stays above threshold even where compilable structure
exists — entropy measures route IDENTIFIABILITY, not COMPILABILITY. The
two gate derivations bracket the design space (relative bars fire
wherever the soft baseline is weak; entropy bars barely fire at all),
and with the single re-derivation spent, the verdict stands. The
pre-registered candidate for any future gate is predictive equivalence
of the posterior's top routes (BMA-vs-MAP behavioral gap) rather than
entropy. Decision-branch outcome per §1 H8: the honest claim is the
third branch — a bound on when compilation is safe — plus the design
lesson; consolidation as specified does not yet deliver the
storage/learning trade.

**Gate shape prediction, pre-registered** (added after external review:
`H_threshold` and `kappa` frozen from only three worlds could pass H8b by
accident of those worlds). A gate that is correct for the right reason
must satisfy more than the endpoints: its firing rate, evaluated
post-hoc across the FULL rho grid on worlds 0-2, is predicted to be
NON-DECREASING in rho — near zero at `rho <= 0.25`, intermediate in the
crossover region, near complete at `rho = 1`. A gate that passes the
endpoints but fires non-monotonically in between is treated as
mis-calibrated, not as H8b-passing, and the thresholds are re-derived
(once, with the re-derivation documented) before any sweep result is
interpreted.

## Model 5 — shared parent + residual (tests H9; from V1 spec §17)

`Q_{tau,l}(z) = Q_{q_tau,l}(z) + R_{tau,l}(z)` with low-rank residuals and an
explicit storage penalty.

**Escape-hatch guard (mandatory):** the residual must not be able to
implement the task alone. Rank small (<= 2 to start), penalty `lambda_Delta`
tuned symmetrically, and the diagnostic `||Delta_{tau,k}|| / ||P_k||`
reported per primitive against that primitive's measured recurrence. The
signature result is residual mass concentrating on low-`rho` primitives; a
flat allocation with a winning loss means the model quietly rebuilt Dense-C
and the result is void.

## Deferred model machinery

Functional descriptors (content-addressed operations), program memory and
retrieval, learned update rules `U_phi`, temporary-residual promotion, and
MDL presence gates (V1 Model 4 — pending an explicit decision on whether
hard-discrete "works," per SPEC_AUDIT.md) are design assets for Phase III.
Descriptors become mandatory the moment any experiment lets the library
change size or membership.

---

# 5. Metrics and objective

Unchanged core: cumulative prequential Gaussian log loss (and its
quantized-target form), examples-to-criterion, frozen-library novel
composition at checkpoints, functional/route recovery, measured functional
recurrence, evaluated int8 retained description, and multiply-add
accounting. Additions:

- **The crossover surface** is the object of study: crossover location as a
  function of (measured recurrence, lifetime length, prior alignment).
  H5a/H5b/H6 each pin one axis.
- **The weighted objective** `J = L_preq + lambda*D + mu*C` graduates from
  reporting convention to selection criterion in exactly one place: Model
  8's compile/don't-compile decision. Report results at more than one
  `(lambda, mu)` setting; do not tune a single magic weighting.
- **Reporting rules** (from the reviews, now binding): per-example nats/bits
  alongside totals; paired per-world deltas; bootstrap intervals only at
  n >= 10 worlds; "crossover," never "phase transition"; falsified
  hypotheses reported as such; training-compute vs inference-compute
  matching stated explicitly wherever "compute-matched" appears.

## Statistical plan

The world remains the unit of replication; all primary comparisons are
paired on identical worlds.

- **Primary estimator:** paired per-world difference in cumulative
  prequential Gaussian log loss, reported as mean, median, and the full
  set of per-world values. At n = 9-10 development worlds, a unanimous
  sign pattern (as in the current sweep) is reported as an exact binomial
  sign test (9/9 wins: two-sided p = 2 * 0.5^9 = 2^-8 ~= 0.0039); bootstrap percentile
  intervals are reported at n >= 10 and labeled exploratory until the
  sealed block.
- **Crossover estimation:** per-world linear interpolation between the two
  bracketing grid points, in BOTH configured-`rho` and measured-recurrence
  coordinates; the cross-world dispersion of the crossing is the H5b
  statistic. Interpolated crossings are descriptive; no parametric curve
  is fit at development n.
- **H9 primary statistic:** the rank correlation (per world) between
  per-primitive measured recurrence and per-primitive mean residual
  fraction `||Delta_{.,k}|| / ||P_k||`, plus the paired lifetime-loss
  comparison against both fixed architectures. Both must hold for H9.
- **Examples-to-criterion:** right-censored at 128 as in V1; the
  model-by-lifetime-position interaction (V1 spec §34's mixed-effects
  form) is reserved for the sealed block, where n supports it.
- **Multiple comparisons:** V2 tests six named hypotheses; confirmatory
  claims on seeds 200-229 will pre-register one primary outcome per
  hypothesis and Holm-correct across that fixed family. Development-stage
  results are never corrected and never called confirmatory.

## Sealed-block point predictions (parameter replication, not just sign)

V1's confirmatory outcomes were sign tests; V2's sealed block (seeds
200-229) can do better, because V1 supplies parameter estimates. Frozen
now, before any seed in 200-229 exists:

- **Slope:** the pooled Dense-minus-Continuous effect against measured
  recurrence on seeds 200-229 will have slope in **[4,000, 7,500] nats
  per unit recurrence** (V1 sealed estimate: 5,716; development: 5,906).
- **Zero crossing:** the pooled-fit root will fall in **[0.40, 0.60]**
  measured recurrence, and the mean of per-world interpolated crossings
  in **[0.42, 0.58]** (V1 sealed: pooled root ~0.46, per-world mean
  0.499; development bridge: ~0.42-0.46 — the dev-vs-sealed drift is
  itself why these are intervals, not points).
- **Linearity:** pooled R^2 in measured-recurrence coordinates >= 0.85
  and exceeding the configured-rho R^2 by >= 0.15 (V1 sealed: 0.935 vs
  0.642).

Falling outside any interval is reported as a parameter-replication
failure even if all sign tests pass. Passing upgrades the claim from
"the phenomenon reproduces" to "the parameters replicate" — the
difference between reproducing an effect and having a measured law.

---

# 6. Constraints carried forward as law

## 6.1 The lifetime/meta-learning line

Any component trained across worlds changes the claim from lifetime learning
to meta-learning. V2 stays on the lifetime side: compilers, update rules,
and consolidation train only on the current lifetime's own experience.
Crossing the line (Phase III, Kaggle scale) requires: equivalent
meta-training budget for dense baselines, held-out evaluation worlds, and
family-mismatch worlds in the evaluation set to bound generator leakage.

## 6.2 No hidden complexity

Task state is charged (bits); controller/compiler capacity is charged (D);
residuals are rank- and penalty-bounded; consolidation cannot free-ride on
unmeasured search compute (peak search cost is reported when wake inference
is expensive). Any new component's capacity must appear in some term of J.

## 6.3 Family-mismatch controls from day one

Every V2 model family ships with its GELU/rank-mismatch variant in the
smoke config. The V1 lesson: this control changed the interpretation more
than any other single run.

## 6.4 Development/confirmation separation

Development worlds 0–9 (and mixed-`rho` development worlds to be drawn from
the same range with distinct sub-seeds). A fresh sealed confirmatory block
(seeds 200–229) is reserved for V2 claims and remains uninspected until V2
configurations, metrics, and exclusions are frozen. V1's seeds 100–129
remain dedicated to V1 confirmation only.

## 6.5 Engineering prerequisites

Batch `forward_tasks` by task ID before any Benchmark D sweep (it is the
standing compute bottleneck and a precondition for useful GPU throughput).
Kaggle sessions require device-agnostic tensors, pinned dependencies, and
checkpoint/resume within the session limit. The V1 batch-size deviation
(effective batch 2 vs suggested 8) was ablated in V1: the advantage
survives at batch 8 but shrinks ~40% (worlds 0-2: +3,463 -> +2,061). V2
keeps batch 2 for continuity with every V1 number; if any V2 experiment
adopts batch 8, every V1-vs-V2 magnitude comparison carries the ~40%
caveat explicitly.

---

# 7. Explicitly outside V2

- Dynamic fork/merge/delete under nonstationarity (Phase III; inherits
  Model 8's consolidation operator and the descriptor machinery).
- Macros, loops, variable-depth execution, HALT (the identity-operator
  ablation was run in V1 and did not materially help; variable depth waits
  for a world that demands it). The binding design constraints for the
  eventual loop benchmark are preserved under Benchmark L in §3.
- Programmable perception / ROW-P (own spec first).
- Cross-world meta-training (Phase III, §6.1 controls).
- Learned optimizers beyond Model 7b/7c's task-code updates.
- Any claim about ARC-AGI or external benchmarks. The north-star ladder
  (ROW -> structured outputs -> variable-length programs -> amortized
  inference -> interactive worlds) is directional context, not scope.

---

# 8. Implementation order

Gate zero: **V1 confirmation closes first** (SPEC_AUDIT.md execution order,
then seeds 100–129). No V2 training before that, with one exception — the
§2 bridge analyses, which consume no compute and no sealed data.

1. **B1–B4** Bridge analyses (§2). Update H5/H6 status; revise this spec.
2. **001** Model 7a exact posterior, post-hoc on frozen artifacts. H7
   verdict. (Also supplies Model 8's compilation gate.)
3. **002** GELU crossover-shift: bracketing `rho` points, worlds 0–2 (H6).
4. **002b** Hypernetwork-gap corollary (promoted per external review — the
   only independent mechanism check on the manifold-vs-primitives story):
   train the hypernetwork at `rho = 0.9`, worlds 0-2, and test whether its
   gap to Continuous is smaller at 0.9 than at 1.0. Three runs.
5. **003** Model 8 consolidation at `rho in {0, 1}`, worlds 0–2 (H8a/H8b),
   with the §1 decision branches pre-registered in this file and the gate
   shape prediction in §4 evaluated across the full grid.
5. **004** Benchmark D generator + validity gates (per-primitive measured
   recurrence, scratch control, output scale) on three development worlds.
6. **005** Model 5 shared-parent+residual on Benchmark D (H9), with the
   escape-hatch guard and the `||Delta||`-vs-recurrence figure as the
   primary deliverable. Dense-C and Continuous run paired as the bounds.
7. **006** Model 7b/7c within-lifetime compiler (H10) on exact-reuse
   worlds; then 7b-dream vs 7b at matched compiler-training compute, with
   the dream-benefit-vs-rho check at both endpoints before trusting the
   augmentation. The dream-vs-rho falsifier is PROTECTED from descoping:
   if time pressure cuts anything from 006, cut 7c, not the dream
   control — it is the self-test of whether the compiler learned the
   world's program distribution.
8. **007** Lifetime-length runs (32 and 128 tasks) at bracketing `rho` if
   and only if B2 showed crossover movement.
9. **008** Benchmark E hierarchical recurrence, only after 005's verdict;
   includes the pre-registered residual-clustering promotion analysis
   (§3).
10. **006b** Mixture-vs-posterior mechanism test (§9.5) — zero compute,
    and it MUST run before 009 so the mechanism framing is timestamped
    ahead of any sealed data.
10b. **006c/006d (pre-registered analyses from reviewer-feedback-13;
    development artifacts only; do NOT gate V2 closure — they may run
    after 009):**
    - **006c, functional-equivalence entropy.** Quotient the route space
      by behavioral equivalence (r ~ r' when mean squared functional
      distance on probes < epsilon), aggregate posterior mass over
      classes, and measure H_func = H([R] | D). Pre-registered
      prediction: at rho = 1 route entropy stays high while functional
      entropy collapses; at rho = 0 neither collapses. If confirmed,
      this is the conceptually correct consolidation signal ("compile
      when the FUNCTION is identified, not its arbitrary
      implementation") and rehabilitates the gate family for V3 — as an
      analysis result, not a third gate derivation.
    - **006d, function-family intrinsic dimensionality across rho.**
      SVD/PCA of task-conditioned functional signatures on common
      probes. Pre-registered prediction: high-dimensional at rho ~ 0, a
      low-dimensional continuous manifold at partial recurrence,
      collapsing toward identifiable atoms at rho = 1 — turning the
      three-regime claim from inference-across-instruments into one
      measured transition.
    Also recorded for V3 from the same review: observable recurrence
    estimators from learning dynamics (gradient alignment, one-step
    transfer benefit) validated against ground truth — the piece a
    self-organizing learner needs to operate the economics without an
    oracle rho; the functional rate-distortion framing R(epsilon) as the
    principled replacement for single-lambda accounting; and the
    amortization scaling analysis G(N) with dream-modulated crossover
    N*, replacing "retry the compiler bigger."
11. **009** Freeze V2 configurations, metrics, exclusions; open seeds
    200–229. Sequencing within the sealed block: Component A completes
    and is analyzed before Component B runs — if Family A's parameter
    intervals miss, that is known before deciding how to PRESENT Family
    B (analysis content for both remains fixed by
    V2_CONFIRMATION_PLAN.md; only reporting order is at stake).
    Durability rule: sealed-block summary JSONs are committed to
    reports/ as each component completes, and a zipped archive of the
    sealed artifacts is stored off-machine (a private Kaggle dataset
    suffices) before the block is declared closed — the repository
    carries reports and code, and confirmatory evidence must not exist
    only on one disk.

Each step has a written outcome before the next begins; negative results
advance the order rather than restarting it.

---

# 9. V2 go/no-go gate (for Phase III)

Do not build dynamic library restructuring (fork/merge, promotion,
descriptors-in-anger) until:

1. H7 is settled — we know whether discrete online cost is inference or
   inherent;
2. H8b holds — consolidation refuses to fire where there is no discrete
   structure (a compile criterion that cannot say no is not a criterion);
3. H9 holds — selective sharing beats both fixed architectures somewhere,
   with the residual-allocation signature intact;
4. the crossover surface (H5/H6) is mapped well enough to say *when* any of
   this machinery should pay;
5. results survive the same nuisance battery V1 survived (order, replay,
   initialization, family mismatch).

If H8 and H9 both fail cleanly, the honest conclusion is that a fixed
continuous basis plus per-task inference is the right representation at this
scale, and Phase III pivots from "restructuring" to "amortized inference at
cross-world scale."

## The Phase III organizing principle: expert iteration over programs

Named here so Phase III inherits a structure rather than a pile of parts.
The AlphaZero loop — search produces better decisions than the raw policy;
the policy distills the search output; the better policy improves the next
round of search — maps exactly onto this project once program spaces
outgrow enumeration (variable length, loops, macros — the Benchmark L era):

    search   = program posterior / enumeration / beam search (Model 7a)
    policy   = amortized compiler C_phi (Model 7b)
    distill  = sleep-training the compiler on solved (D_tau, pi*_tau)
    dream    = fantasy tasks sampled from the learner's own library
               (Model 7b-dream) — self-generated training data whose
               usefulness itself tracks how well the library matches the
               world
    verify   = ground-truth y = f_tau(x), the exactly-checkable outcome
               that makes search a trustworthy teacher (the analogue of
               AlphaZero's perfect simulator — most domains lack it; ROW
               has it by construction)

At V2's scale (512-1,728 programs) "search" is a for-loop and this
structure is scaffolding. It becomes the load-bearing design exactly when
enumeration dies, which is why Models 7a/7b/7b-dream are worth building
correctly now: they are the small versions of Phase III's engine.

What the analogy does NOT supply, recorded to prevent drift: (a) there is
no adversary, so no automatic curriculum — fantasy tasks do not get harder
by themselves, and any self-generated-curriculum or world-generation
mechanism (POET-style "spawn worlds with different libraries/conditions")
crosses the §6.1 lifetime/meta-learning line and takes Tier-3 controls
with it; (b) expert iteration amplifies whatever the verifier accepts —
with exact ground truth this is safe, but any future move to approximate
or learned verification reopens the reward-hacking failure mode and must
be treated as a new hypothesis, not an engineering detail.

---

# 9.5 Development synthesis (running; full text in notes/v2-synthesis.txt)

Five observations, each grounded in at least two independent V2 results,
recorded before the sealed block so they cannot be reshaped by it:

1. **Inference is cheap to do well and expensive to do cheaply.** Exact
   posterior > everything (H7); learned one-shot compiler < plain SGD
   (H10); hard commitment < soft mixing (V1, H8). Working hypothesis:
   the continuous mixture wins because soft weights implement cheap
   approximate Bayesian program inference. Zero-cost test added as step
   006b below.
2. **Allocation is easy; compression is hard — a currency mismatch.**
   Near-perfect allocation from a penalty alone (H9a) versus universal
   failure of bit-savings mechanisms (MDL gates, both consolidation
   gates, two-part reversals): gradient descent natively optimizes nats
   and never sees bits, so compression must be post-hoc and has so far
   always used the wrong instrument.
3. **Non-identifiability dictates: functional instruments only.**
   Identity-based instruments failed every encounter with the
   overcomplete regime (entropy gates, route agreement, slot indices);
   behavioral instruments worked every time. Binding V3 design
   principle; the predictive-equivalence gate is its application.
4. **Amortization is sample-starved, not wrong.** The passing dream
   falsifier (3x benefit ratio between rho endpoints) shows the compiler
   learns the program distribution; 64 tasks cannot beat direct
   optimization. Predicts cross-world scale. Kept idea: dream-benefit as
   a ground-truth-free library-quality probe.
5. **The open problem handed to V3: nothing wins both currencies.**
   Per-task compilation cannot (183 bits/task against a 130k-bit store);
   cross-task PROMOTION of shared residual structure is the only move
   that attacks bits at scale.

**006b (added analysis, zero compute): mixture-vs-posterior
correlation.** Correlate each task's trained soft mixture weights
(existing continuous artifacts) with its exact route posterior
marginals (step 001 machinery) at exact reuse. Alignment upgrades the
V1 headline with a mechanism: the reusable basis wins by approximating
Bayesian program inference. Non-alignment falsifies observation 1's
working hypothesis and is reported as such. Two refinements, decided
before running: (a) 006b runs BEFORE the sealed block, so the mechanism
framing is timestamped ahead of confirmatory data and cannot be
suspected of being fitted to it; (b) report calibration as well as
ranking — whether mixture entropy tracks posterior entropy across
tasks — because "approximately Bayesian" (calibrated) and
"directionally Bayesian" (rank-aligned only) are different claims and
the weaker one may hold alone.

**006b STATUS (run 2026-08-18, BEFORE the sealed block;
reports/v2_mechanism/): FALSIFIED, decisively and informatively.**
Trained mixture weights are uncorrelated with exact route-posterior
marginals over the same basis (mean Spearman -0.03 across worlds 0-2;
fraction of tasks positive ~0.42, chance level), and calibration fails
by three orders of magnitude (mixture entropy ~1.4 nats vs posterior
marginal entropy ~0.001). The mixture is NOT an approximate posterior.
Corrected mechanism statement: soft mixing wins by exploiting the basis
as a CONTINUOUS FUNCTION SPACE — distributed, compensatory solutions —
not by approximating Bayesian route inference. This coheres with 002b
(manifold beats slots at partial recurrence), the recovery-onset result,
and V1's mixture-beats-forced-route finding, and it sharpens H7's
lesson: route-committed representations are better when found (the
posterior beats mixtures by ~3.9k nats), but gradient descent on mixture
codes never finds them — the learner is not doing route inference at
all. Consequences: "premature commitment" was the wrong frame for the
continuous learner (nothing is being committed); the H8 gates found
nothing to compile because there are no latent route beliefs to
harden; and the V3 wake-phase theory must be manifold-first, with route
structure IMPOSED by consolidation rather than read out of the mixture.

**The three-regime claim (named per reviewer-feedback-11).** The V1/V2
results jointly support a representational transition as recurrence
rises: SPECIALIZATION (dense wins; weak recurrence) -> CONTINUOUS SHARED
MANIFOLD (hypernetwork competitive, even beating explicit slots at
rho = 0.9; statistical reuse pays without identifiable primitives) ->
CRYSTALLIZED PRIMITIVES (explicit slots win, recovery sharpens, frozen
recomposition works; near-exact recurrence). Supported by four
independent instruments (paired losses, hypernetwork gap inversion,
recovery-vs-baseline, frozen transfer). The corresponding research map:
ALLOCATION is solved surprisingly well; INFERENCE is solved exactly but
not cheaply; COMPRESSION is unsolved; REPRESENTATIONAL FORM depends on
recurrence.

**Pre-draft V3 constraint set (revised per reviewer-feedback-12/13).**
These observations supersede notes/v3-sketch.txt as the design
constraints for the eventual V3 spec (the sketch predates all verdicts).
Recorded in draft form, timestamped before the sealed-block verdict:

1. **Make bits differentiable (the root fix for observation 5).** Every
   compression failure shares the diagnosis that gradient descent never
   sees bits. The variational MDL / bits-back answer (Hinton & van Camp,
   1993): task state as noisy codes with learned per-parameter precision
   and a KL term to a shared prior — description length IN the gradient,
   during learning, so precision the task does not need collapses
   continuously instead of awaiting a discrete gate. This retroactively
   explains the MDL-gate and consolidation-gate failures in one stroke:
   they imposed discrete bit-decisions on a learner never charged for
   precision. Draft V3 hypothesis: **a variational-coded shared-residual
   learner beats both fixed architectures in mixed worlds UNDER THE
   TWO-PART CODE** — the exact cell where every current model loses.
2. **Slots as an output of compression, not an architectural input.**
   Chaining 006b (mixtures are manifold solutions, not route beliefs),
   002b (the pure manifold beats explicit slots at partial recurrence),
   and recovery onset (primitives crystallize only at exact reuse): the
   V3 wake learner should be manifold-first (hypernetwork-style), with
   discrete structure existing only as an artifact of sleep — route
   hypotheses constructed by posterior machinery over consolidated
   operators, independently of the wake representation. The earlier
   draft hypothesis ("promotion wins both currencies in hierarchical
   worlds") stands, amended by the Benchmark E lesson: the promotion
   testbed requires family structure that varies across task groups,
   which E as built does not provide (section 3).
3. Functional instruments only; amortization at cross-world scale only;
   dream-benefit as the library-quality probe — carried over from the
   synthesis unchanged.
4. **Refinements from reviewer-feedback-12, accepted:** (a) bits enter
   the gradient DURING WAKE (the earlier "no structural pressure awake"
   rule is narrowed: continuous information cost is wake-legal; only
   IRREVERSIBLE DISCRETE restructuring is sleep-only); (b) V3's first
   and only initial structural operation is PROMOTE — merge, fork, and
   delete are staged behind its verdict; (c) discrete primitives are one
   possible ENDPOINT of compression, not the imposed ontology — the
   abstraction ladder is basis directions -> parameterized operators ->
   discrete operators -> macros, with the MDL criterion deciding what
   form each abstraction earns; (d) the promotion testbed uses
   TASK-GROUP families (family(tau), not family(k)) — convergent with
   prediction P-2026-08-18-D and the Benchmark E diagnosis; (e) the LLM
   bridge is staged measure -> factor -> learn (LLM-A: does estimated
   functional recurrence predict sharing benefit among frozen-model
   adapters, using functional-effect probes rather than weight
   similarity; LLM-B: shared-plus-residual factoring; LLM-C: dynamic
   library) — test the law before building the learner.
5. **Draft primary hypothesis for V3 (H11, Abstraction Promotion):**
   when task-specific continuous adaptations contain recurrent
   functional structure, a learner charged for their information content
   (variational coding) reduces total lifetime description length by
   promoting that structure into shared abstractions — with
   D_promoted < D_unpromoted at matched predictive loss,
   J_promoted below shared-residual, Continuous, AND Dense in
   hierarchical/mixed worlds, and NO promotion in matched structureless
   controls (rho = 0). Secondary (H12, Hierarchical Vocabulary):
   repeated promotion recovers the world's latent sharing hierarchy
   functionally. Staged later: fork/merge under nonstationarity — where
   HYSTERESIS and the acquisition-vs-retention distinction
   (V_create < V_retain thresholds; an abstraction worth keeping below
   the recurrence at which it was worth creating) become the sharp
   questions. The smoking-gun promotion figure: task-specific bits fall,
   shared bits rise, total bits fall, prediction cost flat — an
   abstraction being born because naming shared structure became cheaper
   than repeatedly encoding it.

# 10. Anticipated objections and positioning

Written now, while neutral, so the eventual paper answers its reviewers
before they ask. Each objection gets a designed response, not a rebuttal
paragraph improvised at submission time.

## 10.1 "It's a toy" (the scaling mainstream)

16 dimensions, 6 primitives, networks with thousands of parameters. The
response is not apology but design intent: the toyness IS the instrument —
`rho` is a knob and ground truth is known precisely because the world is
synthetic, and no claim in the paper is about scale. State explicitly what
the paper does NOT claim (see 10.5). Do not argue that results transfer to
large models; argue that the *criterion* is scale-free and this is its
existence proof.

## 10.2 "It's circular" (the strongest technical objection)

**[Updated with the H6 result, per the standing instruction.]** The
answer is now quantitative: under activation mismatch the
return-on-recurrence line keeps a positive slope (~74% of the aligned
slope) but is shifted down by an approximately additive penalty of
2-3.4k nats, moving the zero crossing to the edge of attainable
recurrence (parity at rho = 1, not advantage). Alignment is a measured
cost term with a predictable effect on the line — not a gate, and not a
mere caveat. The paper's circularity section states exactly this, with
the three-rho GELU table.

Teacher and learner share a functional family. Response is H6: if
confirmed, prior alignment is a measured cost term that shifts the
crossover predictably — the objection becomes a coordinate of the theory.
The paper must contain a DEDICATED section on this (not a footnote),
reporting the decoupled-alpha, rank-mismatch, and GELU results together
with the H6 crossover-shift experiment, and stating the honest V1-era
boundary: what is demonstrated is that a learner with a suitable
structural prior can discover and exploit reuse, with prior quality
entering the economics as a cost.

## 10.3 "Neural DreamCoder, simpler domain" (program synthesis lineage)

Partly fair; cite generously. The stated differentiators: (a) the
criterion is lifetime code-length, not task solve-rate; (b) H8b — a
consolidation operator that DECLINES to abstract in structureless worlds —
has no analogue demonstrated in that lineage; (c) the continuous-to-
discrete transition is measured on a storage/inference/learning frontier,
not assumed. If H8b fails, drop differentiator (b) rather than diluting it.

## 10.4 "We already knew this" (bias-variance was always true)

Expected-but-never-measured is what benchmarks are for. Nobody had the
crossover located, shown invariant to order/replay/initialization,
shown to move with lifetime length (H5a), or expressed in a measured-
recurrence coordinate that transfers across worlds (H5b). Confirmed
predictions outrank untested intuitions; say so once, without heat.

## 10.5 Claims explicitly not made (print this list in the paper)

- No claim that these results transfer to large-scale or natural-data
  learning; the criterion is the bet, this paper is its controlled test.
- No claim that discrete/modular architectures are superior in general —
  the finding is a regime map, and Dense wins a large region of it.
- No claim of relevance to ARC-AGI or any external benchmark.
- No claim that the learner discovers arbitrary factorizations
  irrespective of structural prior (unless H6 dies AND mismatch controls
  somehow strengthen — do not soften this line otherwise).
- No conflation of lower lifetime cost with clean compositional transfer:
  §0 item 8 shows they dissociate at `rho = 0.9`, so "the model learned
  reusable primitives" is claimed only where frozen-library transfer and
  operator recovery both support it (in practice, `rho = 1`); elsewhere
  the claim is statistical sharing.
- Development-stage numbers are never presented as confirmatory.
- No universality claims as contributions. The eventual substrate being
  Turing-complete is table stakes (see the universality ladder in §3);
  citing expressivity theorems as evidence of capability is the failure
  mode this project exists to avoid. Claims are always about learnability
  and encoding cost, never expressibility.

## 10.6 The provenance question (agent-executed research)

The full history — spec to confirmed result in days, agent-executed with a
human PI — will draw both interest and extra skepticism, and this project
should welcome the scrutiny explicitly: deterministic seeds, per-run
artifacts, fingerprint validation, sealed worlds, and a public audit trail
(SPEC_AUDIT.md, the reviews/ series, a falsified hypothesis on the record)
exist precisely so the work can be checked without trusting its authors,
human or otherwise. One paragraph in the paper, stated plainly, no
defensiveness.

## 10.7 Realistic ceiling, stated internally

This is a strong niche paper for the continual/modular learning and
MDL communities, a benchmark others can adopt, and — if the framing
catches — the founding measurement of an "economics of abstraction"
research program. It is not a capability demo and will not move the LLM
mainstream. (Superseded expectation, kept for the record: this section
originally named rho*(N) movement as the most valuable prospective
result; the bridge analysis returned it PARTIAL — early movement, then
saturation — so that headline is off the table.) The most valuable
results now on offer are the linear return-on-recurrence law (R^2 = 0.97
in development, 0.935 sealed) with its sealed-block parameter-replication
test, and a learner that positions itself on that line (H9). Optimize the
paper to be unimpeachable, not impressive.

---

# 11. Research statement

> **V1 established a law-like regularity: the value of representational
> sharing is approximately linear in measured functional recurrence, with
> a sign change where the return equals the sharing penalty — and it
> established that the sharing which pays is not yet abstraction, which
> crystallizes only near exact recurrence. Amortization over lifetime
> length is a demonstrated early-lifetime component of this picture, not
> its mechanism. V2 asks whether a learner can position itself on that
> line: choosing how much to share per computation (H9), maintaining
> uncertainty over programs instead of committing prematurely (H7),
> compiling soft inference into discrete programs only where discrete
> structure exists (H8), and learning to infer programs in its own
> language within a single lifetime (H10). The criterion throughout is
> unchanged — prospective reduction in lifetime learning cost plus
> retained description length — applied now to the representation's own
> form, with the sealed block asked to replicate parameters, not just
> signs.**

# Appendix A. Verified numbers behind §0 (checked against artifacts,
# 2026-08-17)

All figures below were recomputed from artifact files, not quoted from
notes. Anyone editing §0 must re-verify against these sources.

- Checkpoint sweep, all 10 development worlds
  [artifacts/checkpoints_development/sweep.json]:
  mean 32-shot novel NMSE at 8 tasks — Continuous 0.0228, Dense-C 0.0230,
  Continuous better 4/10; at 64 tasks — Continuous 0.00343, Dense-C
  0.00645, Continuous better 10/10, ratio 1.88x.
- Recurrence sweep, paired worlds 1-9
  [artifacts/rho_development/sweep.json], Dense-minus-Continuous
  prequential Gaussian log loss (positive favors Continuous):
  rho 0.00: 0/9 wins, mean -2,106; 0.25: 0/9, -2,153; 0.50: 0/9, -2,234;
  0.75: 0/9, -1,306; 0.90: 9/9, +1,167; 1.00: 9/9, +3,615.
- Family controls, world 0, current configurations
  [artifacts/family_controls/*/summary.json], Continuous prequential loss
  (Dense-C world-0 reference -166,521):
  decoupled tanh -170,967 (advantage +4,446); teacher rank 16
  -170,093 (+3,296, vs paired rank-16 Dense-C -166,798); GELU
  -166,962 (+441).
- High-priority controls, world 0
  [artifacts/high_priority_controls/*/summary.json]:
  Continuous + identity operator -170,404 (no material help; slightly
  worse than -170,967 baseline); Discrete global anneal -137,321;
  Discrete per-task anneal -146,146 (improved ~8.8k; still ~25k behind
  Continuous).
- Retention [reports/retention/current-retention.json]: Continuous
  total retained 29,248 bits (16,960 shared + 12,288 task state), 6,528
  inference multiply-adds, mean int8-minus-float NMSE 1.1e-6, worst
  single-task increase 3.9e-5 (Continuous; 1.38e-4 worst across families
  per reviewer-07).
- Robustness, ALL TEN development worlds — RESOLVED; supersedes the
  earlier unverifiable worlds 0-2 means from reviewer-feedback-07. Cite
  ONLY [reports/robustness/robustness.json]: Continuous wins 10/10 worlds
  under every condition — replay 0: +3,129 [CI 2,511-3,743]; replay 1:
  +3,698 [3,233-4,177]; replay 4: +3,350 [2,959-3,766]; reverse order:
  +3,456 [3,145-3,791].
- Structural controls
  [reports/structural_controls/structural-controls.json,
  paired_world_effects, 10 worlds]: verified per-world — e.g. world 0:
  Continuous over Hypernetwork +2,820 loss / +0.00188 novel-32;
  Hypernetwork over Dense-32 +1,626 / +0.00094; Dense-24 vs Dense-32
  +46 loss (task-code dimension is immaterial). Ordering Continuous >
  Hypernetwork > Dense holds per SPEC_AUDIT.md in every development
  world.

Known number-hygiene traps, recorded so they are not re-committed:
- "~3,100 nats" (the stage-1-era three-world mean vs Dense-C) must not be
  mixed with current-configuration world-0 numbers (+4,446 baseline); the
  GELU contrast is 4,446 -> 441 under one configuration, not 3,100 -> 441.
- Gain ratios (e.g. "6.4x vs 2.3x") are retired per feedback-08; use the
  equal-start/divergence checkpoint statistics instead.
- The rho sweep's paired records cover worlds 1-9 (world 0's endpoints
  come from the earlier sparse sweep); say "9/9" for sweep claims, not
  "10/10."

Provenance: distilled from `notes/learnings.txt`, `notes/crossover.txt`
(including the 15:09 extension), `notes/on-loops.txt` (Benchmark L
constraints), `notes/next-level.txt` (rev 2), `notes/perception.txt`
(rev 2), `reviews/reviewer-feedback-01` through `-10`, and `SPEC_AUDIT.md`.
Supersession rules encoded here: Benchmark C demoted below D and E
(crossover.txt, feedback-05); the early-transfer three-regime story
falsified (crossover.txt); gain-ratio reporting retired (feedback-08);
Benchmark C's priority in V1 spec §9 and the V1 spec's implementation-order
items 009–010 are inherited by §8 above. Where this document and a note
disagree, this document reflects the later consensus; where this document
and the frozen V1 spec disagree about V1 scope, the V1 spec wins.
