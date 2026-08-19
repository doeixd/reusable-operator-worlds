# Standing predictions ledger

Quantitative predictions for experiments not yet run, committed before
the data exists. Each entry carries a confidence and its freeze commit;
entries are never edited after the experiment runs — outcomes are
appended. The point is to make the repository a falsifiable position
about what will happen, not only a record of what did. (The V1 and V2
sealed-block intervals in CONFIRMATION_PLAN.md and
V2_CONFIRMATION_PLAN.md are the founding entries of this practice; new
predictions accumulate here.)

## P-2026-08-18-A: variational coding wins the two-part cell

A shared-residual learner whose task state is trained as a noisy code
(learned per-parameter precision, KL to a shared prior; variational MDL)
will beat BOTH fixed architectures in canonical mixed worlds under the
literal two-part code at lambda = ln 2 — the cell where every existing
model loses — while retaining at least half of the raw prequential
envelope gain. Confidence: 0.55. Rationale: the failure diagnosis
(gradient descent never sees bits) is well-evidenced; the fix is
principled; the risk is optimization difficulty of learned precisions at
this scale.

## P-2026-08-18-B: the GELU crossing localizes inside (0.9, 1.0)

On a finer configured-rho grid (0.925, 0.95, 0.975), the GELU-continuous
learner's crossing against Dense-C lands strictly inside (0.9, 1.0) on
at least 2 of development worlds 0-2, with the mismatch penalty in
2,000-4,500 nats at every grid point. Confidence: 0.7. Rationale: the
additive-penalty reading of the H6 data.

## P-2026-08-18-C: the crossover is stationary at 256 tasks

A genuine 256-task lifetime at configured rho 0.75 and 0.9 (worlds 0-2)
preserves the sign pattern (Dense 3/3 at 0.75, Continuous 3/3 at 0.9)
with per-task effect magnitudes within a factor of two of the 64-task
per-task effects. Confidence: 0.75. Rationale: step 007's stationarity
across 32-128.

## P-2026-08-18-D: task-grouped families make residuals cluster

In a redesigned promotion testbed where family components are assigned
per TASK GROUP (half the tasks draw family A's perturbation direction,
half family B's — cross-cutting structure a task-invariant basis cannot
absorb), shared-residual task-step residuals will cluster by task group
(within-group functional similarity exceeding cross-group by at least a
factor of 3) in a majority of worlds. Confidence: 0.6. Rationale: the
Benchmark E negative traced to absorbable (task-invariant) family
structure, not to the residuals' inability to carry structure.

## P-2026-08-19-E: the promoter refuses accidental similarity

In a control world with the same marginal statistics, task count, and
rank distribution as the promotion testbed, but where residual
similarity is constructed to be non-predictive of the held-out future
task block, a promotion criterion that includes prospective value
(V_future estimated from held-out or streamed future prefix) will
refuse promotion in a majority of worlds, while the same criterion
fires in a majority of true task-group-family worlds. A
retrospective-only criterion (V_retro alone) will be fooled: it fires
in the accidental control at a rate within a factor of two of its rate
in the true worlds. Confidence: 0.6 for the joint pattern. Rationale:
the accidental control is designed so retrospective compression exists
by construction; only the prospective term can distinguish the cases.

## P-2026-08-19-F: internal promotion economics reproduces the crossover

If V3's promotion value V-hat(A) is frozen operationally before any
promotion run, then across worlds spanning measured recurrence 0.0-1.0,
the recurrence at which mean V-hat(A) crosses zero will fall within
[0.35, 0.60] measured recurrence — bracketing the externally measured
V1/V2 sharing crossover (~0.48) — and V-hat will be monotone
increasing in measured recurrence across the grid. Confidence: 0.5.
Rationale: if the learner's decision variable is a consistent estimate
of the sharing economics, it must inherit the law; but V-hat estimation
noise at 64-task lifetimes may be large, and the promotion value and
the paired-architecture crossover are related, not identical,
quantities.

## P-2026-08-19-G: promotion tracks the remaining horizon

In paired worlds identical through task 32 — one ending after 8 more
tasks, one with 128 remaining, the remaining horizon exposed to the
learner — a promotion criterion whose V_future term uses the horizon
will promote in the long-horizon condition at a rate at least twice
its rate in the short-horizon condition, holding all observed history
fixed. Confidence: 0.65 conditional on H11 passing (this experiment
runs only after V3.1's verdict). Rationale: N_future * s > C is the
investment logic promotion is built on; exposing the horizon removes
the estimation confound, so failure would indicate the criterion does
not actually use prospective value.

## P-2026-08-19-H: an abstraction-promotion amortization threshold in N

Across lifetime lengths N in {16, 32, 64, 128, 256} on the promotion
testbed, the rate of correct promotion (candidate matching a true
family component functionally) increases monotonically with N, and the
ratio D(A)-plus-overheads to realized reuse savings decreases
monotonically, yielding an interpolable threshold N* below which
promotion does not pay under the two-part code and above which it
does. Confidence: 0.6 for the joint monotone pattern; no interval on
N* itself is claimed before development data. Rationale: V1/V2
amortization economics (N * savings > cost) applied one level up; 64
tasks may sit near the threshold, which is precisely why the sweep is
informative. Conditional on the H11 development verdict; runs on
development worlds first.

## P-2026-08-19-I: gated innovation beats Gaussian innovation

Registered BEFORE any gated learner exists, and before any PROMOTE run,
so the transition away from the Gaussian wake code is not fitted to its
outcome. Giving the task code an exact null state — delta = g * R with
a gate over whole RANK COMPONENTS, so rank(R_tau) in {0, 1, 2} and
rank 0 is exactly "reuse the shared computation" — improves the
rate-distortion frontier over the Gaussian variational code on canonical
mixed development worlds: at matched held-out NMSE (within 1e-4), the
gated learner's retained two-part bits are lower in a majority of worlds
0-2. Confidence: 0.6. Secondary and more interesting, registered at
confidence 0.7: the gate becomes an explicit reuse decision that tracks
recurrence — the probability that a task-step carries rank 0 INCREASES
with that step's measured primitive recurrence (Spearman positive in a
majority of worlds). Rationale: the Gaussian code's identity state is
not free (measured: 22.0 KL bits for a task needing nothing), so its
gradient prices precision rather than presence; a null-state code makes
"does this task need an innovation at all" a decision the learner takes
rather than one a post-hoc pruner takes for it. Risk: relaxed-Bernoulli
gates were the mechanism that failed in V1/V2 MDL gating, albeit on
shared library components rather than task innovations.

## P-2026-08-19-J: acquisition-then-freeze versus mutable task state

Two protocols for what replay may change, run as a registered ablation
rather than adopted silently. Option A (current): replay updates task
codes and charges their KL, keeping the KL-to-likelihood pressure ratio
at 1.000 for every task. Option B: a task's code is optimized during its
own 128 examples and then FROZEN, with replay updating only shared
structure. Prediction: Option B reduces total retained task bits by at
least 10% at matched held-out NMSE (frozen codes stop accreting
information under replay pressure) but loses lifetime prequential loss
by less than 1,500 nats per world, in a majority of worlds 0-2.
Confidence: 0.45. Rationale: freezing is the only way "charge the code
once at acquisition" is coherent, and it has an attractive reading as
fast acquisition, medium-term frozen episode, slow consolidation — but
it removes the learner's ability to correct an early task's code once
the shared substrate improves, which is exactly what replay is for.

---
Outcome log (append-only):

- **V3 SEALED BLOCK (seeds 300-329): ALL FIVE REGISTERED OUTCOMES PASS**
  (2026-08-19; plan frozen at bcc8319 BEFORE these seeds were generated;
  reports/v3_sealed.json, reports/v3_sealed_future.json,
  artifacts/v3_sealed/). 90 lifetimes, zero failures.
    O1 two-part gain   PASS  +55,292 nats, positive 30/30 (dev +55,697)
    O2 loss gain       PASS  +1,174 nats,  positive 30/30 (dev +1,350)
    O3 migration       PASS  three-sign 30/30, D_total -63.3% (dev -63.6%)
    O4 prospective     PASS  32-shot +0.00311, positive 30/30 (dev +0.00310)
    O5 refusal         PASS  reuse ratio 1.80x (floor 1.4; dev 2.48x)
  The parameters replicate, not merely the signs: the two-part gain
  reproduces to within 0.7% and the total-bits reduction to within 0.3
  points, on thirty worlds nobody had inspected, against intervals frozen
  in advance. H11 is confirmed in all three mandatory parts — an
  abstraction is born (M2) and it makes future related tasks cheaper to
  learn (M3).
  * SCOPE, stated so the result is not over-read: this holds in a
    deliberately constructed world (saturated 6-slot library, a genuinely
    new hidden primitive introduced at task 16 at a fixed program
    position, two hidden task-group families). Four earlier testbeds
    failed, each because a weaker notion of shared structure certified a
    world where promotion degenerated into routing, quantization, or
    deletion. The claim is that promotion works WHEN recurring structure
    is load-bearing, compressible, and family-specific — the three
    properties the frozen validity gate measures — not that those
    conditions arise on their own.
  * The refusal requirement in its ORIGINAL absolute form (no promotion
    at all in structureless controls) remains FALSIFIED; only the graded
    contrast registered in the plan passed. The promoter builds a smaller
    library on structureless worlds rather than declining to build one.

- **P-2026-08-18-D: FALSIFIED as written, and the reason redirects V3**
  (2026-08-19; reports/v3_group_clustering.json,
  artifacts/v3_taskgroup/). The prediction was that in a task-group
  testbed the shared-residual learner's TASK-STEP RESIDUALS would
  cluster by hidden family, within-group functional similarity
  exceeding cross-group by at least a factor of 3, in a majority of
  worlds. Measured on trained learners over eta in {0, 0.5, 0.7, 0.9},
  worlds 0-2, with similarities centered to remove the task-invariant
  component a shared basis can absorb: residual separation is +0.0007
  (structureless control), +0.0055, +0.0127, +0.0173 — monotone in eta
  and positive 3/3 for every eta > 0, but two orders of magnitude short
  of the registered factor of 3. An operational check confirms the
  structure is not usable: two-means partition recovery reaches 0.672 at
  eta = 0.9 against 0.641 IN THE STRUCTURELESS CONTROL, so the recovery
  figure is selection bias, not signal.
  * WHERE THE STRUCTURE WENT. The same instrument applied to the ROUTE
    codes gives separation -0.0152 / +0.0043 / +0.2006 at eta 0 / 0.5 /
    0.9 — twelve times the residual figure at eta 0.9. Per-group mean
    route distributions at eta 0.9 (world 0) differ by total variation
    0.226 against 0.026 in the control, with group 0 placing 0.032 on
    slot 2 and 0.154 on slot 3 while group 1 places 0.190 and 0.048.
    The learner has absorbed the two hidden family components INTO ITS
    SHARED BASIS as dedicated slots and references them through routes.
    The basis has 8 slots for 6 teacher primitives, so it had exactly
    two spare, and it used them.
  * WHY THIS MATTERS MORE THAN THE FALSIFICATION. Gradient descent
    performed, during ordinary wake learning, the operation V3's core
    experiment was built to test: recurring cross-task structure became
    a shared object referenced by a cheap code. It did so for exactly
    the description-length reason V3 predicts a promoter would — a route
    reference costs 192 bits per task against 17,712 for a residual —
    and it needed no sleep phase to do it. The prediction assumed the
    innovation channel; the learner chose the reference channel, which
    is the cheaper one.
    [CORRECTION, appended same day, before any successor run: the
    description-length clause above overstates what was shown. The eight
    slots were PREALLOCATED and already paid for, so the optimizer was
    never charged 192 bits against 17,712; it simply found the shared
    basis plus task-conditioned routes an easier fit. The supported
    claim is that unused shared capacity gets used for recurrent
    structure, not that MDL drove the choice. Whether a learner CREATES
    shared capacity when creation carries a cost is untouched by this
    result and is exactly what V3 still has to test. The finding is
    nonetheless causal, not merely correlational: substituting the wrong
    family's mean route costs +0.00344 more NMSE than the right
    family's (3/3 worlds), and single-slot ablation damages the families
    differentially in the direction their route mass predicts.] Consequence: H11's premise ("task-specific
    adaptations CONTAIN recurrent functional structure") does not hold
    in this testbed, so the testbed cannot support the V3.1 core
    experiment as specified. The V3 question sharpens to: when is
    recurring structure NOT absorbable by the shared basis, and is that
    the regime where an explicit promotion operator earns its place? The
    natural candidate, being tested now, is capacity — more latent
    families than the basis has spare slots, which makes promotion a
    capacity-ALLOCATION decision rather than a discovery problem.

- **P-2026-08-18-A: FALSIFIED** (2026-08-19; confidence had been 0.55;
  reports/v3_variational.json, artifacts/v3_variational/). Scored at the
  stage-one-selected `description_beta` = 0.3 (grid 0.1/0.3/1.0 plus an
  exploratory 3.0, selected on mean lifetime loss per the V2 protocol) on
  canonical mixed development worlds 0-2. The prediction had two clauses
  and they split:
  * TWO-PART WIN over both fixed architectures: FAILS 0/3. Continuous
    wins the cell by 52-55k nats (world 0: J = -131,324 against the
    variational learner's best -76,120), because it retains 29,248 bits
    against the variational learner's 117,534-118,278 inside the
    behavioral margin.
  * ENVELOPE RETENTION ("at least half of the raw prequential envelope
    gain"): PASSES, 0.80 / 0.87 / 0.83, mean 0.83.
  The conjunction fails, so the prediction is falsified. Mechanism,
  measured rather than inferred: (1) the bits are STRUCTURAL, not
  precision — an independent post-hoc audit of the untouched H9 baseline
  needs 89% of dense bits at the same margin in 3/3 worlds, and
  variational coding recovers 11%, the same order, so it bought no
  structural compression; (2) a Gaussian code mischarges the identity
  state, paying 22 bits to say "this task needs nothing"; (3) the
  per-tensor-type prior collapses winner-take-all and took the ROUTE
  mechanism — the cheap shared-reference channel — to uniform mixtures in
  2/3 worlds at beta = 1 and 3/3 at beta = 3. The beta sweep is monotone
  and never crosses: more beta buys fewer bits, less route structure, and
  worse loss together. Reading: continuous information penalties alone
  cannot create structural compression, because shrinking a KL does not
  produce a shared object plus a reference to it. Only PROMOTE changes
  the representation class, which makes H11 a sharper hypothesis rather
  than a weaker one.
  * NARROWING (appended 2026-08-19, same day, before any successor run):
    the reading above was first written as "continuous information
    penalties alone cannot create structural compression", which is
    wider than the experiment supports. The supported claim is that a
    continuous penalty over a FIXED representational topology does not
    by itself create shared-object-plus-reference structure, and that
    this Gaussian implementation did not induce useful structural
    sparsity indirectly. A continuous penalty over a different
    parameterization might; that was not tested. The durable distinction
    is that changing the cost of values is not the same operation as
    changing the vocabulary of representations.

# V4.1 H14 — RETRACTED (recorded 2026-08-19)

The first V4.1 result reported that lifecycle compaction beats V3 by
+2,928 nats in structured worlds against +1,098 in controls, and the
exact behavioral-cover oracle then reported that a library of 4-6
abstractions compacts to a cover of size 1 in 3/3 worlds for 3,371-5,578
nats. Both are ARTIFACTS of the substitution tolerance and are retracted
here rather than edited out above.

The defect. Substitutability was tested as

    ||f_j(x) - f_cur(x)||^2 / Var[f_cur(x)]  <=  epsilon = 0.02

normalizing the deviation against TOTAL OUTPUT VARIANCE. On this testbed
an abstraction contributes about 0.2% of that variance, so the
denominator is roughly a thousand times larger than the quantity being
judged. Every abstraction therefore substituted for every other, which
is why the oracle's causal control came back degenerate: functional,
usage, and random retirement all stranded 0 dependents at identical
cost. The control was working correctly and was reporting that the
relation it had been handed carried no information.

The diagnostic that exposed it. Removing an abstraction ENTIRELY scored
a smaller deviation (0.0016-0.0024) than replacing it with a different
one (0.0022-0.0039). A tolerance under which deleting the whole library
is admissible cannot certify anything about which members are redundant.

The corrected measurement. Normalizing instead against what the
abstraction itself contributes, ||f_cur - f_no-abs||^2, substitution
costs 0.86, 1.25, and 1.60 of the contribution in worlds 0-2 — as much
as or more than deletion — and 99-100% of ordered pairs exceed a 10%
relative tolerance. Priced in the objective's own currency, compacting
to a single abstraction costs 4,474 / 7,883 / 5,769 held-out Gaussian
nats to save 3,294 / 5,490 / 4,392 bits: net -1,180, -2,393, and -1,377,
mean -1,650 nats.

Reading. On the frozen V4 testbed, V3's 4-6 abstractions are NOT
redundant estimates of one or two concepts. The representation-
fragmentation hypothesis that motivated V4.1-as-compaction is not
supported here, and compaction is net negative in all three development
worlds. The abstractions are individually weak (0.2% of output variance)
but mutually distinct, which is a different problem from fragmentation
and points at V4.2 consolidation — a REFITTED synthetic merge that could
capture several distinct contributions at once — rather than at
retiring redundant copies, which there are none of.

Durable methodological rule. A functional tolerance must be normalized
against the quantity whose loss it is licensing, not against total
output scale. Any epsilon under which the null edit (delete everything)
passes is measuring nothing, and a degenerate causal control is the
symptom to look for.

# V4.2 economic factorization — FAILS its independent-compression null (2026-08-19)

The V4.2 gate as first scored PASSED 3/3: a shared centre plus a rank-1
or rank-2 functional family beat both the behavioral ceiling cost and
the rank-0 collapse null by 1,192-1,637 net nats on worlds 0-2, with
both currencies priced (bits saved minus held-out Gaussian nats paid).

It does not survive the matched-bit independent-compression control.
Give each abstraction the SAME total bit budget the factorization costs,
spent privately as coarser symmetric per-tensor quantization, and
private storage wins in 9/10 development worlds. Net nats, shared versus
matched-bit: 459/1,055, 1,192/3,153, 1,637/2,043, 1,450/2,028,
1,584/2,129, -34/-34 (tie), 2,643/3,861, 1,610/2,974, 676/3,090,
1,709/3,040. The single non-loss is a tie at a negative value.

Reading. The apparent factorization gain was NOT cross-abstraction
reuse. V3's abstractions are individually overparameterized -- each one
survives coarse quantization nearly intact -- and a shared family
captures less than private precision reduction does for the same bits.
The rank-2 excess over the isotropic null (+15.7, +26.2, +27.3 points)
is real geometry, but real low-dimensional geometry is not the same
thing as an economically preferable representation.

The prospective test is correspondingly weak. Leave-one-abstraction-out
(fit centre and rank-2 basis on m-1 abstractions, infer ONLY the held-out
one's two arguments, evaluate on disjoint audit probes) beats the
centre-only baseline in 9/10 worlds, but recovers a mean of only about
7.5% of the centre-only deficit (range -2.9% to +16.4%). A real operator
family should let a new member be acquired from a couple of scalars;
this one does not.

Durable rule. A structural sharing claim needs a MATCHED-BUDGET private
baseline, not only a no-sharing baseline. "Shared beats unshared at full
precision" and "shared beats unshared at equal bits" are different
claims, and only the second supports reuse.

# V4R census, RETAIN cell — BLOCKED by V3's promotion false-positive rate (2026-08-19)

Three dormancy designs have now failed to instantiate retention value,
and the third identifies the structural cause rather than adding another
null.

  1. Gap (32, 64): the returning arm resumes at the final task, so both
     arms are byte-identical for the whole lifetime and score alike.
  2. Gap (32, 62), two returners, below `minimum_cluster = 3`: the OTHER
     task group keeps promoting through the return window, so post-gap
     births occur in both arms and the dormant family's option is not
     isolated.
  3. Single family (`--task-groups 1`), gap (32, 48), registered in
     V4R §2.1 as the fix for (2): STILL fails. Abstractions are born at
     or after sleep 48 in the PERMANENT arm — worlds 0 and 2 — where the
     family never returns and every post-gap task is background noise.

Diagnosis. The confound is not the world. It is a known property of the
V3 learner, already recorded: PROMOTE fires in structureless controls,
creating 2.9-3.0 abstractions where there is no family at all. A
retention experiment asks whether the learner needed to KEEP an
abstraction; that question is unanswerable while the learner
manufactures replacement abstractions from noise at a comparable rate,
because deletion is never actually costly.

Consequence for V4R. The census's RETAIN cell cannot be measured on this
learner in the online setting. Two admissible routes, neither yet taken:
freeze the library after the gap so the retain-versus-delete comparison
is not contaminated by fresh promotion, which keeps the census offline
and oracle-only as V4R §1 requires; or reduce PROMOTE's false-positive
rate first, which is a change to the frozen V3 substrate and therefore a
separate rung with its own gate.

Durable rule. A refusal control is only as clean as the baseline rate of
the behavior it asks the learner to refuse. Before building a world in
which an operator must NOT fire, measure how often the mechanism fires
with no cause present; if that rate is comparable to the effect, no
world design can rescue the control.
