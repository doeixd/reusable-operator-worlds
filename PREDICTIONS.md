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

# V4R RETAIN cell — GATE PASSES under a frozen-library oracle (2026-08-19)

Taking the first of the two admissible routes recorded above: freeze the
library after the gap so retain-versus-delete is not contaminated by
fresh promotion. This keeps the census offline and oracle-only as
V4R §1 requires, and it is the first clean refusal control in the V4/V4R
arc.

Question: for tasks arriving AFTER the return, is referencing a pre-gap
abstraction better than referencing none?

| arm | world | post-gap tasks | value of best pre-gap abstraction |
| --- | --- | --- | --- |
| returns | 1 | 9 | +2,828 nats |
| returns | 2 | 13 | +2,433 nats |
| permanent | 2 | 4 | -90 nats |

The separation is exactly the registered direction and is large. Where
the regime RETURNS, the dormant abstraction is worth 2,433-2,828 nats to
tasks that arrive after it; where the regime is PERMANENTLY gone, it is
worth nothing (-90). Against a carry cost of one residual, about 1,098
nats, `V_retain = P(return) * C_reacquire - C_carry` is comfortably
positive in the returning arm and negative in the permanent arm.

So the retention OPPORTUNITY exists. What blocked it was never the
world's economics; it was that online PROMOTE manufactures replacement
abstractions from noise, so deletion was never costly. Freezing the
library removes that confound without modifying the frozen V3 substrate.

LIMITS, and they are real. Only three of six cells are scoreable: the
other three have no abstraction born before task 32 at all, because
`family_onset = 16` leaves just sixteen tasks in which to promote. The
permanent arm contributes a single usable cell. This is a
three-observation sign separation, not an interval estimate. Before this
becomes a rung it needs an earlier family onset so pre-gap abstractions
form reliably, the full ten development worlds, and a gap sweep
establishing the crossover `g*` that V4R §2.1 registers as the
prerequisite for implementing any online retention policy.

# PROMOTE false positives are separable by the carry cost (2026-08-19)

The confound blocking every online lifecycle experiment is that PROMOTE
fires on noise. Measured directly: per-abstraction VALUE, defined as the
held-out Gaussian loss its dependents would pay if it were removed,
in structured worlds versus structureless controls where every promotion
is by construction a false positive.

| condition | n | mean | median | fraction below one residual (1,098 nats) |
| --- | --- | --- | --- | --- |
| structured (eta=0.9) | 15 | 1,245 | 1,281 | 33% |
| control (eta=0) | 10 | 651 | 432 | 90% |

Per-world values, sorted:

    structured 0:  1437 1281 1194  -60
    structured 1:  2757 2241 2156 1572 1266  163
    structured 2:  1566 1335  987  859  -82
    control    0:   923  653  326  123
    control    1:   385  354  325
    control    2:  2302  643  479

Reading. A threshold at ONE RESIDUAL'S CARRY COST -- 8 bits x the
residual width, about 1,098 nats, the price of keeping the abstraction
to the end of the lifetime -- retains 67% of structured abstractions and
rejects 90% of control abstractions. That threshold is not tuned: it is
the abstraction's own storage cost, so the rule is simply "an
abstraction must be worth at least what it costs to keep", which is the
MDL criterion the project already uses everywhere else.

Why PROMOTE misses this. Its acceptance criterion is behavioral
deviation within epsilon plus a V_future generalization check. Neither
compares the abstraction's value against its storage price. An
abstraction can fit its members well, generalize to held-out members,
and still not be worth its bits.

CAVEAT, and it is the difference between a diagnostic and an operator.
This is measured post hoc on final models with full dependent sets. An
online rule must estimate value at promotion time, when few dependents
exist and the estimate is noisiest. `LifecycleLibraryLearner` already
accrues `realized_savings_bits` for exactly this purpose, so the
instrument exists; whether an online estimate separates the
distributions as cleanly as the post-hoc one is untested and is the next
thing to measure. Note also that control world 2 contains one
2,302-nat abstraction, so the threshold is not a perfect classifier and
should be reported with its error rates, never as a clean gate.

# Value filter at promotion time — FAILS (2026-08-19)

The post-hoc diagnostic above separated real abstractions from PROMOTE's
false positives cleanly at an untuned threshold (one residual's carry
cost: 90% of control abstractions below it, 33% of structured). I
implemented that rule as an online promotion filter and it does not
work.

`prune_by_value` refuses a promotion made at the current sleep whose
value to its dependents is below its carry price. Reversal is free only
for a fresh promotion, because the dependents' private residuals are
still current -- reversing an OLD promotion pays n private residuals to
reclaim one shared one and can never pay, which is the trap V4.1's
compaction gate fell into. So the operator is a filter, not a collector.

Result on worlds 0-2, live library after filtering versus baseline:

| condition | filtered | baseline |
| --- | --- | --- |
| structured | 0, 2, 0 | 4, 6, 5 |
| control | 0, 0, 0 | 4, 3, 3 |

It refuses almost everything in BOTH conditions. The library collapses
where it should be preserved.

Why, and it is not a coding error. At the moment of promotion an
abstraction has its MINIMUM dependent set -- a cluster of about three
tasks -- so its measured value is at its lowest, while its price is the
full carry cost to the end of the lifetime. Value accrues later as
further tasks join. The post-hoc measurement that separated the
distributions used FULL dependent sets, which is information the learner
does not have at birth. The threshold is right; the estimate available
at decision time is not.

A units bug was found and fixed on the way, and is recorded because the
first version looked like the same result. Value was computed as a mean
squared deviation on a probe and compared directly against a price in
nats, so a quantity of order 1e-2 was tested against one of order 1e3
and everything failed trivially. Converting through the Gaussian
likelihood, `d * deviation * examples / (2 sigma^2)`, changed one cell
(structured world 1 keeps 2 abstractions) and left the conclusion
intact. Same class of error as the V3 KL-scaling bug: comparing two
quantities in different currencies.

Reading. Refusing bad promotions requires PROSPECTIVE value -- expected
future dependents -- not realized value at birth. That is exactly V3's
H11.3 territory and is not free. Until it exists, the promotion
false-positive rate cannot be reduced by a retrospective MDL test, and
the online retention experiment stays blocked. The frozen-library
oracle remains the only clean route to the RETAIN question.

# V4R FORK census cell — bound too loose to be a gate (2026-08-19)

Attempted the FORK cell with a conservative upper bound: a forked
abstraction A' can do no better for a subgroup than giving each member
its own PRIVATE residual, so if that bound fails to exceed one
abstraction's bits, FORK has no opportunity.

The bound "passes" in 24/33 abstraction-cells across F=2 and F=4. It
should NOT be reported as a FORK opportunity, because the bound does not
discriminate. In nearly every cell the best subgroup is ALL dependents,
which means the quantity measured is "private residuals beat the shared
abstraction", priced against the cost of ONE abstraction. But serving n
tasks privately costs n residuals, not one. A single A' can only match
per-task private residuals if the subgroup is functionally homogeneous,
and nothing here tests that.

So the cell is UNRESOLVED, not positive. Deciding it requires actually
fitting A' to each candidate subgroup and pricing the fit -- the same
functional-fit machinery `factorize` uses -- rather than bounding it.

Incidental finding, worth its own follow-up: private residuals beat the
shared abstraction for essentially every dependent, by 857-5,869 nats.
That is not alarming on its face, since PROMOTE exists to trade
prediction for description length, but the magnitude has not been
audited against the bits it buys, and it bears on why the V4.1
compaction and V4.2 factorization gates both failed -- the shared
objects are cheaper than private state but meaningfully worse at
prediction.

Method note. An upper bound is only a gate in the direction it bounds. A
FAILING loose bound is decisive; a PASSING loose bound says nothing.
This one was built to fail-closed and then read as though it could pass,
which is the same error as scoring a single dormancy arm.

# RETRACTION: "PROMOTE false positives are separable by the carry cost"

The entry above is WRONG and is retracted here rather than edited out.
It used raw ablation as the counterfactual — remove the abstraction and
give the dependent NOTHING — which is not an alternative the learner
would ever face. The correct comparison is against the best admissible
alternative, which in this substrate is the task's own private residual,
and it must be a two-part comparison because the whole point of the
abstraction is that it replaces n private residuals with one shared one:

    V(A) = lambda * n * D_residual - (loss_with_A - loss_private)
    RETAIN iff V(A) > T_A, where T_A = lambda * D(A), untuned

Recomputed on development worlds 0-2:

| condition | abstractions repaying their own cost | V(A) range |
| --- | --- | --- |
| structured | 15/15 | 3,241 - 15,085 |
| control    | 10/10 | 4,777 - 21,692 |

EVERY abstraction repays, in both conditions, by margins of 3x to 20x
the threshold. The economic survival rule does not discriminate at all
here, and the earlier "90% of control abstractions fall below carry
cost, versus 33% structured" was an artifact of the wrong counterfactual.

Why the reversal is so large. Raw ablation measures only the prediction
A buys against nothing, of order 400-2,300 nats. The correct measure is
dominated by the DESCRIPTION term: not storing n private residuals saves
`n * 1,098` nats, which for a typical dependent set of 6-16 tasks is
6,600-17,600 nats. Against that, the prediction A gives up relative to
private residuals (656-10,148 nats) is real but smaller. Any abstraction
with more than about two dependents repays trivially.

Reading, and it is consistent with every other V4 gate. PROMOTE is NOT
creating uneconomic abstractions. Its libraries are not inflated in
description-length terms; they are economically justified even in
structureless controls, where an abstraction over noise still beats
storing that noise n times. So there is no lifecycle slack here either
— which is now the fourth independent measurement saying the same thing
about this regime.

This retraction is itself the strongest evidence for the rule that
produced it, adopted as constitutional alongside the other two:

    STRUCTURAL VALUE IS ALWAYS MARGINAL VALUE AGAINST THE CHEAPEST
    LEGITIMATE ALTERNATIVE.

Three counterfactual errors have now produced three false positives in
this project: a tolerance normalized against total output variance, a
sharing claim scored against full-precision atoms instead of matched
bits, and this one. In all three the favorable number came from
comparing against something the learner would never actually do.

# The exchange rate discriminates where V(A) does not (2026-08-19)

V(A) is dominated by its description term — not storing n private
residuals saves `n * 1,098` nats — so every abstraction repays in every
condition and the economic survival rule cannot sort them. The quantity
that survives that domination is the RATE of the trade:

    exchange rate = prediction nats GIVEN UP / description nats SAVED

Development worlds 0-2, world-level means (abstractions within a world
are not independent, so the abstraction-level Mann-Whitney p = 0.0079
across 15 versus 10 abstractions OVERSTATES the evidence and is not the
headline):

| world | structured | control | control - structured |
| --- | --- | --- | --- |
| 0 | 0.194 | 0.321 | +0.127 |
| 1 | 0.190 | 0.300 | +0.110 |
| 2 | 0.164 | 0.275 | +0.111 |

Structured is lower in 3/3 worlds, mean paired difference +0.116. Per
project rule at n=3, the paired deltas are the result; no interval is
reported.

Reading. Abstractions over real recurring structure give up about 0.18
nats of prediction per nat of description saved; abstractions over noise
give up about 0.30. Both trades are profitable at lambda = ln 2 — 0% of
abstractions in either condition reach the break-even rate of 1.0 — so
PROMOTE is not miscalibrated, and this does NOT revive the idea that
control promotions should be refused. What it shows is that abstraction
QUALITY is visible in the rate even when it is invisible in the net.

This is the first measurement in the V4/V4R arc that separates
structured from structureless on a per-abstraction economic quantity.
It is a diagnostic, not an operator: a rate threshold would be a tuned
hyperparameter, unlike `T_A = lambda * D(A)`, which is untuned but
non-discriminating. Whether the rate has any decision-theoretic use is
untested.

# V4R FORK cell — RESOLVED NEGATIVE, 0/30 (2026-08-19)

Replaced the too-loose upper bound with an actual refit: split each
abstraction's dependents by their behavioral residual functions using the
learner's own clustering, fit A' functionally to one part with
`_fit_abstraction` (the same machinery PROMOTE uses), reassign that
subgroup, and price one extra abstraction.

FORK pays in 0/30 abstraction-cells across F=2 and F=4. The best gain is
670 nats against a cost of 1,098, and five splits are NEGATIVE: a
refitted A' is worse for its own subgroup than the parent it was carved
from, because the parent was fitted to more data.

This overturns the bound's apparent 24/33 "pass". The bound measured
"private residuals beat the shared abstraction", priced against one
abstraction rather than n, so it was answering a question about
un-promotion, not forking. Confirms the method note recorded with it: a
failing loose bound is decisive, a passing one says nothing.

The census is now complete: COMPRESS dominates 12/12, FACTORIZE
negative, RETIRE negative, FORK negative 0/30, RETAIN positive only as a
frozen-library oracle. At this scale, under stationary recurrence with
cheap reacquisition, the optimal representation IS a static library of
independently compressed abstractions, and lifecycle machinery costs
more than the slack it recovers.

# The remaining slack is CODING, not topology (2026-08-19)

Direct test of the hypothesis that V3's library is minimal-sufficient at
the FUNCTIONAL level but wasteful at the CODING level. Symmetric
per-tensor quantization of the abstractions, structured worlds 0-2,
loss increase over the 8-bit reference:

| bits/scalar | 1 | 2 | 3 | 4 | 6 |
| --- | --- | --- | --- | --- | --- |
| mean nats paid | 6,224 | 2,681 | 599 | 127 | -7 |

At SIX bits the behavioral cost is zero (-7 nats, i.e. marginally
better). At four bits it is 127 nats across an entire library. Priced
against the description saved at lambda = ln 2, the net is positive
everywhere and is MAXIMIZED at the most aggressive setting tested:

    1 bit: save 38,428, pay 6,224 -> net +32,204
    2 bit: save 32,938, pay 2,681 -> net +30,258
    4 bit: save 21,959, pay   127 -> net +21,832

So `D_min(A) ~ 1-2 bits/scalar` against a stored `D(A) = 8`. The
abstractions carry roughly a quarter to an eighth of the information
their encoding pays for. CONFIRMED: the expensive thing is the encoding
of each learned function, not the number of distinct functions.

Consequences, and the second is a caution about this project's own
accounting.

First, it explains the whole census. FACTORIZE must beat COMPRESS at
matched bits; if each atom is 4-8x overparameterized, private
requantization harvests that slack with no fixed cost, while a shared
basis pays `D(C) + D(B)` up front for structure that is weaker than the
waste. RETIRE and FORK are unaffected either way. Local compression wins
because there is a large, cheap, purely numerical reservoir sitting in
front of every structural edit.

Second, the 8-bit retention proxy used throughout V1-V4 OVERSTATES
description length by roughly 4-8x in absolute terms. Every paired
comparison in this project survives, because the proxy is applied
symmetrically to all models and the comparisons are relative. But any
ABSOLUTE two-part claim -- "the library costs N bits", "V(A) repays its
storage by 3x to 20x" -- is inflated by that factor and should be
restated at the measured frontier rather than at 8 bits. This is worth
an audit before the paper quotes any absolute figure.

Reading. The next bottleneck for this architecture is neural coding
efficiency, not library topology. Restructuring the computational
language is not yet rational: the existing words are simply written in
too many bits.

# C_reacquire measured ONLINE, and it decays with gap length (2026-08-19)

First online measurement of reacquisition cost, and the first time the
dormancy pressure has produced a graded signal rather than a null.

Two arms per cell, identical single-family worlds with a returning
regime, differing only in whether the live library is force-deleted at
the gap (task 32). Deletion had to be made meaningful first: V3's
`select_reference` scans every abstraction ever created, so a "deleted"
abstraction was still adoptable by new tasks and deletion was a no-op.
`LifecycleLibraryLearner` now overrides it to reuse only from the LIVE
library — an override, never an edit to the frozen V3 class.

Paired cumulative prequential Gaussian log loss, deleted minus retained
(this project reports loss as a large negative number, so a POSITIVE
difference means the deleted arm is worse):

| gap | world 0 | world 1 | world 2 | mean |
| --- | --- | --- | --- | --- |
| 8  | +1,841 | +1,636 | +1,210 | **+1,563** |
| 16 | +1,283 | +1,198 |   +841 | **+1,107** |
| 32 |   +280 |    +11 |   +105 |   **+132** |

C_reacquire is positive in 9/9 cells and DECAYS MONOTONICALLY with gap
length, 1,563 -> 1,107 -> 132. That is the real-options structure the V4
dormancy pair was built to produce and never did: the longer the family
stays away, the less a retained abstraction is worth, because the
learner has time to rebuild one.

WHAT THIS DOES NOT SHOW. Retention still does not pay on net here,
because the intervention deletes the WHOLE library, so the carry cost
being avoided is roughly 5 abstractions at 1,098 nats each, about 5,490,
against a reacquisition cost of at most 1,563. `V_retain < 0` at every
gap. The clean test deletes only the DORMANT family's abstraction and
charges only its carry; that is the next cell and it is not yet run.

So the correct reading is narrow and real: reacquisition is no longer
free, and its decay curve in `g` is measurable online. Whether it ever
exceeds carry cost is still open, and the frozen-library oracle's
earlier crossover between gap 8 and 16 remains the only positive
retention result.

Reporting note: the first printout of this table carried an inverted
legend, reading a positive difference as "no cost". The sign convention
here is that more negative loss is better, so deleted-minus-retained
POSITIVE means deletion hurt. The numbers were right; the label was not.

# Single-abstraction retention probe: V_retain not reliably positive (2026-08-19)

The decisive RETAIN test, with every confound removed: one abstraction,
no redundancy, no re-homing, no alternative representative. Only the
most-depended-upon abstraction is deleted at the gap, and only its own
forward-looking carry is charged — historical creation cost is sunk and
correctly excluded, so the comparison is future carry versus
P(return) * C_reacquire.

    V_retain = (loss penalty of deleting) - (description ACTUALLY saved)

| gap | loss penalty | V_retain per world | mean |
| --- | --- | --- | --- |
| 8  | +1,841 / +1,636 / +1,210 | +1,841 / +538 / -2,084 | +99 |
| 16 | +1,283 / +1,198 / +841 | -913 / +2,296 / -2,453 | -357 |
| 32 | +280 / +11 / +105 | +280 / +11 / -993 | -234 |

Outcome 3, with a caveat that matters. V_retain is NOT reliably positive
at any gap. But the two terms behave very differently and should not be
reported as one number:

  * The LOSS term is clean — positive 9/9, decaying monotonically with
    gap (1,563 -> 1,107 -> 132 in the whole-library probe). Reacquisition
    genuinely costs something and the option genuinely decays.
  * The DESCRIPTION term is noisy, and structurally so. Deleting one
    abstraction at task 32 perturbs the whole downstream promotion
    trajectory: arms end with 5/5, 6/5, 5/2, and in one cell 6/7, where
    the DELETED arm finishes with MORE abstractions than the retained
    one. Final description is therefore not a controlled quantity in
    this design, and it swings V_retain by +/-2,000.

Methodological consequence. A deletion intervention mid-lifetime is not
a paired comparison in the sense the rest of this project uses: the two
arms stop being matched the moment the intervention changes what gets
promoted next. Only the pre-divergence window is genuinely paired. The
right instrument is the RETURN CURVE — cumulative loss over the first
returning tasks, where the arms are still comparable — rather than an
end-of-lifetime J difference. That is the measurement to build next,
and it is what "area between the curves" was asking for.

Standing reading. Reacquisition is not free, but in this learner it is
cheap enough, and library evolution divergent enough, that persistent
retention has no demonstrated net value at any tested gap. n=3 worlds
per gap; the deltas are reported, no interval is claimed.

# RETURN CURVE: V_retain goes POSITIVE at short gaps (2026-08-19)

The end-of-lifetime V_retain above is superseded as an instrument, not
as data. A mid-lifetime deletion stops the arms being paired the moment
it changes what gets promoted next, so final description is
uncontrolled. The paired window is the RETURN CURVE: prequential loss
over tasks arriving after the regime resumes.

Deleted minus retained, cumulative over the full return window:

| gap | returning tasks | per world | mean | vs carry (1,098) |
| --- | --- | --- | --- | --- |
| 8  | 24 | +1,807 / +1,581 / +1,214 | +1,534 | ABOVE -> RETAIN |
| 16 | 16 | +1,168 / +1,179 / +829 | +1,059 | at parity |

3/3 worlds in both cells. Over the first eight returning tasks the
deleted arm is worse on EVERY task (~60 nats each) with no catch-up
inside the window: the relearning curve has the predicted shape, and
the area between the curves is the value of memory.

This is the first POSITIVE online retention result. The crossing lies
between gap 8 and gap 16 and independently reproduces the
frozen-library oracle's crossover, from an unrelated measurement.

CORRECTION to the entry above. "C_reacquire decays monotonically,
1,563 -> 1,107 -> 132" was partly an artifact. The g=32 cell uses
dormancy (32, 64), which resumes at the FINAL task, so its returning
window is empty; the +132 was no-returning-tasks, not a decayed option.
The decay between gaps 8 and 16 is real (+1,534 -> +1,059); the
collapse to +132 is withdrawn.

Standing rule, third occurrence of the same error: a dormancy gap must
CLOSE strictly before the lifetime ends, with enough tasks after it to
measure. This has now produced a false reading in V4.1, in the retention
sweep, and here. Any dormancy config must be validated by counting
returning tasks BEFORE its numbers are read.

# g* WITHDRAWN. The boundary is a HORIZON, not a dormancy length (2026-08-19)

The interpolated "g* = 15.2" from the fine gap sweep is withdrawn. It is
an artifact of a fixed 64-task lifetime: as the gap grows, the return
window shrinks (28, 26, 24, 20, 16 tasks), so cumulative value falls
even if dormancy has no effect at all.

Two checks, both from the same runs:

PER-TASK RATE across gaps 4, 6, 8, 12, 16:
    63.5  64.6  63.9  62.2  66.2   (mean 64.1, sd 1.3, cv 2%)

MATCHED 16-TASK WINDOW, identical number of returning tasks per gap:
    +971  +976  +979  +981  +1,059

Both are FLAT. Dormancy length has no detectable effect on what the
stored abstraction is worth. What varies is only how many post-return
uses remain to amortize it.

THE LAW, and it is derived rather than fitted:

    retain iff  H_R * s_bar  >  lambda * D(A)
    H_R* = lambda * D(A) / s_bar = 1,098 / 64.1 = 17.1 returning tasks

Observed: 24 returning tasks retains clearly (+1,534); 16 returning
tasks is break-even (+1,059 against 1,098). The data bracket the
prediction at 16-24 with no retention threshold fitted anywhere. This is
the SAME amortization logic as abstraction birth in V1/V3 — enough
repeated future use to repay a code cost — now applied one level up, to
whether a learned abstraction is worth carrying.

So the correct statement is not "memories decay while dormant" but
"a dormant abstraction is worth carrying when expected remaining reuse
exceeds its amortization threshold".

TWO LIMITS, both real. (1) The measured quantity is C_reacquire, not
V_retain: the marginal carry counterfactual is endogenous, because when
deletion triggers re-promotion the final description is unchanged and
the true saving is 0, not 1,098. A controlled test must suppress
re-promotion during the evaluation window so that
D_retain - D_delete = D(A) exactly. (2) n=3 worlds per cell; deltas
reported, no interval claimed.

SEPARATE FINDING, not noise. Deleting ONE abstraction at task 32 changes
later library size to 5/5, 6/5, 5/2, and in one cell 6/7 — the deleted
arm ending LARGER. Library state is PATH-DEPENDENT: removing one object
changes what later looks worth promoting, so V(A_i) is not independent
of the rest of the library or of history. That is a substantive result
about lifecycle management being sequential rather than per-object, and
it is why the paired return window, not endpoint J, is the right
instrument.

# CONFIRMED: retention obeys the amortization law (2026-08-19)

The controlled horizon sweep. Gap fixed at (32,40); library FROZEN at
task 32 (last sleep at the gap start), so no replacement abstraction can
be born and `D_retain - D_delete = D(A)` exactly. Verified in every
inspected cell: zero births after task 32, retained arm carries exactly
one more abstraction than deleted. This is the counterfactual the
earlier probe could not supply.

Registered BEFORE the sweep, from independently measured quantities:

    H_R* = lambda * D(A) / s_bar = 1,098 / 64.1 = 17.1 returning tasks

Result:

| H_R | C_reacquire | per-task rate | V_retain | verdict |
| --- | --- | --- | --- | --- |
|  8 |   476 | 59.6 |   -621 | DELETE |
| 12 |   691 | 57.6 |   -407 | DELETE |
| 16 |   967 | 60.5 |   -131 | DELETE |
| 20 | 1,238 | 61.9 |   +140 | RETAIN |
| 24 | 1,502 | 62.6 |   +404 | RETAIN |
| 32 | 1,880 | 58.8 |   +782 | RETAIN |

Monotone in H_R, 3/3 worlds agreeing in every cell. Interpolated
crossing 17.9 against a prediction of 17.1 — within 5%, with NO fitted
retention threshold. Using this sweep's own rate (60.2 nats/use) the
prediction is 18.2, closer still. The per-task saving is flat across
horizons (57.6-62.6), which is what makes the law linear.

    RETAIN A  iff  H_R * s_bar  >  lambda * D(A)

This is the same amortization logic as abstraction birth in V1/V3 --
enough repeated use to repay a code cost -- now operating one level up,
on whether a learned abstraction is worth carrying. Retention is not a
separate mechanism requiring its own theory; it is the birth criterion
applied to a future horizon instead of a past one.

It also settles what the dormancy experiments were failing to find. The
boundary is NOT a dormancy length. Gap length has no detectable effect
on the stored abstraction's value (per-task rate flat across gaps 4-16);
what matters is expected remaining reuse. Every earlier dormancy design
failed because it varied the wrong quantity.

LIMITS. n=3 worlds per cell, deltas reported, no interval claimed. The
law is established with promotion SUPPRESSED after the gap; with
endogenous re-promotion the marginal carry cost falls toward zero and
the crossing must move. Measuring that shift is the next step, and it is
the difference between retention economics in isolation and retention
under a library that keeps evolving.

# Open library: C_reacquire survives, the DECISION RULE does not (2026-08-19)

Same horizon sweep with re-promotion restored (sleeps inside the return
window), so the marginal carry cost is endogenous rather than fixed.

| H_R | C_reacquire | marginal carry | V_retain | vs frozen-library |
| --- | --- | --- | --- | --- |
|  8 |   473 |     0 |   +473 | frozen said -621 |
| 16 |   971 | 1,098 |   -127 | frozen said -131 |
| 24 | 1,529 | 1,098 |   +431 | frozen said +404 |
| 32 | 1,960 |   366 | +1,594 | frozen said +782 |

TWO SIDES BEHAVE DIFFERENTLY, and this is the result.

The LOSS side is invariant. C_reacquire is 473 / 971 / 1,529 / 1,960
against 476 / 967 / 1,502 / 1,880 with the library frozen — within 5% at
every horizon. Allowing re-promotion does NOT materially reduce the cost
of having deleted the abstraction, at least within these windows. The
per-use saving and the amortization arithmetic are robust properties of
the learner.

The CARRY side collapses. What deleting an abstraction SAVES is not
lambda*D(A); it is whatever the altered trajectory ends up costing. Here
it is 0, 1,098, 1,098, and 366 nats across the four horizons — zero when
deletion simply triggers a replacement promotion, full when it does not,
and fractional when the library sizes diverge by a non-integer mean.
V_retain therefore becomes NON-MONOTONE in horizon (+473, -127, +431,
+1,594) even though C_reacquire rises smoothly.

Reading. The amortization law
`RETAIN iff H_R * s_bar > lambda * D(A)` is correct as an economic
statement and predicts the crossing to within 5% when the counterfactual
is controlled. But in a library that keeps evolving, `D(A)` is the wrong
carry term: the quantity that matters is the marginal description of the
whole trajectory, which is endogenous to the decision being evaluated.

This is the same path-dependence recorded earlier, now with a
consequence. A per-object retention rule is not well-posed in an
evolving library, because the value of retaining A depends on what the
learner would build instead — which depends on whether A was retained.
The decision is sequential, not per-object: the right object is
something like Q(L_t, e_t), not a scalar attached to each abstraction.

LIMITS. n=3 worlds per cell, four horizons, deltas only. The carry
figures are cross-world means of integer library-size differences, so
the fractional 366 reflects worlds disagreeing, not a partial
abstraction. A larger sweep would be needed to say whether V_retain is
genuinely non-monotone or merely noisy at this n.

# Audit: which absolute two-part figures are affected (2026-08-19)

Triggered by the coding-frontier result (`D_min ~ 1-2 bits/scalar`
against 8 stored). Checked every absolute description-length claim in
`paper/draft.md`.

ROBUST (ratios and orderings, unaffected by a rescaling):
  * "per-task residuals retain ~130,624 bits (9x the Continuous task
    state)" -- a ratio between two components at the same proxy.
  * The retention/compute frontier ORDERING (Discrete < Continuous <
    Hypernetwork < Dense-24 < Dense-C).

NEEDS RELABELLING, not correction:
  * "int8 retention: Discrete 26,208 bits, Continuous 29,248, ..." is
    presented as the retention cost when it is one point on a frontier
    that was never swept downward. It is an upper bound at 8 bits, and
    should say so. Caveat on my own generalization: the 1-2 bit frontier
    was measured on the V3/V4 ABSTRACTION tensors, not on Dense-C or
    Hypernetwork parameters. Whether those tolerate the same depth is
    untested, so no absolute figure should be restated downward without
    measuring that model's own frontier.

CHECKED AND CONSERVATIVE:
  * "total retained description length falls 63.3%" survives, and if
    anything UNDERSTATES the effect. The ratio is only rescaling-proof
    if task-private and shared components compress equally, and they do
    not. Nats paid per task at 4 bits/scalar, each component scored only
    on the tasks it actually affects:

        shared abstractions   2.0 / 1.2 / 4.0
        task-private          24.5 /  -  / 11.1   (world 1 has no live
                                                   private tasks)

    Shared state is 6-14x more compressible per task than private state.
    Promotion moves description FROM private INTO shared, so the 8-bit
    proxy overcharges the post-promotion side more than the
    pre-promotion side, and the true reduction exceeds 63.3%.

METHOD NOTE. The first version of this measurement returned exactly 0.0
for task-private at every bit depth, which reads as "infinitely
compressible" and is instead "never measured": the loss was scored over
`refs`, the tasks that USE abstractions, which are precisely the retired
tasks whose private residuals are inactive. Second vacuous guard this
session -- the other printed "CONTROL HOLDS" over zero rows. A check
that cannot fail is worse than no check; both now score each component
only where it acts.

# CODING-FRONTIER AUDIT: V3's description claim SURVIVES (2026-08-19)

The audit that had to be settled before anything else: does promotion's
description reduction survive when BOTH representations are placed on
their own behavioral rate-distortion frontiers, rather than compared at
a fixed 8-bit serialization?

Method. For each arm, find the minimum bits/scalar per component -- task
-private residuals and shared parameters separately -- such that total
held-out Gaussian loss rises by no more than a matched per-task budget.
Each component is scored ONLY on the computations that depend on it.
Paired arms on development worlds 0-2: promotion disabled
(`shared_residual`) versus promotion enabled (`lifecycle`).

| world | promoted / private (AFTER) | reduction @8bit | reduction @frontier |
| --- | --- | --- | --- |
| 0 | 56 / 9  | 71.2% | 73.1% |
| 1 | 64 / 1  | 79.4% | 80.1% |
| 2 | 49 / 16 | 60.2% | 55.8% |
| mean | | 70.3% | 69.7% |

Tolerance sensitivity, mean reduction at the frontier: 71.1% at 2
nats/task, 69.7% at 10, 73.1% at 30 -- stable across a 15x range.
Selected component depths are 4-5 bits for task-private state and 4-5
for shared, so both sides do carry slack; they simply carry comparable
amounts, which is why the ratio holds.

OUTCOME 1. Promotion reduces description length even after each
representation is independently compressed near its behavioral frontier.
The absolute bit counts in V1-V4 are inflated by roughly 2x at this
tolerance, but the STRUCTURAL claim is unaffected: PROMOTE is not merely
substituting one naive numerical encoding for another. This answers the
main available criticism of the two-part accounting, and it makes the
V3 result harder to dismiss rather than easier.

A PROVENANCE BLOCKER, and the first attempt at this audit was void
because of it. V3's SEALED artifacts (300-329) do not persist
`task_reference` or `retired` -- both are plain Python containers absent
from `state_dict`, the same defect recorded for the V4.1 oracle. Without
them every task reloads as private in BOTH arms, promotion's entire
saving disappears, and the audit reported reductions of -9.6%, -5.5%,
-5.5%. Those figures are VOID; they measure a loading bug, not a
representation. The audit is therefore run on development worlds where
the lifecycle learner persists the table, and the sealed worlds CANNOT
be audited this way without re-running them. Any future artifact whose
analysis depends on the promoted/private split must persist that split.

LIMITS. Development worlds 0-2, n=3, deltas reported. The sealed-world
figure (63.3%) is not itself re-derived here; what is shown is that on
matched development worlds the same comparison survives frontier
optimization, with the fixed-precision and frontier numbers agreeing to
within 1 point on average.

# V5 proposed hypotheses — preregistered before any V5 experiment (2026-08-20)

These six hypotheses form a ladder, not a single version. They are
committed here before any V5 world is generated, in the same spirit as
the V1/V2/V3 confirmation plans. None has been run.

## H19 — Code-cost invariance

The V4 retention law H* = lambda*D(A)/s_bar predicted 17.1 returning
tasks and observed 17.9 (within ~5%). V5.1 deliberately and
independently manipulates D(A) (store equally functional abstractions
at 2, 3, 4, 6, 8 bits/scalar) and s_bar (vary per-reuse savings while
holding D fixed), predicting before running that the empirical
retention threshold follows H*(D) = lambda*D/s_bar -- i.e., H* is
proportional to D and inversely proportional to s. If halving the code
cost approximately halves the threshold, and halving s_bar approximately
doubles it, then N*·s ~ C is an empirical law of neural abstraction
economics, not one lucky boundary. No fitted threshold.

Prediction: the threshold moves quantitatively as C/s predicts across
at least two independent manipulations of C and two of s.

Confidence: high. The single-point V4 result was unfitted and within 5%.
The strongest risk is that s_bar changes when D changes (coarser
operators may be slightly less useful per reuse), which would bend but
not break the proportionality.

## H20 — Higher-order amortization

At sufficiently high abstraction-level recurrence (r_meta) and library
scale (M), FACTORIZE overtakes matched-budget COMPRESS. V4 found no
factorization crossing below ~16 abstractions. V5.2 controls r_meta
and M independently while holding task support per abstraction fixed.
Predicted: M small => COMPRESS wins even at decent r_meta; M↑ and
r_meta↑ eventually produce FACTORIZE. The experiment discovers
M*(r_meta), which is V1's recurrence experiment one abstraction level
higher.

Prediction: a COMPRESS-to-FACTORIZE crossing exists and M*(r_meta) is
monotone non-increasing in r_meta.

Confidence: moderate. The existence of a crossing is theoretically
expected from the amortization law, but the required M may exceed the
216-program world's ceiling unless primitives are added.

## H21 — Prospective schema reuse

A learned operator family A(z;alpha) makes a novel family member
cheaper to acquire by learning only its arguments alpha_new rather
than a complete new operator. Measured by prequential cost, samples to
criterion, retained bits, and held-out behavior, always against a
matched-budget independently compressed operator.

Prediction: family acquisition beats full acquisition on at least
prequential cost and retained bits when the family was fit on enough
members.

Confidence: moderate. V4's leave-one-abstraction-out test recovered
only ~7.5% of the centre-only deficit, so the current rank-2 family is
weak. A richer family parameterization or higher r_meta may be needed.

## H22 — Economic edit selection

One prospective scoring rule selects the correct representation edit
across regimes where different edits are oracle-optimal (low recurrence
=> KEEP, repeated innovation => PROMOTE, bloated atoms => COMPRESS,
related atoms => FACTORIZE, future return => RETAIN, obsolete => RETIRE).
Initially hand-designed, not learned.

Prediction: a single scoring rule matches the oracle-optimal edit in
a majority of (regime, world) cells.

Confidence: moderate. The rule must generalize across structurally
different edits; the risk is that no single hand-designed estimator
covers all regimes.

## H23 — Structural planning

When edits alter future library formation, finite-horizon structural
planning beats myopic per-object decisions. V4 showed deleting A can
trigger replacement promotion A', saving 0 bits -- path-dependence.
V5.5 compares myopic (argmin_e J_{t+1}), short-horizon rollout Q_h, and
clairvoyant oracle.

Prediction: under library evolution or nonstationarity, rollout
structural regret < myopic structural regret. In stationary worlds,
myopic ~ oracle.

Confidence: high for the qualitative claim (myopic is insufficient
under path-dependence); moderate for the quantitative margin.

## H24 — Learned restructuring

A policy trained on restructuring trajectories reduces structural regret
on held-out worlds/economic conditions not seen during training (e.g.,
held-out code cost, held-out horizon). The first test of "train the
optimizer with the model."

Prediction: the trained policy's structural regret is lower than the
hand-designed estimator's on held-out conditions.

Confidence: low-moderate. This is the most ambitious rung and depends
on H22 and H23 producing usable signal first.

# V5 preflight diagnostics — NOT H19 (2026-08-20)

Scored existing artifacts as development diagnostics. Neither grid is
the H19 protocol in notes/v5-sketch.txt (unpaired seeds, slots=12,
no s-arm, no rank 4). Outcomes appended so the scores are not
mistaken for the registered test.

Rank 1, artifacts/v5_causal, worlds 500-509, N in {48,56,64},
slots=12: s_bar = 36.5 nats/use (flat across H_R=8/16/24), crossing
15.2 against predicted 15.0 at 8-bit carry 549. Zero post-gap-birth
exclusions.

Rank 2, artifacts/v5_horizon, worlds 0-2, N in {48,52,56,60,64,72},
slots=12: s_bar = 60.1, crossing 17.9 against predicted 18.3 at carry
1098. This cell reproduces the V4R O4 development numbers
(s_bar ~61, crossing 17.9).

Unpaired rank-2/rank-1 ratios: D = 2.00, H* = 1.18, s_bar = 1.65.
If this survived a paired grid it would be sketch T1 (s_bar moves
with rank, so H* does not track D). It is not paired. H19 as
specified remains unrun.

Compositional closure, Continuous checkpoints worlds 0-2: Hungarian-
matched slots vs teacher primitives, random programs. Median NMSE
depth-8 / depth-3 = 4.39 / 4.26 / 4.09, all under the draft 5x gate;
neither teacher nor learned saturates at depth 8. V6 is not blocked.
The gate is tight: error grows smoothly, it does not explode.

# Frontier audit on ALL 30 SEALED worlds: the claim strengthens (2026-08-19)

The development-world audit above is superseded by the real thing. The
sealed worlds could not initially be audited because V3 artifacts do not
persist `task_reference`/`retired`. Resolved WITHOUT re-opening anything:
`LifecycleLibraryLearner` is behaviourally identical to the promoting
learner with its flags off, so re-running seeds 300-329 reproduces the
same trajectory and persists the table. Verified BIT-EXACT against the
original artifacts in the first three worlds (delta 0.0 nats in each);
these are already-scored worlds, and sealed seeds 400-429 were not
touched.

Componentwise behavioral rate-distortion, each component scored only on
the computations depending on it, tolerance 10 nats/task:

    MEAN over 30 sealed worlds:  @8bit 67.6%   @frontier 71.1%
    frontier reduction >= fixed-precision reduction in 29/30 worlds

The reduction is LARGER once both representations are independently
compressed near their behavioral frontiers. Promotion is therefore not
buying its description saving from numerical slack that quantization
could have harvested anyway; the saving grows when that slack is removed
from both sides. This is the strongest available form of H11.1 and it
answers the main criticism of the two-part accounting.

DISCREPANCY TO RECONCILE BEFORE PUBLICATION. This audit's
fixed-precision figure is 67.6%, while the paper reports 63.3% for the
same sealed block. The gap is an accounting-scope difference, not a
disagreement about the data: this audit prices task-private residuals
plus abstractions plus shared basis, and the paper's retained-description
figure evidently includes or excludes a component differently (route or
reference state being the likely candidate). The two must be reconciled
and a single accounting fixed before either number is quoted. Until then
the defensible claim is the RELATIVE one: whatever the exact scope,
frontier optimization does not reduce the effect and increases it in
29/30 worlds.

LIMITS. Single tolerance (10 nats/task) on the sealed block; the
development-world check showed stability from 2 to 30 nats/task but that
sweep was not repeated here. Uniform per-component scalar quantization,
not an optimal code, so every D* is an upper bound on the true minimum
description length -- which makes the comparison conservative in the
right direction for both arms.

# Reconciled: frontier audit at the paper's own accounting scope (2026-08-19)

The 67.6% vs 63.3% discrepancy is resolved, and my first explanation for
it was wrong. I guessed `reference_bits_total`; including or excluding
those changes the reduction by 0.1 points (63.3% -> 63.5%), so they are
not the cause.

The actual cause is ROUTE/CODE STATE. `score_v3_sealed.py` prices
retained task state including per-task route codes (1,170 scalars),
which are retained in BOTH arms. My audit omitted them. Adding the same
constant to numerator and denominator moves the ratio toward 1, so
omitting it INFLATES the apparent reduction — which is exactly the
4-point gap.

Re-run with route codes included as a third independently-quantized
component, matching the scorer's scope:

    MEAN over 30 sealed worlds:  @8bit 62.6%   @frontier 68.7%
    frontier reduction >= fixed-precision reduction in 30/30 worlds

The fixed-precision figure now reproduces the paper's 63.3% to within
0.7 points (residual: the scorer's `task_state_scalar_count` and my
live-residual-plus-codes count differ slightly in which tasks' residuals
count as retained). The two accountings are therefore the same
measurement, and the earlier 67.6% is superseded by 62.6%.

FINAL RESULT. At the paper's own scope, placing every component on its
own behavioral rate-distortion frontier raises the description reduction
from 62.6% to 68.7%, in 30/30 sealed worlds. Promotion's economic claim
does not depend on both representations being stored wastefully; it is
larger when they are not. H11.1 survives in its strongest form.

Method note: three components (task-private residuals, shared
abstractions plus basis, route codes), each scored only on the
computations depending on it, uniform scalar quantization at a matched
10 nats/task behavioral budget. Uniform quantization is not an optimal
code, so each D* is an upper bound on true minimum description length —
conservative for both arms.

# V5.1 CAUSAL TEST: the law holds, the proportionality prediction FAILS (2026-08-20)

Registered prediction: `H_R* proportional to D(A)`, manipulated at the
generator via residual rank (99 / 198 / 396 scalars at rank 1 / 2 / 4).
200 lifetimes, 10 worlds per cell, controlled protocol, 0 excluded.

| rank | D(A) | carry | s_bar | crossing | predicted (carry/s_bar) |
| --- | --- | --- | --- | --- | --- |
| 1 |  99 |   549 | 36.8 | 15.2 | 14.9 |
| 2 | 198 | 1,098 | 61.0 | 18.0 | 18.0 |
| 4 | 396 | 2,196 | 68.3 | 32.4 | 32.2 |

PROPORTIONALITY: FALSIFIED. D(A) ratio rank4/rank1 = 4.00; observed
crossing ratio = 2.13; relative error 46.8%.

THE LAW ITSELF: CONFIRMED, and this is not the same statement. At each
rank, carry and s_bar are measured independently and the crossing lands
within 2% of carry/s_bar (14.9 vs 15.2, 18.0 vs 18.0, 32.2 vs 32.4).
The relation predicts across three regimes it was not fitted to, so it
is not the accounting identity the single-point V4R confirmation could
not rule out.

WHY PROPORTIONALITY FAILS. `s_bar` co-varies with D(A): 36.8, 61.0,
68.3. A higher-rank abstraction is both more expensive to carry AND more
useful per use. Factoring that out closes the arithmetic exactly:
D ratio 4.00 / s_bar ratio 1.86 = 2.15 against an observed 2.13.

Reading. The governing relation is `H_R* = lambda * D(A) / s_bar(D)`,
and in this substrate D(A) is NOT an independently manipulable knob --
residual rank sets an abstraction's cost and its expressive capacity
together. Testing proportionality alone requires a manipulation that
inflates the ENCODING of an abstraction without changing what it can
compute (padding, or a coarser code), which is a different intervention
and is not yet run.

This is the second time in this project that a registered prediction
failed while the mechanism behind it survived (cf. H5b, where the mean
recurrence curve smoothed but cross-world dispersion worsened). Report
both halves; the failure of the simple scaling form is as informative as
the success of the underlying relation.

# H25-H27 — recursive amortization (registered 2026-08-20, before any V5.2 world)

Registered after review 44 and before the V5.2 generator exists. The
V5.1 outcome licenses exactly one form of the law -- `H* = lambda D*(A)
/ s_bar`, with cost and utility measured SEPARATELY -- so the
higher-order version is registered in the same form and never as a
proportionality.

## H25 — the schema crossing is predictable from unfitted quantities

With `A_i = S(alpha_i) + eps_i`, private cost `sum_i D*(A_i)` and schema
cost `D*(S) + sum_i [D(alpha_i) + D*(eps_i)]`:

    FACTORIZE  iff  M * s_bar_schema > D*(S)
    M*(rho)    =    D*(S) / s_bar_schema(rho)

PREDICTION: with `D*(S)` and the leave-one-out per-member saving
`s_bar_schema` measured independently at a fixed meta-recurrence rho,
the observed COMPRESS-to-FACTORIZE crossing in a sweep over realized
library size M lands within 15% of `D*(S)/s_bar_schema`. Both
quantities are recorded before the M sweep is run; fitting the crossing
and then explaining it with them does not count.

Falsified by: a crossing outside 15%, or no crossing at any reachable M
when the predicted M* is reachable.

## H26 — meta-recurrence moves the boundary, and only the boundary

    dM*/drho < 0

PREDICTION: the more related the abstractions, the fewer family members
are needed before naming the family. Registered jointly with the
generator BALANCE GATES, which are a precondition on reading H26 at all
and are frozen here at +/- 10%: across the swept rho, each of
`D*(A_i)`, `s_bar(A_i)`, per-abstraction behavioral contribution, and
promotion rate must be constant to within 10%. If any gate fails, the
sweep is UNSCOREABLE, not weak evidence -- the boundary would then move
for the same confounded reason proportionality failed in V5.1.

## H27 — shared scalars are individually cheaper, and it is a mechanism

V5.0 measured `D*_shared ~ 3.9` bits/scalar against `D*_private ~ 5.0`.
PREDICTION: this gap is not a selection artifact. Specifically, promoted
abstractions have lower effective functional rank than private residuals
of matched participant count, and the per-component `D*` gap is
predicted by that spectral difference (rank-order correlation across
worlds |r| >= 0.5) rather than by abstraction size or usage count.

Falsified by: no spectral difference; or a spectral difference that does
not track `D*`; or the gap disappearing once participant count is
matched, which would make it a selection effect.

Not registered: any padding/dead-bit manipulation of `D(A)`. Review 44's
objection is accepted -- under the rate-distortion currency this project
now uses, `D*(A + dead bits) = D*(A)`, so padding tests a storage tax
rather than abstraction economics. The V5.1 note proposing it as the
clean proportionality test is superseded.

# H25 protocol amendment and H29 (registered 2026-08-20, reviews 46-47)

## H25 amendment, before any V5.2 world

H25 registered `M* = D*(S)/s_bar_schema` within 15%. Review 47 shows
the protocol as written could not deliver that: re-fitting S at every
M makes both inputs functions of M, so the prediction would use
quantities that depend on what it predicts. The interval is unchanged;
the protocol is now fixed. `D*(S)` and `s_bar_schema` are measured on a
calibration set of M_0 = 4 members with S FROZEN thereafter, written
down before any member is added, and members are added one at a time
without re-fitting. A crossing scored against a re-fit schema (H20b) is
descriptive and does NOT score H25.

## H26 amendment

The four balance gates become three. Promotion rate is removed: it is a
learner response, not a generator property, and gating on it would
discard the outcome where PROMOTE births fewer atoms because the
lower-level representation already absorbs the commonality. That
outcome is now registered as a reading in its own right (H20b outcome
3), not a failure. `D*(A_f)`, `s_bar(A_f)` and behavioral contribution
remain hard preconditions at 10%.

## H29 — abstraction changes coding geometry, decomposed

Registered before the audit runs. On the same residual clusters, with
PROMOTE-rejected clusters of matched size as the control:

    P_0   private residuals as they stood before promotion
    P_1   one functionally fitted shared residual, no further training
    P_2   that shared residual after post-promotion SGD

PREDICTION: `D*(P_2) < D*(P_1) < D*(P_0)`, and the selection term
`D*(P_0 selected) < D*(P_0 rejected)` accounts for less than half of
the measured 5.0 -> 3.9 bits/scalar gap. That is: the gap is mostly
made, not selected.

Falsified by: selection accounting for the majority of the gap; or
`D*(P_2) >= D*(P_1)`, which would mean continued learning does not
restructure the code even though sharing helps.

This is the causal form of H27, which stays as the cheap correlational
audit. The stronger sequel — matched lifetimes with PROMOTE disabled
versus enabled, compared at matched behavioral performance — is NOT
registered here; it waits until the decomposition says which term
dominates.

# S0 protocol, frozen before Stage 1 (2026-08-20, review 48)

Registered before any `p_reuse`, `s_conditional`, or fixed-window
`s_bar(g)` measurement exists. Seven `g=0.5 / N=56 / retained` cells and
one `g=1.0 / N=56` pair survive an aborted batch; they are development
scratch and are NOT the Stage-1 estimate, which runs at a separate
fixed window.

## What S0 is, stated so it cannot drift

S0 is NOT a pure s-only intervention and is not claimed as one. The gain
scales the family primitive's residual scale, so at `g != 1` the
post-gap operator is a SCALED version of the pre-gap one:

    g changes the value/match regime while D(A) remains fixed.

`D(A)` is fixed in the strong sense — the carried tensor is
bit-identical across `g`, so `lambda * D*(A)` is the same number, not a
number within tolerance. That is what H19 needs; the weaker semantic
story ("same function, worth more") is not claimed.

## Registered refusal threshold

    p_reuse(g) = P(retained A is selected | returning task)

PREDICTION and REFUSAL: `p_reuse(g) >= 0.5` at every gain. Below that,
the retained arm has effectively stopped using the abstraction, S0 is
measuring selection collapse rather than payoff, and the gain is
reported UNSCOREABLE rather than folded into the law.

## Registered decomposition

    s_bar(g) ~= p_reuse(g) * s_conditional(g) + residual

with `s_conditional` the mean saving on returning tasks that actually
route to `A`. Three readings, all legitimate for the law because it uses
measured `s_bar`, but distinguishable in mechanism:

    A  s_conditional up, p_reuse flat   the abstraction became more
                                        economically important
    B  p_reuse down                     the scaled target drifted far
                                        enough that routing stopped
                                        trusting A
    C  both                             mixed; report as mixed

## Two-stage protocol, deterministic and committed in advance

STAGE 1 estimates the predictor ONLY, at a fixed window (N=72,
H_R=32) large enough for a stable rate. It does not locate a crossing.

STAGE 2 estimates the crossing from SEPARATE runs. Using one horizon
sweep for both the predictor and the outcome would partially reintroduce
the identity concern that V5.1 exists to rule out.

The Stage-2 grid is a deterministic function of Stage 1, fixed here
before any Stage-2 outcome exists:

    1. measure s_hat(g) at the fixed window
    2. H_hat*(g) = lambda * D*(A) / s_hat(g)
    3. H_low  = largest admissible horizon <= H_hat*(g) - 4
       H_high = smallest admissible horizon >= H_hat*(g) + 4
    4. add interior points at the two horizons nearest H_hat*(g),
       rounded to the N grid (N = H_R + 40, even N only)

## Registered statistic: the dimensionless collapse

    chi = H*_observed * s_bar / (lambda * D(A))

PREDICTION: chi = 1 across every operating point — ranks 1, 2, 4 from
the D-arm and every gain from S0 — with all points falling on the
identity line of observed against predicted threshold. Registered
tolerance: mean |chi - 1| <= 0.15 across operating points, and no single
point outside [0.7, 1.3].

## Retained, not scored here: reuse under drift

`delta(g)` = functional distance between the carried `A` and the scaled
post-gap operator, recorded alongside `s_bar(g)`. S0 incidentally
produces the first reuse-under-drift curve, which is the FORK /
specialization question in V6+. Measured and stored; no prediction is
registered on it in V5.

## S0 amendment at baseline, before any gain data (2026-08-20)

The absolute `p_reuse >= 0.5` registered above FAILS at `g = 1.0` — the
unmodified world — which makes it a specification error, not a result.
Measured on the first paired cell (N=56, world 500): 4 / 16 returning
tasks route to the retired abstraction, while 16 / 16 route to SOME
abstraction, and `s_bar` is a healthy 63.1 against V4R's sealed 61.0.

The threshold was meant to catch "the gain made routing abandon A". A
bound that the baseline already violates cannot detect that. Amended to
the relative form the review offered, fixed now while only `g = 1` data
exists and before any `g != 1` cell is scored:

    REFUSE gain g if  p_reuse(g) < 0.5 * p_reuse(g = 1)

reported alongside the any-abstraction rate and `s_conditional`. The
absolute form is withdrawn and recorded here rather than deleted.

## Two properties of the retention instrument, measured not assumed

Both are inherited from the V4R O4 protocol and apply to its sealed
result as much as to S0. Recorded because they were found by looking.

1. DELETING ONE ABSTRACTION COLLAPSES THE REUSE PATHWAY. The deleted
   arm keeps abstraction 0 live, yet 0 / 16 of its returning tasks adopt
   anything, against 16 / 16 in the retained arm (12 to id 0, 4 to the
   retired id 1). So `C_reacquire` prices the loss of the reuse
   behaviour, not the direct use of one object. Consistent with that,
   the saving is flat across the split: 61.3 nats/use on tasks routed to
   the retired abstraction, 63.7 on tasks routed elsewhere.

2. `D_retain - D_delete = D(A)` IS AN ACCOUNTING CONVENTION, NOT A
   MEASURED DIFFERENCE. Both arms' checkpoints store identical scalar
   counts — 2 abstractions (396 scalars), 57 task residuals (11,286),
   57 task codes (2,052). Retirement is logical; the deleted arm is
   charged as though it need not store `A`, and referencing a live
   abstraction does not reduce a task's stored residual count.

Neither blocks S0, and S0 is unusually robust to both: whatever the
carry convention is, it is IDENTICAL across `g` by construction, so the
s-arm varies only the measured saving. They do bear on how the D-arm's
absolute crossings should be described, and on any future per-object
carry claim.

# S0 STAGE 1 RESULT and the Stage-2 grid it determines (2026-08-20)

Stage 1 measured the PREDICTOR only, at a fixed window N=72 (H_R=32),
10 worlds per gain, both arms, 0 excluded, 0 pre-intervention leaks.
No crossing was located and none is claimed; Stage 2 estimates the
threshold from separate runs.

| g | s_bar (nats/use) | C_reacquire | p_reuse (w500) | s_conditional |
| --- | --- | --- | --- | --- |
| 0.5 | 28.9 |   925 | 0.41 | 27.0 |
| 1.0 | 58.1 | 1,858 | 0.44 | 52.8 |
| 1.5 | 85.2 | 2,725 | 0.44 | 80.2 |

CARRY INVARIANCE: the abstraction checksum is identical in 10/10 cells.
The carry term is the same number across gains, not a number within
tolerance, which is what makes this a clean s-arm.

MONOTONE in g: PASS. s_bar ratio g=1.5/g=0.5 = 2.95 against a gain ratio
of 3.00 — very nearly linear, better than the sub-linear behaviour
expected from the tanh and the mismatch channel.

MECHANISM, against the registered decomposition: reading A. `p_reuse` is
flat (0.41 / 0.44 / 0.44) while `s_conditional` rises (27.0 / 52.8 /
80.2). The gain made the same abstraction more economically important;
it did not make routing abandon it. Reading B (selection collapse) is
excluded, and the relative refusal threshold never fired.

`g=1.0` reproduces the V4R sealed operating point: s_bar 58.1 across 10
worlds (60.1 on world 500) against the sealed 61.0.

## Stage-2 grid, computed by the frozen rule, committed before any Stage-2 outcome

H_hat*(g) = lambda * D(A) / s_hat(g) at the 8-bit carry of 1,098 nats,
then H_low <= H_hat* - 4, H_high >= H_hat* + 4, plus the two nearest
interior even horizons. N = H_R + 40.

| g | s_hat | H_hat* | N grid |
| --- | --- | --- | --- |
| 0.5 | 28.9 | 38.0 | 72, 76, 78, 82 |
| 1.0 | 58.1 | 18.9 | 54, 58, 60, 64 |
| 1.5 | 85.2 | 12.9 | 48, 52, 54, 58 |

PREDICTION: each gain's observed crossing lands within 15% of its
H_hat* above. These are parameter-free — every input was measured at a
window that contains no crossing.

At the D* currency (535 nats) the predicted crossings are 18.5 / 9.2 /
6.3, which the grids above do NOT bracket. The 8-bit currency governs
grid selection, as it did for the D-arm; D* crossings are reported where
a grid happens to bracket them and are otherwise declared unscoreable
rather than interpolated.

## slots=6 protocol robustness, g=1.0, N=72, 10 worlds (2026-08-20)

The registered constant is `operator_slots = 6`; the scored D-arm ran at
12. Both artifact roots were internally consistent at 12, so this is
external protocol robustness, not repair of an invalid experiment.

    slots=12   s_bar 58.1   library 3.3 abstractions   59.0 tasks referencing
    slots= 6   s_bar 44.3   library 2.2 abstractions   46.7 tasks referencing

`D(A)` is unchanged (residual rank 2 = 198 scalars either way), so the
law predicts the crossing moves purely as 1/s_bar: H* 18.9 -> 24.8.

FINDING: the protocol constant is NOT neutral for magnitude. Halving the
slot budget costs 24% of the per-use saving, and it does so by producing
a smaller library (3.3 -> 2.2 abstractions) that fewer tasks reference
(59.0 -> 46.7). The D-arm's absolute crossings are therefore
protocol-dependent and are not transferable between slot budgets.

NOT YET TESTED: whether `H* = lambda D / s_bar` still holds at slots=6.
That needs its own bracketing grid around 24.8, and until it is run the
slots=6 arm establishes that s_bar moves, not that the law survives.
Carry invariance held 10/10 here as well.

## Stage-2 grid: one correction, made before any Stage-2 cell ran

The committed g=0.5 grid began at N=72 (H_R=32). Two reasons to move
that point to N=74 (H_R=34), both applied before running:

  1. The registered rule says `H_low` is the LARGEST admissible horizon
     <= H_hat* - 4. With H_hat* = 38.0 that is 34, not 32; the helper
     script's floor arithmetic emitted the conservative value. N=74
     follows the registered rule more exactly.
  2. N=72 at g=0.5 is the fixed window Stage 1 measured s_bar on. Using
     it again as a Stage-2 outcome point would let one cell serve as
     both predictor and outcome, which is precisely the shared-noise
     path the two-stage split exists to close.

Final Stage-2 grids, unchanged otherwise:

    g=0.5   N 74, 76, 78, 82     (H_R 34, 36, 38, 42)   H_hat* 38.0
    g=1.0   N 54, 58, 60, 64     (H_R 14, 18, 20, 24)   H_hat* 18.9
    g=1.5   N 48, 52, 54, 58     (H_R  8, 12, 14, 18)   H_hat* 12.9

No Stage-2 cell shares a run with Stage 1. 240 cells.

# WHAT CLOSES H19 — frozen 2026-08-20, before g=1.5 was scored

Written while batch 3 was still running and before any g=1.5 cell was
read. Not to be modified after that point lands.

## The claim, stated so it admits what actually happened

NOT "we independently manipulate D and s". That is false and the
project has already paid once for a claim of that shape.

    The D-arm manipulates abstraction CAPACITY, jointly changing
    storage cost and per-use utility. The S-arm holds the stored
    abstraction EXACTLY fixed — bit-identical tensors — while
    manipulating its post-return utility regime. Across both
    interventions the measured threshold is predicted by
    C_carry / s_bar.

The invariant is neither `H* ~ D` (falsified in V5.1 at 46.8% error)
nor `H* ~ 1/g`. It is

    H* = C_carry / (expected saving per future use).

## H19 PASSES iff all of the following hold on the g=1.5 grid

1. Routing validity: `p_reuse(1.5) >= 0.5 * p_reuse(1.0)`.
2. The predicted crossing 12.9 is bracketed by the grid (H_R 8, 12, 14,
   18) with >= 4 returning tasks on both sides.
3. Observed-vs-predicted relative error < 0.25, the registered H19
   falsifier. The tighter Stage-2 registration of 15% is reported
   separately; missing 15% while meeting 25% is a PASS with the miss
   stated, not a quiet pass.
4. The S-arm delivers >= 1.5x variation in s_bar across gains. (Already
   satisfied at 2.95x from Stage 1; restated so it is part of the
   criterion rather than an assumption.)
5. Zero post-gap births, zero pre-intervention leaks, in every scored
   cell.
6. No off-grid interpolation. A crossing outside the grid is refused,
   not extrapolated.
7. The slots=6 pairing stays a SEPARATE robustness debt. It is not
   evidence for or against the s-arm verdict, and H19 does not borrow
   its result either way.

If all seven hold: H19 PASS, and the rung moves off partial P1 + P3.
The claim licensed is then, within this testbed:

    The lifetime over which a reusable computation is worth retaining
    is quantitatively determined by its representation cost divided by
    its realized per-use saving.

## Why this matters beyond RETAIN

The decision rule is not "is this abstraction big?" nor "is this
abstraction useful?" — neither is sufficient. It is "how many future
uses are required to amortize what this representation costs?" A small
abstraction of little use may deserve deletion; a large one with a large
per-use benefit may deserve retention.

The same form is what H25 and V6 are registered against, which is why
H19 is foundational rather than a local threshold:

    atom birth      N_uses      * s_bar_A > D(A)
    atom retention  H_R         * s_bar_A > D(A)      <- H19
    schema birth    M           * s_bar_S > D(S)      <- H25
    macro birth     N_programs  * s_bar_M > D(M)      <- V6

# H19 VERDICT: PASS (2026-08-20), scored against the criterion frozen above

Stage 2 ran 240 cells, 3 gains x 4 horizons x 10 worlds x 2 arms, zero
failures, zero post-gap births, zero pre-intervention leaks.

| operating point | D(A) | s_bar | carry | H* predicted | H* observed | err | chi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-arm rank 1 |  99 | 36.8 |   549 | 14.9 | 15.2 | 2.0% | 1.019 |
| D-arm rank 2 | 198 | 61.0 | 1,098 | 18.0 | 18.0 | 0.0% | 1.000 |
| D-arm rank 4 | 396 | 68.3 | 2,196 | 32.2 | 32.4 | 0.6% | 1.008 |
| S-arm g=0.5  | 198 | 28.9 | 1,098 | 38.0 | 38.7 | 1.8% | 1.019 |
| S-arm g=1.0  | 198 | 58.1 | 1,098 | 18.9 | 19.2 | 1.6% | 1.016 |
| S-arm g=1.5  | 198 | 85.2 | 1,098 | 12.9 | 13.0 | 0.8% | 1.009 |

    mean |chi - 1| = 0.012   (registered <= 0.15)
    chi range      = [1.000, 1.019]   (registered inside [0.7, 1.3])

## The seven frozen conditions

1. Routing validity. PASS, and more strongly than required: p_reuse is
   IDENTICAL between g=1.0 and g=1.5 in every world inspected
   (0.22/0.22, 0.00/0.00, 1.00/1.00 at the common N=58). The gain moved
   payoff without touching selection.
2. Bracketing. PASS. 12.9 sits inside H_R {8, 12, 14, 18} with 4.9 and
   5.1 returning tasks on either side.
3. Relative error < 0.25. PASS at 0.6%, and also inside the tighter
   Stage-2 registration of 15%. No miss to report.
4. s_bar variation >= 1.5x. PASS at 2.95x.
5. Zero post-gap births, zero leaks. PASS, 240/240 cells.
6. No off-grid interpolation. PASS; every crossing is interior to its
   own grid.
7. slots=6 stays a separate robustness debt. Honoured; the verdict
   borrows nothing from it in either direction.

## What is licensed, and what is not

LICENSED, within this testbed:

    The lifetime over which a reusable computation is worth retaining is
    quantitatively determined by its representation cost divided by its
    realized per-use saving.

The D-arm manipulates abstraction CAPACITY, moving storage cost and
per-use utility together. The S-arm holds the stored abstraction
bit-identical and moves only its post-return utility regime. Across both
interventions the threshold is predicted by C_carry / s_bar. Neither
`H* ~ D` (falsified, 46.8%) nor `H* ~ 1/g` is the law; the ratio is.

NOT LICENSED, stated so the result is not over-read:

  * SIX POINTS ON A LINE, NOT A 3x3 GRID. Every S-arm point shares
    D(A) = 198, so all cost-axis spread comes from the D-arm. A crossed
    design (each gain at each rank) is not run.
  * ONE CURRENCY. Every crossing is priced at the 8-bit proxy. At D*
    the predicted gain crossings are 18.5 / 9.2 / 6.3, which these grids
    do not bracket, so the law is confirmed in one currency and untested
    in the other.
  * ONE PROTOCOL POINT. All of it is slots=12, rank 2 for the S-arm,
    worlds 500-509, one dormancy geometry. slots=6 changes s_bar by 24%
    and its crossing is unmeasured.
  * DEVELOPMENT, NOT CONFIRMATORY. Worlds 500-509 are contaminated V5
    development seeds. Nothing here is sealed, and 600-629 remain
    untouched.
  * The instrument's two known properties still hold: deleting one
    abstraction collapses the reuse pathway, and D_retain - D_delete =
    D(A) is an accounting convention. Both are identical across gains,
    which is why the S-arm is unaffected, but both bear on the absolute
    numbers.

# H20a / H25 / H26 FIRST RESULT — the amortization law one level up (2026-08-20)

Exogenous atoms, schema fitted on M_0 = 4 and FROZEN, every piece coded
to the same distortion budget. F=12, K=2, worlds 0-2, probe 256.
Offline; no learner, no lifetimes.

| r_meta | s_bar_schema | M* predicted | M* observed | matched-budget winner |
| --- | --- | --- | --- | --- |
| 0.00 |  -33 |  inf | none | COMPRESS |
| 0.50 |   -9 |  inf | none | COMPRESS |
| 0.70 |   24 | 39.7 | none (M* unreachable at F=12) | FACTORIZE |
| 0.90 |  103 |  8.0 | 7.5 | FACTORIZE |
| 0.95 |  157 |  5.1 | 5.7 | FACTORIZE |
| 1.00 |  495 |  1.6 | 2.0 | FACTORIZE |

## H26 — dM*/dr_meta < 0: SUPPORTED

M* falls monotonically from unreachable through 39.7, 8.0, 5.1 to 1.6.
The more related the abstractions, the fewer members are needed before
naming the family. The COMPRESS -> FACTORIZE boundary sits between
r_meta 0.5 and 0.7, and the structureless control behaves: at r_meta = 0
the schema has NEGATIVE per-member saving and never pays, which is the
V4R result reproduced as the low-relatedness limit of a knob.

## H25 — the frozen-schema point prediction: 2 PASS, 1 MISS, 3 not applicable

    r_meta 0.90   observed 7.5 vs predicted 8.0    6.4%   PASS
    r_meta 0.95   observed 5.7 vs predicted 5.1   10.1%   PASS
    r_meta 1.00   observed 2.0 vs predicted 1.6   25.7%   MISS

The miss is reported as a miss. It is also, on inspection, at the edge
of what the metric can resolve: observed M is an INTEGER, so with a
predicted 1.6 the attainable values are 1 and 2 and the finest possible
relative error is 25%. ceil(1.6) = 2 is exactly what was observed, so
the law is consistent with the data at that point while the registered
15% criterion is not attainable there.

REGISTERED FOR NEXT TIME, not applied retroactively: at small M* the
criterion should be `observed == ceil(predicted)` or
`|observed - predicted| <= 1`, whichever is stated in advance. The 15%
relative form is only meaningful for M* >~ 7. The r_meta 0.90 and 0.95
cells satisfy both forms.

Also recorded: at r_meta = 0.90 only 2/3 worlds produced a crossing
within the 8 unseen members available; the third needs F > 12.

## What this does and does not establish

DOES: a schema over abstractions has an economic region; it is entered
by relatedness, not by scale alone; and its crossing is predicted by
`D*(S) / s_bar_schema` measured on a frozen schema before the members
are counted. That is the same ratio form H19 just confirmed one level
down, which is the first evidence for the recursive reading.

DOES NOT: say anything about whether PROMOTE reaches this region. These
are exogenous atoms by design (D16). H20b is the separate experiment,
and outcome 3 there — the learner collapsing meta-structure into fewer
atoms rather than forming a schema — remains a live and interesting
possibility.

## H20b first reading: PROMOTE does not deliver M ~ F (2026-08-20)

Cheap-first slice, F=4, m=16, K=2, N=72, worlds 0-2, learned library,
sleeps 16/24/32/48/64.

    r_meta = 0.0    realized M = 5, 6, 7   mean 6.0
    r_meta = 1.0    realized M = 7, 9, 5   mean 7.0

Against the three registered outcomes, this is none of them cleanly.
Outcome 3 (the learner collapsing meta-structure into fewer atoms) does
NOT occur: M does not fall as relatedness rises, it is flat to slightly
higher. Nor is it outcome 1 or 2 as stated, because those assume the
learner "produces ~F atoms" and it produces MORE — 1.5 to 1.75x F.

So the premise carried over from review 44, that F explicit recurring
families make PROMOTE yield M ~ F, is NOT satisfied at this operating
point. The learner fragments each family across several abstractions.
n = 3 worlds and the spread is wide (5 to 9), so this is a direction,
not an estimate.

CONSEQUENCE FOR G2, recorded before the accounting is built: a schema
fitted over the LEARNED library would be fitting over fragments of
families rather than over one object per family. That is not a reason
to skip H20b — it may be the most interesting thing about it — but it
does mean H20b's G2 is not the same measurement as H20a's, and the two
must not be reported as one number.

## Why H20b's G2 is not yet run, stated rather than left implicit

A learned abstraction is a flat 198-vector of (u, v, b) over three
steps. Fitting a schema to those vectors is parameter-space fitting,
which is exactly the gauge-sensitive move this project forbids: two
abstractions computing the same function can have unrelated parameter
vectors, and a schema that "captures" them may be capturing coordinates.
H20a avoided this because the teacher's family operators share hidden
features by construction, so a mixture of U matrices IS a mixture of
functions. The learned atoms have no such shared frame.

A correct H20b G2 therefore needs a functional accounting: fit the
schema on probe-set effects, and charge bits for what must be stored to
reproduce those effects to the matched budget. That is a real piece of
design, and V4.2 is the standing lesson about what happens when a
sharing claim is scored with the convenient accounting instead of the
matched-budget one.

## H20b G2 RESULT: the schema economy exists, the learner does not reach it

Cheap-first slice, F=4, N=72, worlds 0-2, learned libraries. Schema
fitted in EFFECT space (gauge-free) and charged in PARAMETER space, with
uncovered members paid at full private price so no leftover code is
invented in FACTORIZE's favour.

    r_meta = 0.0   FACTORIZE wins 0/3 worlds   3,157 vs 1,630 bits
    r_meta = 1.0   FACTORIZE wins 0/3 worlds   3,814 vs 2,258 bits

G2 FAILS, 0/6. And the reason is not the bit accounting — the schema
covers 0% of unseen atoms at BOTH relatedness levels, which was checked
against the vacuous-guard failure mode before being reported:

    unexplained fraction of a held-out atom's effect variance
        r_meta = 0.0   0.909
        r_meta = 1.0   0.873

So learned abstractions do NOT lie in a shared low-dimensional
functional family, even at r_meta = 1.0 where the TEACHER's family
operators lie in one perfectly (R_LOO = 1.000, H20a). The absolute
residual is only ~5x the distortion budget, so a looser budget would
flip the binary "covered" flag; the 87-91% unexplained fraction is what
carries the result and it is not budget-sensitive.

## Why this is the result the H20a/H20b split was built to produce

Read together with the realized-M reading, the picture is specific:

    H20a   over TEACHER family operators, a schema pays, and its
           crossing is predicted by D*(S)/s_bar_schema (H25 passes at
           r_meta 0.90 and 0.95, H26 monotone)
    H20b   over the LEARNED library, PROMOTE produces 1.5-1.75x more
           atoms than families, and those atoms share almost none of
           the family's functional structure, so no schema pays

The representation class is not the problem; the promoter is. Had H20
been run only on the learned library — the design before review 46 and
47 — this would have read as "higher-order factorization does not pay"
and closed the question. The exogenous-atom arm is what makes the two
distinguishable, and D16 is vindicated.

REGISTERED READING, before any follow-up: this is n=3 worlds at one
operating point (F=4, K=2, M_0=3, slots=12). It is a direction. The
obvious next question is whether PROMOTE can be made to produce
family-aligned atoms at all — which is a question about the promoter's
clustering, not about schemas — and that is not a V5.2 rung as
currently written.

## H21 PROSPECTIVE SCHEMA REUSE: PASS at r_meta 1.0, FAIL at 0.9 and 0.0

Leave-one-family-out on teacher operators, F=8, K=2, worlds 0-2, target
held-out functional MSE 1e-3.

    r_meta   examples to target: schema / independent   verdict
      0.0    never (floor 8.4e-3) / 8                   FAIL
      0.9    never (floor 1.21e-3) / 8                  FAIL
      1.0    1 / 8                                      PASS

At r_meta = 1.0 a frozen schema acquires an unseen family member from a
SINGLE example, against 8 for independent acquisition, and retains 2
arguments against 128 operator scalars. Both halves of the registered
falsifier are satisfied.

The two failures are informative rather than surprising, and they are
the falsifier working as written ("a win on loss at extra bits is
capacity"): at r_meta < 1 a rank-2 schema cannot express a member that
is only partly in the subspace, so its error PLATEAUS above the target
(1.21e-3 at r_meta 0.9) no matter how many examples arrive. That is a
capacity limit of the schema, not a sample-efficiency loss.

TWO CAVEATS, recorded with the result:

1. The independent baseline here is unrealistically strong. Hidden
   features are shared and known, so acquiring the operator is an
   exactly-determined linear solve that hits machine precision the
   moment support reaches 8 (128 unknowns, 16 outputs per example). A
   learner that must discover its own features would need far more.
   The 1-vs-8 win at r_meta = 1.0 is therefore a LOWER BOUND on the
   schema's advantage, and the r_meta < 1 failures are measured against
   a baseline no real learner attains.
2. The registered target is a fixed absolute MSE. A target above the
   schema's expressible floor would flip r_meta 0.9 to a pass on
   samples. Not done here, because changing a threshold after seeing
   where the floor lies is precisely what this project forbids. The
   honest follow-up is a schema+leftover arm at matched bits, which is
   registered as the H20a ladder and not yet run prospectively.

## H27 VERDICT: the D* gap is real, the spectral mechanism is FALSIFIED

16 frozen artifacts, shared abstractions against an equally sized
sample of private residuals, matched participant count.

    mean D* gap (private - shared)     +0.253 bits/scalar
    mean spectral gap (private - shared)  -0.157
    rank correlation of the two gaps      -0.009   (registered |r| >= 0.5)

Both halves of the registered prediction fail, and they fail in
different ways:

1. DIRECTION. H27 predicted promoted abstractions would have LOWER
   effective functional rank — faster spectral decay — than private
   residuals. Observed the OPPOSITE: shared objects run at
   sigma_2/sigma_1 ~ 0.70 against ~0.54 for private ones. A shared
   abstraction uses MORE of its available functional dimension, not
   less. "Effective-dimensionality reduction" is out as the mechanism.
2. RELATIONSHIP. The registered link between the two gaps is absent:
   rank correlation -0.009 against a required |r| >= 0.5. Whatever
   makes shared scalars cheaper, it is not tracked by how many
   functional directions the object uses.

WHAT SURVIVES. The sign of the D* gap itself: private residuals cost
more per scalar than shared abstractions, reproducing the direction of
V5.0's 5.0-vs-3.9 headline on a different probe and budget (the
magnitude here is smaller, +0.25, and the two are not directly
comparable). So the phenomenon is real and one of its four candidate
explanations is now eliminated. Noise purification, the PROMOTE
selection effect, and representation restructuring remain live, and
they are exactly the three H29 was designed to separate.

This is the third time in this project a registered prediction has
failed while the phenomenon behind it survived (H5b, H19(a), now H27).
The pattern is worth naming: a mechanism guess attached to a real
measurement is the cheapest thing to get wrong and the most useful
thing to have registered.

## H29 BLOCKED by a provenance gap, not by cost

The P_0 / P_1 / P_2 decomposition needs the same cluster at three
moments. A finished lifecycle artifact holds P_2 (the trained shared
abstraction) and the surviving private residuals, but NOT P_0 — the
residuals as they stood at the sleep that promoted them. Nothing
checkpoints them there.

This is the same class of defect as the missing `task_reference` table
that once voided a coding audit and forced a 30-world re-run. Fixing it
is a one-field change to the promoting learner's sleep path: record the
member residuals it consumed. H29 is not attempted until then, and no
approximation is substituted.

# SEALED BLOCK, C2 (schema crossing) — seeds 600-629 opened 2026-08-20

Scored against `V5_CONFIRMATION_PLAN.md`, frozen at 1ed227d and hashed
into `tools/check_prereg.py` before these worlds were generated.

| r_meta | predicted M* | worlds with a crossing | within 15% | observed == ceil(pred) |
| --- | --- | --- | --- | --- |
| 0.00 | inf | 0/30 | — | — |
| 0.50 | inf | 0/30 | — | — |
| 0.70 | inf | 0/30 | — | — |
| 0.90 | 20.2 | 14/30 | 11 | 10 |
| 0.95 |  6.5 | 26/30 | 19 | 20 |
| 1.00 |  1.7 | 30/30 | 13 | **30** |

## Verdicts

MONOTONE M*(r_meta): PASS. inf, inf, inf, 20.2, 6.5, 1.7.

r_meta = 0 NEVER PAYS: PASS. Negative per-member saving in every world;
the V4R negative reproduced as the low-relatedness limit of a knob.

r_meta = 1.0 UNDER THE REGISTERED CEIL CRITERION: PASS, 30/30. The plan
excluded this point from the 15% test in advance, because integer M
cannot resolve 15% at a predicted 1.7, and specified
`observed == ceil(predicted)` instead. It is exact in every world.

r_meta in {0.90, 0.95} AT 15%: **AMBIGUOUS AS REGISTERED, and reported
as such.** The plan says "within 15% ... in >= 4 of 6 sealed cells" and
does not say what the denominator is when a world produces no
observable crossing:

    of worlds that CROSSED     r=0.90  11/14 = 79%   r=0.95  19/26 = 73%
    of ALL worlds              r=0.90  11/30 = 37%   r=0.95  19/30 = 63%

On the first denominator both clear 4/6 = 67% and C2 passes. On the
second both fall short and C2 fails. I will not pick the flattering
reading of my own plan after seeing the numbers, so C2's 15% clause is
recorded as UNRESOLVED, with both figures reported and the cause named:

REACHABILITY, not a mechanism failure. At F = 12 with a calibration set
of 4 there are only 8 unseen members, so any world whose M* exceeds 8
cannot show a crossing however well the law holds. Sealed r=0.90
predicted M* = 20.2 on average — well past 8 — which is why 16 of 30
worlds are silent there rather than wrong. Development did not hit this
because its r=0.90 predicted 8.0, right at the edge.

REGISTERED FOR ANY REPLICATION, not applied retroactively: the
denominator must be stated with the threshold, and F must satisfy
F - M_0 > M* at the r_meta being tested. The clean rerun is F = 32 at
r_meta 0.90, which is a generator change and a new fingerprint.

## What C2 establishes despite the unresolved clause

The three unambiguous predictions all pass — monotonicity, the r=0
negative, and the ceil criterion at r=1.0 in 30/30 worlds. Where a
crossing was observable at all, roughly three quarters of worlds landed
within 15% of a prediction made from a frozen schema before the members
were counted. The recursive form
`M* = D*(S) / s_bar_schema` survives its first out-of-sample test; the
part that failed is my specification of how to count it.

## H29 UNBLOCKED, and one of its three terms is structurally zero

The provenance gap is fixed: `LifecycleLibraryLearner` now overrides
`sleep` to snapshot, per promotion, the member task ids and their
residuals AS THEY STOOD BEFORE the promotion consumed them (P_0), plus
the abstraction as born (P_1). The frozen V3 learner is untouched — the
override lives in the subclass and changes no decision, only
bookkeeping. Two tests pin it.

Building it surfaced something the registered prediction assumed away.
H29 reads

    D*(P_0 selected) < D*(P_0 rejected)   ->  SELECTION
    D*(P_1) < D*(P_0)                     ->  PURIFICATION
    D*(P_2) < D*(P_1)                     ->  RESTRUCTURING

but promoted abstractions are created with `requires_grad=False` and
appear in no optimizer group, so they NEVER change after birth:

    P_2 == P_1, identically, in this learner.

The restructuring term is therefore not small or hard to measure — it is
structurally zero, and no run of this architecture can produce evidence
for or against it. That also eliminates a second of the four candidate
mechanisms for the D* gap: "representation restructuring of the shared
object by continued learning" cannot be the explanation, because the
shared object does no continued learning. Combined with H27 eliminating
effective-dimension reduction, two of four are now out and the live
candidates are NOISE PURIFICATION and the PROMOTE SELECTION EFFECT —
which are exactly the two terms the snapshot makes measurable.

A caveat against over-reading: the SYSTEM still restructures around a
frozen abstraction, because task residuals and routes keep adapting to
it. What is ruled out is restructuring OF the shared object, which is
what H29 registered. A version of the claim about the surrounding
representation is a different hypothesis and is not registered here.

Existing artifacts predate the snapshot and cannot be rescored; H29
needs fresh runs, which is the honest cost of having found the gap late.

# SEALED C1 STAGE 1, and the Stage-2 grid it fixes (2026-08-20)

Seeds 600-609, fixed window N=72 (H_R=32), 60 cells, 0 failures,
0 post-gap births, 0 pre-intervention leaks. The predictor only; no
crossing is located here and none is claimed.

| g | s_bar sealed | s_bar development | carry invariance |
| --- | --- | --- | --- |
| 0.5 | 26.6 | 28.9 | identical 10/10 |
| 1.0 | 57.3 | 58.1 | identical 10/10 |
| 1.5 | 84.3 | 85.2 | identical 10/10 |

s_bar monotone in g: PASS. The sealed worlds reproduce the development
per-use savings to within 8% at every gain, which is itself a
replication of the S0 intervention on untouched seeds.

## Stage-2 grid, deterministic from Stage 1, committed before any Stage-2 cell runs

H_hat*(g) = lambda * D(A) / s_hat(g) at the 8-bit carry of 1,098 nats;
H_low <= H_hat* - 4, H_high >= H_hat* + 4, plus the two nearest interior
even horizons. N = H_R + 40.

| g | s_hat | H_hat* | H_R grid | N grid |
| --- | --- | --- | --- | --- |
| 0.5 | 26.6 | 41.3 | 36, 40, 42, 46 | 76, 80, 82, 86 |
| 1.0 | 57.3 | 19.1 | 14, 18, 20, 24 | 54, 58, 60, 64 |
| 1.5 | 84.3 | 13.0 |  8, 12, 14, 18 | 48, 52, 54, 58 |

PREDICTION (C1, as frozen in V5_CONFIRMATION_PLAN.md): each gain's
observed crossing lands so that chi = H*_observed * s_bar / (lambda D)
has mean |chi - 1| <= 0.15 across operating points, with no point
outside [0.7, 1.3]. Every input above was measured on runs that contain
no crossing, and no Stage-2 cell shares a run with Stage 1.

240 cells. Report: `reports/v5_sealed_c1_stage2_grid.json`.

## Sealed C1, first crossing (g=0.5), scored 2026-08-20

Grid committed before the batch ran; predictor from Stage 1's separate
fixed-window runs. 80 cells, 0 failures, all 10 worlds contributing at
every horizon.

| H_R | n | C_reacquire | V_retain |
| --- | --- | --- | --- |
| 36 | 10 |   941 | -157 |
| 40 | 10 | 1,042 |  -56 |
| 42 | 10 | 1,081 |  -17 |
| 46 | 10 | 1,204 | +106 |

    observed H* 42.5   predicted 41.3   error 3.0%   chi 1.031

Monotone in H_R with a single sign change. Inside the registered
[0.7, 1.3] and inside the tighter 15%. One of three C1 points; the
verdict waits on g=1.0 (predicted 19.1) and g=1.5 (predicted 13.0),
which are queued.

# SEALED C1 VERDICT: PASS (2026-08-20)

Seeds 600-609, two-stage, 300 cells total (60 Stage 1 + 240 Stage 2),
zero failures, zero post-gap births, zero pre-intervention leaks, every
crossing interior to its own grid.

| g | s_bar | H* predicted | H* observed | error | chi |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 26.6 | 41.3 | 42.5 | 3.0% | 1.031 |
| 1.0 | 57.3 | 19.1 | 18.9 | 0.8% | 0.989 |
| 1.5 | 84.3 | 13.0 | 12.9 | 0.5% | 0.993 |

    mean |chi - 1| = 0.016   (registered <= 0.15)
    chi range      = [0.989, 1.031]   (registered inside [0.7, 1.3])

Every protocol condition the plan made mandatory held: the predictor was
measured at a fixed window containing no crossing; the Stage-2 grid was
a deterministic function of Stage 1 and committed before any Stage-2
cell ran; carry invariance was verified by checksum rather than
tolerance; no cell served as both predictor and outcome.

The crossing moves by a factor of 3.3 across the sweep (42.5 to 12.9)
while the carried abstraction is bit-identical in every arm. Nothing
about the three predictions was fitted to the data that tested them.

## What the sealed block licenses

    Within this testbed, the lifetime over which a reusable computation
    is worth retaining is quantitatively determined by its
    representation cost divided by its realized per-use saving.

Stated to admit what each arm did: the development D-arm manipulated
abstraction CAPACITY, moving cost and utility together; the S-arm here
holds the stored abstraction exactly fixed and moves only its
post-return utility regime. Across both, the threshold is predicted by
C_carry / s_bar. It is not `H* ~ D` (falsified at 46.8% in V5.1) and not
`H* ~ 1/g`.

## Scope, unchanged by the pass

Ten sealed worlds at one protocol point (slots=12, rank 2, one dormancy
geometry, 8-bit currency). The D-arm was NOT rerun on sealed seeds, so
the sealed evidence covers the s-arm only; the six-point collapse
reported for H19 remains development plus this. The slots=6 crossing
grid and the D* currency grids remain the named debts they were.

# SEALED C3 and C4 (2026-08-20)

## C3 — the learned library does not realize the schema economy: PASS, 3/3 clauses

12 sealed lifetimes, seeds 600-605, r_meta 0 and 1, F=4.

    FACTORIZE fails vs matched-budget COMPRESS   0/6 wins at BOTH r_meta
                                                 (registered: fails >= 5/6)
    unexplained fraction at r_meta = 1.0         0.921
                                                 (registered: > 0.6)
    realized M > F                               12/12 cells, mean 7.2 and
                                                 7.7 against F = 4
                                                 (registered: >= 4/6)

Every clause replicates, and the unexplained fraction is HIGHER on
sealed worlds (0.921) than in development (0.873). The negative is a
real property of the promoter, not a development artifact.

## C4 — the coding-geometry mechanism: MIXED, and one clause FAILS

    D* gap positive (private costs more)   6/12 cells   registered >= 5/6
                                                        -> FAIL
    shared shows FASTER spectral decay     0/12 cells   registered: should
                                                        not -> PASS
    rank correlation of the two gaps       +0.888       registered within
                                                        [-0.5, 0.5] -> FAIL

Two of three clauses miss, in opposite directions, and both are
reported rather than reconciled.

THE SIGN CLAUSE FAILS. Private residuals cost more per scalar than
shared abstractions in only 6 of 12 sealed cells — a coin flip. The
mean gap is +0.087 bits/scalar, down from +0.253 in development and far
below V5.0's headline 1.1. So the D* gap that motivated H27 does NOT
survive as a per-world regularity at this operating point. V5.0's
measurement stands on its own worlds and accounting; what fails is the
generalization I registered from it.

THE CORRELATION CLAUSE FAILS IN THE INTERESTING DIRECTION. Development
found no relationship between the D* gap and the spectral gap
(-0.009), and I registered that it would stay absent. On 12 homogeneous
sealed artifacts the rank correlation is +0.888. So there IS a
relationship between how much cheaper a shared object is and how its
functional spectrum compares to a private one — the opposite of what
the development null suggested.

WHAT SURVIVES CLEANLY. Shared abstractions never show faster spectral
decay: 0/12 cells, and 0/16 in development. H27's original direction is
falsified twice over, on separate worlds. Shared objects consistently
use MORE of their functional dimension, not less.

READING, registered before any follow-up: H27's mechanism question is
now in worse shape than a clean negative. The phenomenon it was
explaining (a robust per-scalar D* gap) is itself unstable across
worlds at this operating point, while a relationship the development
data said was absent appears strongly in the sealed data. Both point at
the same methodological fact -- 16 development artifacts pooled across
two different roots were too heterogeneous to estimate either quantity.
The honest next step is not another mechanism guess but a
properly powered measurement of whether the D* gap exists per-world at
all, which V5.0's aggregate never established.

# D* CURRENCY GRIDS: two of three viable, committed before running (2026-08-20)

At the frontier carry (3.9 bits/scalar x 198 scalars = 535 nats) the
predicted crossings sit at roughly half the 8-bit ones, so each gain
needs its own grid. Derived by the same frozen rule from the SEALED
Stage-1 s_bar values, and committed here before any cell runs:

| g | s_bar | H* at D* | H_R grid | N grid | status |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 26.6 | 20.1 | 16, 20, 22, 26 | 56, 60, 62, 66 | viable |
| 1.0 | 57.3 |  9.3 |  4,  8, 10, 14 | 44, 48, 50, 54 | viable |
| 1.5 | 84.3 |  6.3 |  2,  6,  8, 12 | 42, 46, 48, 52 | **NOT VIABLE** |

g = 1.5 IS UNREACHABLE AT THIS GEOMETRY, and that is a structural fact
rather than a budget choice. Bracketing 6.3 by four returning tasks
requires a low point at H_R = 2, and the promotion protocol's
`minimum_cluster = 3` means a two-task return window cannot be scored at
all. Measuring the D* crossing at high gain would need a different
dormancy geometry — an earlier gap, so more tasks remain after it —
which is a different world and a new fingerprint, not a re-scoring.

PREDICTION for the two viable gains: the observed crossing lands within
15% of the D* prediction, i.e. chi computed at the frontier carry stays
inside [0.7, 1.3]. If it does, the law is currency-independent and the
8-bit result is not an artifact of the proxy. If the D* crossings miss
while the 8-bit ones hit, then `lambda D(A)` is doing work that the
frontier accounting does not reproduce, and P2 of H19's registered
partials ("the law is currency-dependent") is the reading.

# FRAGMENTATION AUDIT (review 48, Hypothesis A): REJECTED

Sealed C3 artifacts, 12 cells. Unexplained fraction of a TEACHER family
operator by the best k-subset of the learned library, measured on the
state distribution the operator actually acts on:

    condition   atoms   F     k=1     k=2     k=3
    r0            7.2   4   0.938   0.926   0.923
    r100          7.7   4   0.935   0.918   0.911

Fragmentation predicts k=1 poor and k=2 or 3 good. Observed: k=3 barely
improves on k=1 (drop 0.015 and 0.025) and all of it stays above 0.91.
One family is NOT split across several promoted atoms. M > F is not
oversegmentation.

INSTRUMENT NOTE, recorded because it nearly produced a wrong number.
The first run probed at N(0, I) and reported 0.98 unexplained at every
k. The family primitive fires at the LAST program step, so its input is
the state after two base-primitive steps — a distribution neither object
acts on at the origin. Probing at the operating distribution moved the
figure to 0.92-0.94. The conclusion is unchanged, but the first version
was measuring the wrong thing and would have overstated the effect.

## What this does NOT settle, and why the next test is global

A promoted abstraction is a RESIDUAL on top of the learner's own shared
basis, which already models much of the computation. So it should not be
expected to equal the teacher's family operator even in principle — it
should equal whatever the basis leaves over. This audit therefore
rejects Hypothesis A but cannot separate:

    B  the learner found a DIFFERENT, equally economical basis, and the
       teacher's decomposition is simply not the one it uses
    C  promotion preserved the computation but in per-object coordinates
       a plain schema cannot see (the H28 reading)
    D  PROMOTE's objective is myopic and its objects are locally
       economical but globally poor building blocks

Distinguishing them needs the GLOBAL comparison review 48 names:
`D*(L_learned) + D*(programs | L_learned)` against the teacher-aligned
representation at matched behaviour. If the learner's library is equally
cheap overall, B is right and there is nothing to fix at the task level
— the cost is in future extensibility. If it is dearer, PROMOTE is
leaving description on the table and REFACTOR has something to do.

Registered before that test runs: I expect B. The learner's basis is
fitted to its own residual structure and there is no gradient pressure
anywhere in V3's PROMOTE toward the teacher's decomposition, so
recovering it would be a coincidence rather than a consequence.

# WHOLE-LIBRARY ECONOMY, FIRST ATTEMPT: INVALID INSTRUMENT, RESULT RETRACTED

The run produced learner/teacher cost ratios of 14-17x, 0/6 worlds to
the learner at both r_meta, which would read as B3 — "PROMOTE leaves
enormous compression on the table". That reading is WITHDRAWN. The
instrument is rigged and the number means nothing.

THE FLAW. I defined each task's target as the innovation it needs at the
family step, and took that from the world as
`residual_effect(teacher_operator, z)`. But that target IS the
teacher-aligned library's own object, so its residual is identically
zero and its per-task cost collapses to a reference index, while the
learner has to code a large leftover against a target defined in the
other side's terms. The comparison asks "how well does each library
reproduce the teacher's operators", which is teacher recovery — the
exact question review 48 said to stop asking, and which the
fragmentation audit already answered.

A 16x gap should have been the tell. No representation that solves the
same tasks to the same behaviour is sixteen times dearer; a ratio that
large means the two sides were not being asked the same question.

WHAT A VALID VERSION NEEDS. The target must be the task's own required
BEHAVIOUR — its held-out outputs — not either library's internal
object. Each library is then charged for reproducing that behaviour to
a matched tolerance:

    J_rep(L) = D*(L) + D*(per-task state | L) + L_behaviour(holdout)

with the teacher-aligned arm REFIT to the tasks rather than handed the
answer: its operators are a proposed library like any other, and its
per-task residuals must be fitted under the same budget the learner's
were. Only then are both sides solving the same problem with different
vocabularies.

Registered before the rebuild, unchanged from before: I still expect B
(roughly tied). Nothing in the flawed run bears on that either way.

STANDING LESSON, and the third time this class of error has appeared in
this project (V4.1's tolerance, the raw-ablation counterfactual, now
this): when one arm of a comparison is defined in terms of the other
arm's objects, the winner is decided by the definition. Check which side
the TARGET belongs to before reading any ratio.

# slots=6 DEBT CLOSED: the protocol moves the numbers, not the relation

Sealed worlds 600-609, g=1.0, grid N {60, 64, 66, 70}, 80 cells, 0
failures.

| H_R | n | C_reacquire | V_retain |
| --- | --- | --- | --- |
| 20 | 10 |   913 | -185 |
| 24 | 10 | 1,118 |  +20 |
| 26 | 10 | 1,179 |  +81 |
| 30 | 10 | 1,351 | +253 |

    s_bar 45.7   observed H* 23.6   predicted 24.8   error 4.8%
    chi 0.982

Halving the slot budget cuts s_bar 24% (58.1 -> 45.7) through a smaller
library that fewer tasks reference, while D(A) is unchanged at 198
scalars. The law predicted the crossing would move purely as 1/s_bar,
18.9 -> 24.8. It moved to 23.6.

So the protocol constant changes the NUMBERS substantially and leaves
the RELATION intact. Absolute crossings stay protocol-dependent and must
be quoted with their slot budget; the ratio survives the change. This
converts the slots=6 item from a named debt into a seventh operating
point.

## Seven points on the identity line

| point | D(A) | s_bar | H* pred | H* obs | chi |
| --- | --- | --- | --- | --- | --- |
| D-arm rank 1 |  99 | 36.8 | 14.9 | 15.2 | 1.019 |
| D-arm rank 2 | 198 | 61.0 | 18.0 | 18.0 | 1.000 |
| D-arm rank 4 | 396 | 68.3 | 32.2 | 32.4 | 1.008 |
| sealed g=0.5 | 198 | 26.6 | 41.3 | 42.5 | 1.031 |
| sealed g=1.0 | 198 | 57.3 | 19.1 | 18.9 | 0.989 |
| sealed g=1.5 | 198 | 84.3 | 13.0 | 12.9 | 0.993 |
| slots=6      | 198 | 45.7 | 24.8 | 23.6 | 0.982 |

Spanning a 4x range in D(A) and a 3.2x range in s_bar, with mean
|chi - 1| = 0.015. Three of the seven are on sealed seeds.

# H29 RESULT: the structure is mostly gone BEFORE promotion (2026-08-20)

Six fresh lifetimes at r_meta = 1.0 with P_0 snapshots, worlds 600-605.
Leave-one-out subspace capture, matched-count control, isotropic null.

    mean R_meta before promotion (matched count)   0.095
    mean R_meta after promotion                    0.052
    isotropic null                                 0.001

Neither of review 48's two clean readings holds exactly, and the module
refused to pick one. The honest statement is a third thing:

BOTH populations are ~10x above the null, so there IS a trace of shared
structure in the learner's residuals. But both are ~10x BELOW what the
same instrument reports on the teacher's family operators at this
r_meta, where capture is 1.000 by construction. So roughly 90% of the
available meta-structure is already missing in the private residuals,
before PROMOTE runs at all.

Promotion then roughly halves what little remains (0.095 -> 0.052). That
is a real secondary effect and it is in the destructive direction, but
it is not the main story: PROMOTE cannot destroy structure that its
inputs never carried.

## What this does to the REFACTOR hypothesis

Review 48's proposal was that a global sleep oracle might reorganize the
promoted library into a fertile one. This result says the ceiling on
that is low: a refactor of the promoted objects can only recover
structure present in them, and by the time PROMOTE sees its inputs, 90%
is already gone. REFACTOR-after-PROMOTE is repairing the wrong stage.

The question moves UPSTREAM, to what the wake learner encodes. Its
per-task residuals are fitted against a shared basis with no pressure of
any kind toward representing related tasks relatedly; two tasks drawing
on the same family operator can land on residuals that compute the same
thing in unrelated coordinates. That is Hypothesis C (lost canonical
coordinates) relocated one stage earlier than review 48 placed it, and
it makes the cheap-adapter alignment test (test C) the most informative
next measurement: if cheap e_i, d_i recover the family structure from
the PRE-promotion residuals, the information is there and the wake
learner's coordinates are the problem.

## Instrument note, and a near-miss worth recording

Two versions of this scorer reported "no structure before promotion"
for reasons that had nothing to do with the science.

1. The first snapshot read `record.supporting_tasks` from the lineage,
   which the parent syncs AFTER the sleep returns, so it stored zero
   member residuals and the audit reported pre_objects = 0.
2. The corrected snapshot worked, but the scorer then timed out
   silently -- the null does LOO refits over a ~70-object population of
   3,072-wide vectors, thousands of SVDs -- and left a STALE report on
   disk from run 1, which I read as a result.

The second is the more dangerous: a scorer that dies after writing
nothing leaves the previous answer in place, and nothing in the output
says so. Capping the P_0 population at 24 and the null at 10 draws makes
it tractable. Any audit that writes to a fixed path should either write
a run stamp or refuse to read a report older than its inputs.

## H29 INTERPRETATION CORRECTED (review 49): I measured the wrong unit

The entry above says "roughly 90% of the available meta-structure is
already missing in the private residuals, before PROMOTE runs at all",
and then slides into treating that as a statement about the LEARNER.
Those are not the same claim, and only the first is earned.

WHAT IS ESTABLISHED: the learner's RESIDUAL OBJECTS carry little of the
teacher's meta-family geometry before PROMOTE.

WHAT IS NOT: that the learner as a whole has lost it. The wake learner
has more places to put a family than the private residual. A task is
solved as

    shared basis  +  task route  +  private residual

so two tasks drawing on the same teacher family can carry residuals that
look unrelated because the common computation is split differently
between the route and the residual. If so, R_pre = 0.095 does not mean
the structure vanished; it means THE RESIDUAL TENSOR IS THE WRONG
COMPUTATIONAL UNIT to call "the task's innovation" — which is a mistake
this project has made before in a different costume (learner slot
indices vs teacher primitive indices; parameter means vs functions).

The claim that the REFACTOR ceiling is low is therefore also withdrawn
pending the corrected measurement. It follows from the over-claim, not
from the data.

## The measurement that decides it: effective-operator H29

Define, at the step where the family fires and on the on-trajectory
state distribution,

    F_tau(z)   the learner's ACTUAL transformation for task tau —
               basis mixture, route, residual, everything
    F_0(z)     the same with task-specific information nulled, not an
               arbitrary parameter baseline
    I_tau(z)   = F_tau(z) - F_0(z), the effective task-conditioned
               innovation

then run the same leave-one-out capture instrument on {I_tau} instead of
{Delta_tau}, giving R_effective.

REGISTERED FORK, before running it:

    R_effective high (~0.7+)   the information was never lost, only
                               DISTRIBUTED across the wake
                               representation. PROMOTE has been
                               promoting the wrong object, and
                               abstraction boundaries should be found
                               functionally rather than inherited from
                               parameter boundaries.
    R_effective low, cheap
    adapters recover it        coordinate-equivalent computation, at the
                               stage where it originates (H28/Hypothesis
                               C, one stage earlier than review 48 put
                               it).
    R_effective low, local
    adapters fail, one cheap
    GLOBAL rotation Q works    the meta-structure exists at the level of
                               the population, not in individual atoms.
    nothing recovers it        the wake objective does not identify the
                               fertile representation at all, and
                               post-hoc REFACTOR cannot be sufficient.

My expectation, registered: R_effective is materially higher than 0.095
but not high — the route carries some of the family and the residual
carries some, so neither unit alone shows it. That would put the answer
between fork 1 and fork 3 and would make the global-rotation test the
decisive one.

# H29 CORRECTED RESULT: the residual was hiding half of it (2026-08-20)

Effective task-conditioned operator, I_tau = F_tau - F_0 with the route,
the shared reference and the residual all included and task-specific
information nulled by the population mean route. Six worlds, r_meta =
1.0, 24 family tasks each, same LOO instrument.

    mean R_effective            0.190
    mean R_residual (H29)       0.095
    isotropic null              0.003
    teacher operators at r=1    1.000 by construction

Per world: 0.320, 0.314, 0.048, 0.225, 0.141, 0.090 — wide, so the mean
is a direction rather than an estimate.

VERDICT against the registered fork: the MIDDLE branch, which is what I
predicted before running it ("materially higher than 0.095 but not
high"). Measuring the right unit DOUBLES the visible family structure,
so the review is right that the residual tensor was the wrong
computational unit and the earlier 0.095 understated what the learner
holds. And it is still 5x below the teacher's, so the information is not
merely relocated — most of it is genuinely not in the learner's
task-conditioned computation either.

## What this leaves standing

The "90% lost before PROMOTE" claim is now properly retired: the honest
figure is that the learner's effective operators carry about a fifth of
the family geometry the teacher's do, with roughly half of what they
carry invisible in the residuals alone.

Two hypotheses remain live and are now cleanly ordered:

    global rotation   the family may exist at the POPULATION level —
                      one cheap Q over the library/innovations rather
                      than per-object adapters. This is the decisive
                      test, because a basis like B_1 = A_1 + A_2,
                      B_2 = A_1 - A_2 defeats every local adapter while
                      one global reparameterization recovers everything.
    wake objective    nothing in the wake objective rewards representing
                      related tasks relatedly, so SGD is free to pick
                      any member of a large equivalence class of
                      task-solving implementations, most of which
                      destroy cheap higher-order abstraction.

The two are distinguishable: if a cheap global Q lifts R_effective from
0.19 to ~0.8, the structure was there and the coordinates hid it. If no
cheap Q helps, the wake objective never encoded it and post-hoc
refactoring — global or local — cannot be the remedy.

Calibration note, since it cuts the other way as often as not: the
registered expectation was correct here, and the value of registering it
was that "0.190" could otherwise have been told as either a success
(doubled!) or a failure (still 5x short) after the fact.

# POPULATION SPAN: the global-rotation hypothesis is DEAD (2026-08-20)

Review 49's global-rotation idea, made well posed. A global ORTHOGONAL
rotation cannot change leave-one-out capture at all — capture is a
property of the point set — so the testable version is the span
question: does the teacher's family operator lie in span{I_tau}, the
learner's effective task-conditioned innovations? A basis like
B_1 = A_1 + A_2, B_2 = A_1 - A_2 defeats every local adapter while
leaving the spans identical, so this is exactly the hypothesis that
local tests cannot reach.

Unexplained fraction of a teacher family operator, six worlds, 24
innovations each:

    best single innovation      0.801
    top-2 principal directions  0.864
    top-4                       0.839
    top-8                       0.812
    FULL span (all 24)          0.707

The full-span figure is an UPPER BOUND on what any global linear
refactor could achieve, which is what makes this decisive rather than
another similarity measurement. At 0.707 the answer is no: 71% of a
teacher family operator lies outside everything the learner's population
spans. No reparameterization, cheap or expensive, local or global,
recovers it.

## Where that leaves V5's closing question

Four hypotheses were on the table for why a schema pays over teacher
operators and fails over the learned library. Three are now eliminated:

    A  fragmentation          rejected: k-subsets barely beat k=1
    C  local coordinates      not reachable: the objects are not
                              coordinate-variants of the teacher's
    global rotation           dead: not in the span

What remains is the wake objective. Nothing in it rewards representing
related tasks relatedly, so SGD is free to choose any member of a large
equivalence class of task-solving implementations, and the one it picks
does not contain the family structure in any linearly recoverable form.

This makes the ordering explicit and, I think, is V5's most consequential
negative: POST-HOC REFACTORING CANNOT BE THE REMEDY. Not local
alignment, not global reparameterization, not a sleep oracle over the
promoted library — none of them can recover structure that is not in the
span of what wake produced. If a fertile representation is wanted, the
pressure has to be present while the representation is being formed.

Registered before any follow-up: the corresponding positive claim — that
adding prospective pressure during wake DOES yield a representation with
recoverable family structure — is NOT established by any of this. It is
the natural next experiment and it is a V6 question, not a V5 one.

## One caveat that limits the strength

The learner's innovations are measured against its own shared basis,
which already absorbs part of every computation. A teacher operator
could in principle be reconstructible from innovations PLUS basis
outputs while not being in the innovation span alone. This audit does
not test that, and the honest scope of the claim is: not recoverable
from the task-conditioned innovations, which are the objects any
refactor over the library would actually manipulate.

# D* CURRENCY, first crossing: the law is not an artifact of the 8-bit proxy

Grid committed before the batch ran, derived from the sealed Stage-1
s_bar by the same frozen rule, at the frontier carry
(3.9 bits/scalar x 198 = 535 nats).

    g = 0.5, D* currency

    H_R   n   C_reacquire   V_retain
     16  10          441         -94
     20  10          549         +14
     22  10          597         +62
     26  10          703        +168

    s_bar 27.3   observed H* 19.5   predicted 20.1   error 3.0%
    chi 0.994

Every crossing scored in V5 until now used the 8-bit serialization
proxy, and H19's registered partial P2 named the risk explicitly: the
law could be currency-dependent, with `lambda D(A)` doing work the
frontier accounting does not reproduce. At this gain it is not. The same
lifetimes, re-priced at 3.9 bits/scalar instead of 8, put the crossing
at a completely different horizon (19.5 against 42.5 at 8-bit) and the
ratio still lands on 1.

That is a stronger form of the result than another gain at the same
currency would have been: it varies the ACCOUNTING rather than the
world, and the relation survives.

g = 1.0 at D* is running. g = 1.5 at D* is structurally unreachable at
this dormancy geometry (predicted crossing 6.3 needs a low grid point at
H_R = 2, below `minimum_cluster = 3`), and that was recorded before the
grids were derived rather than discovered as a gap afterwards.

## Two corrections to the claims above (review 50)

### 1. The refactoring negative was stated too broadly

I wrote "POST-HOC REFACTORING CANNOT BE THE REMEDY" and listed a sleep
oracle among the things ruled out. The span result does not support that
scope. What it rules out is:

    no SPAN-PRESERVING post-hoc refactor of the learned OBJECTS can
    recover the structure

i.e. `L -> QL`, or any relabelling, rotation, merge, split, or adapter
applied to the finished library. Those all live inside the span, and the
teacher operator is 0.707 outside it.

It does NOT rule out a sleep phase that returns to the original
experience and RE-SOLVES it under a different objective:

    (D, L) -> L'

That is not refactoring, because it is not a function of the library
alone. The distinction matters for what V6 may attempt, and the earlier
wording would have foreclosed the most promising architecture on
evidence that does not reach it.

### 2. The D* result is a robustness check, not a fourth causal point

I described the D* crossing as testing whether the law is
currency-dependent, which is right, but the surrounding language drifted
toward treating chi = 0.994 as another independent confirmation. It is
not: the same lifetimes are re-priced, so the outcome shares all its
noise with the 8-bit reading.

The disciplined statement:

    The amortization law survived replacement of the project's
    historical 8-bit proxy with the independently measured component
    rate-distortion currency, with the economically relevant crossing
    moving from 42.5 to 19.5 and chi staying at 0.994.

The CAUSAL evidence for H19 still comes from the independently generated
D and s regimes — the three residual ranks and the three return gains —
and the seven-point collapse should be quoted on those, not padded with
re-priced versions of cells already counted.

# D* CURRENCY COMPLETE: partial P2 closes favourably (2026-08-20)

Both viable gains, sealed worlds 600-609, grids committed before the
batches ran, frontier carry 535 nats.

| g | s_bar | D* predicted | observed | error | chi |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 27.3 | 20.1 | 19.5 | 3.0% | 0.994 |
| 1.0 | 58.4 |  9.3 |  9.3 | 0.3% | 1.018 |

g = 1.5 remains structurally unreachable: its D* crossing at 6.3 needs a
grid point at H_R = 2, below `minimum_cluster = 3`. Recorded when the
grids were derived, not discovered afterwards.

H19 registered partial P2 as the live risk — "the law is
currency-dependent; report both and do not pick after seeing". It closes
in the favourable direction. The economically relevant crossing moves by
more than a factor of two when the carry is re-priced from the 8-bit
proxy to the measured frontier (42.5 -> 19.5 at g=0.5, 18.9 -> 9.3 at
g=1.0), and chi stays on 1 at both.

STATED WITH THE DISCIPLINE REVIEW 50 ASKED FOR: this is a currency
robustness check, not a third and fourth causal point. The same
lifetimes are re-priced, so these results share all their noise with the
8-bit readings. The defensible sentence is:

    The amortization law survived replacement of the project's
    historical 8-bit proxy with the independently measured component
    rate-distortion currency.

The causal evidence remains the seven independently generated regimes —
three residual ranks and three return gains, plus the slot budget — and
the collapse should be quoted on those alone.

## V5.1's registered debts are now all discharged

    slots=6 crossing grid    closed, chi 0.982
    D* currency grids        closed at both viable gains
    s-arm (S0)               closed, three gains, sealed
    rank cap / B0            closed

H19 therefore stands as a full pass rather than partial P1 + P3: the
D-arm and an informative s-arm both ran, the s-arm on sealed seeds, and
the protocol and currency robustness checks both hold.

# H30-H36 — V6, prospective representation formation (registered 2026-08-21)

Registered before any V6 code exists, and before the V6 worlds are run.
V5 is the control: the same H20 worlds, the same architecture, only the
objective changes.

    CENTRAL CLAIM: ordinary task loss UNDERDETERMINES the representation.
    Prospective pressure breaks that degeneracy toward representations
    that make related future learning cheaper.

## The primary endpoint is adaptation cost, not geometry

    Phi(R) = E_{T'~F}[ C_adapt(T' | R_baseline) - C_adapt(T' | R) ]

with F a future-task distribution never used to fit the current task.
`R_effective` is a SECONDARY, mechanistic endpoint. A prospective
learner may find a representation unlike the teacher's and better for
future learning, and V4/V5 have twice punished this project for treating
geometry as the pass. The pass is economic.

## H30 — fertility exists (load-bearing)

At matched current-task loss and matched D*, prospective pressure
reduces adaptation cost on unseen related tasks:
`Phi(R_prospective) > Phi(R_standard)`.

## H31 — fertility is structurally specific

`Phi_related >> Phi_unrelated`. Registered as: the related-minus-
unrelated difference is at least half of the related gain. If both move
equally the finding is generic plasticity, not abstraction, and must be
reported as such.

## H32 — prospective pressure raises accessible structure

`R_effective(prospective) > 0.19`. Predicted range 0.4-0.6, explicitly
NOT ~1.0. A large Phi with a flat R_effective is a legitimate outcome
and means the learner found a different mechanism for cheap adaptation;
that is a finding about the library model being too restrictive, not a
failure.

## H33 — fertility makes schema economics realizable (the bridge)

V5: teacher FACTORIZE pays, learned FACTORIZE 0/6. Prediction: at least
2/6 cells become FACTORIZE > COMPRESS at matched frontier coding.
Genuinely uncertain. If H30 passes and H33 fails, meta-learning and
explicit language formation are distinct steps, which is itself a
result.

## H34 — prospective structure obeys the amortization law

With p = Pr(a future task belongs to a reusable family), organizing cost
C_F and per-use saving s_bar over horizon H:

    fertile organization pays iff  p * H * s_bar > C_F

Prediction: sweeping p produces a threshold, and the crossing is
predicted by independently measured C_F and s_bar within 25%. This would
put V5's law one level further up — even learning a good way of looking
at things amortizes.

## H35 — too much fertility pressure hurts

A non-monotonic optimum in the prospective weight: current loss and
unrelated-task adaptation both degrade at large pressure while related
adaptation saturates. Registered as a U-shape in total lifetime cost,
not merely "large eta is worse on something".

## H36 — meta-learned updates rediscover it without labels

An updater trained across lifetimes on lifetime prequential cost forms
representations with greater Phi than SGD on HELD-OUT worlds, with no
future information at update time.

## Anti-fooling guards, fixed now

* strict support/query split on every adaptation probe; the prospective
  gradient never sees the query set;
* the sibling used for prospective pressure is never the sibling used
  for evaluation;
* at evaluation both arms are adapted by the SAME standardized
  optimizer from frozen representations, so an advantage cannot be the
  optimizer;
* matched parameter budget and matched D*, plus a larger-ordinary-learner
  arm, so the effect cannot be spare capacity;
* four arms, because the failure pattern localizes the cause:
  ordinary / replay-joint / oracle-prospective / supervised-family.
  Supervised failing means a substrate problem; supervised working while
  prospective fails means an objective problem; prospective working
  while replay does not means future-adaptation pressure specifically;
  replay matching prospective means continual multitask learning was
  enough.

## The transfer kernel, registered as V6's structural instrument

    K_ij = saving when adapting to task j after learning task i

A fertile representation should have an internal geometry that PREDICTS
K. The registered analysis is the rank correlation between a
representation-space distance d_R(i,j) and K_ij. This matters beyond
V6: it defines relatedness as "learning one makes the other cheaper"
rather than "the teacher gave them the same label", which is the only
form available in data that has no teacher.

## V6.1 design conflict, found in the smoke test (2026-08-21)

The first four-arm smoke run returned BIT-IDENTICAL lifetimes for
ordinary, replay and prospective: cumulative loss -178,669 and M = 7 in
all three, with the hook firing 56 times and changing nothing.

Cause, and it is a genuine conflict rather than a coding error. V5's
protocol passes `--freeze-basis-at 8`, which sets `requires_grad=False`
on every shared parameter at task 8 — and families begin at task 8. V5
wanted that: with the basis frozen, new structure has nowhere to go but
task-local innovations, which is what makes an explicit promoter
necessary. V6's mechanism is the opposite: prospective pressure acts by
SHAPING the shared representation, so a frozen basis makes every
prospective gradient dead on arrival.

CONSEQUENCE FOR THE CONTROL, recorded because it weakens a claim I made
when registering V6. I wrote "V5 is the control: the same H20 worlds,
the same architecture, only the objective changes". That is no longer
exactly true. V6 must leave the basis trainable, so the honest control
is `ordinary` WITHOUT the freeze, run alongside the other three arms.
V5's numbers become a reference point rather than the control arm.

All four arms are re-run without the freeze. The comparison stays
internally matched — same world, same architecture, same schedule, one
knob — which is what the four-arm design needs. What is lost is the
ability to quote V5's 0.19 as the ordinary baseline for R_effective; the
V6 ordinary arm supplies its own.

# V6.1 FIRST RESULT: H30 NOT SUPPORTED at this pressure (2026-08-21)

Four arms x three worlds, unfrozen basis, oracle sibling selection.
Fertility measured on an UNSEEN FAMILY from the same shared subspace,
adapted by one standardized routine (Adam at the task learning rate, 60
steps) from frozen shared parameters.

Adaptation cost on related futures, by support size:

| arm | k=1 | k=2 | k=4 | k=8 | k=16 |
| --- | --- | --- | --- | --- | --- |
| ordinary    | 0.47 | 0.26 | 0.20 | 0.18 | 0.13 |
| replay      | 0.40 | 0.27 | 0.23 | 0.19 | 0.12 |
| prospective | 0.42 | 0.24 | 0.23 | 0.18 | 0.12 |
| supervised  | 0.50 | 0.30 | 0.23 | 0.18 | 0.12 |

Phi at k=1, per world:

    replay        +0.087, +0.029, +0.099   mean +0.072  sd 0.031  3/3 positive
    prospective   +0.216, -0.173, +0.087   mean +0.043  sd 0.162  2/3 positive
    supervised    +0.095, -0.216, +0.017   mean -0.035  sd 0.132  2/3 positive

H30: NOT SUPPORTED. The prospective mean is positive but smaller than
its own spread and one world reverses. On world 0 alone Phi_related was
+0.216, which I would have reported as a pass had I stopped there; it
did not replicate.

H31: not evaluable while H30 fails.

The scorer's threshold has been tightened accordingly, and the change is
recorded rather than silent: `Phi > 0` on a mean of three worlds is not
a pass. It now requires every world to agree in sign AND the mean to
exceed the spread. The earlier version printed PASS for +0.043 with
sd 0.162.

## What this does and does not say

DOES NOT say prospective pressure fails. The intervention as run is
weak: one hook per task, four inner steps, four outer steps, weight 1.0,
and the resulting lifetime differs from ordinary by 54 nats in 191,907
(0.03%). An intervention that barely moves the lifetime cannot be
expected to move a downstream endpoint, and H35 already registers that
the pressure has an optimum which this run does not attempt to find.

DOES say the pipeline is now able to detect an effect and this
configuration has none to detect. Three measurement failures had to be
fixed before that sentence was true: a frozen basis that made every
prospective gradient dead, a supervised arm whose penalty had no
gradient into shared state, and a probe with no dynamic range -- first
because the "future" tasks were tasks the lifetime had trained on, then
because the adaptor moved nothing.

NOTABLE: replay is the only arm positive in 3/3 worlds (+0.072, sd
0.031). Weak, but it is the arm review 52 warned would deflate the
result -- if merely seeing relatives suffices, the meta-objective is
unnecessary. Registered follow-up: sweep the prospective weight before
concluding anything about the mechanism, since the current weight is
demonstrably too small to change the representation.

## H33 is not evaluable in V6's unfrozen regime, and why that matters

Realized library sizes on the V6 arms: M = 2, 5, 3 per world (identical
across arms). `audit_learned_schema` needs M > M_0 + 1 = 4 to fit a
schema over abstractions and hold members out, so H33 can be scored in
at most one of three worlds.

This is a structural tension in V6's design, not a scoring accident:

    frozen basis    M = 7, schemas definable, but every prospective
                    gradient is dead (measured: all arms bit-identical)
    unfrozen basis  prospective gradients live, but the basis absorbs
                    the structure that used to become abstractions and
                    M collapses to 2-5

So the protocol that lets prospective pressure act is the protocol in
which the higher-order library barely exists. H33 asks whether
prospective pressure makes FACTORIZE pay over that library; with two
abstractions there is nothing for a schema to be a schema OF.

REGISTERED RESOLUTION, before running it: partial freeze. The runner
already supports `--freeze-slots N`, which freezes the first N basis
slots and leaves the rest trainable — introduced in V3 as the promotion
oracle. That gives prospective pressure somewhere to act while still
forcing recurring structure into promotions. The V6.1 configuration
should therefore be a partial freeze, and the fully-unfrozen runs
reported above are the wrong operating point for H33 even though they
are the right one for H30.

H32 is unaffected: it reads the effective operator, which exists
regardless of library size. First reading on world 0 is ordinary 0.137
-> prospective 0.176 (+0.039), consistent in direction with H32 and far
too small and too singular to claim.

## The prospective WEIGHT is not a pressure knob (2026-08-21)

The registered follow-up to H30's null was "sweep the prospective
weight before concluding anything about the mechanism". The sweep is
uninformative, and the reason is a property of the optimizer rather
than of the learner.

    weight    lifetime change vs ordinary
       1      0.01%, 0.07%, 0.15%
      10      0.02%, 0.10%, 0.07%
     100      0.03%

A hundredfold weight buys nothing. The hook constructs a FRESH AdamW on
each call, and Adam divides by its own running second-moment estimate,
so with a handful of steps from a cold start the update is essentially
`lr * sign(gradient)` and is INVARIANT to the loss scale. Multiplying
the penalty by 100 multiplies both the gradient and the normalizer.

CONSEQUENCE: the pressure in this design is set by learning rate and
step count, not by `--prospective-weight`, and the H35 sweep must vary
those instead. The weight argument is retained for the loss-scale
bookkeeping it does provide but is no longer described as the pressure
knob.

This is the gradient check that review 53 asks every apparent null to
survive. H30's null at weight 1 remains a null AT A PRESSURE THAT WAS
NEVER RAISED — the three weight cells are three samples of the same
pressure, not a sweep, so they do not strengthen the null at all.

## CODE AUDIT: the prospective arm was not applying the prospective objective

Asked to double-check the V6 implementation, and the check found a bug
that invalidates every prospective cell run so far.

`prospective_penalty` adapted the sibling's task-local parameters with
SGD at lr 0.05 for a handful of steps. Measured on a trained model:

    support loss before  0.840895
    support loss after   0.840891     reduction 0.000%
    task code moved by   8.06e-04

So the "adapted" code was unadapted, and the penalty was the query loss
of a task the learner had NOT adapted to. That is not the registered
objective. "Make a sibling predictable without adaptation" is the
EXPLICIT-FAMILY-SHARING objective; the prospective arm is supposed to
charge the cost of ADAPTING. The two arms were, in effect, running the
same pressure under different names.

This is the same failure that the fertility scorer had — gradients here
are ~1e-3, so SGD at any sane learning rate moves nothing — and I fixed
it there without checking whether the learner shared it. Fixed: the
inner loop now uses Adam at the task learning rate, which is what the
lifetime itself uses to fit a task code. Verified: the adapted code now
moves the support loss materially.

WHAT THIS INVALIDATES. Every prospective cell in `artifacts/v6`,
`artifacts/v6_pressure/w*` and `artifacts/v6_pressure/s*`. H30's null
was measured on an arm that was not implementing H30's intervention, so
the null carries no information about H30. It is withdrawn rather than
downgraded.

WHAT SURVIVES. The ordinary and replay arms are untouched by this bug —
replay does not call `prospective_penalty` — so replay's Phi = +0.072
positive in 3/3 worlds stands. So does the finding that the weight is
inert under Adam, and the frozen/unfrozen allocation contrast.

Two regression tests now pin the property: the inner loop must move the
support loss by more than 10%, and the default inner optimizer must be
Adam. The first is the test that would have caught this on day one.

## The prospective pressure is now calibrated against task training

The corrected arm at 2 outer steps still moved the lifetime by only 5
nats in 191,907 (0.003%), so "is the pressure material?" needed a
measurement rather than another guess. Measured on a trained model, in
shared-parameter displacement:

    one task of ordinary training      2.51   (100%)
    prospective hook,  2 outer steps   0.34   (13%)
    prospective hook,  8 outer steps   1.31   (52%)
    prospective hook, 32 outer steps   4.99   (199%)
    prospective hook, 64 outer steps   9.35   (373%)

So outer steps IS a real pressure knob — unlike the weight, which Adam's
scale invariance makes inert — and the earlier configuration was
applying about an eighth of one task's worth of gradient per task, then
having it largely overwritten by the next task's training.

Worth noting the discrepancy this exposes: 13% relative displacement per
task produced a 0.003% lifetime change, which means the prospective
gradient direction is largely orthogonal to, or cancelled by, ordinary
training. That is itself informative and is the thing the sweep should
resolve — whether more pressure accumulates into a different
representation or simply fights the task loss.

The registered sweep is now `outer in {2, 8, 32}` across three worlds,
spanning an order of magnitude around parity with task training, which
is the range H35's predicted optimum should live in if it exists.

# CODE REVIEW 55: THREE PUBLISHED CONCLUSIONS RETRACTED (2026-08-21)

An independent code review found conclusion-impacting bugs in audits I
had already reported results from. The retractions come first, before
the fixes, because they change what this project currently claims.

## RETRACTED 1 — R_effective = 0.190, and everything read off it

`audit_effective_operator` builds each task's innovation vector on THAT
TASK'S OWN `eval_x`, then compares the vectors coordinate-by-coordinate
with an SVD. Coordinate 20 therefore means a different input for every
task, so the "shared subspace" is fitted across unaligned coordinates.
On a common aligned probe the reviewer's spot check moved capture from
~0.41 to ~0.65.

WITHDRAWN: R_effective ~ 0.19; the corrected-H29 claim that the learner
holds "about a fifth of the family geometry"; and V6's H32 reading of
0.137 -> 0.176. All three are artifacts of the misalignment.

## RETRACTED 2 — "no post-hoc linear refactor can recover the structure"

`audit_population_span` has the same defect and one more: it uses
mutually unaligned innovation vectors as a common regression basis for
every target. A valid test evaluates every candidate innovation on each
TARGET's state set before fitting. The spot check moved unexplained
variance from ~0.59 to ~0.43 against my reported 0.707.

WITHDRAWN: the 0.707 figure and the conclusion that the global-rotation
hypothesis is dead. That conclusion was the basis for saying
post-hoc refactoring cannot be the remedy and for pointing V6 upstream
at the wake objective. It is now UNRESOLVED, and V6's premise is
weaker than I stated.

## RETRACTED 3 — every V6 arm's lifetime comparison

`_sibling_of` draws the prospective sibling from `world.tasks`, i.e.
from the ORDINARY LIFETIME, and the hook trains shared parameters on
that task's support AND query labels before the task legitimately
arrives in the prequential stream. The comment claims the sibling is
held out; it is not. So every non-ordinary arm received future
supervision, and the cumulative-loss comparisons between arms are void.

WITHDRAWN: replay's Phi = +0.072 as a clean control result, and all
arm-versus-arm lifetime deltas. The world already generates
`held_out_family_tasks` for exactly this purpose and the hook must draw
from there.

## Also confirmed, and affecting the same audits

* `load_learner` restores `task_reference` but never `retired`, so for
  retired tasks the effective-operator audit adds BOTH the promoted
  abstraction and the private residual that retirement was supposed to
  remove. Most H29 tasks are retired.
* The "on-trajectory" state runs only the routed basis for earlier
  steps, omitting promoted references and residuals, so it is not the
  trajectory `forward()` actually produces.
* Replay is not a matched control: it uses one AdamW at lr 0.003 for
  shared, code and residual together, while ordinary acquisition uses
  route LR 0.05 and residual LR 0.01, and prospective additionally runs
  an inner loop per outer step.
* The fertility scorer reports an endpoint query MSE after 60 steps on
  one batch. That supports "better few-shot endpoint", NOT the
  registered acquisition gate of prequential loss, samples AND
  description cost.
* H31's unrelated futures are PRE-ONSET TASKS THE LIFETIME TRAINED ON,
  while its related futures are genuinely novel. The asymmetry cannot
  separate structural specificity from generic plasticity.
* Protocol knobs (arm, freeze settings, inner steps, sleeps, meta-world
  spec) are absent from the resolved fingerprint, and resume only checks
  that `summary.json` exists, so a differently-configured artifact can
  be silently accepted.
* Compute accounting omits `lifecycle` and `prospective` from the
  shared-residual branch, so their multiply-add counts use the wrong
  fallback.
* `family_operators` now generates `families + held_out_families`
  operators by default, so V5 audit scripts rerun today are not
  semantically identical to the reports they produced before that API
  change.

## What is NOT affected

V1-V4 stand. V5's primary causal results stand: the amortization law
across seven operating points, the sealed C1 pass, the schema-crossing
economics of H20a/H25/H26, and the C3 finding that FACTORIZE loses to
matched-budget COMPRESS on the learned library. None of those depend on
the misaligned audits or the leaking hook.

What moves to UNRESOLVED: V5's H29 localization, the population-span
conclusion, and V6's H30-H32.

# CORRECTED H29: THE STRUCTURE WAS THERE ALL ALONG (2026-08-21)

With coordinates aligned — every task's innovation evaluated on a COMMON
on-trajectory state set, retirement state restored, and the rollout
matching `forward` — the effective task-conditioned operator carries:

    R_effective   0.762      (retracted value: 0.190)
    teacher       1.000      by construction at r_meta = 1
    null          0.003

The registered fork's FIRST branch fires, and it is the branch I
reported as excluded:

    the information was never lost, only DISTRIBUTED across route and
    residual. PROMOTE is promoting the wrong object; abstraction
    boundaries should be found functionally rather than inherited from
    parameter boundaries.

## What this reverses

WRONG, and withdrawn: "roughly 90% of the meta-structure is already
missing before PROMOTE runs"; "the learner holds about a fifth of the
family geometry"; and the whole reading in which V5 ends by pointing
upstream at the wake objective because the structure was never encoded.

RIGHT, on the corrected measurement: the wake learner DOES encode the
family structure — 76% of it, against a teacher ceiling of 100% and a
null of 0.3%. What fails is the promoted library: C3's FACTORIZE 0/6 and
the 0.921 unexplained fraction over promoted atoms are measured on
`abstractions.*` tensors and are NOT affected by this bug. So the gap is
between what the learner COMPUTES and what PROMOTE EXTRACTS, which is a
statement about the promoter's unit of abstraction rather than about the
wake objective's blindness.

That is a better result than the one it replaces, and it is not the one
I predicted: I registered "materially higher than 0.095 but not high"
and the answer is high.

## What it does to V6

V6's premise was "ordinary wake does not form representations preserving
the structure". That premise is now false as stated. The V6 question
becomes narrower and sharper: not "can prospective pressure create
structure the learner lacks", but "can it make the structure the learner
ALREADY HAS extractable as reusable objects".

H30-H32 stay UNRESOLVED. H32 in particular was registered as
"R_effective(prospective) > 0.19", a threshold that is now meaningless —
the ordinary arm alone is at 0.762. A corrected H32 has to ask whether
prospective pressure moves an already-high number, and the honest answer
may be that there is little headroom.

## Method note

Three separate defects had to be fixed before this number was
trustworthy: per-task probe coordinates (the SVD compared incomparable
axes), missing retirement state (retired tasks got both the abstraction
and the residual retirement removed), and a rollout that ran only the
routed basis. Each individually depressed the estimate. The reviewer's
spot check on one world predicted 0.41 -> 0.65; the full fix gives
0.762 across three.

## CORRECTED POPULATION SPAN: 0.491, and the verdict flips

Same coordinate fix — one common on-trajectory state set for every
innovation and every target, retirement state restored, rollout matching
`forward`.

    mean single-innovation unexplained   0.695
    mean top-2 span                      0.773
    mean top-4 span                      0.737
    mean top-8 span                      0.676
    mean FULL span                       0.491   (retracted value: 0.707)

WITHDRAWN: "NOT in the span. No global linear reparameterization of this
population recovers the teacher family, cheap or otherwise, so the
global-rotation hypothesis is dead."

CORRECTED VERDICT: about half of a teacher family operator IS linearly
recoverable from the learner's population of innovations, and
essentially none of it from any single one (0.695 for the best single
object against 0.491 for all 24). The structure is DISTRIBUTED across
the population, which is precisely the hypothesis the retracted number
declared dead.

## The three corrected numbers now tell one coherent story

    R_effective        0.762   the learner's task-conditioned operators
                               carry most of the family structure
    full span          0.491   half of a teacher operator is linearly
                               recoverable from the innovation population
    promoted library   0.921 unexplained, FACTORIZE 0/6

The learner computes the structure. It is spread across route and
residual and across tasks rather than concentrated in any one object.
And PROMOTE, which extracts single task residuals into single
abstractions, captures almost none of it. The failure is the extraction
step's UNIT, not the wake objective and not the absence of structure.

This also revives what review 48 called Hypothesis B and what review 49
called the global-rotation branch, both of which I reported as
eliminated. They are live again, and a sleep phase that refactors over
the POPULATION — rather than promoting one residual at a time — is now
the intervention the evidence points at.

# REVIEW 55 REMEDIATION STATUS: FIXED INSTRUMENTS, NO NEW V6 VERDICT (2026-08-21)

The code paths identified above have now been changed: prospective
siblings come from `held_out_family_tasks`; replay uses the lifetime's
separate shared, route, and residual learning rates; H31 uses unseen
unrelated tasks; the fertility scorer exposes cumulative adaptation-
trajectory cost, endpoint error, and steps-to-target; lifecycle and
prospective compute accounting use the shared-residual branch; and V6
provenance records the intervention knobs.

These repairs do NOT reinstate any old result. All artifacts produced by
the leaking hook, dead inner SGD loop, unmatched replay optimizer, or
asymmetric scorer remain invalid. In particular:

    H30 fertility exists                 UNRESOLVED
    H31 fertility is specific            UNRESOLVED
    H32 pressure raises structure         UNRESOLVED
    replay Phi = +0.072                   WITHDRAWN
    old prospective pressure sweeps       WITHDRAWN

The corrected H29 results change the question those hypotheses address.
Ordinary wake is already at `R_effective = 0.762`, so the registered H32
threshold `> 0.19` is vacuous and must not be reused. Any successor test
must ask whether prospective pressure makes existing distributed
structure more EXTRACTABLE or more economical on genuinely unseen tasks,
not whether it creates structure absent from ordinary wake.

One provenance debt remains explicit: `rho_profile.json` now records the
full V6 intervention record, but the current early resume check compares
only the arm name. Before a clean result is accepted, resume must validate
the entire stored protocol (or the run must use a fresh empty output
path), including inner/outer steps, support, freeze allocation, sleeps,
lifecycle settings, and the meta-world specification.

# V6.1 FIRST VALID RESULT: H30 FAILS, and prospective pressure HURTS

12 cells, 4 arms x 3 worlds, serialized, every review-55 fix applied.
This is the first V6 measurement in which the prospective arm actually
implemented the prospective objective and no arm saw future labels.

Adaptation cost on UNSEEN related futures (prequential acquisition cost,
one standardized adaptor, shared parameters frozen):

| arm | k=1 | k=2 | k=4 | k=8 |
| --- | --- | --- | --- | --- |
| ordinary    | 13.36 | 10.44 |  8.66 | 7.68 |
| replay      | 16.17 | 10.92 | 10.09 | 9.39 |
| prospective | 21.94 | 14.85 | 12.72 | 9.80 |
| supervised  | 13.09 | 11.02 |  9.90 | 8.91 |

Phi against ordinary, per world at k=1:

    replay        -7.83, -2.77, +2.19    mean -2.81   1/3 positive
    prospective  -10.05, -13.53, -2.14   mean -8.58   0/3 positive
    supervised    -0.51, +1.53, -0.22    mean +0.27   1/3 positive

H30: FAILS. Prospective Phi_related = -8.58, negative in 3/3 worlds.
Prospective pressure does not produce fertility; it DESTROYS it.

H31: not a pass, and informative anyway. Phi_specific = -6.6, because
prospective damages RELATED futures (-8.6) far more than unrelated ones
(-2.0). The harm is structurally specific in the direction opposite to
the hypothesis: the intervention aimed at making relatives cheap makes
relatives expensive.

## The precondition also fails, and that matters for the reading

H30 was registered "at matched current-task performance and matched
D*". The arms are NOT matched: every intervention makes the lifetime
loss worse than ordinary by 1,000-2,000 nats (ordinary -191,907 against
prospective -189,953 on world 0). So this is not a clean
fertility-at-equal-cost comparison. The honest statement is stronger
than a failed H30 and weaker than a matched refutation:

    the intervention degrades BOTH current-task loss AND future
    adaptation, so there is no trade to evaluate.

## Reading, registered before the follow-ups

The most likely mechanism is H35's over-alignment, arriving early rather
than at high pressure. The objective rewards a shared representation
that makes a sibling's query loss low AFTER a short adaptation, and the
cheapest way to do that appears to be collapsing capacity toward the
family mean — which is exactly what would make a NEW relative harder to
fit quickly, because the coordinates that distinguish family members
have been squeezed out. That predicts the harm should be worst at small
support, and it is: -8.6 at k=1 falling to -2.1 at k=8.

Replay also hurts on average (-2.81, 1/3 positive), which withdraws the
earlier claim that replay was the one arm producing fertility. That
claim came from the leaking configuration and does not survive.

Supervised is the only arm near neutral (+0.27, 1/3 positive) and is
within noise of ordinary.

## What this does NOT say

It does not say prospective pressure cannot work. It says THIS
prospective objective, at this pressure, on this substrate, is harmful.
The pressure was chosen at 52% of one task's training movement, which
H35 registered as inside the plausible band but which the over-alignment
reading suggests may already be past the optimum. A lower-pressure cell
is the cheapest discriminating follow-up, and its prediction is
registered here: if over-alignment is the mechanism, Phi should rise
toward zero as pressure falls and never become positive.

## H32: passes its threshold, and the threshold is the wrong instrument

    R_effective   ordinary 0.762 -> prospective 0.791  (+0.029)
                  replay 0.784, supervised 0.817, null 0.003

Against the registered "> 0.19" this is a PASS, but that threshold was
written when the ordinary baseline was believed to be 0.19 and is
meaningless now that the corrected baseline is 0.762. The honest reading
of +0.029 on an already-high number is "no material change".

THE INFORMATIVE PART IS THE DISSOCIATION. Geometry went slightly UP in
every intervention arm while fertility went sharply DOWN
(Phi_related -8.58 for the same prospective arm). Review 52 warned that
a large Phi with flat geometry would be a legitimate outcome; the
observed pattern is the mirror image, and it is worse for the geometric
endpoint. R_effective rising while adaptation cost rises means
R_effective is NOT tracking the thing V6 cares about.

That vindicates making Phi primary. Had H32 been the pass criterion,
V6.1 would have been reported as a success on the basis of +0.029 while
the intervention was making related futures 65% more expensive to learn.

## H33: NOT EVALUABLE, and the reason is structural

Realized library sizes across all four arms and three worlds:

    ordinary     2, 5, 3        replay       2, 5, 4
    prospective  4, 2, 4        supervised   4, 3, 2

The schema audit needs M > M_0 + 1 = 4 to fit a schema and hold members
out. Exactly one cell of twelve qualifies. This is the same tension
recorded earlier: an unfrozen basis absorbs recurrence continuously and
the discrete library never grows large enough for a schema over
abstractions to be defined, while a frozen basis makes every prospective
gradient dead.

H33 therefore cannot be answered in either extreme regime, which is what
the partial-freeze allocation sweep exists to fix.

# H34 and H36: NOT ATTEMPTED, with reasons (2026-08-21)

Recorded as an explicit decision rather than left as unfinished work,
because "untested" and "declined for cause" are different statuses and
only the second is honest here.

## H34 — the recurrence-probability threshold: NOT ATTEMPTED

H34 predicts fertile organization pays iff `p * H * s_bar > C_F`, with
p the probability a future task belongs to a reusable family. It needs a
new generator knob (a mixture of family and non-family futures) and a
measurement of `C_F`, the cost of organizing prospectively.

Declined because `C_F` is not measurable in the current substrate: the
one intervention that produces prospective organization makes BOTH
current loss and future adaptation worse, so its "cost" has no
well-defined benefit side to trade against. A threshold experiment over
a quantity whose favourable side does not exist would produce a crossing
at p = 1 by construction. V4's standing lesson applies: do not build an
operator before the opportunity census shows there is something to buy.

## H36 — meta-learned updater: NOT ATTEMPTED

H36 asks whether an updater trained across lifetimes forms
higher-Phi representations on held-out worlds without family labels. It
is the largest build in the V6 ladder — differentiating through the
update process across many lifetimes — and it is downstream of H30,
which failed.

Declined because the hand-specified version of the same pressure is
actively harmful (Phi -8.58, 0/3 worlds). Meta-learning a pressure whose
explicit form damages the endpoint would be optimizing a proxy this
project has just shown to be miscalibrated. If a later result shows
prospective pressure helping in SOME regime, H36 becomes worth building
against that regime.

## H35 — partially answered, and the partial answer matters

H35 predicted a non-monotonic optimum: Phi rising then falling with
pressure. Observed at 52% of one task's training movement, Phi is
already sharply negative and the harm is largest at the smallest support
(-8.6 at k=1 falling to -2.1 at k=8). That is consistent with being PAST
the optimum rather than below it, so the registered U-shape may exist
entirely at pressures lower than the one tested.

The cheap discriminating cell is registered: at a materially lower
pressure, over-alignment predicts Phi rises toward zero and never
becomes positive, while "pressure was too low" predicts Phi becomes
positive. This is the one V6 follow-up worth running, and it is one
sweep rather than a new mechanism.

# MECHANISM AUDIT: over-alignment is NOT what happened (2026-08-21)

Review 56 named over-alignment as the leading explanation for
Phi_prospective = -8.58 and gave it ~65% — the representation collapsing
family members toward a shared mean, erasing the coordinates that
distinguish them. It also named the discriminating measurement, which
runs offline on the existing artifacts.

| arm | discrimination d_between/d_within | \|\|df/dc_task\|\| |
| --- | --- | --- |
| ordinary    | 0.0088 | 0.1688 |
| replay      | 0.0088 | 0.1716 |
| prospective | 0.0097 | 0.1710 |
| supervised  | 0.0099 | 0.1747 |

Over-alignment predicts BOTH fall under prospective pressure. Both
rose: discrimination +11.1%, code sensitivity +1.3%.

So the prospective representation separates family members slightly
MORE than ordinary, and is slightly MORE responsive to the task code —
the very channel few-shot adaptation moves. The members did not collapse
and the argument channel was not crushed.

VERDICT: over-alignment is REFUTED as the mechanism for V6.1's harm.
What remains is the alternative review 56 flagged as
indistinguishable-in-Phi: the adaptation problem became harder to
OPTIMIZE — a conditioning effect — without any collapse of the
representation's content.

## This changes what H35 should be expected to show

I registered the over-alignment reading as predicting Phi rises toward
zero as pressure falls and never turns positive. That prediction was
conditional on a mechanism now refuted, so it is withdrawn as a
prediction while the sweep still runs as a measurement.

Conditioning damage should scale with how far the prospective gradient
drags the shared parameters, which means the low-pressure cells have a
genuine chance of being neutral or positive rather than merely
less-negative. H35 is now a more open question than it was an hour ago,
which is the opposite of what I expected from running a mechanism audit.

## Why this measurement was worth running before the sweep finished

Phi says an intervention hurt. It cannot say whether the representation
lost information or merely became awkward to optimize, and those imply
different successors: the first wants an architecture that protects
argument coordinates, the second wants a better-conditioned objective
over the same architecture. Review 56's proposed successor — separating
SHARED SCHEMA from FAST ARGUMENT from PRIVATE INNOVATION — is a response
to the first. On this evidence it is not yet motivated, because the
argument channel is intact.

## Correction to the mechanism audit's wording (same day)

I wrote that the prospective representation "separates family members
slightly MORE than ordinary, and is slightly MORE responsive to the task
code". Per-world numbers do not support the direction:

    prospective, discrimination vs ordinary   -4.2%, +19.1%, +23.4%
    prospective, code sensitivity vs ordinary  +4.5%,  -4.1%,  +3.2%

Neither is sign-consistent, and the sensitivity mean of +1.3% sits well
inside a world-to-world spread of about +/-4%. The same discipline I
applied to Phi — same sign in every world, mean exceeding the spread —
rejects both directions here.

WHAT THE AUDIT SUPPORTS, stated correctly: neither discrimination nor
task-code sensitivity FELL. Over-alignment predicts a clear fall in
both, and there is no fall in either, in any world. So over-alignment
remains refuted as the mechanism. What is NOT established is the
opposite claim that the prospective arm improved these quantities.

INSTRUMENT CHECKS, run because this conclusion is load-bearing: the
sensitivity estimate is in its linear regime (0.1597 at eps=0.01 through
0.1587 at eps=1.0, so not saturated), and the task code is genuinely
influential (zeroing it moves the output by 1.86 against an output norm
of 14.84, i.e. 12.5%). The measure has dynamic range; the arms simply do
not differ on it.

The refutation therefore stands on an absence of the predicted effect,
which is weaker than a measured reversal and is the honest form of the
claim.

# H35 RESULT: THE REGISTERED NON-MONOTONIC OPTIMUM IS NOT SUPPORTED (2026-08-21)

The registered low-pressure cells are complete: outer steps 1 and 2
(approximately 6% and 13% of one task's ordinary parameter movement),
paired with the existing pressure-0 ordinary arm and pressure-8 arm on
development worlds 0--2. Every checkpoint was scored by the same frozen-
representation adaptor used for the first valid V6 result: 40 Adam steps,
learning rate 0.05, disjoint support/query data, and k in {1,2,4,8}. The
pressure-0 and pressure-8 anchors reproduce that result exactly.

At the primary k=1 support:

| outer steps | Phi related | per-world Phi | positive worlds | Phi unrelated | current-loss delta |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | -0.432 | +0.559, -1.569, -0.286 | 1/3 | +0.173 | -46.1 |
| 2 | -0.833 | +1.800, -6.697, +2.397 | 2/3 | -0.324 | +63.8 |
| 8 | -8.575 | -10.050, -13.532, -2.144 | 0/3 | -1.978 | +1,281.4 |

Positive Phi means cheaper future acquisition. Negative current-loss delta
means a lower (better) cumulative lifetime loss. Neither low-pressure cell
meets the standing replication rule of one sign in every world with a mean
larger than its spread. Pressure 1's apparently favourable -46.1 current-
loss mean is mixed (+75.8, -7.3, -206.7) and smaller than its 118.6 standard
deviation. Pressure 2 is mixed on both endpoints. Pressure 8 robustly harms
both related-future acquisition and current lifetime loss.

VERDICT: H35 is NOT SUPPORTED in the tested range. There is no reliable
beneficial interior cell followed by deterioration, either in Phi or in total
lifetime cost. "Large pressure hurts" is supported; the preregistered U-shape
is not. H30 is likewise unsupported at every tested nonzero pressure, but this
does not establish that every possible pressure or objective must fail.

The mechanism remains unresolved. The earlier audit ruled against a clear
loss of family discrimination or task-code sensitivity, leaving conditioning
damage plausible, but H35 does not identify it. The next move should not be a
finer sweep around a nonexistent positive cell; it requires a changed,
independently justified objective or optimizer and a new prospective
prediction.

METHOD CORRECTION DURING SCORING: the first H35 scorer run silently used its
60-step source default, whereas the valid V6 report had been invoked with 40
steps. That changed absolute costs but not the signs. The scorer now defaults
to 40, records the full evaluation protocol in the report, exposes related,
unrelated, and within-family paired effects, and writes atomically before
console presentation. Only the corrected 40-step report is interpreted.

# V6 ALLOCATION RESULT AND CLOSURE (2026-08-21)

The independently motivated allocation matrix is complete: free shared slots
in {0,1,2,3,6}, three arms, and development worlds 0--2. Every cell freezes the
basis at task 8, uses 12 slots, 16 prospective outer steps, 16 inner steps, and
the same sleep schedule. The allocation result is descriptive and cannot rescue
the separately closed H30/H35 pressure claim.

## Protocol repair before scoring

Preflight found that a pooled launcher had accidentally reused H35's 8-step
inner setting for nine allocation cells, while the allocation protocol used 16.
The nine artifacts were quarantined rather than deleted, then rerun along with
the one cell that had failed from memory pressure. All 45 active artifacts now
pass the complete protocol check. Rerun ordinary/replay checkpoints were
bit-exact to their quarantined versions, confirming that the inner-step knob is
inert for those arms; the prospective cells were outcome-changing and had to be
replaced.

## The allocation frontier

Mean results by free capacity:

| free slots | arm | M | R_effective | Phi_related | FACTORIZE | current-loss delta vs ordinary |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | prospective | 7.00 | 0.485 | 0.000 | 0/3 | 0 |
| 1 | prospective | 6.33 | 0.525 | -5.182 | 0/3 | +107 |
| 2 | prospective | 5.33 | 0.673 | -11.827 | 0/3 | +1,965 |
| 3 | prospective | 6.00 | 0.660 | -9.991 | 0/2 | +1,999 |
| 6 | prospective | 3.67 | 0.716 | -8.433 | 0/1 | +3,362 |

At zero free slots all three arms are bit-identical: prospective and replay
gradients have nowhere shared to act. At every plastic setting prospective Phi
is negative. It is negative in all three worlds at free slots 2, 3, and 6; the
free-1 mean is also negative with only 1/3 worlds positive. No allocation
setting produces reliable fertility.

The structural relationship is graded, not a sharp phase boundary. From zero
to six free slots, mean library size falls 7.00 -> 3.67 in ordinary and
prospective and 7.00 -> 4.67 in replay. Every one of the nine paired arm-world
endpoints has a smaller library at six free slots than at zero, while every one
has higher R_effective. Intermediate means are not monotone, so the supported
claim is an overall allocation frontier: plastic shared capacity tends to
absorb recurrence continuously and leaves fewer explicit promoted objects.

H33 DOES NOT CROSS. FACTORIZE wins 0/12 scoreable prospective cells and 0/34
scoreable cells across all arms. Every measured margin favors COMPRESS. Greater
plasticity makes the mean margin less negative, but it also makes more worlds
unscoreable by shrinking M; no intermediate regime combines fertility with a
paying explicit schema.

## V6 closed in development; no sealed block

The final status is:

    H30  NOT SUPPORTED   no beneficial prospective setting
    H31  NOT SUPPORTED   related futures are harmed more, not helped more
    H32  NON-DIAGNOSTIC  geometry and fertility dissociate
    H33  NOT SUPPORTED   FACTORIZE 0/12 scoreable prospective cells
    H34  NOT ATTEMPTED   no positive benefit exists to amortize
    H35  NOT SUPPORTED   high pressure harms; no beneficial interior optimum
    H36  NOT ATTEMPTED   gated on a positive fertility mechanism

V6 will not open confirmatory seeds. It falsified this prospective-learning
strategy in development and localized the next question without earning a
confirmation block. The durable result is `geometry != fertility`: useful
structure can be present or increase while the fixed learning interface becomes
less economical. What V6 has NOT established is whether the deficit is a loss
of representational opportunity or a failure of the adaptor to find an
available solution.

# H37 — V6R ADAPTATION GEOMETRY (REGISTERED 2026-08-21)

Registered after V6 closure and before any V6R audit code or measurements. The
full frozen protocol is `V6R_ADAPTATION_GEOMETRY_PLAN.md`.

H37 asks whether V6's prospective deficit is representational or algorithmic.
On the existing valid ordinary/prospective checkpoints and the same six unseen
related future tasks, compare the registered 40-step adaptor with fixed
high-budget support-only optimizers at k=1 and k=128.

The registered fork is:

    k=128 prospective remains worse
        -> representational opportunity was damaged

    k=128 equal; k=1 remains worse after high-budget fitting
        -> sparse member identifiability/generalization was damaged

    k=128 and k=1 equal under high budget; 40-step gap remains
        -> optimizer/findability failure; fertility is a property of (R,U)

    gates disagree across optimizers/worlds
        -> unresolved; measure conditioning directly

The primary optimizer is canonical-start Adam at learning rate 0.01 for 2,000
fixed updates. Adam 0.05 and LBFGS are fixed robustness checks. Query labels
never choose the optimizer, restart, checkpoint, or stopping point. The audit
must reproduce the original 40-step anchor exactly before the new result is
read.

# H37 RESULT — REPRESENTATIONAL OPPORTUNITY LOSS (2026-08-21)

The anchor reproduced all 12 registered ordinary/prospective task costs exactly
(maximum absolute difference 0). All primary and robustness fits were finite;
no task, optimizer, or endpoint was excluded. The frozen decision tree gives the
same result for primary Adam and both robustness optimizers:

| optimizer | k | ordinary | prospective | gap | per-world gaps |
| --- | ---: | ---: | ---: | ---: | --- |
| Adam 0.01, 2,000 | 1 | 0.2387 | 0.2730 | +0.0343 | +0.0315, +0.0366, +0.0346 |
| Adam 0.01, 2,000 | 128 | 0.01814 | 0.02976 | +0.01162 | +0.01930, +0.00906, +0.00651 |
| Adam 0.05, 2,000 | 128 | 0.01771 | 0.02916 | +0.01145 | +0.01840, +0.00873, +0.00721 |
| LBFGS | 128 | 0.01783 | 0.02991 | +0.01207 | +0.02021, +0.00941, +0.00660 |

Values are final query MSE divided by `2 * 0.1^2`; positive gaps mean the
prospective checkpoint is worse. At k=128, primary prospective cost is 64%
higher than ordinary. Every one of the six task-paired gaps is positive, every
world mean is positive, and the mean exceeds both its population SD and the
registered 10% threshold. The endpoints are nearly invariant to optimizer:
Adam 0.01, Adam 0.05, and LBFGS all reach ordinary means near 0.018 and
prospective means near 0.030. Primary final support MSE is also higher for
prospective (`0.000414` versus `0.000250`).

VERDICT: H37 localizes the repaired V6 deficit as **operational
representational opportunity loss**. A large support set and 2,000 updates do
not recover an ordinary-quality solution using the allowed fresh task code and
private residual under the frozen prospective representation. This rejects the
specific optimizer/findability branch proposed after V6; changing only the
optimizer is not the earned next mechanism.

This is deliberately not a claim of global impossibility: every "near-oracle"
fit is finite. The conclusion is strong because three substantially different
registered optimizers converge to the same arm-specific gap, not because any
one optimizer proves a global optimum. The k=1 primary and LBFGS gaps are also
replicated worse, and support-selected perturbation restarts preserve the sign,
so sparse generalization may contribute too. The registered tree assigns the
result to opportunity loss first because the deficit already survives k=128.

The next architecture is therefore earned: separate slow shared schema from a
cheaply inferable fast argument and exact-null private innovation. Its first
experiment must test existence and use of that factorization before asking a
learner to discover it or reviving prospective amortization.

# H38-H46 — THE POST-V6R FERTILITY DECISION TREE (REGISTERED 2026-08-21)

Registered from review 58 (`reviews/reviewer-feedback-58.txt`). H37 above IS
review 58's H37: the frozen `V6R_ADAPTATION_GEOMETRY_PLAN.md` is the
standardized-versus-near-oracle adaptation audit the review calls for, and its
registered fork is the plan's fork. Review 58's four outcome rows refine the
reading: oracle removes the disadvantage (findability bottleneck), oracle
reduces but does not eliminate it (both representation and optimizer), oracle
leaves the full disadvantage (representational damage), and prospective
becoming BETTER than ordinary under the oracle (V6 built good representations
paired with the wrong adaptor).

STATUS AT REGISTRATION, recorded because the review was written before the
V6R audit ran and its outcome is now KNOWN (report
`reports/v6r_adaptation_geometry.json`, H37 RESULT immediately above): H37
has resolved to the REPRESENTATIONAL OPPORTUNITY LOSS branch. At k=128 the
high-budget gap remained — primary Adam 0.01 endpoints 0.01814 ordinary versus
0.02976 prospective (+64%), every task and world gap positive, Adam 0.05 and
LBFGS agreeing — so the reviewer's core equality prediction
`L*_prospective ≈ L*_ordinary` is falsified. Its narrower prediction that
high-budget fitting would shrink but not erase the k=1 deficit is supported by
primary Adam (88% smaller than the standardized endpoint gap) and approximately
by LBFGS (80% smaller), while Adam 0.05 is unstable at k=1. The ACTIVE branch
of the tree is therefore the representation branch, H39 -> H40 -> H41 -> H44.
H38/H42/H46 remain the findability research line and stay registered — V6R
establishes an opportunity-loss component but does not exclude an additional
findability component — so optimizer work is not licensed as the next
mechanism. H39-H46 themselves have no code, artifacts, or measurements at
registration.

The tree's discipline rule, registered as binding on the project: earn each
component separately. Do not build SHARED SCHEMA + FAST ARGUMENT + PRIVATE
INNOVATION + LEARNED UPDATER as a unit because it sounds good; H37 chooses
the branch and each later hypothesis licenses its own mechanism.

## H38 — fertility is a property of the representation-updater pair

Branch: H37 says findability. Phi = Phi(R, U), not Phi(R). Frozen
representations; compare generic standardized Adam, a per-representation
tuned preconditioner, a learned updater U_phi, and possibly Fisher-whitened
adaptation; train updaters on development families, evaluate on UNSEEN
families. Prediction: C_adapt(R_prospective, U_learned) <
C_adapt(R_prospective, U_generic), with the stronger form
C_adapt(R_prospective, U_learned) < C_adapt(R_ordinary, U_learned).
Registered outcomes: only prospective recovers (V6 built useful coordinates
the generic adaptor cannot exploit); both improve equally (the updater was
generically poor, no evidence for prospective value); ordinary still wins
(prospective genuinely shaped a worse representation); learned updater works
only on known families (meta-overfitting, needs a structural architecture).

## H39 — a fertile representation requires an explicit fast argument channel

Branch: H37 says representational damage. T_i = S(alpha_i) + eps_i with S
slow/shared, alpha_i a low-dimensional fast task argument, eps_i private
exact-null innovation. Prediction: C_future^{S+alpha+eps} < C_future^current
at matched present performance, matched parameter budget, matched adaptation
steps, and matched D*, against an equal-sized generic task code control, with
unrelated-family refusal intact. Registered outcomes: strong help (V6's
problem was the shared/variation interface); help only with teacher family
labels (expression without discovery); no help even with oracle grouping
(explicit factorization insufficient); helps related but badly hurts
unrelated (schema too rigid, innovation/refusal required). REVIEWER
PREDICTION: decent chance of working on seen meta-families, but private
innovation will be necessary to avoid forcing every task through S.

## H40 — exact-null private innovation is necessary for fertile sharing

T_i = S(alpha_i) + g_i eps_i with g_i = 0 a literal no-deviation state.
Prediction: on strongly related tasks D*(eps) falls AND C_adapt falls,
without hurting unrelated tasks (which activate innovation). Registered
outcomes: strong benefit (typed schema/argument/innovation channels
justified); storage benefit only; adaptation benefit only (exact-null is
about optimization, not MDL); no benefit (the V3 null-state lesson does not
transfer to this level).

## H41 — the critical quantity is argument identifiability

Not merely T_i = S(alpha_i), but the map few observations -> alpha_i must be
easy. Prediction: C_adapt is proportional to C_identify(alpha), measured by
task-code uncertainty after k samples, Fisher information in alpha, argument
Jacobian conditioning, or alpha recovery error from k examples; more fertile
representations show lower argument error after k shots. If an
identifiability metric predicts Phi while R_effective does not, the standing
diagnostic changes: shared geometry is secondary, identifiable coordinates
are primary. REVIEWER PREDICTION: fairly likely.

## H42 — expressibility and findability are independent axes

E(R, T) = best achievable future-task cost; F(R, T, U) = cost of finding it
with U. Prediction: there exist R_1, R_2 with E(R_1) approximately E(R_2)
but F(R_1) << F(R_2), and possibly the reverse. REVIEWER PREDICTION:
strongly expected to exist. If it does, objectives must carry explicit
search cost: J = D*(R) + C_express + beta * C_find. This is the bridge to
program synthesis: a short program existing in the language is distinct from
the synthesizer finding it.

## H43 — plasticity allocation determines representation level

Prediction: n_free up implies M down, with information migrating into the
shared substrate, and possibly an intermediate n* where the shared basis
stays adaptable, explicit atoms still form, and FACTORIZE becomes
economically viable. STATUS AT REGISTRATION, recorded because the review was
written while the sweep was in flight and its data now EXISTS: the
directional claim is SUPPORTED (mean M fell 7.00 -> 3.67 prospective and
7.00 -> 4.67 replay from free0 to free6, with R_effective rising in all nine
paired comparisons); the intermediate fertile regime did NOT appear (Phi
negative at every plastic setting; FACTORIZE 0/34 scoreable cells). H43's
remaining open content is interpretive, not prospective: whether "allocation
moves the storage level but creates no fertility" is a general law beyond
this operating point. See the V6 ALLOCATION RESULT AND CLOSURE entry.

## H44 — good sharing requires preserving independent innovation

The best representation lies on a Pareto frontier between shared compression
and capacity for cheap specialization; too little sharing raises D* and
hurts transfer, too much raises C_identify and hurts novelty. Prediction:
an intermediate shared/private allocation minimizes
J = L_current + lambda D* + C_future, where the manipulation is the
ARCHITECTURE's sharing capacity, not an objective coefficient — registered
as the better form of H35's question.

## H45 — prospective value should be measured across branching futures

J_option(R) = E_{T' ~ p(T'|H)} C_adapt(T' | R). Prediction:
Phi_branching > Phi_single, especially on held-out families. Gated: do not
run until H37-H41 establish a positive fertile representation. If it works
it is the bridge to dreaming (real future branches -> imagined plausible
branches); if it fails, breadth alone is not the missing mechanism.

## H46 — learned update rules favor fertile representations without future leakage

R_{t+1} = U_phi(R_t, D_t), outer objective min_phi E[sum_t L_preq(T_t) +
lambda D*(R_T)], no future information at update time. Prediction:
C_adapt^{U_phi} < C_adapt^{SGD} on held-out generator families. Registered
outcomes: works across new generators (major evidence for model+optimizer
co-training); works only within the training generator (meta-overfitting);
no effect while an oracle architecture works (update-policy capacity
problem); no effect AND the explicit fertile architecture fails (the grand
hypothesis needs revision). Sequencing: late, not next.

## The decision tree, fixed

    H43 (answered above) -> H37 (answered above: representation branch) ->
        findability:     H38 -> H42 -> H46     (inactive; see H37 status)
        representation:  H39 -> H40 -> H41 -> H44   (ACTIVE)
    either branch positive -> H45 -> compositional language (V7+)

REVIEWER OVERALL PREDICTION, registered verbatim: a hybrid — V6 failed
partly because fertility is relational between representation and update
rule, and partly because task variation lacks a clean fast coordinate
system; H37 removes some but not all of the prospective deficit. STATUS:
still plausible but not localized cleanly. High-budget fitting removes most
of the k=1 endpoint gap for primary Adam and LBFGS, yet a replicated residual
gap remains and abundant-support harm survives every optimizer. The pure
findability account is falsified; a mixed account remains live, while the
registered decision tree makes the fast-coordinate representation branch
H39-H41 the next one to test.

None of H38-H46 opens or implies a sealed block. All are development-stage
hypotheses; any confirmatory run requires its own frozen plan.

# REVIEW 59 (2026-08-21): H37 READ AS OPPORTUNITY LOSS; REPRESENTATION BRANCH CONFIRMED ACTIVE

Registered from `reviews/reviewer-feedback-59.txt`, written after the V6R
result. This entry appends; it registers no new sealed block and opens no
seeds.

## Numbering reconciliation (binding)

Review 59 labels its ladder H39-H42 with a different assignment from the
review-58 tree already registered above. The registered numbers stand. The
review's content maps as follows:

- review-59 "H39" (existence test: explicit S/alpha/eps channels, no teacher
  values, L*_future lower at matched present loss and D*) -> registered H39.
- review-59 "H40" (matched-capacity generic task-code/residual control) ->
  the equal-sized generic-code CONTROL already required inside registered
  H39; it is not a separate hypothesis number.
- review-59 "H41" (argument identifiability predicts C_acquire better than
  R_effective) -> registered H41.
- review-59 "H42" (exact-null innovation, g_i = 0, unrelated tasks not
  forced through the schema) -> registered H40.
- Registered H42 (expressibility vs findability axes) is untouched by
  review 59.

## Reviewer position, registered verbatim in substance

The optimizer-only account is "increasingly implausible"; the V6 prospective
objective damaged the set of useful future solutions reachable through the
task-local interface. Learned optimizers are deprioritized for the main ROW
line (H38/H42/H46 remain registered but inactive). The next earned mechanism
is SHARED SCHEMA + FAST ARGUMENT + PRIVATE INNOVATION, built first as the
smallest ORACLE-FORM architecture on the V6 worlds — not as learner-discovered
decomposition, which is a later question, and not as program synthesis, which
is later still.

## Registered ordered checklist for the H39 existence experiment

To be frozen in the H39 plan before any lifetime runs; listed here so the
order cannot be rearranged after data:

1. present-task parity with the ordinary arm;
2. D* preserved or reduced;
3. near-oracle (k=128, three-optimizer, V6R instrument) future endpoint
   improves;
4. finite few-shot (k=1) acquisition improves;
5. gain specific to related futures (unrelated futures unchanged);
6. alpha identifiable quickly (registered H41 instruments);
7. exact-null eps refuses inappropriate sharing (registered H40);
8. learned library exposes schema economics.

Items 1-5 are the H39 gate; 6-8 belong to H41/H40/H44 and are not H39
pass/fail criteria.

## Long-term criterion, registered as a CANDIDATE, not a rule

Review 59 proposes replacing `PROMOTE iff N * s_bar > D*(A)` with
`Value(A) = compression savings + search savings - lost variation
opportunity`. No instrument for "lost variation opportunity" exists yet;
V6R's k=128 frozen-representation gap is the closest measurement. This is
recorded so that any future claim that the amortization law was "always"
three-term can be checked against the date it was first written.

## Reviewer predictions attached for scorekeeping

- H39 existence test: if it fails with the architecture handed to the
  learner, "the entire schema/argument idea is probably wrong for this
  substrate" (reviewer's own falsification condition).
- H41: identifiability cost tracks C_acquire "much better than R_effective";
  review 58 rated this "fairly likely".
- Registered H40 (exact-null): related tasks use S+alpha, atypical tasks
  activate eps, controls not forced through the schema.

# H39 RESULT (2026-08-21): NOT RUN — CENSUS C0 NEGATIVE IN 3/3 WORLDS

Plan: `H39_EXISTENCE_PLAN.md`, frozen at `b6fc27d`, Amendment 1 at
`16906ff` (appended before any census code or data; corrected the C0 fit set
from leave-one-family-out to all trained family objects, since the F arm
trains on all families and only the siblings are held out). Instrument:
`census_h39_schema.py` at `63a8e9e`. Report: `reports/h39_census.json`.

Registered gate: alpha-only k=128 B1 endpoint through a rank-8 linear schema
over the ordinary learner's own residual vectors, at most 1.5x the ordinary
V6R endpoint, in at least 2 of 3 worlds.

Observed ratios: 3.531, 2.243, 3.305 (alpha-only 0.07860 / 0.03452 / 0.05537
versus ordinary 0.02226 / 0.01539 / 0.01675). 0/3 worlds pass. Per the
frozen verdict table: **H39 NOT RUN: census negative.** No factorized or
generic-control lifetime was launched. H39 is neither supported nor
falsified; its existence lifetimes were not licensed.

Instrument non-vacuity: every fit reduced the k=0 query loss 3-10x and the
curves are flat from update 1,000 to 2,000; the unit tests include a zero-
channel companion guard.

Exploratory calibration, NOT registered (`reports/h39_census_exploratory_
rank16.json`, `..._rankmax.json`): at rank 16 the ratios are 3.441, 2.236,
3.078; at the MAXIMUM available rank — the full affine span of every live
residual and promoted abstraction in the artifact (26, 12, 34 dimensions) —
they are 3.075, 2.250, 2.806. The shortfall is therefore not a rank-8
limitation: the held-out siblings' best residual solutions lie OUTSIDE the
affine span of the ordinary learner's residual population. A contributing
fact is that the ordinary lifetimes retired 39/64, 56/64, and 32/64 family
tasks, so the population is small, and in worlds 0 and 2 its spectrum is
flat (rank-8 variance 0.67 and 0.51, no knee).

Status of the branch: the representation branch H39 -> H40 -> H41 -> H44
is not advanced. The census is a statement about a schema fitted POST HOC
to a finished ordinary learner; it does not measure what a jointly trained
schema would contain. Any successor plan must say how the schema is to be
obtained other than from the final residual population, and must be frozen
before running.

# CORRECTION (2026-08-21) to the H39 RESULT entry: "siblings" are unseen-family members

The H39 RESULT entry and `H39_EXISTENCE_PLAN.md` describe the held-out
future tasks as "siblings" of trained families. Per `meta_world.py`, at
`r_meta = 1` every family operator lies in one shared rank-2 functional
subspace, and `novel_family_tasks` are members of two families the lifetime
never trains on; held-out members of SEEN families were found
non-discriminating in V6 design. Census C0, its exploratory calibrations, and
the V6R anchors all used `novel_family_tasks`, so every number stands. The
corrected reading is stronger, not weaker: the unseen family's operator is
in the span of the trained families' teacher operators by construction, and
the ordinary learner's residual population does not contain it at any rank.
The H39 existence plan's rationale for `a = 8` ("four rank-2 family
subspaces") was also wrong about the generator; the oracle-form schema rank
is 2. See `H39_PILOT_PLAN.md` Amendment 2.

# H39 PILOT RESULT (2026-08-21): BRANCH C — PRIVATE RELEARNING, NOT A FERTILE ARGUMENT

Plan: `H39_PILOT_PLAN.md` frozen at `effaf6b`; Amendments 1-3 appended and
re-frozen before the corresponding data was read (eps init stationarity;
pooled a=2 primary after the unseen-family correction; alpha-only fit freezes
eps at init rather than zero). Report: `reports/h39_pilot.json` (EXPLORATORY,
world 0). The first scorer run was refused by the registered non-vacuity
check (alpha did not move) and is preserved as
`reports/h39_pilot_discarded_stationary_protocol.json`.

Preconditions: the ordinary history rerun is bitwise identical to
`artifacts/v6_clean/ordinary/world_0/lifecycle` (loss -191,906.786); the
ordinary V6R k=128 anchor (0.02226) reproduced to 1e-12; every non-vacuity
check passes for both factorized arms.

Primary arm, pooled a=2 (the oracle form): present parity PASS (+302 nats
versus ordinary). Alpha-only k=128 endpoint 0.0931 = **4.18x** ordinary
(Adam 0.05: 4.17x, LBFGS: 4.17x) — FAIL against 1.5x. Full task-local fit
(code + alpha + eps) k=128 = 1.063x ordinary, within the registered 1.2x.
Branch **C**: the architecture preserved the ability to relearn privately;
the jointly formed linear schema provides no cheap coordinate for the unseen
family. Secondary arm, pooled a=8: parity +460 nats; alpha-only 3.22x
(robust 3.21x, 3.20x); full fit 0.892x ordinary.

Channel use, end of lifetime, live family tasks: zeroing W alpha and eps
together moves mean NMSE from 0.00303 to only 0.00310 (a=2) — the entire
residual channel carries about 2% of the fit; 56/64 (a=2) and 36/64 (a=8)
family tasks were retired into abstraction references, for which neither
channel fires. The registered `schema_share` (0.836 at a=2; undefined at
a=8 where ablation improves NMSE) is a ratio of these small differences and
is reported, not leaned on.

Historical-span diagnostic (ordinary arm, pre-retirement residuals of all
64 family tasks): alpha-only ratios 4.02 (rank 8), 3.82 (rank 16), 2.75
(rank 63). Registered reading: **ordinary wake never formed those
directions**; retirement did not discard them. This matches the reviewer's
stated prior.

Reviewer-prediction scorekeeping (review 60): A ~50%, partial ~30%,
insufficient ~20%. Observed: not A; the alpha-only endpoint is WORSE than
the post-hoc census (4.18x versus 3.53x on the same world), so "helps but a
gap remains" is not supported either. Closest registered outcome: "simple
linear schema fundamentally insufficient", via branch C.

Licensing: per the plan, branch C licenses nothing. H40-H44 are not
advanced. The representation branch of the review-58 tree is not falsified
in general — only the LINEAR pooled schema at this operating point — and the
reviewer's registered next direction (nonlinear parameterized operators /
conditional primitives) requires its own frozen plan.

# H39b PILOT RESULT (2026-08-21): BRANCH U BY THE FROZEN TABLE; THRESHOLD MISCALIBRATED; FIRST ARGUMENT CHANNEL THAT IS USED

Plan: `H39B_PSLOT_PILOT_PLAN.md` frozen at `83ac418` (no amendments).
Report: `reports/h39b_pslot_pilot.json` (EXPLORATORY, world 0). Instrument
and scorer at `6f33a2b`/later.

Preconditions: the frozen-argument control P2-frozen reproduces the ordinary
V6 artifact BITWISE on every ordinary tensor and in loss (-191,906.786), so
the only difference between P2 and ordinary is the argument channel. The
ordinary anchor (k=128 0.02226; per task 0.02672 / 0.01780) reproduced to
1e-12. All non-vacuity checks pass for both arms.

Primary arm P2 (K=2): present-task loss -192,413.8, i.e. **507 nats BETTER
than ordinary** (parity PASS). Alpha-only k=128 = 3.94x ordinary (robust
3.92x, 3.92x): FAIL against 1.5x. Full task-local fit k=128 = 0.936x
ordinary. Channel use over all 64 family tasks: max-step mean route mass on
P 0.093 (uniform reference 0.083); alpha-zeroed NMSE ratio 1.381.
Secondary arm P8 (K=8): loss -192,934.5 (**1,028 nats better**); alpha-only
k=128 = **1.89x** (robust 1.88x, 1.88x); per task 0.0624 / 0.0217 against
ordinary 0.0267 / 0.0178; full fit 0.790x ordinary; max-step route mass
0.169; alpha-zeroed ratio 2.677.

Verdict by the frozen table: **U** — `route_mass_P_max_step < 0.2`.

ANNOTATION, appended rather than re-judged: the plan's P-CARRIES and
UNUSED thresholds (route mass on P >= 0.5 / < 0.2) were registered without
checking the baseline learner's route concentration. The ordinary learner's
routes are diffuse — mean maximum coefficient 0.44 / 0.29 / 0.32 by step,
mean mass on any one slot about 0.08-0.10 — so >= 0.5 on a single slot was
unattainable for this learner and < 0.2 is near the uniform reference. This
is the same error class as the S0 `p_reuse >= 0.5` bound (learnings). The
functional criterion registered alongside it, alpha-zeroed ratio >= 1.25,
PASSES in both arms (1.38, 2.68). The route-mass criterion is therefore
reported as miscalibrated, and the branch U description "the learner never
routed family tasks through P" is NOT supported by the functional evidence.
No branch is re-assigned; the substantive readings below are exploratory.

What the pilot shows regardless of the table: (1) an argument channel in
the basis is the first one this project has built that the learner USES
and that LOWERS present-task cost; (2) the full task-local interface
becomes better than ordinary (0.79-0.94x); (3) the alpha-only gap falls
from 3.5x (post-hoc census) and 4.2x (residual schema) to 1.89x at K=8,
still short of 1.5x; (4) K=8 beats K=2 on every endpoint, so the
generator's rank-2 coordinate is not the learner's coordinate.

Historical-span diagnostic unchanged (2.75x at full rank). Licensing: none.
A successor must (a) register the channel-use criterion against the
baseline's measured route concentration, (b) treat K as an independent
variable, and (c) be frozen before running; worlds 1-2 remain unopened.

Correction to the entry above (same day, before any further use): the
ordinary per-task k=128 anchors are 0.02524 / 0.01928 (mean 0.02226), not
"0.02672 / 0.01780" as written. P8's per-task alpha-only endpoints 0.0624 /
0.0217 therefore sit at 2.47x / 1.13x; the mean ratio 1.89x and every
decision are unaffected.

# H39c RESULT (2026-08-21): VERDICT P — THE IN-BASIS ARGUMENT IS REAL, LEARNED, AND CAPACITY-LIMITED AT K <= 16

Plan `H39C_KSWEEP_PLAN.md` frozen at `1e99904`, no amendments. Report
`reports/h39c_ksweep.json`. 15/15 cells completed with zero failures; all
three ordinary anchors reproduced to 1e-12; every cell passed the fail-closed
non-vacuity checks (G_8's argument matrices bitwise at init; U_k moved in
every P_K; alpha moved in every fit).

Alpha-only k=128 ratio to the ordinary endpoint (worlds 0 / 1 / 2):

    K=2   3.94 / 2.53 / 3.64
    K=4   3.56 / 2.15 / 2.49
    K=8   1.89 / 2.02 / 1.57
    K=16  1.75 / 1.58 / 1.35
    G_8   3.32 / 2.84 / 3.22   (U_k frozen at init, alpha learns)

Robustness optimizers agree within 0.02 everywhere. TREND: non-increasing in
K in 3/3 worlds. LEARNED DIRECTIONS: G_8 - P_8 = +1.44 / +0.82 / +1.65
(3/3 positive, mean 1.30 > 0.2); G_8's alpha-zeroed NMSE ratios are
1.03-1.06 — the frozen random channel is essentially unused — so the effect
is the learned U_k, not K extra task-local scalars. Present-task loss
improves monotonically with K in every world (K=16: -1,119 / -708 / -844
nats); the full task-local future interface beats ordinary in 13/15 cells
(K=16: 0.75 / 0.88 / 0.73). FERTILE_K: no K reaches 1.5x in >= 2 worlds;
K=16 passes in world 2 only (1.35).

Channel use: alpha-zeroed ratios 1.37-3.59 for every P_K (pass); the
baseline-relative route-mass term (>= 2x ordinary slot-12 mass) is not met
in any cell, though P_8 / P_16 reach 0.96x / 0.91x of it in world 2. The
verdict does not depend on that term: fertile_K already fails on the
ratio. Registered as a strictness observation, not re-judged.

Verdict by the frozen table: **P**. Per the plan this licenses a larger-K
or multi-slot development plan, NOT confirmation. Seeds 700-729 remain
closed. The gap has fallen from 3.5x (post-hoc census) / 4.2x (residual
schema) to a three-world mean of 1.56x at K=16 with no sign of saturation
between K=8 and K=16.

# H39d RESULT (2026-08-21): VERDICT A VIA THE TWO-SLOT ARM; SINGLE SLOT SATURATES

Plan `H39D_CAPACITY_PLAN.md` frozen at `c6b1b7b`, no amendments. Report
`reports/h39d_capacity.json`. 12/12 cells completed with zero failures;
anchors reproduced to 1e-12; every cell passed fail-closed non-vacuity
(both slots' argument matrices moved, both slots' alphas nonzero, alpha
moved in every fit). The first scorer invocation crashed on serializing a
two-slot alpha before writing any report; the fix (flatten) touched no
numerics.

Alpha-only k=128 ratio to ordinary (worlds 0 / 1 / 2; mean):

    P16 (H39c)  1.75 / 1.58 / 1.35   1.56
    P32         1.74 / 1.56 / 1.47   1.59
    P64         1.71 / 1.54 / 1.45   1.57
    M2K16       1.38 / 1.79 / 1.60   1.59
    M2K32       1.27 / 1.74 / 1.36   1.46   <- fertile in worlds 0 and 2

Every arm passes parity and the functional usage criterion (alpha-zeroed
NMSE ratios 2.4-4.4); present-task loss improves in every cell, most for the
two-slot arms (M2K32: -1,991 / -975 / -1,162 nats); the full interface beats
ordinary in 12/12 cells (0.68-0.87x).

Decision rules: fertile_M2K32 TRUE (2/3 worlds <= 1.5 with robustness and
usage). Single-slot TREND holds in 2/3 worlds but is essentially FLAT
(means 1.56 -> 1.59 -> 1.57): the single linear-in-U slot has saturated at
K >= 16. SLOT_STRUCTURE is NOT supported (M2K16 < P32 in 1/3 worlds; M2K32
< P64 in 2/3). Improvement of the best mean over H39c: +0.104.

Verdict by the frozen table: **A**. Per the plan this licenses only the
WRITING of a frozen confirmation plan on seeds 700-729 around M2K32, with
the frozen-direction control and a parity gate. Seeds 700-729 are NOT
opened by this entry.

CAVEATS, registered with the verdict: (1) A is met at the minimum — one arm,
two worlds, margins 0.23 and 0.14 below threshold, and world 1 is WORSE
under both two-slot arms (1.74-1.79) than under any single-slot arm. (2)
The +0.104 improvement clears the P+ line by 0.004. (3) The second slot's
contribution is not monotone in K per world (M2K16 loses to P32 in two
worlds). (4) M2K32 adds 8,192 shared scalars to a 3,576-scalar basis; D* is
reported, not gated. A confirmation plan is the correct instrument for
exactly this situation: the development evidence is positive but marginal,
and only a sealed 30-world block with a preregistered interval can decide
whether the two-slot effect is real.

# H39 CONFIRMATION BLOCK OPENED (2026-08-21); REVIEW 61 → AMENDMENT 1 BEFORE ANY CELL WAS READ

`H39_CONFIRMATION_PLAN.md` was frozen at `1c98017` after PI approval and
the 90-lifetime block on seeds 700-729 launched (`tools/run_h39_confirmation.py`,
pool of 3, prereg check enforced at launch). Review 61 arrived minutes later
and asked for one change to the decision rule; since the change touches
only the scorer and no sealed artifact had been opened or scored, it was
appended as Amendment 1 and the plan re-frozen at `f78f209` (current hash in
`check_prereg.py`). The scorer (`score_h39_confirmation.py`) implements the
amended rule and was committed before any sealed number existed.

Registered decision rule (final): E1 mean(R_G − R_M) > 0, CI excludes 0,
predicted [0.8, 2.2]; E2 geometric mean of R_M in [1.2, 1.8] with its CI
below 1.8, the 1.5x fraction reported as C0 continuity only; E3 mean
present-task gap < 0, CI excludes 0, predicted [−2,000, −500] nats; E4 mean
full-interface ratio < 1, CI excludes 1, predicted [0.65, 0.90]; E5
alpha-zeroed ratio ≥ 1.25 in ≥ 27/30 worlds. CONFIRMED = all; CONFIRMED-
RELATIVE = E1/E3/E4/E5 with E2 missing its interval; FAILED = E1, E3, or
E5 fails. Interval misses with passing signs are PARTIAL, never upgraded.
Scope: existence and use, not discovery.

Our prediction, registered before scoring: CONFIRMED-RELATIVE more likely
than CONFIRMED, driven by world-to-world spread in R_M (development 1.27 /
1.74 / 1.36); E1, E4, E5 pass; E3 passes with its mean inside [−2,000, −500].
Reviewer 61's predictions are in the plan's Amendment 1.

# H39 SEALED RESULT (2026-08-21, seeds 700-729): CONFIRMED — WITH THE MARGIN AND THE AMENDMENT STATED

Report `reports/h39_confirmation.json`, scorer `score_h39_confirmation.py`
at `aa5a5b2`, plan `H39_CONFIRMATION_PLAN.md` Amendment 1 (`f78f209`).
90/90 cells completed with zero failures; all records and protocols
validated; every cell passed the fail-closed non-vacuity checks; prereg
check passed at launch and at scoring.

    E1  mean(R_G - R_M) = 0.886   CI [0.717, 1.054]   pred [0.8, 2.2]   PASS
    E2  geomean R_M     = 1.636   CI [1.495, 1.7997]  pred [1.2, 1.8]   PASS
    E3  mean L          = -1,545  CI [-1,649, -1,435] pred [-2000,-500] PASS
    E4  mean F          = 0.815   CI [0.764, 0.866]   pred [0.65, 0.90] PASS
    E5  used            = 30/30 worlds (alpha-zeroed ratio 2.86-4.34)   PASS

    C0 continuity: fraction of worlds with R_M <= 1.5 = 0.433 (13/30);
    arithmetic mean R_M 1.696; per-world R_M min 1.084, median 1.582, max
    3.499; 24/30 worlds below 2.0. R_G < R_M in one world (717); F > 1 in
    two (710, 728).

Verdict by the frozen (amended) table: **CONFIRMED**. All five estimands
pass on sign, bootstrap interval, and predicted interval. No PARTIAL marks.

TWO THINGS STATED WITH THE VERDICT, NOT AFTER IT.

1. The E2 bootstrap upper bound is 1.7997 against a registered ceiling of
   1.8 — a margin of 0.0003. The point estimate (1.636) sits inside its
   interval with room; the interval test did not. Had the CI crossed 1.8
   the verdict would have been CONFIRMED-RELATIVE. The E2 magnitude claim
   is therefore confirmed at its registered bound and not one digit
   stronger.
2. The amendment changed the verdict. Under the ORIGINAL E2 rule
   (fraction of worlds with R_M <= 1.5 at least 0.5) this block reads
   0.433 and the verdict would have been CONFIRMED-RELATIVE. Amendment 1
   was registered before any sealed cell was read or scored, on the
   review-61 argument that 1.5 is an inherited operational threshold with
   no theoretical meaning and that the magnitude is the claim; the
   reviewer predicted the fraction would be the least stable quantity, and
   it was. Both verdicts are recorded here so no reader has to reconstruct
   this.

What is confirmed, as the plan states it: on 30 unseen worlds, two routed
parameterized operators P(alpha) with K = 32 LEARNED argument directions,
formed online under the ordinary objective with no family labels, (i) make
an unseen-family member expressible through the arguments alone at a
geometric-mean 1.64x the ordinary full-residual cost — an upper bound on
multiplicative opportunity loss of 1.8 at 95%; (ii) do so because the
directions are learned (a matched fixed-direction channel is 0.89 worse in
ratio, 29/30 worlds); (iii) lower present-task cost by 1,545 nats on
average, in every world; (iv) leave a full schema + innovation interface
18.5% better than ordinary on the future task, in 28/30 worlds; and (v)
are used (zeroing the arguments raises family NMSE 2.9-4.3x in every
world).

What is NOT confirmed: discovery (the architecture was supplied); the 1.5x
absolute level (13/30 worlds); any description-length economy of the
argument matrices (8-bit proxy only); slot structure at matched capacity.

Reviewer-61 scorekeeping: E1 "very likely" — passed, near its lower
predicted edge; E3 "likely, with world variability" — passed, every world
negative, mean inside its interval; E4 "cleanest strong positive" — passed
with the tightest relative interval; E5 "very likely" — 30/30; E2 geomean
"about 1.4-1.7" — 1.636; fraction "least stable" — 0.433, below the old
bar. Our own registered prediction (CONFIRMED-RELATIVE more likely) was
wrong under the amended rule and right under the original one.

# REVIEW 62 (2026-08-22): DISCOVERY REGISTERED AS H47-H49; PRE-DESIGN AUDIT OF THE SEALED REPORT

Filed as `reviews/reviewer-feedback-62.txt`. Registers the next rung and the
reviewer's predictions. Nothing here opens a seed or licenses a run; each
of H47-H49 needs its own frozen development plan on seeds 0-9.

## H47 — membership discovery (review 62 D1)

With the number of parameterized slots FIXED at the confirmed architecture
(two, K = 32) and NO family information of any kind, the learner discovers
which tasks share a slot well enough that its economics approach the
supplied-architecture arm: J_present, R_alpha, R_full within registered
tolerances of the confirmation arm M on the same worlds. Teacher membership
recovery (ARI / NMI / route agreement) is diagnostic, never deciding.
Reviewer prediction: likely solvable; soft or annealed routing with an
MDL/usefulness pressure beats hard early commitment. Note for design: the
confirmed arm M already used NO family labels (pooled slots; oracle
grouping was withdrawn in H39 pilot Amendment 2) — so membership in M is
already discovered by soft routing. H47 must therefore be stated as a
comparison against a labelled-routing oracle (tasks told which slot), not
against M, or it is already answered. To be resolved in the H47 plan.

## H48 — cardinality discovery (review 62 D2)

Given a candidate pool of K_max parameterized slots and an explicit charge
for active ones, J = L_preq + lambda D*(active schemas) + lambda D*(routes,
arguments, innovation), the number of ACTIVE parameterized objects tracks
the economic recurrence condition (birth when N s_bar_schema > D*(S))
rather than the teacher count or the available capacity. Controls: fixed
too-few, fixed correct, fixed too-many, discovered. Discovery wins only if
it approaches the oracle architecture without using all available
capacity. Reviewer prediction: harder; failure modes are too many objects
(local specialization is easy) or too few (a flexible schema absorbs
several families); the D* charge should matter a lot. A granularity sweep
F_schema in {1, 2, 4, 8} at constant support per schema tests whether the
discovered count tracks F.

## H49 — representation-type discovery (review 62 D3)

The learner chooses per recurring computation among ordinary private
object, parameterized schema + argument, and private innovation, and the
choice tracks economy and future usefulness. Not next. Reviewer prediction:
much harder; probably requires a sleep/consolidation decision rather than
pure online gradient descent.

Binding safeguard for all three (review 62): reward economy and future
usefulness, never teacher-looking grouping; a grouping that differs from
the teacher but yields J_present <= J_oracle and C_future <= C_oracle is a
valid discovery. Functional/economic equivalence before ontology recovery.

## Pre-design audit: review 62's six attack points, answered from the sealed report where possible

(2) DOES E3 REFLECT CAPACITY OR OPTIMIZER BUDGET? No. The frozen-direction
arm G has exactly M's parameter count, per-task scalars, optimizer groups,
and budget; its present-task gap is +53 nats mean, negative in 12/30
worlds, range -827 to +834 — i.e., nothing. M is -1,545 in 30/30. The
present-task gain is the learned directions.
(3) DOES E4 SURVIVE A MATCHED TASK-LOCAL BUDGET? Partly. G's full-interface
ratio is 0.917 (22/30 worlds below 1), M's 0.815. So about half of M's
full-interface gain over ordinary is available to a matched fixed channel
(the 64 extra task-local scalars) and about half (paired M - G = -0.10) is
the learned directions. E4 stands as registered (M vs O) and is now also
stated against G: the learned-direction share of the full-interface gain
is ~0.10 in ratio. This qualification is recorded with the result.
(6) ORACLE GROUPING? None in the sealed block. M and G use two POOLED
slots; no family label, family parameter, or grouping enters any arm. The
only supplied structure is the architecture (two parameterized slots,
K = 32). "Parameterized representation" is qualified as
architecture-supplied, not supervised.
(5) E2 TRANSPARENCY: both verdicts and the 0.0003 margin are in the H39
SEALED RESULT entry; nothing further to add.
(1) CENSUS VS ONLINE APPLES-TO-APPLES and (4) ASYMMETRIC COMPENSATION IN
THE ALPHA-ZERO ABLATION are left for an outside reviewer. For (1), the
two measurements share worlds 0-2, futures, instrument, support, and
baseline, but differ in the learner (O's population versus M's slots) and
in the alpha-only protocol's eps (zero in the census, task-free init
online); a version of the census run ON M's final argument matrices would
close the gap and is cheap. For (4), the ablation measures reliance
without re-adaptation; the alpha-only versus full future fits partially
address compensation, but a "zero alpha then re-fit everything else" cell
on the trained family tasks has not been run.

G's own alpha-zeroed ratio is 1.29 in the sealed block (two slots), versus
~1.05 for single-slot G_8 in development: a fixed random two-slot channel
is slightly used, which is consistent with E4's partial budget share.

# STAGE A RESULT (2026-08-22): BOTH AUDITS PASS; "MADE, NOT MINED" STANDS IN ITS NARROWED FORM

Plan `H39_NEXT_STEPS_PLAN.md` Stage A, frozen with Amendments 1-2 before
the registered run (`27e5a36`). Report
`reports/h39_confirmation_followups.json`. Reads only the 30 sealed
ordinary and M artifacts and the sealed report; no new lifetimes.

A1 — post-hoc extraction versus online formation, same worlds, same
futures, same instrument (geometric means over seeds 700-729):

    online alpha-only, M (sealed E2)             1.636
    census on O's realized population, max rank  2.592   (+0.96)
    census on M's realized population, max rank  6.334   (+4.70; rank 8: 6.714)

Registered rule: PASS iff both census geomeans exceed online M by >= 0.5.
Both do. The O-census beat online M in 2/30 worlds (700, 728); its
per-world range is 1.52-5.53; O's realized populations held 11-36 objects.
Verdict: **PASS**. Registered sentence (Amendment 1): useful coordinates
were not recoverable from the final EXTENSIONAL task-object population;
they had to be maintained in an explicit intensional channel during
learning. Not claimed: that nothing useful can be mined from the finished
M learner — its learned argument matrices are in it and are what the
online fit uses. The M-population census at 6.3x is consistent with the
confirmation's E5: in M the family structure lives in the arguments, and
M's private residuals carry even less of it than O's.

A2 — compensation at matched re-fit budget (Amendment 2: alpha-zeroed
re-fit / alpha-free re-fit, both 2,000 updates on the task's own 128
examples): mean of world medians **2.014**, per-world medians 1.69-2.54,
all finite, maximum plateau change between updates 1,000 and 2,000 of
3.9e-4 NMSE. Above the 1.50 line, so the robustness band was not entered
and no second optimizer was required. Verdict: **PASS**. Re-optimizing
route and private residual with the arguments removed recovers less than
half of what the arguments carry. E5's "causally used" reading is kept,
now with the qualification it needed: used, and not replaceable by the
other task-local pathways at matched budget under one finite optimizer
(2.0x is far from the 1.25 bar; H37-style agreement was not required by
the rule and was not run).

Predicted intervals: A1 census(M) [2.2, 3.5] — MISSED high (6.33; we
under-predicted how little family structure M's residuals hold); census(O)
[2.2, 3.5] — 2.59, inside. A2 [1.4, 2.5] — 2.01, inside. The A1 miss is a
prediction error on a diagnostic quantity and does not touch the rule.

Stage A gate: satisfied. Stage B (H47) may now be frozen as its own plan,
with L as an exact route-mask oracle, H_early / H_late, and relative
tolerances set after measuring M's route statistics on worlds 0-2.

# H47 BASELINES (2026-08-22): M DOES NOT ASSIGN TASKS TO SLOTS, AND THE WORLD HAS NO MEMBERSHIP TO DISCOVER

`reports/h47_baselines.json` (`audit_h47_baselines.py`, read-only on the
H39d two-slot K=32 artifacts, worlds 0-2). Conditional route entropy over
the two parameterized slots at the dominant step: late means 0.922 /
0.943 / 0.949 bits of a 1.0 maximum (medians 0.974 / 0.974 / 0.990);
early 0.944 / 0.961 / 0.973. Median margins 0.19 / 0.19 / 0.12. Teacher-
family ARI 0.00 / 0.03 / -0.00, NMI 0.04 / 0.05 / 0.02. Economics: J gap
-1,991 / -975 / -1,162; R_alpha 1.27 / 1.74 / 1.36; R_full 0.71 / 0.82 /
0.76. Cross-world SD of log R_alpha 0.136, of log R_full 0.060.

Finding, registered: the confirmed architecture's family representation
is one DISTRIBUTED 64-direction argument channel spread across two routed
operators; no task commits to a slot, early or late, in any world. And
the generator (`meta_world.family_operators`) places every family
operator, trained or unseen, on one circle in a single rank-2 subspace of
U-space at `r_meta = 1`, so the {0,1} -> S1, {2,3} -> S2 split in the
Stage B design was an arbitrary partition of a continuous family, not a
teacher fact. H47 as "the economic price of discovering membership" is
ill-posed on this world: there is no membership. Review 64's baseline
step caught this before a plan was frozen.

Consequences: (1) E1 and E5 of the sealed result were produced by
distributed arguments, not discovered clusters — consistent with every
number and stated now so no later reading invents clusters; (2) H47 is
redrafted in two parts (`H47_MEMBERSHIP_PLAN.md`, DRAFT): B1, the cost of
discreteness and of premature commitment on the existing world (M,
L_arbitrary, H_early, H_late), testable now; B2, true membership on a
world with two disjoint family subspaces, requiring a generator extension
`schema_groups` whose `G = 1` case must reproduce the current world
bitwise before any lifetime. Neither part is frozen.

Registered predictions for B1: L_arbitrary a COST on R_alpha (32 directions
per task instead of 64; H39d single-slot K=32 1.59 versus two-slot 1.46),
NEUTRAL on J and R_full; H_early a COST on R_alpha, H_late NEUTRAL — the
premature-commitment pattern. For B2, review 64's prediction stands as
written (J and R_full neutral, R_alpha taxed: innovation buffers
imperfect discovery), to be re-registered with that world's baselines.

# REVIEW 66 (2026-08-22): H39 REREAD AS A CONTINUOUS MANIFOLD; B1 FROZEN; H48 SPLIT

Filed as `reviews/reviewer-feedback-66.txt`. Registered reading of the
sealed H39 result: joint learning created an intensional CONTINUOUS
coordinate system P(alpha), alpha in R^64, over one family of related
computations; it did not discover clusters (none exist in the confirmed
world). A new level enters the representation ladder between private
solutions and discrete primitives: the continuous parameterized manifold.

B1 is frozen with `H47_MEMBERSHIP_PLAN.md` Amendment 1: the cost of
imposing discrete commitment on a continuous family, arms M / L_arb /
H_early / H_late, result matrix CONTINUOUS / COMPILE-AFTER-FORMATION /
WRONG-ONTOLOGY / REDUNDANT, tolerances from the measured baselines.
Predictions registered in the plan (ours: COMPILE-AFTER-FORMATION with
L_arb also costly on R_alpha; review 66: M < H_late < H_early on R_alpha,
L_arb > M).

H48 is split: H48a discrete schema cardinality (functional manifold
count, never occupied slots), H48b within-schema dimensionality; both
under one D*. B2 acquires generator tests (G = 1 bitwise; oracle
functional-separation audit) before it can be frozen.

# H47 B1 RESULT (2026-08-22): MIXED BY THE TABLE; THE PREDICTED ORDERING HOLDS IN SIGN AT SUB-TOLERANCE MAGNITUDE

Plan `H47_MEMBERSHIP_PLAN.md` Amendment 1 (B1), frozen `11aff52`. Report
`reports/h47_b1.json`, scorer `score_h47_b1.py`. 9/9 new cells completed
with zero failures; M reused from H39d. Non-vacuity: L_arb masked 64/64
family tasks in every world; H_early / H_late reached median conditional
entropies of 0.004-0.011 bits (target <= 0.20) at the registered final
temperature 0.1, within 0.05 bits of each other (pair comparable); alpha
moved in every fit; argument matrices moved in every arm.

Per world (J relative to M in nats; R_alpha; R_full; median entropy bits):

    world 0   M  +0     1.272  0.710  0.912 | L_arb +725 1.458 0.593 0.000 | H_early +405 1.449 0.704 0.007 | H_late +210 1.318 0.752 0.004
    world 1   M  +0     1.741  0.823  0.939 | L_arb +676 1.909 0.958 0.000 | H_early +245 1.859 0.938 0.004 | H_late +137 1.909 0.881 0.006
    world 2   M  +0     1.355  0.757  0.960 | L_arb +654 1.616 0.781 0.000 | H_early +134 1.537 0.711 0.004 | H_late  +97 1.328 0.806 0.011

Classifications by the frozen rules (COST needs >= 0.25 of |J_M - J_O|, or
log-ratio >= +0.15 on R_alpha / +0.08 on R_full, in >= 2 of 3 worlds):

    L_arb    J COST (0.36 / 0.69 / 0.56)   R_alpha NEUTRAL (+0.137 / +0.092 / +0.176)   R_full MIXED (-0.18 / +0.15 / +0.03)
    H_early  J NEUTRAL (0.20 / 0.25 / 0.12) R_alpha NEUTRAL (+0.130 / +0.066 / +0.127)  R_full NEUTRAL
    H_late   J NEUTRAL (0.11 / 0.14 / 0.08) R_alpha NEUTRAL (+0.036 / +0.092 / -0.020)  R_full NEUTRAL

Label: **MIXED** (no row of the matrix is satisfied: L_arb is a COST on J
but not on R_alpha; neither H arm is a COST on anything).

What the numbers say, stated beside the label and not in place of it:
- The ordering predicted by review 66 and by us, M < H_late < H_early on
  R_alpha, holds in sign in every world on the means (dlog R_alpha 0.036 /
  0.108 / 0.135 for H_late / H_early / L_arb) and on J (+148 / +261 / +685
  nats), but every hardening effect on R_alpha sits BELOW the registered
  tolerance of +0.15 log units. Forcing one slot per task costs about
  10-15% in alpha-only future acquisition and 7-25% of M's present-task
  gain over ordinary; committing late costs roughly half of committing
  early. The effects are real in direction and small in size.
- L_arb is the only arm with a registered COST: an arbitrary partition
  imposed from task onset forfeits 36-69% of M's present-task gain — the
  largest effect in the block — while leaving alpha-only acquisition only
  ~14% worse. On this world, imposing the wrong ontology costs the
  PRESENT more than it costs the future.
- The tolerances were set at about one cross-world SD of M (0.136 on log
  R_alpha). The measured effects are of that same size, so a three-world
  block with this rule cannot resolve them; that is a property of the
  rule we froze, not a retrospective complaint. B2, or more worlds, would.

Scorekeeping: review 66 — ordering right in sign (M < H_late < H_early;
L_arb > M), magnitudes below its implied "cost" expectation; ours —
L_arb COST on R_alpha: no (+0.135 mean, under 0.15); H_early COST on
R_alpha: no; H_late NEUTRAL: yes. The registered COMPILE-AFTER-FORMATION
reading is not licensed; the data are closest to "discretization after
learned directions is cheap, and earlier is a little less cheap".

Licensing: B1 licenses nothing by itself. B2 (true membership on a two-
subspace world) proceeds as designed, with the generator tests required
by review 66 before freezing.

# H47 B2 OPPORTUNITY GATE (2026-08-23): NO-OPPORTUNITY — THE LEARNER ABSORBS TWO ORTHOGONAL FAMILIES INTO ONE DISTRIBUTED CHANNEL

Plan `H47_MEMBERSHIP_PLAN.md` Amendments 2-3 (frozen before any B2
lifetime). Report `reports/h47_b2_gate.json`; teacher gates
`reports/h47_b2_world_gates.json` (PASS 3/3). 6/6 cells completed; all
non-vacuity checks passed (L_true masked 64/64 family tasks onto their
GROUP's slot in every world; argument matrices and alphas moved in every
arm and fit).

The G = 2 world has two ORTHOGONAL rank-2 family subspaces (cross-group
least-squares residual 1.0; held-out families exactly in their group's
span; group classification from teacher behaviour 1.000). Membership is
real and, at the teacher level, perfectly separable.

Membership tax, log E_alpha(M) - log E_alpha(L_true), k=128 B1 (robust
optimizers within 0.002): **+0.271 / +0.099 / -0.027** (worlds 0 / 1 / 2).
Gate: >= +0.15 in >= 2 of 3 worlds — **1/3**. Verdict **NO-OPPORTUNITY**;
per the plan, H_early / H_late are NOT launched on this world.

Secondary: full-interface tax +0.159 / +0.094 / +0.058 (L_true better,
small, consistent); present-task J(M) - J(L_true) = -479 / -529 / -522
nats (M BETTER in every world: the mask costs the present). M_G2's route
diagnostics against the true group labels: median conditional entropy
0.913 / 0.940 / 0.956 bits of 1.0, ARI -0.01 / -0.01 / 0.05. The learner
did NOT separate the two groups even though they exist and are
orthogonal; it again used both slots as one distributed 64-direction
channel — and paid only a small, inconsistent alpha-only tax and NO
present-task cost for doing so.

Reading, registered: at this argument capacity (two slots x K = 32), one
distributed channel is expressively sufficient for the union of two
rank-2 family subspaces, so discrete identity buys nothing the learner
needs. Naming two manifolds is not cheaper than stretching one when the
one has 64 directions to stretch with. This is the amortization question
review 67 deferred ("when does it become cheaper to name two manifolds
than to stretch one?") answering itself at the generous end: not here.
It is also H48b's variable (within-schema dimensionality) showing up
before H48a (schema count) was even asked.

Scorekeeping: review 67 predicted R_alpha,M > R_alpha,L_true "by a
measurable amount" — true in 2/3 worlds but below the registered gate in
2/3; and J_M ~ J_L — wrong in direction: M beats L on J in 3/3. Ours
("gate passes; M_G2's entropy falls well below 0.93 and group ARI rises
above 0.5") — wrong on all three counts. The learner does not find
orthogonal groups because it does not need to.

Licensing: nothing. The registered next question, for a separate frozen
plan: make discrete identity ECONOMICALLY necessary by restricting
argument capacity — K per slot at or near the group rank (K = 2 or 4), so
that one distributed channel can no longer absorb both groups — and
re-run the gate there; if membership then has value, B2's H arms follow
on that world. A separation knob (review 67's rho_group) remains
deferred.

# H48b RESULT (2026-08-23): MIXED BY THE TABLE — THE FUTURE CHANNELS PAY FOR DISCRETE IDENTITY AT K <= 8; THE PRESENT NEVER QUITE DOES

Plan `H48B_WIDTH_SWEEP_PLAN.md` frozen `377579a`, no amendments. Report
`reports/h48b_width.json`. 30 cells (24 new + the K = 32 pair), all
records validated on the G = 2 world, all non-vacuity checks passed
(L masks 64/64; M_K's alpha-zeroed usage 1.42-2.40 at K <= 4, above the
1.25 floor). The sweep was interrupted once by a session exit and resumed
idempotently (6 cells skipped, none reread).

Oracle-minus-learner deltas (positive = told-identity better), worlds 0/1/2:

    K    d_alpha (log)            d_full (log)            d_J (nats)          alpha  present  full
    2    +0.151 +0.356 +0.117     +0.078 +0.181 +0.107    -66  -111   +57     yes    no       yes
    4    +0.297 +0.122 +0.517     +0.258 +0.132 +0.183    -115 -121   -26     yes    no       yes
    8    +0.177 -0.162 +0.289     +0.227 +0.086 +0.221    -347 -257  -104     yes    no       yes
    16   -0.026 +0.123 +0.511     -0.010 +0.020 +0.269    -732 -265  -303     no     no       no
    32   +0.271 +0.099 -0.027     +0.159 +0.094 +0.058    -479 -529  -522     no     no       yes

Rules: alpha pays at K = 2, 4, 8 (3/3 worlds at K = 2 and 4; robust
optimizers agree in sign); full pays at K = 2, 4, 8 (3/3 each) and 32
(2/3); PRESENT pays at no K — d_J >= 0 in 1/3 worlds at K = 2 (+57) and
0/3 elsewhere. K* is undefined. Outcome: **MIXED (reported per K)**.

What the table says beside its label:
- Restricting argument width makes discrete identity valuable for FUTURE
  acquisition: with K <= 8 per slot the told-identity oracle is better on
  alpha-only by 0.12-0.52 log units (one world at K = 8 reversed) and on
  the full interface by 0.08-0.26 log units in every world.
- The oracle's present-task cost shrinks ~10x as K falls — from ~-510
  nats at K = 32 to -66 / -111 / +57 at K = 2 — but crosses zero in only
  one world; the registered "present pays" bound (d_J >= 0 in 2/3) was
  not met. The mask is nearly free, not free.
- M_K's routing sharpens as capacity shrinks (median entropy 0.66-0.83
  bits at K = 2 versus 0.91-0.97 at K >= 16) but still does not align
  with the groups (ARI 0.05 / -0.01 / 0.27 at K = 2). The learner pays
  for ignoring the groups at small K and still does not find them.
- K = 16 is non-monotone (alpha and full do not pay; world 2 alone pays
  strongly), and K = 32's full-pays is carried by two worlds. With three
  worlds the high-K end is noisy; the low-K end (K = 2, 4) is consistent.

Reading, registered with the label: the reviewer's phase boundary exists
for the future channels — discrete identity starts to pay for alpha-only
and full-interface acquisition somewhere between K = 8 and K = 16 per
slot — while on the present-task axis the boundary has not been reached
by K = 2. This is outcome (3) ("structured channel wants discreteness")
extended: the FULL interface wants it too at small K; only present cost
declines to flip. Our registered prediction (outcome 3 at K = 4, possibly
4 at K = 2) was right about alpha and full at K = 4 and wrong about
present crossing zero at K <= 4 (it nearly did: -115 / -121 / -26).
Review 68's crossover prediction: true for the future channels, false for
the present.

Licensing: MIXED licenses nothing. The natural next plan, to be frozen
separately and NOT read from this data's thresholds: B2's discovery arms
(H_early / H_late) at K = 4 — the width where the oracle's future
advantage is largest and most consistent (alpha +0.30 / +0.12 / +0.52,
full +0.26 / +0.13 / +0.18) and its present cost smallest (-115 / -121 /
-26) — with "present" registered as a TOLERANCE band around zero rather
than a one-sided bound, since the quantity of interest there is whether
discretization is nearly free, not whether it is profitable.

# H49 REGISTERED (2026-08-23): STRUCTURAL DISCOVERABILITY — IS THE FERTILE PARTITION VISIBLE IN PAST EXPERIENCE?

From review 69 (`reviews/reviewer-feedback-69.txt`). Plan
`H49_DISCOVERABILITY_PLAN.md`, frozen `f806c8c`; instrument
`audit_h49_discoverability.py`; running at registration, no value read.

Motivation, registered: at K = 4 on the two-subspace world the true
grouping has FUTURE value (alpha +0.30 / +0.12 / +0.52, full +0.26 /
+0.13 / +0.18 log units) and no PRESENT value (-115 / -121 / -26 nats).
M's non-discovery is therefore rational under its objective, and
annealing it (the planned K = 4 H_early / H_late block) would force a
discrete decision its objective never rewarded — withdrawn before any
run. Three economic quantities are now distinguished: current utility,
future fertility, and representational cost; discovery needs a signal
computable from PAST experience that tracks the second.

Hypothesis H49: on the label-free learner's own frozen representation
(M_4), at least one retrospective quantity computable without labels or
future tasks — leave-one-out reacquisition C_LOO under a candidate
partition, a two-part description proxy D*, or own- versus other-group
substitutability — prefers the true grouping over the distributed policy
and over every wrong or random partition.

Registered rules: SIGNAL (C_LOO) iff log C_LOO(DISTRIBUTED) - log
C_LOO(TRUE) >= +0.15 AND the margin over the best wrong/random partition
>= +0.10, in >= 2 of 3 worlds; SIGNAL (D*) iff D*(DISTRIBUTED) > D*(TRUE)
and the two-part score prefers TRUE in >= 2 of 3. Instrument check: on
L_4 (representation trained with the grouping) TRUE must beat both wrong
pairings by >= 0.15 or nothing is read. Outcomes: A DISCOVERABLE (signal
on M_4) -> build a sleep operator on the signalling quantity, then
H_early / H_late; B UNDERDETERMINED (no signal on M_4 or L_4) -> option
value not inferable from current observations; C SIGNAL NEEDS
ORGANIZATION (signal on L_4 only) -> discovery needs a proposal-and-
reorganize step, not just a score.

Predictions: review 69 — A (a signal exists that wake ignores). Ours —
C: M_4's two slots are interchangeable (entropy ~0.75, ARI ~0), so
masking onto either slot re-fits about equally and P_LOO sits near zero,
while L_4 shows a clear signal.

Candidate sleep objective, registered as a design target and not as a
result: J_sleep = lambda D*(L) + eta sum_i C_reacquire(T_i | L_{-i}) —
description economics (V1-V5) plus retrospective fertility (H39/H48b),
with no future label. Its program-synthesis form: invent A iff D*(A) <
the cross-validated reduction in reacquisition cost. Neither is tested by
H49; H49 asks only whether the quantities they would optimize carry a
signal.

# H49 RESULT (2026-08-23): "A" BY THE FROZEN TABLE VIA A NON-DISCRIMINATING BRANCH; SUBSTANTIVE READING C — THE SIGNAL NEEDS ORGANIZATION

Plan `H49_DISCOVERABILITY_PLAN.md` frozen `f806c8c`, no amendments.
Report `reports/h49_discoverability.json`. All ~2,300 re-fits finite with
alpha moving and support falling; the instrument check passed (on L_4 the
true partition beats the best wrong partition by +0.601 / +0.304 / +0.646
log units).

    artifact  P_LOO (DIST - TRUE)        margin vs best wrong       P_D (nats)        S_subst
    M_4       -0.464 -0.525 -0.489       +0.059 -0.034 -0.043       +16.9 +15.0 +9.4  +0.119 -0.019 +0.207
    L_4       -0.065 -0.086 -0.045       +0.601 +0.304 +0.646       +24.3 +26.6 +22.5 +1.273 +0.815 +1.303

Frozen verdict: **A — DISCOVERABLE**, because SIGNAL (D*) fired on M_4
(P_D > 0 and two-part TRUE < DISTRIBUTED in 3/3 worlds).

ANNOTATION, recorded with the verdict and before any use of it: the D*
branch of the frozen rule does not discriminate the true partition. On
M_4 every MASKED candidate — true, wrong, or random — has nearly the same
description proxy (281-288 nats; the ~15-nat saving is 4 paid alphas
instead of 8, a property of masking itself) and nearly the same C_LOO
(0.0043-0.0064, with RANDOM-2 beating TRUE in two worlds). The C_LOO
branch required a margin over the best wrong partition; the D* branch,
as frozen, did not — the same registered-threshold error class as the
route-mass criteria (S0, H39b), now in its third appearance. The A
verdict therefore stands in the table but licenses nothing: a sleep
operator built on this D* term would prefer ANY partition over none.

Substantive reading, which the registered outcome C describes exactly:
every quantity that discriminates the true partition from wrong ones —
the C_LOO margin (+0.30 to +0.65) and own-versus-other-group
substitutability (+0.8 to +1.3) — appears ONLY on L_4, the
representation trained with the grouping. On the label-free M_4 those
same quantities sit at noise (margins -0.04 to +0.06; S_subst -0.02 to
+0.21), and the distributed policy re-acquires past tasks better than
any mask (P_LOO -0.46 to -0.53) because masking halves the directions of
an interchangeable-slot representation. The fertile partition leaves a
detectable signature in experienced tasks only once the representation
has been organized around it.

Scorekeeping: our registered prediction (C, with a clear signal on L_4
and none on M_4) is what the discriminating statistics show; review 69's
prediction (A, a signal wake ignores) is met only by the letter of the
non-discriminating branch.

Consequence, registered: discovery on this world cannot be a scoring
problem alone. A sleep operator would have to PROPOSE a partition,
REORGANIZE the representation under it (re-fit task states, possibly
briefly re-train), and RE-SCORE — the bootstrap loop outcome C implies —
or the world's statistics must change (more worlds, more diverse
experience, meta-priors). Building that proposal-reorganize-score
operator is the next design; it requires its own frozen plan, and its
score must carry a wrong-partition margin in every branch.

# H50 REGISTERED (2026-08-23): PROPOSE -> REORGANIZE -> SCORE (review 70)

Hypothesis: a structural hypothesis h that is future-fertile but
invisible in the current representation becomes distinguishable from
wrong hypotheses, using PAST data only, after a bounded matched-budget
reorganization of the representation under h. Central quantitative
variable: the migration budget m at which the separation first appears
(m small -> sleep-like restructuring is plausible; m near full
retraining -> wake must preserve more reorganizable state).

Registered outcomes (review 70): (1) TRUE beats WRONG/RANDOM after equal
reorganization — counterfactual restructuring is a valid discovery
objective; next problem is proposal/search. (2) Indistinguishable —
experience underdetermines the fertile structure. (3) Everything
improves — the loop is only extra optimization; a SHAM no-split control
and stronger charges are required before any claim. Non-negotiables
carried from H49/H50 design review: identical start checkpoint,
parameters, optimizer, steps, data, and D* accounting per candidate; a
SHAM arm; wrong-structure margins in EVERY branch; future siblings never
select the hypothesis.

Predictions: review 70 — outcome 1, with the open question being the
size of m*. Ours — outcome 1 at a large-but-partial m (the L_4 endpoint
is reachable without full retraining because the argument matrices and
task states carry most of the group structure and the basis is shared),
with the SHAM arm improving C_LOO uniformly but showing no TRUE-vs-WRONG
separation.

Plan: `H50_REORGANIZATION_PLAN.md`, DRAFT at registration — the reviewer
asked for careful design; it is not frozen and nothing runs until it is.

# H50 RESULT (2026-08-24): UNDERDETERMINED — BOUNDED MIGRATION RECOVERS ESSENTIALLY NONE OF THE SEPARATION

Plan `H50_REORGANIZATION_PLAN.md` (Amendments 1-2, frozen `aff9042`).
Report `reports/h50_reorganization.json`. All migrations and ~2,700
re-fits finite; every non-vacuity check passed; identical budgets
verified across arms.

    C_LOO margins, TRUE vs best wrong (need >= +0.15 in 2/3):
        m = 16:  +0.008 / -0.029 / +0.016
        m = 64:  +0.023 / -0.003 / +0.003
    TRUE vs SHAM: -0.39 to -0.58 at both budgets (SHAM far better).
    Recovery fraction of the L_4 separation: -0.09 to +0.09 — zero.
    Substitutability at m = 64: TRUE 0.46 / 0.29 / 0.66 versus best wrong
        0.39 / 0.24 / 0.61 — migration under ANY partition manufactures
        slot specialization; the TRUE-minus-wrong difference (~0.05) is
        far below the 0.30 margin.
    Sibling diagnostics (computed after all past-data decisions, selecting
        nothing): TRUE 0.0734 / 0.0570 / 0.0505 versus SHAM 0.0681 /
        0.0430 / 0.0542 — migrated-TRUE does not even recover a FUTURE
        advantage (worse than SHAM in 2/3 worlds).

Verdict by the frozen table: **UNDERDETERMINED** (no SEPARATION at any
scored m; SHAM's own m0 -> m64 gain, 0.20 log units, is below the 0.30
that would have read OPTIMIZATION-ONLY).

What the numbers say beside the label: 4,096 steps of budget-matched
reorganization under the true partition recover approximately 0% of the
retrospective separation that whole-lifetime organization produces
(L_4 margins 0.60 / 0.30 / 0.65), and none of the future advantage
either. Masking under any hypothesis creates the surface signature of
organization (elevated substitutability) without the substance
(discrimination or fertility). The chicken-and-egg from H49 is stronger
than "the score needs some reorganization": at these budgets the score
needs essentially the FULL formation process. Within the tested
operator, the propose-reorganize-score loop degenerates to
retrain-under-each-hypothesis-and-select — valid (L_4 exists and works)
but priced at one full lifetime per candidate.

Scorekeeping: review 71 predicted separation beginning at m = 16 with
25-50% recovery — wrong; our prediction of separation at large-but-
partial m — also wrong; our SHAM expectation (uniform improvement, no
ordering) — right. Review 70's fallback branch ("if it requires
near-total retraining, discovery is much more expensive and the
architecture needs to preserve more reorganizable state during wake") is
the branch the data chose.

Instrument disclosures, recorded with the result: (1) optimizer state
resets at the m = 4 / 16 / 64 snapshot boundaries (three AdamW instances,
identical for every arm); (2) retirement was lifted during migration, so
previously retired tasks start from the reference-plus-stored-residual
double-count state and migrate out of it (identical across arms; the
scoring re-fit resets residuals and is unaffected). Neither favours an
arm; both are matched deviations from an idealized single continuous
migration.

Licensing: nothing. The registered fork for the successor (each side
needs its own frozen plan): (a) test the retrain-and-select reading
directly — one full lifetime per candidate hypothesis is the price; or
(b) change WAKE so that reorganizable state is preserved during
formation (review 70's fallback), making future migrations cheaper —
which is a new architecture question, not a new score.

# H51 REGISTRATION (2026-08-24): THE REORGANIZABILITY TESTBED

Review 72 (`reviews/reviewer-feedback-72.txt`) answers the H50 fork: branch
(b), change wake so reorganizable state is preserved; retrain-and-select is
kept only as an oracle ceiling because C_search ~ N x C_lifetime makes it
unusable as a mechanism. Plan drafted at `H51_REORGANIZABILITY_PLAN.md`;
nothing runs until it is frozen with its commit hash in
`tools/check_prereg.py`.

NUMBERING MAPPING (appended, never renumbered): review 72 labels its three
successors H51 (provenance), H52 (decomposable innovation basis), and H53
(overcomplete provisional schemas). Our ledger uses H51 for the TESTBED the
same review recommends running first; the reviewer's H51 and H52 are its arms
`R_1` and `R_2`. The reviewer's H53 keeps its label and is out of scope for
this plan.

New endpoint, registered: `C_restructure(R) = min m in the scored grid at which
the H50 SEPARATION rule holds when the unchanged H50 migration operator is
applied to representation R`. The independent variable is the WAKE
representation; the migration operator and the H49 scoring instrument move by
nothing. Arms: `R_0` ordinary (re-used M_4), `R_1` provenance-preserving,
`R_2` decomposable-innovation, `R_3` oracle organized (re-used L_4, the
positive control that separation is expressible at m = 0).

Balance gates before any causal reading (the V5.1 confound rule and the
matched-budget constitutional rule): present loss within 10%, retained D*
within 20%, parameter budget within 20%, migration-variable count within 20%
with the step count matched exactly and a frozen-extra-variable companion where
an arm exposes more migration freedom. Failing arms are unscoreable for the
causal claim, not weak.

Registered predictions. Reviewer: `ordinary ~ provenance < decomposable <
oracle`; preserving residual vectors alone will not suffice; highest prior on
`R_2`. Ours: `R_1` moves nothing (H50 showed the retrospective OBJECTIVE, not
missing evidence, is what fails to discriminate); `R_2` improves the recovery
fraction but does not reach SEPARATION at m <= 64; modal outcome 2 (PARTIAL),
second outcome 3 (FORMATION-TIME CONFIRMED). If `R_2` does separate, we predict
a measurable present-performance cost near the G1 boundary, and that cost is
the result, not the separation alone.

# H51 RESULT (2026-08-24): FORMATION-TIME-CONFIRMED — REORGANIZABILITY IS NOT BOUGHT BY PRESERVING STATE

Plan `H51_REORGANIZABILITY_PLAN.md` (Amendments 1-5, frozen `b03a1be`).
Report `reports/h51_reorganizability.json` (merged from the three
per-representation files). The H50 migration operator and the H49 LOO
scorer were used UNCHANGED; the wake representation was the only
independent variable.

    C_restructure:  R_0 None   R_1a None   R_1b None   R_2 None
    Recovery of L_4's separation at m = 64 (worlds 0/1/2):
        R_0   -0.066 / +0.091 / +0.067      (H50, re-used)
        R_1a  -0.074 / +0.173 / +0.076
        R_1b  -0.119 / +0.021 / +0.050
        R_2   +0.047 / +0.189 / -0.003
    Margin vs best wrong (need +0.15 in >= 2/3):
        R_1a  +0.019 / +0.025 / +0.009
        R_1b  -0.005 / -0.027 / -0.009
        R_2   +0.084 / +0.030 / -0.045
    Margin vs SHAM (need +0.15): every arm, every world, -0.33 to -0.54.

Verdict by the frozen table: **FORMATION-TIME-CONFIRMED** (outcome 3). No
arm separates at any scored budget, and no pairwise recovery gain reaches
the 0.25 that would have read PARTIAL. Preserving task provenance — as
initialization (R_1a) or as a re-fittable recombination channel over the
traces of co-assigned tasks (R_1b) — and storing innovation in a
separately addressable component basis learned during the lifetime (R_2)
all leave the fertile counterfactual as unreachable as ordinary wake did.

Three findings beside the label.

1. USEFULNESS AND INFORMATIVENESS COME APART. R_1b's trace channel cut
   absolute reacquisition cost 18-38% below R_0 (e.g. world 2 TRUE
   0.00276 versus R_0's 0.00329) while being ANTI-informative about
   grouping: the wrong partition scored better in all three worlds. A
   channel can be worth having and carry no evidence about which
   structure produced it.
2. THE PARAMETER GATE EARNED ITS PLACE. R_2's only strong margin
   (+0.084) is in world 0 — the world whose G3-shared reading fails at
   +28.9% because R_2 promoted three more abstractions there. The two
   gate-PASSING worlds give +0.030 and -0.045. Without the registered
   gate this would have been reported as "decomposable wake improves
   discrimination in 2 of 3 worlds"; with it, the effect is confined to
   the cell that is unscoreable for the causal claim. G1 passed
   everywhere (present loss changed by 0.09-0.19%), so R_2 is a genuinely
   matched representation on cost.
3. THE MANUFACTURED SIGNATURE REPLICATES ACROSS REPRESENTATIONS.
   Substitutability rose under every partition in every arm, with
   TRUE-minus-best-wrong gaps of 0.008-0.119 against a required 0.30, and
   the WRONG partition scoring HIGHER in R_1b world 1 and R_2 world 2.
   Sibling endpoints show no recovered future advantage either: migrated
   TRUE is worse than SHAM in 2/3 worlds for R_1a, R_1b, and R_2 alike.

Scorekeeping. Review 72 predicted `ordinary ~ provenance < decomposable <
oracle`; the ordering in recovery is within noise (all |values| < 0.19)
and its qualitative claim — that provenance alone would not suffice — is
upheld, while the expectation that an explicitly decomposable wake WOULD
buy reorganizability is not. Ours: R_1a moves nothing — CORRECT; R_1b
improves recovery without separating — WRONG in sign (it lowered
recovery and made the margin negative); no arm separates at m <= 64 —
CORRECT; modal outcome PARTIAL — WRONG, outcome 3.

Instrument disclosures. (a) An arm's own m = 0 row is measured for R_1a
and R_2 and re-used from H49 for R_1b (its coefficients start at zero, so
its m = 0 IS R_0's; verified bitwise on the real artifact). R_1a's three
measured m = 0 rows reproduce H49's recorded M_4 values exactly
(0.00560 / 0.00549 / 0.00447), which independently validates this
re-implementation of the H49 scorer. (b) H50's two matched deviations
(optimizer state resets at snapshot boundaries; retirement lifted during
migration) are inherited unchanged, by design. (c) Amendments 4 and 5
record two defects found by the standing pre-run audit while the scorers
ran — R_1b's basis originally contained the task's own trace (worth ~2%,
arm-neutral, fixed and re-run) and `factorized_fit` would have crashed
R_2's sibling step and lost the whole report (fixed and re-run).

Licensing: nothing new. What H51 removes is a class of explanations: the
H50 negative is NOT an artifact of ordinary wake discarding task-local
evidence, nor of storing innovation densely rather than in addressable
components. Whatever L_4 has that M_4 lacks, it is not preserved state
and not storage form.

# H53 REGISTRATION (2026-08-25): PARALLEL FORMATION OF STRUCTURAL HYPOTHESES

Review 73 (`reviews/reviewer-feedback-73.txt`) answers H51 with the PI's call:
change the substrate. Plan `H53_PARALLEL_FORMATION_PLAN.md`; nothing runs
until it is frozen with its commit hash in `tools/check_prereg.py`.

Numbering: this is the reviewer's own H53 label. The reviewer's H52
(decomposable innovation basis) was already run as H51's arm `R_2` and is
closed; H52 is therefore retired as a label and not reused.

Question: can several candidate organizations develop CONCURRENTLY within one
lifetime, so the true one becomes retrospectively distinguishable at
substantially less than one full-lifetime cost per candidate? Six externally
supplied heads (TRUE, WRONG-A/B, RANDOM-1/2 with H49's seeds, SHAM), no
learned proposer. Each head predicts every task, is scored before update, and
is updated on the same data; the objective is the SUM of the heads' ordinary
prequential objectives, so `H = 1` is exactly the ordinary lifetime.

Independent variable: the sharing level. L1 branches only the task-local slot
arguments; L3 additionally branches the argument matrices, route codes and
residuals, sharing only the basis and the promoted library. L1 and L3 are run
first because they BRACKET the predicted frontier `s*`; L2 runs only if they
disagree. Disclosed in advance: PROMOTE and retirement stay shared at every
level, so H53 spans `s = 1` toward but not to `s = 0` — the separate-lifetime
endpoint is `M_4` and `L_4` themselves.

Endpoints: `C_LOO` per head under H50/H51's UNCHANGED margins (+0.15 vs every
wrong head AND vs SHAM, >= 2/3 worlds, substitutability +0.30), recovery
against `L_4` with `M_4`'s m = 0 baseline, and the AMORTIZATION RATIO
`A = (C_shared_wake + H C_candidate) / (H C_lifetime)`. Outcomes A (works
cheaply, `A <= 0.5`), B (works only at L3, `A > 0.5` — concurrent
retrain-and-select), C (sharing collapses the heads; frontier localized
between the levels), D (no discrimination anywhere, with heads demonstrably
distinct and learning — registered stop condition for this architecture).

Equivalence controls, required before any verdict: `H = 1` SHAM must reproduce
`M_4` BITWISE and `H = 1` TRUE must reproduce `L_4` BITWISE, at each level;
heads must be functionally distinct; every head must learn; every head's
policy must fire on the expected task count.

Registered predictions. Reviewer: a frontier exists, heads collapse at high
sharing, and the hoped-for result is `A << 1` with TRUE developing `L_4`-like
signal. Ours: outcome C — collapse at L1 (H47's baselines showed this world's
learner absorbs imposed structure into one distributed channel: entropy ~0.93,
ARI ~0), and at L3 a TRUE margin clearly above H51's (>= +0.10) but short of
`L_4`'s 0.30-0.65 and probably short of +0.15, because the shared basis and
shared library still carry most of the computation. We further predict SHAM
remains the best head on raw `C_LOO` at both levels, as in H50 and H51, making
the SHAM clause the binding one.

# H53 AMENDMENT 1 (2026-08-25, review 74, before any code): WHAT H53 CAN AND CANNOT CLAIM

Review 74 rejected the first freeze and its framing objection is accepted in
full. H53 does not compare a shared trajectory with six independent ones; it
introduces a NEW COUPLED LEARNER in which every head contributes gradient to the
parameters TRUE must develop through. The measured quantity is renamed the
SHARED-PARAMETER CO-FORMATION FRONTIER, and no H53 verdict may claim that
computation in general cannot be amortized.

Six fixes registered before any code (`H53_PARALLEL_FORMATION_PLAN.md`
Amendment 1): (1) shared parameters take the MEAN of head gradients while
head-specific parameters keep their own gradient at full scale — stricter than
review 74's literal proposal, which would have divided head gradients by H and
changed each head's effective learning rate relative to M_4/L_4; at H = 1 both
rules reduce to the ordinary objective, so the bitwise controls stay exact.
(2) L3's partition is specified object by object, optimizer state included, with
PROMOTE clustering over the SHAM head's residuals at L3 so the library stays one
head-agnostic object. (3) The collapse probe CLEARS every head's mask and
resets its route temperature, so it compares learned state rather than six
hard-coded policies, referenced against the divergence between the independent
M_4 and L_4 artifacts. (4) Cost splits into A_train, A_state and A_total in
device-seconds, with selection cost included — LOO scoring exceeds a lifetime
per head, so selection rather than formation may be the binding cost, which is
itself a registered possible finding. (5) Outcomes C_1 (L1 fails, L3 passes:
frontier bracketed) and C_2 (L3 beats L1 by >= 0.05 in >= 2/3 worlds but does
not separate: depth matters, frontier unlocalized) replace the old C, and D now
licenses only "the frontier, if it exists, lies deeper than L3"; the
stop-pushing-this-architecture recommendation is WITHDRAWN from D because this
design never reaches independent libraries. (6) The L2 trigger is machine-
testable and printed before any L2 cell launches.

Predictions after the amendment. Review 74 (PI): L1 collapses, with neutralized
head state very similar across heads; L3 either lifts the TRUE margin without
reaching 0.15 or separates at a poor cost ratio; low probability on clean
separation with A_train << 1. Ours, now stated against the narrowed split: **C_2
specifically** — collapse at L1, and at L3 a TRUE margin >= +0.10 that still
misses +0.15, with SHAM the binding clause.

# EXPORT BRANCH REGISTRATION (2026-08-25, review 75): EXPORT -> COMPOSITION -> SYNTHESIS

Review 75 opens the successor branch. Program document
`EXPORT_BRANCH_PROGRAM.md`, frozen as the branch's DECISION TREE and
TERMINOLOGY CONTRACT; each rung gets its own frozen plan before its own code.
The running formation line (H53) is untouched and its result may not be used to
change any threshold registered here.

Question: does learning produce frozen computational objects that remain useful
when exported into programs the lifetime never optimized? Constitutional
criterion: synthesis starts when a FROZEN learned object composes in a program
never jointly optimized with it. Binding terminology from now on: an object is a
SHARED OBJECT / BASIS ELEMENT / SLOT / ABSTRACTION until it passes an export
test; PRIMITIVE is reserved; unseen-task reuse is not COMPOSITION without unseen
PROGRAM COMBINATIONS; route optimization is not SYNTHESIS without a compact
program variable.

Rungs and order: Phase 0 E0/E4/E7 on existing artifacts (no new lifetimes);
Phase 1 E1 frozen-library export (arms O/R/S/F/D/W, `G_export` reported with
absolute losses); Phase 2 E2 support-split composition, whole programs withheld,
strata H1 triple-novel and H2 pair-novel; Phase 3 E8 length generalization;
Phase 4 E3 writability then E5 program recognizer; Phase 5 E6 primitive
invention under an explicit birth charge with refusal controls. Registered STOP:
if E1's oracle arm fails, the synthesis interpretation of this substrate ends.

Predictions. Review 75: partial functional identity; residuals more load-bearing
than hoped; reasonable chance E1-oracle passes; lower confidence in E1-route, so
the modal interesting outcome is ORACLE EXPORT WORKS, ROUTE INFERENCE LAGS,
localizing the next problem to writing/search; E2 genuinely uncertain and not
implied by E1. Ours: we agree on E1's modal outcome; we are PESSIMISTIC on E2,
because V1's crossover shows the basis is preferred only above `r ~ 0.83` and
H47 showed this learner absorbing distinct latent structure into a single
channel — the opposite of the stable per-object interface E2 needs. If E2's
oracle arm fails, our registered reading is that the OPERATOR ABI, not the router
and not the library's contents, is what must be redesigned. E7 is the rung we are
least able to predict and we say so in advance.

# EXPORT BRANCH AMENDMENT 1 (2026-08-25, before any rung runs)

Five changes to `EXPORT_BRANCH_PROGRAM.md`, made while no data from the branch
exists.

1. E1 must include a ROUTE-EXPRESSIBLE substrate (the discrete-library family)
   beside the strongest-loss one, and the branch STOP requires the oracle arm to
   fail THERE. Our V1 measurements already show the continuous learner solution
   is not expressible as a teacher route through matched slots (0.00191 own
   mixture versus 0.00937 matched-slot route, ~4.9x) while the discrete learner
   is (0.00456 versus 0.00484, ~1.06x). Selecting on lifetime loss alone would
   have produced a negative about mixture routing and licensed a STOP about
   export.
2. New gate E1.0: oracle routing on SEEN programs must recover within 2.0x of the
   intact model before any unseen-program claim. Threshold set against those two
   measured baselines (geometric midpoint 2.28x), not guessed.
3. E1 gets an adaptable/frozen tensor table per arm, and arm R splits into
   R-route (residual frozen; PRIMARY) and R-full (residual adaptable; capacity
   reference), so `G_export` cannot be manufactured by a 198-scalar patch.
4. New Phase 0 gate E2-feas: enumerate whether the E2 pair-novel stratum EXISTS
   at usable size under the coverage constraints, before designing the world.
5. New rung E9, EXPORT-CONSTRAINED FORMATION, registered now with its trigger,
   matched controls (including a wrong-constraint arm) and endpoint, so an E1
   negative lands on a plan instead of a pause. Rationale: H29, H50 and H51 all
   say a property not built into formation is not recoverable afterward, and
   exportability is such a property.

Registered predictions for the new material: E1.0 passes on discrete and fails on
the mixture substrate; E9 works where the audit fails and costs present-task
loss, the same trade H39 found for the argument channel. Ceiling stated in the
document: a clean sweep licenses a fixed-depth sequential-composition DSL, a
foundation claim, not program synthesis in the usual sense.

# E2-FEAS RESULT (2026-08-25): E2 IS CONSTRUCTIBLE; THE STRATA TRADE OFF AGAINST EACH OTHER

Plan `E0_PHASE0_AUDIT_PLAN.md` (Amendment 1, frozen). Report
`reports/e2_feasibility.json`. Pure combinatorics, no model, seconds of compute.

At the TRUE lifetime size (64 distinct programs of 216, with every primitive in
every position, at least 3 distinct surrounding contexts each, and frequency
balance <= 2.0), over 64 seeded constructions per fill objective:

    pair-minimising fill:  best |H1| =   4    best |H2| = 152   (64/64 valid)
    neutral random fill:   best |H1| =  31    best |H2| = 121   (62/64 valid)
    registered thresholds: |H1| >= 16 AND |H2| >= 16

**CONSTRUCTIBLE = True**, and no designed program schedule is required: the
neutral fill reaches both thresholds on its own (an example construction has
|H1| = 111, |H2| = 41, balance 1.4, minimum 25 contexts per primitive). E2 may
therefore keep `K = 6, D = 3` and stay comparable with every existing artifact.

The informative part is the TRADE-OFF the two arms expose. Minimising newly
covered adjacent pairs maximises the pair-novel stratum (152) and collapses the
triple-novel one (4): concentrating training on few pairs pushes nearly every
unseen triple into H2, leaving almost nothing that is triple-novel with all
pairs seen. The two strata are in direct tension, so E2's plan must FREEZE its
target split rather than maximise either -- a lifetime tuned for H2 has almost
no H1, and vice versa.

Instrument disclosure, recorded because the first run was wrong: the initial
implementation satisfied the coverage constraints with 9-13 programs and
reported |H1| = 19, |H2| = 203 -- void, because E2's training set IS the
lifetime (64 tasks), not something bounded by it. Amendment 1 fixed the
quantifier before any verdict was recorded; that first run is preserved as an
instrument dry run. A second reporting bug (the verdict read only the
pair-minimising arm, printing False while the random arm passed) was fixed in
the same pass, before this result was recorded.

Registered prediction scored: ours said this was the rung we could not guess and
that |H2| >= 16 might be impossible. It is possible, comfortably, under a
neutral schedule -- so the prediction was right to abstain and wrong in its
worry.

# E0.1 + E1.0 RESULT (2026-08-25): THE SUBSTRATE THAT WINS THE ECONOMICS IS NOT THE ONE THAT ROUTES

Plan `E0_PHASE0_AUDIT_PLAN.md` (Amendments 1-2, frozen). Report
`reports/e0_export_audit.json`. Existing artifacts only, no new lifetimes.

    substrate                      E0.1 matched  random  null-edit  margin  spread   E1.0 intact  oracle-route  ratio
    DISC  discrete, seed 0             0.094     0.390     1.000    0.003    0.25      0.00948      0.00967      1.02
    MIX   continuous, rho=1 world 1    0.075     0.546     1.000    0.000    0.22      0.00163      0.00739      4.53
    MIX   continuous, rho=1 world 2    0.075     0.662     1.000    0.023    0.29      0.00182      0.00738      4.06

**E1.0 (eligibility for E1): DISC PASSES (1.02), MIX FAILS (4.53 / 4.06)** against
the registered gate of 2.0x, with both controls far worse in every case (random
assignment 3.5-31x, shuffled library 3.2-30x), so the gate itself is non-vacuous.

This is exactly the case the branch amendment was written for. Selecting
artifacts by lifetime economics — as the branch originally specified — would
have put E1 on MIX alone, produced a negative, and cashed it as "these objects
are not exportable program primitives". The correct reading of that negative is
narrower: **the continuous mixture's solution is not expressible as a route
through individual objects.** Registered branch consequence: only DISC may carry
the branch STOP; MIX's later E1 rows are UNINTERPRETABLE as export failures.

The dissociation is sharp and is the finding worth keeping: MIX beats DISC by
~5.8x on its own intact task loss (0.00163 versus 0.00948) and is the substrate
the whole project selected on economics, yet under oracle routing the two are
comparable (0.00739 versus 0.00967). The mixture's advantage lives precisely in
the part that is NOT a program.

**E0.1 (terminology): both substrates pass substitutability, with a caveat that
matters.** Matched contextual distances are 0.075-0.094 against a null edit of
1.000, i.e. a matched object is about 11-13x closer to the teacher operation
than omitting that operation entirely, and 4-9x better than a random assignment;
per-position spread is 0.22-0.29, inside the registered 50%. The licensed
sentence is therefore available: LEARNED OBJECTS ARE FUNCTIONALLY SUBSTITUTABLE
FOR STABLE TEACHER OPERATIONS ACROSS MULTIPLE PROGRAM CONTEXTS.

The caveat: the ASSIGNMENT MARGIN is 0.000-0.023. The identity of the matching is
barely determined — several assignments are nearly as good as the best one. So
substitutability holds while one-to-one identity does not, which is review 75's
prediction of "partial functional identity rather than beautiful one-to-one
teacher recovery", confirmed quantitatively. Under the terminology contract this
licenses "substitutable object", not "recovered primitive".

Scorekeeping. Ours: DISC passes E1.0 near 1.1 (actual 1.02) and MIX fails at 3-5
(actual 4.53 / 4.06) — CORRECT, and it reproduces the V1-era matched-slot
measurement (1.06 and 4.9) on different artifacts with today's loader. Ours also
said MIX would show weaker alignment than DISC; that is WRONG, MIX's matched
distance is slightly better (0.075 versus 0.094) — alignment quality and route
expressibility are independent, which we had not anticipated. Review 75's
partial-identity prediction: CORRECT.

Instrument disclosures. (a) Amendment 2 corrected the distance denominator from
total output scale to the teacher operation's CONTRIBUTION before any number
existed; the null-edit guard it added reads exactly 1.000 for every primitive on
every substrate, as it must by construction, and no matched object scored above
its own null edit. (b) The discrete artifact predates learnable operator scales,
so the alpha convention is read from the checkpoint rather than assumed; the
continuous artifacts are legacy pickle checkpoints and are loaded through the
provenance-checked fallback the quantizer already used. Both facts are recorded
per artifact in the report. (c) DISC exists for one world only; every DISC number
here is n = 1, and E1 requires two further DISC lifetimes before any E1 verdict.

# E0.2 / E7 RESULT (2026-08-25): THE PRIVATE RESIDUAL IS INERT AS TRAINED, NOT INCAPABLE

Plan `E0_PHASE0_AUDIT_PLAN.md`. Report `reports/e0_residual_audit.json`.
Existing artifacts only.

**Structural, for the two substrates E1 actually concerns.** DISC and MIX carry
`task_codes` and nothing else: there is NO private residual channel, so task
identity lives wholly in the routed library. Reported as an architectural fact
rather than as a measured zero.

**Measured, on the modern substrate that does have residuals** (`w_m4`, all 64
family tasks plus the 8 pre-onset tasks per world):

    world   L_full     L_no_residual   L_no_library   L_refit    retired   R_residual   live-only R
    0       0.00579    0.00580         1.61801        0.00253    57/72     0.0000       0.0000  (15 tasks)
    1       0.00537    0.00540         1.59600        0.00222    59/72     0.0000       0.0001  (13 tasks)
    2       0.00457    0.00457         1.63119        0.00208    55/72     -0.0000      -0.0000 (17 tasks)

Disabling every private residual moves the loss by 0.0-0.5%; disabling the
routed library moves it by a factor of ~280-357. `R_residual` is zero to four
decimals in every world, and the LIVE-ONLY slice gives the same answer, so the
aggregate is not an artifact of most tasks being retired.

**The refit condition is the informative one.** Re-fitting that same residual
under the frozen library and route reaches 0.00253 / 0.00222 / 0.00208 — 41-46%
of the trained loss, i.e. LESS THAN HALF. The channel is therefore not
incapable; it is UNUSED AS TRAINED. Online learning plus retirement left a
capacity the task never exploited.

Registered consequence for E1, already anticipated: an export arm that leaves
the private residual adaptable can recover a large fraction of loss for reasons
that have nothing to do with the library. Export-branch Amendment 1's split of
arm R into **R-route** (residual frozen, PRIMARY) and **R-full** (residual
adaptable, capacity reference) is not a formality — this result gives it a
number, and any `G_export` computed with an adaptable residual must be read as
a capacity measurement.

Instrument disclosure, important: the refit condition LIFTS RETIREMENT so that a
retired task has a residual to fit at all. That configuration — a promoted
reference AND a private residual on the same task — is one the lifetime never
used, and it is exactly the state the review-55 loader bug produced by accident.
Here it is deliberate and necessary (there is otherwise nothing to re-fit), but
`L_refit` is therefore a counterfactual capacity probe, not a reconstruction of
anything that existed during training, and no claim rests on it beyond
"the channel had unused capacity". `L_full`, `L_no_residual` and `L_no_library`
all use the true retirement state.

Scorekeeping. Review 75 predicted residuals would be "more load-bearing than we
would like ... the first warning sign" — WRONG, decisively: they are inert.
Ours registered the rung as unpredictable and declined to give a number, which
was honest but scores as no prediction. The one substantive thing we said —
that H39's ~2% family-computation figure said nothing about ordinary task
identity — is now answered: ordinary task identity is in the library too.

# H53 RESULT (2026-08-25): OUTCOME D — NO DISCRIMINATION AT EITHER SHARING LEVEL

Plan `H53_PARALLEL_FORMATION_PLAN.md` (Amendment 1, frozen `68688b7`). Report
`reports/h53_parallel_formation.json`. Six candidate organizations co-formed in
one lifetime at two bracketing sharing levels, scored by the UNCHANGED H49 LOO
instrument and H50/H51 margins.

    margin vs best wrong (need +0.15 in >= 2/3):
        L1  -0.050 / -0.025 / -0.036
        L3  -0.072 / +0.027 / -0.150
    margin vs SHAM (need +0.15):
        L1  -0.133 / -0.108 / -0.080
        L3  -0.137 / -0.062 / -0.150
    depth effect (L3 minus L1, need >= +0.05 in >= 2/3):
        -0.022 / +0.052 / -0.114        -> 1 of 3 worlds
    recovery of L_4's separation:
        L1  -0.202 / +0.026 / +0.010
        L3  -0.242 / +0.179 / -0.155

Verdict by the frozen table: **D — NO DISCRIMINATION AT EITHER LEVEL.** The
machine-testable L2 trigger evaluates to FALSE, so L2 is not run and the
decision was made without discretion.

Licensed conclusion, exactly as narrowed by Amendment 1: **the shared-parameter
co-formation frontier, if it exists, lies deeper than L3.** NOT that structural
alternatives cost full retraining on this substrate — this design never reaches
independent libraries, and the stop-this-architecture recommendation was
withdrawn from D before the data existed.

The controls that make D readable rather than vacuous all pass:

- **Heads are demonstrably distinct.** Mask-neutralised pairwise divergence is
  0.0068-0.0302 against an independent-development reference (M_4 versus L_4 on
  the same probe) of 0.0596-0.0747 — that is 10-44% of genuine independence, far
  above the registered 5% collapse threshold. NO WORLD COLLAPSED at either
  level. So this is not "sharing destroyed the hypotheses"; the heads really did
  develop differently and still cannot be told apart retrospectively.
- **Every head learned** (first-to-last-quarter prequential improvement, all six
  heads, all six cells) and every head's policy fired on all 64 family tasks.
- **`H = 1` reproduces `M_4` and `L_4` bitwise**, so the substrate contains both
  reference mechanisms exactly.

**The cost accounting is the other half of the result.**

    A_train (device-seconds, six heads vs six independent lifetimes)
        L1  0.767 / 0.833 / 0.833      L3  0.859 / 0.782 / 0.847
    A_state (unique live state vs six independent)
        L1  0.188 (5.3x amortised)     L3  0.848-0.860 (barely amortised)

Co-formation saves 15-23% of training compute at best. The one place amortisation
is real is L1's STATE (heads share everything but their per-task arguments,
0.19), and L1 is precisely the level with no discrimination and — structurally —
no possible future differentiation: at L1 a head's only private state is
per-task alphas, which do not exist for an unseen task, so all six heads give
IDENTICAL sibling endpoints (0.0703 / 0.0553 / 0.0742, equal to four decimals
across heads). L1 co-formation cannot produce differential future fertility even
in principle. At L3, where heads own real state, sibling endpoints finally differ
— TRUE beats SHAM in 2 of 3 worlds, by 0.0026 and 0.0024 log units, which is
noise beside L_4's separation.

Substitutability corroborates nothing, again: at L1 the best WRONG head is MORE
substitutable than TRUE in all three worlds (TRUE -0.010/+0.026/-0.012 versus
0.091/0.077/0.060); at L3 TRUE leads in only 1 of 3.

Scorekeeping. Review 74 (the PI) predicted L1 would COLLAPSE with neutralised
head state very similar across heads — WRONG: L1 heads reached 21%, 21% and 9%
of independent divergence and never fired the collapse test. Review 73 predicted
a frontier `s*` between the levels — NOT FOUND in this range. Ours predicted
outcome C_2 (collapse at L1, L3 margin >= +0.10 short of +0.15) — WRONG on both
halves: no collapse, and L3's margins are mostly NEGATIVE. Ours also predicted
SHAM would remain the best head and be the binding clause — CORRECT, in all six
cells. That is now four consecutive rungs (H50, H51, H53) in which the
structureless control beats every structural hypothesis.

Licensing: nothing new. What H53 removes is the hope that developing candidate
organizations CONCURRENTLY is a cheap substitute for developing them
independently: at the sharing levels reachable here it is neither cheap (A_train
0.77-0.86) nor discriminating (no margin anywhere near +0.15), and the failure
is not explained by the heads collapsing into one another.

# EXPORT BRANCH AMENDMENT 2 (2026-08-25, review 76, before the E1 rung plan exists)

Phase 0's refit measurement (residual inert as trained, 41-46% of trained loss
recoverable when re-fitted) identifies a false-positive route through E1 that the
branch as written would not have caught: an adaptable private residual can
manufacture export-looking numbers while the frozen program contributes little.

Registered in `EXPORT_BRANCH_PROGRAM.md` Amendment 2:

1. Three interfaces on every E1/E2 test — **E1-P** pure program (PRIMARY),
   **E1-PR** program+residual (full-system endpoint, never evidence on its own),
   **E1-R** residual-only with the library neutralised — with
   `Delta_library = L_R - L_PR` as the performance uniquely attributable to the
   library. This subsumes Amendment 1's R-route/R-full split.
2. `L_oracle-program, eps = 0` is the cleanest vocabulary test, and its failure
   branch is frozen now: oracle+residual working while oracle-no-residual fails
   means THE RESIDUAL IS REPAIRING THE LIBRARY and no export claim is licensed.
3. E7 becomes a continuing control reported per cell rather than a Phase 0 audit.
4. Every write-up must separate WHAT THE REPRESENTATION EXPORTS from WHAT A
   DOWNSTREAM ADAPTOR RECONSTRUCTS AROUND IT — 2,000 steps of private adaptation
   grants a capability the lifetime policy declined to use.
5. E1's question is restated: not whether anything is worth exporting (Phase 0
   settled that) but whether a library that fully specifies its trained tasks
   retains stable semantics on an unseen program. Live alternatives: stable
   reusable operations versus a distributed routed basis that only works in
   familiar contexts.

Our note, registered with it: the confound cannot bite on the substrate that
actually passed E1.0. DISC has NO private residual channel, so E1-P is the only
interface it has and E1-PR/E1-R are undefined there. The three-way split becomes
load-bearing when a residual-bearing substrate enters — i.e. at E9, whose export
endpoint is therefore fixed to E1-P by this amendment.

# E1 RESULT (2026-08-26): THE FROZEN LIBRARY EXECUTES PROGRAMS IT NEVER TRAINED ON, AND THE ROUTE IS FINDABLE

Plan `E1_FROZEN_EXPORT_PLAN.md` (Amendments 1-2, frozen `dc161fc`), under
`EXPORT_BRANCH_PROGRAM.md` (Amendments 1-2). Report `reports/e1_export.json`.
Substrate: the three fresh DISC lifetimes, all three eligible by the E1.0 gate
(1.17 / 1.09 / 1.20 against 2.0). Interface E1-P throughout — frozen library,
route only; DISC has no private residual channel, so E1-PR and E1-R are
undefined here rather than skipped.

Geometric-mean query NMSE over 12 held-out programs per stratum per world:

    H1 (triple-novel)      O        O-W      R        R-W      S        F
        world 0          0.00760  0.06874  0.00735  0.05842  0.04432  0.07035
        world 1          0.00190  0.05936  0.00190  0.04810  0.03096  0.03955
        world 2          0.00418  0.06886  0.00374  0.05605  0.04139  0.05310
    H2 (pair-novel)
        world 0          0.00610  0.06520  0.00503  0.05889  0.04379  0.05686
        world 1          0.00451  0.05455  0.00501  0.04859  0.03413  0.04020
        world 2          0.00473  0.07298  0.00334  0.05826  0.04009  0.05064

**E1a — VOCABULARY EXPORTS: PASS**, on both strata, in all three worlds.
`O` beats scratch by +1.76 to +2.79 log units and beats an incompatible world's
library by +2.20 to +3.44, against a registered margin of +0.15. The oracle
figures (0.0019-0.0076) sit at or below the substrates' own TRAINED-task loss
(0.0038-0.0073): executing a program the lifetime never saw costs the frozen
library nothing measurable.

**E1b — THE ROUTE IS FINDABLE: PASS**, on both strata, in all three worlds.
`R` beats scratch by +1.80 to +2.48, and `|log R - log O| <= 0.15` in 5 of 6
cells (H2 world 2 is -0.346 — inferred routing BEATS the teacher route there).
`G_export` is 0.983-1.039: support-only route inference closes essentially all
of the scratch-to-oracle gap.

**H2 is as strong as H1.** The pair-novel stratum — held-out programs containing
an adjacent primitive pair that never occurred in training — matches or exceeds
the triple-novel one on every margin. Whatever the library holds does not depend
on having seen a particular adjacency.

What this licenses under the terminology contract, exactly: the frozen learned
objects CONTAIN TRANSFERABLE COMPUTATION and a task can cheaply IDENTIFY AND USE
that computation. It does NOT yet license the word COMPOSITION: E1's held-out
programs are opportunistic leftovers of the world's own 64-program sample
(constrained so every primitive appeared in every tested position), while the
contract reserves the composition claim for E2's constructed support-split
lifetime. E1's `O` arm passing on H2 is strong evidence that E2 will not fail
for lack of stable objects.

Two boundaries stated so the result is not over-read:

1. **E1b is OFFLINE route inference** — 2,000 Adam steps on 128 support examples
   against a frozen library — not the ONLINE routing that V2 found hard during a
   lifetime. These are different problems and the result does not overturn V2.
2. `F` (full finetune) drives its training objective to ~0 (objective reduction
   1.000 in every cell) and generalises WORSE than route-only adaptation
   (`C_repair` = -0.035 to -0.063). At this budget, unfreezing the library on one
   task's support OVERFITS rather than repairs. Reported as measured; no
   threshold depends on it.

Non-vacuity, all satisfied: claim-bearing arms `R` and `S` reduce their own
objective by more than 1% in every cell (0 weak cells); held-out programs
verified absent from training in code; every primitive of every held-out program
appeared in training in that position; support and query drawn from disjoint
seeded streams; no query label entered any selection.

Scorekeeping. **E1a: ours CORRECT** (we predicted a pass, reasoning from the
1.02 trained-program gate, and predicted `L_O` near trained-task NMSE — it landed
at or below it). Review 75 also gave it a reasonable chance. **E1b: BOTH
PREDICTION SETS WRONG.** Ours said route inference would fail or sit near the
boundary, citing V2's online-routing difficulty and DISC's own weak online
routing; review 75 called "oracle export works, learned route inference lags"
its modal interesting result. Instead inference matched the oracle almost
exactly. The error in both cases was importing an ONLINE result into an OFFLINE
question — the same category slip this project has made before in the other
direction. Our stratum prediction (H2 no worse than H1 for the oracle arm, worse
for inference) is CORRECT for the oracle arm and WRONG for inference: H2's
inference margins are the larger ones.

Licensing: E1a and E1b both pass, so the branch proceeds to **E2** as the frozen
run order specifies. The registered E1a-failure STOP is not triggered, and E9
(export-constrained formation) is not triggered — its trigger was an E1 oracle
failure or an E2 oracle failure, and the first has not occurred.

Instrument disclosure: a first full pass with a MODE-MIXED support-reduction
diagnostic was VOIDED before any verdict was recorded and is preserved at
`reports/e1_export_void_firstpass.json`; its query NMSEs were computed
consistently and match this pass, but its non-vacuity statistic was meaningless.
Amendment 2 replaced it with two mode-consistent reductions and fixed the
clause's conflation of "adapted" with "improved" before the data was read.

# E1-R RESULT (2026-08-26): EXPORT IS RECURRENCE-DEPENDENT

Plan `E1R_RECURRENCE_CONTROL_PLAN.md` (frozen `bed3b91`). Report
`reports/e1r_recurrence.json`. Held-out TASK protocol, uniform across `rho`;
12 held-out tasks per cell; interface E1-P.

    rho   measured r        O        R        S      M_R (log S - log R)
    1.0   +1.000 (3/3)   0.00674  0.00683  0.04214   +1.819
                         0.00219  0.00225  0.03235   +2.665
                         0.00498  0.00314  0.03321   +2.360
    0.9   +0.654/+0.628/+0.639
                         0.02385  0.02307  0.03964   +0.541
                         0.01848  0.01909  0.03535   +0.616
                         0.02368  0.02206  0.03611   +0.493
    0.0   -0.001/-0.001/+0.002
                         0.03859  0.03676  0.03786   +0.029
                         0.04458  0.03848  0.03777   -0.019
                         0.04523  0.03910  0.04010   +0.025

Verdict by the frozen rule: **RECURRENCE-DEPENDENT.** `M_R(1.0) >= +1.0` in 3/3
worlds and `M_R(0.0) <= +0.3` in 3/3, with `rho = 0.9` strictly between.

**At zero recurrence the frozen library is worth nothing.** `M_R` is +0.03 /
-0.02 / +0.03 — a trained 12-operator library gives a held-out task no advantage
over a library initialised at random and trained on that task's own 128 support
examples. The oracle arm is if anything slightly NEGATIVE (`M_O` -0.02 / -0.17 /
-0.12): with independent per-task teachers there is no operator to route to.

So E1's result is about LEARNED REUSE, not about the architecture's function-space
coverage. The alternative reading — twelve tanh-residual operators span enough of
this function class that any trained basis serves — is refuted: the same
architecture, the same operator budget, the same adaptation budget, and the same
scorer produce zero export margin when the world has no recurrence to learn.

**The reproduction check passes.** The `rho = 1` cells were built by an
independent protocol (the world's own tasks 64-75 of a 76-task world, verified
bitwise-identical in its first 64) and land within 0.13 log units of E1's
corresponding margins (+1.797 / +2.793 / +2.403 there against +1.819 / +2.665 /
+2.360 here), inside the registered 0.15 tolerance.

**Export is graded in recurrence, and grows faster than linearly.** Mean `M_R`
is 0.011 at r ~ 0, 0.550 at r ~ 0.64, and 2.281 at r = 1.0. V1's lifetime
advantage was LINEAR in measured recurrence (`R^2 = 0.935`); export margin is
not — most of it appears between r = 0.64 and r = 1.0. Recorded as an
observation, not a law: three grid points cannot fit a curve, and the two
quantities are different (lifetime cost versus held-out-task export).

Scorekeeping. Ours: `M_R(1.0)` reproduces E1 at +1.8 to +2.8 — **CORRECT**
(+1.82 to +2.67). `M_R(0.0)` near zero — **CORRECT** (+0.03 / -0.02 / +0.03),
including the subsidiary prediction that absolute losses would be high for both
arms and the MARGIN would vanish (S 0.038-0.040, R 0.037-0.039). `M_R(0.9)`
above +1.0 — **WRONG**: it is +0.49 to +0.62. We over-estimated the intermediate
point by treating V1's crossing (r ~ 0.484) as though export would be near its
ceiling just above it; instead r = 0.64 delivers under a quarter of the r = 1.0
margin.

Non-vacuity: prefix property verified in code for all nine cells; `R` and `S`
each reduce their own objective by more than 1% in every cell (0 weak); measured
recurrence read from the project's OWN registered instrument
(`world_functional_reuse.json`) rather than recomputed.

Instrument disclosure: two earlier attempts to compute measured recurrence
inside this scorer were wrong (cross-task centering read 0.000 at `rho = 1`
where the true value is exactly 1.0; removing it read 0.983 at `rho = 0`). Both
were caught in the dry run, before any cell was scored. The scorer now reads the
artifact's registered diagnostic, which is the definition every other rung of
this project uses. The `O` arm's per-task functional assignment uses teacher
information and is a CEILING; the verdict is read from `R`, which uses none.

Licensing: E1's interpretation stands and is now causally anchored. E2 proceeds
as the frozen run order specifies, and inherits a stronger prior: whatever E2
finds about composition will be about learned reusable structure, since the same
measurement says the structure vanishes when recurrence does.

# E2 RESULT (2026-08-26): COMPOSITION HOLDS, INCLUDING IN POSITIONS THE OPERATOR NEVER OCCUPIED

Plan `E2_COMPOSITION_PLAN.md` (Amendment 1, frozen `36d7dbf`). Report
`reports/e2_composition.json`. Support-split worlds 0-2, discrete substrate,
interface E1-P. 12 held-out programs per stratum per world.

    stratum / world      O        O-W      R        R-W      S       O-S     R-S     R-O
    H1 triple-novel  w0 0.00315  0.06418  0.00315  0.05739  0.04630  +2.688  +2.688  +0.000
                     w1 0.00347  0.06158  0.00342  0.05221  0.03283  +2.246  +2.263  -0.017
                     w2 0.00287  0.06608  0.00290  0.05730  0.03809  +2.584  +2.576  +0.008
    H2 pair-novel    w0 0.00283  0.06135  0.00283  0.05643  0.04149  +2.684  +2.684  +0.000
                     w1 0.00373  0.06470  0.00321  0.05455  0.02861  +2.037  +2.186  -0.149
                     w2 0.00374  0.05846  0.00377  0.05332  0.03653  +2.280  +2.270  +0.010
    H3 position-novel w0 0.00313 0.06902  0.00293  0.05882  0.04533  +2.674  +2.739  -0.065
                     w1 0.00403  0.05958  0.00309  0.05247  0.03399  +2.132  +2.398  -0.266
                     w2 0.00393  0.06797  0.00351  0.06128  0.03446  +2.171  +2.284  -0.113

**All three strata pass in all three worlds**, against a registered +0.15 margin
on three separate comparisons (oracle vs scratch, oracle vs an incompatible
world's library, inference vs scratch). Under the terminology contract this is
the rung licensed to use the word COMPOSITION, and it is earned: a
pre-specified split with verified coverage, balance <= 2.0, and >= 21 distinct
contexts per primitive, frozen and written to the artifact before training.

**H3 is the result worth stating on its own.** Three `(primitive, position)`
placements per world were withheld from training ENTIRELY — verified in code
that no training program contains one — and held-out programs then placed those
primitives exactly where they had never occurred. Performance is
INDISTINGUISHABLE from the familiar-position strata: H3's oracle losses
(0.0031 / 0.0040 / 0.0039) sit inside H1's range (0.0029-0.0035), and its
margins (+2.13 to +2.67) match H1's (+2.25 to +2.69).

    A learned operator retains its semantics in a program position it never
    occupied during training.

The withheld positions included the LAST position in all three worlds (0,1),
(1,2) and (3,2) among them, so the prediction that any positional penalty would
grow with depth was testable and found nothing to grow.

**Inference beats the teacher's own route on H3, consistently** (`R - O` =
-0.065 / -0.266 / -0.113, negative in all three worlds, and negative or zero in
7 of 9 cells overall). The functional assignment from E0.1 is not optimal, and
support-only inference finds a better route through the same frozen library.
This sits with E0.1's weak assignment margins (0.002-0.019 here): the mapping
from learned object to teacher primitive is barely determined, yet execution and
composition work regardless. Substitutability without sharp identity is
sufficient for export — which is a substantive finding about what "primitive"
needs to mean, and an argument for the terminology contract's insistence that
identity claims and use claims are different.

Scorekeeping. `H1`/`H2` pass as predicted, with margins within ~0.5 log units of
E1's — CORRECT. On `H3` we registered "near even, slight lean to PASS for the
oracle arm and FAIL-or-marginal for the inference arm", plus a prediction that
any failure would be larger for later positions. The oracle lean was CORRECT;
the inference call was WRONG in the interesting direction — inference is the
BETTER arm on H3, not the weaker one; and the depth prediction was untestable
because there was no failure to grade.

Non-vacuity, all satisfied: withheld placements verified absent from every
training program; every held-out program verified absent from training; every
H3 program verified to contain a withheld placement; `R` and `S` each reduce
their own objective by more than 1% in every cell (0 weak of 108); split
diagnostics (balance 1.69-2.00, contexts 21-26) recorded in the artifact before
the lifetime ran.

Licensing: the branch may now say COMPOSITION for this substrate at exact reuse.
It may NOT yet say synthesis (no compact program variable has been priced, and
route optimization is not synthesis under the contract) and it may not say
anything about depth beyond 3. The frozen run order sends the branch to E8
(length generalization) next.

# E8 REGISTRATION (2026-08-26, review 77): IS THE OPERATOR INTERFACE CLOSED UNDER LENGTH?

Plan `E8_LENGTH_PLAN.md`; frozen before any code. Review 77 accepted in full,
with its two design additions folded in before the plan was written.

Question: E2 established same-depth systematicity; E8 asks whether the interface
is closed under a CHANGE OF LENGTH, separating "a learned length-3 algebra" from
"operators that iterate as a DSL".

Two preconditions VERIFIED before the plan was frozen. (a) `Primitive.random`
does not depend on `program_length`, so depth-2 and depth-4 worlds from the same
seed carry EXACTLY the same six teacher operators (checked elementwise on U, V, b
for all three worlds) — no confound between new depth and new operators. (b) The
executor hardcodes depth (`for step in range(self.task_steps)`), so a
variable-depth executor is required before any length failure is interpretable;
its registered control is BITWISE reproduction of the current executor at depth 3
on the real artifacts, and E8 does not run if that fails.

Conditions, per review 77: **E8a** depth 2, familiar positions; **E8b** depth 4,
whose fourth operation occupies an execution position that never existed in
training — the direct continuation of E2's H3. Arms are E1/E2's unchanged, oracle
computed and reported FIRST. Registered diagnostic: per-step error `e_1..e_D`
against the teacher's intermediate states, so a depth-4 failure can be read as
fourth-step breakage (`e_4` alone large) versus compounding (`e_t` growing).

Registered branches: both depths closed -> the interface extrapolates in length;
depth-4 oracle closed but inference not -> the WRITER is missing, successor
E3/E5, library unchanged; depth-4 oracle not closed while depth 2 is ->
composition is bounded by the executor and the successor is a variable-length
executor, not a better recognizer; depth 2 not closed -> re-examine the
equivalence control before concluding anything.

Predictions. Review 77: depth 2 very likely; depth-4 oracle 60-70%; depth-4
inference lower. Ours: depth 2 passes; **we lean slightly AGAINST depth-4 oracle
passing at the registered margin**, on distributional rather than architectural
grounds — each operator was fitted on states produced by at most two prior
operators, and a fourth step feeds it a state one composition deeper than
anything it saw. We therefore predict `e_t` GROWING with `t` rather than a clean
fourth-step break, and that depth-4 inference partially compensates (as inference
beat the oracle on H3), possibly landing closer to the threshold than the oracle.
If depth 4 passes cleanly, our distributional worry was wrong.

Also recorded from review 77, as a refinement of the terminology contract: an
OPERATIONAL definition of primitive-like status — survives freezing, is
load-bearing, works in unseen combinations, retains meaning across positions, and
needs no operator repair — in place of teacher matching. E1, E1-R and E2 together
satisfy all five for this substrate at exact reuse.

# E8 RESULT (2026-08-26): THE OPERATOR INTERFACE IS CLOSED UNDER VARIABLE-LENGTH COMPOSITION

Plan `E8_LENGTH_PLAN.md` (frozen `473d34b`). Report `reports/e8_length.json`.
Libraries are the E1 discrete artifacts, trained at depth 3 and FROZEN; only the
executor changed, and its registered gate passed first.

**Equivalence control: BITWISE EQUAL AT DEPTH 3 in all three worlds** (16 tasks
each, 48 total). The variable-depth executor reproduces the shipped one
tensor-for-tensor, so a length result here is about the interface and not about
an architecture that hardcoded `D = 3`.

    condition / world      O        O-W      R        R-W      S       O-S     R-S     R-O
    E8a depth 2   w0    0.00333  0.03743  0.00334  0.03416  0.03213  +2.267  +2.265  +0.002
                  w1    0.00151  0.03601  0.00150  0.03005  0.02188  +2.671  +2.679  -0.008
                  w2    0.00283  0.03702  0.00264  0.03375  0.02392  +2.136  +2.204  -0.068
    E8b depth 4   w0    0.00959  0.10409  0.01018  0.08926  0.05924  +1.821  +1.761  +0.059
                  w1    0.00292  0.09518  0.00359  0.07747  0.03731  +2.549  +2.342  +0.208
                  w2    0.00676  0.10892  0.00574  0.08564  0.05830  +2.155  +2.319  -0.164

**LENGTH-CLOSED at both depths**: oracle and inference both clear the registered
+0.15 margin over scratch and over an incompatible world's library, in 3 of 3
worlds, at a length SHORTER and a length LONGER than anything trained. Depth 4
introduces a fourth execution position that never existed during training, and
it is handled.

**The per-step diagnostic answers the question it was designed for.** Oracle
errors after each step, geometric means:

    depth 4, world 0:  0.0013  0.0038  0.0066  0.0096
    depth 4, world 1:  0.0005  0.0011  0.0020  0.0029
    depth 4, world 2:  0.0012  0.0029  0.0049  0.0068

Error GROWS SMOOTHLY with step index — roughly linearly, about 6-7x from step 1
to step 4 — rather than staying flat and breaking at the fourth. So the interface
degrades by COMPOUNDING, not by failing at an unseen execution position. Two
consequences worth stating. First, the error at step 3 OF A FOUR-STEP PROGRAM
(0.0066 / 0.0020 / 0.0049) is comparable to these libraries' own trained-task
loss at the end of a three-step program (0.0073 / 0.0038 / 0.0043), so the
executor is not degraded by being asked to continue. Second, the degradation is
gentle enough that depth-4 final error stays 6-20x better than scratch.

**Inference tracks the oracle at both depths** (`R - O` = +0.002 / -0.008 /
-0.068 at depth 2 and +0.059 / +0.208 / -0.164 at depth 4; inference is better in
one world at each depth). The writer is not the bottleneck here, which is the
outcome review 77 called less likely than the oracle-only branch.

Scorekeeping. Review 77: depth 2 "very likely passes" — CORRECT; depth-4 oracle
"60-70%" — CORRECT and better calibrated than ours; depth-4 inference "lower than
oracle" — PARTIALLY CORRECT (worse in 2 of 3 worlds, better in the third, and it
passes everywhere). Ours: depth 2 passes — CORRECT; **we leaned AGAINST the
depth-4 oracle passing at the registered margin — WRONG**, and it passed by
+1.82 to +2.55. Our reasoning was that each operator had been fitted on states at
most two compositions deep and a fourth step leaves that distribution; the
distributional effect is REAL and visible in `e_t`, but an order of magnitude too
small to matter. **Our prediction about the SHAPE was CORRECT**: `e_t` grows with
`t` rather than breaking at step 4, exactly as registered, and that is the
diagnostic review 77 asked for.

Non-vacuity: equivalence control bitwise in all worlds before any cell scored;
test programs verified of the right depth and absent from training; `R` and `S`
each reduce their own objective by more than 1% in every cell (0 weak of 144);
the frozen libraries are the E1 artifacts, unmodified.

Licensing, under the terminology contract. The branch may now say that these
learned objects behave as a COMPOSITIONAL VOCABULARY CLOSED UNDER LENGTH for this
substrate at exact reuse: they export (E1), the effect is caused by recurrence
(E1-R), they compose systematically including in positions never occupied (E2),
and they compose at lengths never trained (E8). It may still NOT say SYNTHESIS —
no compact program variable has been priced (E3) and no recognizer has been
built (E5). The frozen run order sends the branch to E3 next.

# E3 REGISTRATION + BRANCH ORDER (2026-08-26, review 78)

Review 78 accepted. Official branch order, registered before any of it runs:

    E3 development -> freeze the sealed export confirmation -> open 800-829
    -> close the composition/program claim -> E5 synthesizer -> E6 invention

with the CONFIRMATION BLOCK treated as more important than E3: E1, E1-R, E2 and
E8 are all development results on worlds 0-2 and none has been banked, which by
this project's own standards is overdue. E5 does not start before the block.

E3 plan: `E3_PROGRAM_ECONOMY_PLAN.md`. Three separate claims — E3a syntax
sufficiency (store nothing but `(z_1..z_D)`), E3b two-part economy (the
headline), E3c semantic controls including the GAUGE control where route and
library are permuted CONSISTENTLY and behaviour must be preserved. Every
component charged: `D*(library)` by the established interpolated
rate-distortion instrument at three contribution-relative budgets (1%, 5%, 25%),
plus a length code `log2(D_max)` charged even at constant depth, plus route bits
`D log2(slots)`. Correctness is FUNCTIONAL throughout; no claim reads route
agreement against teacher IDs.

Registered predictions. E3a passes structurally; E3c passes with the gauge
control exact. **E3b: we predict PROGRAM ECONOMY HOLDS DECISIVELY, and we
disagree with review 78's hedge that the library is "the elephant in the room".**
The shared library is 12 operators; the private alternative needs 64 x 3 = 192
operator-equivalents for the same tasks, so sharing wins by ~16x on the only
expensive term while the program corpus costs ~688 bits. The failure condition is
therefore TASK COUNT, not library size — the amortization law one level up, a
vocabulary paying when `H x s_bar > lambda D*(V)` — and we predict the measured
amortization point lands in the low single digits to low tens of tasks. If it
lands above 64, review 78 is right and we are wrong. Least confident:
`D_program` versus `D_continuous`, where both amortize a library and the contest
is 10.75 route bits against the continuous route's `D*`; we expect the program to
win by a much smaller factor than against the private alternative.

Also registered from review 78, to be honoured when the sealed plan is drafted:
seeds 800-829; orthogonal estimands (C1 export, C2a/b/c the three novelty strata
with H3 as flagship, C3 depth-4 closure, C4 drift quantified rather than merely
monotone, C5 optional E3 economy); a HIERARCHICAL verdict so that a drift miss
cannot erase a composition pass; and structural assertions that FAIL THE RUN
rather than warn.

# E3 RESULT (2026-08-26): PROGRAM REPRESENTATION - SUFFICIENT, ECONOMICAL AND CAUSAL

Plan `E3_PROGRAM_ECONOMY_PLAN.md` (Amendments 1-2, frozen `97f2413`). Report
`reports/e3_program_economy.json`. Existing artifacts only.

**E3a SYNTAX SUFFICIENCY: PASSES, and structurally.** Storing nothing but the
literal sequence `(z_1, z_2, z_3)` reproduces the model's predictions BITWISE for
64/64 tasks in all three worlds (max NMSE deviation 0.00e+00). The task solution
really is the discrete syntax; there is no residual task state doing hidden work.

**E3b PROGRAM ECONOMY: PASSES on the primary clause, 3/3 worlds.** All rates are
BEHAVIOURAL (Amendment 1): the smallest interpolated bits/scalar whose COMPOSED
task NMSE stays within 10% of the unquantized model, so every representation is
behaviour-preserving by construction of its own rate.

    world 0: rate 4.99 b/scalar | library  15877 + programs 880 =  16758   private  236214   amortize@4.3 tasks
    world 1: rate 5.00 b/scalar | library  15889 + programs 880 =  16769   private  257687   amortize@4.0 tasks
    world 2: rate 4.70 b/scalar | library  14962 + programs 880 =  15842   private  234075   amortize@4.1 tasks

A program costs **10.75 bits of route plus a 3-bit length code = 13.75 bits per
task**; the whole 64-task corpus is 880 bits against a library of ~15-16k. The
private alternative — each task keeping its own copies of the operators it uses,
each compressed until THAT TASK degrades by the tolerance — costs 234-258k bits.
The shared vocabulary plus explicit programs is **14-15x cheaper**, and the
measured amortization point is **4.0-4.3 tasks**.

Note the private operators DO compress harder in two of three worlds (4.60-5.06
b/scalar against the shared 4.70-5.00): a private operator serves one
distribution and a shared one serves all of them, which is the V4R mechanism and
the real risk this comparison was built to expose. It is simply swamped by the
count: 12 shared operators against 64 x 3 = 192 private ones.

**E3b SECONDARY (versus a continuous-route learner): FAILS, and the failure is
informative.**

    world 0: continuous artifact ABSENT (not substituted)
    world 1: D_continuous  14775 = library  10852 + routes  3923   (8 slots, 2.55 b/scalar routes)
    world 2: D_continuous  14579 = library  10229 + routes  4349   (8 slots, 2.83 b/scalar routes)

The discrete program wins the ROUTE term decisively — 13.75 bits per task against
61.3 (3923/64), a **4.5x** advantage for explicit syntax — and loses the total
because the continuous artifact's library is smaller: **8 slots against 12**, for
the same six teacher primitives. That is a capacity choice made when those
lifetimes were configured, not a property of program representation, so the
aggregate comparison is CONFOUNDED BY SLOT COUNT and is reported as failing
exactly as registered rather than reinterpreted. A matched-slot continuous
comparison is the obvious successor and is not run here.

**E3c SEMANTIC CONTROLS: PASSES, 3/3 worlds.** The GAUGE control — library and
route permuted CONSISTENTLY — preserves predictions BITWISE for 64/64 tasks in
every world. The three wrong-code controls, all at identical bit cost, collapse:
wrong route +1.94 to +2.13 log units, shuffled library +1.80 to +2.16, wrong
depth +1.72 to +2.12. So the symbol names are arbitrary while the
syntax-semantics relation is causal - which is exactly what the tiny
teacher-assignment margins (0.001-0.019) demanded be shown separately.

Scorekeeping. E3a and E3c as predicted. **E3b primary: OURS CORRECT and review
78's hedge WRONG** - we predicted a decisive win with the binding variable being
TASK COUNT rather than library size, and registered that the amortization point
should land "in the low single digits to low tens of tasks". Measured: 4.0-4.3.
Review 78 called `D*(L) ~ 25k bits` "the elephant in the room"; at 64 tasks the
elephant amortizes in four. **E3b secondary: BOTH WRONG in the same direction** -
we expected the program to beat the continuous representation by a smaller
margin; it loses outright, on a slot-count difference neither of us anticipated.

Licensing. With E1 (export), E1-R (recurrence-caused), E2 (composition including
position-novel), E8 (length closure) and now E3, the five conditions review 78
set for the word PROGRAM are met for this substrate at exact reuse: compositional
semantics, a frozen vocabulary, compact explicit syntax, syntax that causally
determines computation, and no operator repair. The branch may say LEARNED
PROGRAMS OVER A LEARNED COMPUTATIONAL VOCABULARY. It may NOT say SYNTHESIS, which
is E5.

All of this remains DEVELOPMENT evidence on worlds 0-2. The registered next step
is unchanged and is now overdue: freeze the sealed export confirmation and open
seeds 800-829.

# EXPORT CONFIRMATION REGISTRATION (2026-08-26, review 79): SEEDS 800-829

Plan `EXPORT_CONFIRMATION_PLAN.md`, frozen before any world in the band is
generated. **Band verified untouched mechanically** before drafting: no artifact
carries a world seed in 800-829 and no report references it.

Thesis: at exact reuse, lifetime learning produces a compact learned PROGRAM
LANGUAGE whose frozen operations systematically compose on unseen programs and
unseen execution lengths. Components preregistered independently, with a
HIERARCHICAL verdict (PROGRAM / EXPORT / COMPOSITION / LENGTH-CLOSED COMPOSITION
/ FULL BLOCK / DRIFT MECHANISM) so one mechanistic miss cannot erase a direct
behavioural result.

Protocol: ONE support-split discrete lifetime per world carries every estimand -
trained tasks for C1, the three held-out strata for C2/C3, depth-2/4 programs for
C4/C5. Consequence recorded because it changes provenance: C2's interval derives
from E2's H1 stratum (support-split) rather than E1's opportunistic holdouts.
8 programs per stratum/condition, 30 worlds; the replication unit is the world.

Intervals derived MECHANICALLY from the development statistics, not from
illustrative numbers:

    C1a  bitwise 64/64 in >= 28/30                  (dev 64/64, 3/3)
    C1b  mean Q_D >= 2.0, >= 28/30 above 1.5        (dev 2.646/2.732/2.693)
         mean N* in [3.0, 6.0]                      (dev 4.32/3.96/4.11)
    C1c  gauge bitwise >= 28/30; each wrong-code
         control >= 1.0 log unit in >= 28/30        (dev bitwise 3/3; 1.72-2.16)
    C2   mean G_export in [0.75, 1.15]; and
         log L_S - log L_O >= 1.5 in >= 28/30       (dev 0.983-1.039; 2.04-2.79)
    C3   per stratum mean >= 1.5, >= 28/30 above 1.0
         H1 dev 2.688/2.263/2.576, H2 2.684/2.186/2.270,
         H3 (flagship) 2.739/2.398/2.284
    C4   mean Delta_D4 >= 1.25, >= 28/30 above 0.75 (dev 1.761/2.342/2.319)
         depth 2 mean >= 1.5 (secondary)            (dev 2.204/2.265/2.679)
    C5   mean b in [0.3, 0.9]; mean q in [0.45, 0.95];
         q < 1.5 in >= 28/30                        (dev b 0.661/0.566/0.566,
                                                     q 0.611/0.745/0.687)

**A correction to review 79's own proposal, made before freezing.** It suggested
predicting `q ~ 1` on the grounds that the fourth call should behave like another
ordinary application. Our development data says otherwise: the step ratios
DECELERATE (2.99 -> 1.74 -> 1.45 in world 0), giving `q` = 0.611 / 0.745 / 0.687.
Registering `q ~ 1` would have preregistered a prediction our own data
contradicts, so the interval is [0.45, 0.95] and the registered statement is that
depth-4 drift is a DECELERATING continuation rather than a novel-position
discontinuity.

Registered predictions. Ours: C1a, C1c and C3 pass comfortably (development
effects are 10-20x the registered floors); C1b passes near Q_D 2.65 and N* 4;
C2 passes. **C4 is the estimand we would bet against most readily** - its
development minimum (1.761) sits closest to its floor and depth 4 is the only
condition asking the vocabulary to leave the distribution it was fitted on.
**C5's q is where a sealed miss would be most informative**: if q returns near 1,
the deceleration we measured was a three-world accident.

Review 79's: expects replication, with shrinkage rather than absence as the risk.

E5 does not start before this block closes.

# SEALED EXPORT CONFIRMATION RESULT (2026-08-27): FULL PROGRAM-LANGUAGE BLOCK CONFIRMED

Plan `EXPORT_CONFIRMATION_PLAN.md`, frozen and hashed at `4b1f8cd` BEFORE any
world in the band existed. Seeds 800-829, thirty support-split discrete
lifetimes, one per world, all 30 completed. Report
`reports/export_confirmation.json`. Verdict read once, from the frozen table.

    VERDICT
      PROGRAM CONFIRMED                       yes   (C1a + C1b + C1c)
      EXPORT CONFIRMED                        yes   (C2)
      COMPOSITION CONFIRMED                   yes   (C2 + C3)
      LENGTH-CLOSED COMPOSITION CONFIRMED     yes   (C2 + C3 + C4)
      FULL PROGRAM-LANGUAGE BLOCK CONFIRMED   yes   (C1-C4)
      DRIFT MECHANISM CONFIRMED               yes   (C5)

    estimand                     sealed (n=30)              registered        development
    C1a syntax sufficiency       bitwise 64/64, 30/30       >= 28/30          64/64, 3/3
    C1b Q_D                      2.714  [2.667, 2.771]      mean >= 2.0       2.646-2.732
    C1b N*                       4.024  [3.800, 4.225]      mean in [3, 6]    3.96-4.32
    C1c gauge                    bitwise 64/64, 30/30       >= 28/30          bitwise 3/3
    C1c collapse (route/lib/dep) 2.460 / 2.470 / 2.162      each >= 1.0       1.72-2.16
                                 mins 1.995 / 1.817 / 1.749
    C2  G_export                 1.0049 [0.998, 1.044]      mean in [.75,1.15] 0.983-1.039
    C2  oracle leg               2.379  min 1.836           >= 1.5, 28/30     2.04-2.79
    C3  H1 / H2 / H3             2.411 / 2.457 / 2.406      each mean >= 1.5  2.19-2.74
                                 mins 1.933 / 1.771 / 1.670; 0 worlds below 1.0
    C4  depth 4                  2.286  min 1.796           mean >= 1.25      1.761-2.342
                                 0 worlds below 0.75
    C4  depth 2 (secondary)      2.452                      mean >= 1.5       2.204-2.679
    C5  b                        0.581  [0.470, 0.782]      in [0.3, 0.9]     0.566-0.661
    C5  q                        0.785  [0.574, 1.107]      in [0.45, 0.95]   0.611-0.745
                                 2 worlds >= 0.95; 0 worlds >= 1.5

Every clause passed with no world below any per-world floor, no fatal structural
assertion, and no arm failing its adaptation check across 30 worlds.

**What is now confirmed, in the terminology the contract permits.** At exact
reuse, on this synthetic operator substrate, lifetime learning produces a
COMPACT LEARNED PROGRAM LANGUAGE: a frozen vocabulary whose operations execute
programs the lifetime never trained on (C2), compose systematically on unseen
triples, unseen adjacent pairs and — the flagship — in POSITIONS THE OPERATOR
NEVER OCCUPIED (C3), remain executable at program lengths never trained
including a fourth execution position that did not exist (C4); whose task
solutions are literally the discrete sequence, bitwise (C1a), 14-15x cheaper
than private coding and amortizing after four tasks (C1b), with symbol names
arbitrary but the syntax-semantics binding causal (C1c). SYNTHESIS remains
unclaimed: E5 has not run.

**Scorekeeping.**
- **C4 was the estimand we said we would bet against most readily.** WRONG:
  2.286 mean, min 1.796, zero worlds below the floor - and above its own
  development minimum. Our stated reason (operators fitted on states at most two
  compositions deep) has now been wrong twice in the same direction, at E8 and
  here. Recorded as a standing calibration error: we over-weight distributional
  arguments against this substrate.
- **C5's q was where we said a miss would be most informative.** It did not
  miss, but it moved: development 0.681 -> sealed 0.785, with two worlds above
  0.95 and a maximum of 1.107. The deceleration is real and slightly weaker than
  three development worlds implied.
- **The mechanical derivation of C5's interval is what made it pass.** Review 79
  proposed registering `q ~ 1` on the reasoning that the fourth call behaves like
  an ordinary application. We refused, derived [0.45, 0.95] from development, and
  recorded the disagreement. Sealed `q` is 0.785 - which a `q ~ 1` registration
  (any interval centred there) would have MISSED. Deriving intervals from the
  data rather than from a plausible story is not a formality; here it decided a
  clause.
- Review 79 expected replication with SHRINKAGE as the risk. Shrinkage was
  minimal: composition means moved from 2.19-2.74 (development) to 2.41-2.46
  (sealed), and the MDL quantities barely moved at all - `Q_D` spans 0.104 across
  thirty untouched worlds and `N*` sits at 4.02 +/- 0.21.

**Instrument disclosures, all made before the band opened.** Two defects were
caught by dry-running the sealed scorer against development artifacts: a WORLD
MISMATCH in which `World.generate` produced the same 64 opaque task IDs but
different programs (63/64), silently pairing a support-split model's routes with
another world's targets; and a PROCESS-DEPENDENT SEED derived from Python's
built-in `hash()`, which gave different values in different processes. Both were
fixed, the first with a fatal assertion that the reconstructed programs equal the
recorded split. The frozen plan was not changed.

**Licensing.** The export branch is banked. E5 (the program recognizer /
synthesizer) is now unblocked, as is E6. No development verdict is altered by
this block; it confirms them.

# E5 REGISTRATION (2026-08-27, review 80): CAN THE LEARNER WRITE THE PROGRAMS?

Plan `E5_SYNTHESIZER_PLAN.md`, frozen before any code. Development worlds 0-2;
the sealed band 800-829 is spent and is NOT reused as development data.

The sealed block closed `C_express`. E5 measures `C_find` and `C_amortize`:
can learned inference recover FUNCTIONALLY GOOD programs at substantially less
search or sample cost than optimization? Correctness is functional throughout -
E0.1's assignment margins and E2/E8's `R < O` cells established that the learned
vocabulary uses a different gauge, so exact-route recovery is a diagnostic and
gates nothing.

Arms: **O** oracle, **ENUM** exhaustive search over all `slots^D` programs,
**OPT** the sealed route optimization, **REC** a deep-sets recognizer
`q_phi(p | D_support)` with top-`k` re-ranking (`k in {1, 5, 25}`), **S**
scratch. The recognizer trains only on a world's 64 training programs against
each task's OWN argmax route (its program under E3's syntax sufficiency), never
a teacher label, with the library frozen.

Registered outcomes: SYNTHESIS DEMONSTRATED requires BOTH an oracle gap <= 0.15
AND `C_find(REC) <= 0.1 x C_find(best of ENUM, OPT)` in executions AND
device-seconds; otherwise AMORTIZATION WITHOUT QUALITY, QUALITY WITHOUT
AMORTIZATION, or NO SYNTHESIS. The word SYNTHESIS is licensed by the first only.

Registered predictions. Ours: **`ENUM` beats `OPT`** - 1,728 forward executions
against 2,000 forward+backward passes - so the meaningful comparison is `REC`
versus `ENUM`, registered now rather than after seeing it. **Modal outcome:
QUALITY WITHOUT AMORTIZATION.** A recognizer should reach the oracle gap easily
(the target is three integers and E3 showed the behaviour-to-program mapping is
well determined), but beating 1,728 cheap executions by 10x once training is
charged is a different matter. Registered consequence if that is what happens:
at `D = 3, slots = 12` this domain is too small for amortized synthesis to pay,
and the correct successor is a DEEPER or WIDER program space where enumeration
is infeasible - not a better recognizer.

Review 80's: expects `C_find(REC) << C_find(OPT)` with `L_REC ~ L_O`, licensing
learned program synthesis.

Also registered from review 80, as standing rules now in `AGENTS.md`: do not
infer a binding generalization barrier from unseen internal-state distributions;
and register the region the data supports rather than the story that sounds
clean. E6's successor law is recorded as `H s_desc + beta H s_search > D*(A)` -
a primitive may now pay through description AND search savings.

# E5 (2026-08-27): the writer. VERDICT AMORTIZATION WITHOUT QUALITY at BOTH settings

Plan `E5_SYNTHESIZER_PLAN.md` frozen at `72a4ae6` (Amendments 1-2). Development
worlds 0-2. Report `reports/e5_synthesizer.json`. Decision rules registered
before any code; outcome computed per setting by the scorer.

    D = 3 (space 1,728, ENUM feasible)     AMORTIZATION WITHOUT QUALITY
    D = 6 (space 2,985,984, ENUM infeasible) AMORTIZATION WITHOUT QUALITY

The word SYNTHESIS is NOT licensed. `EXPORT_BRANCH_PROGRAM.md`'s five layers
stand where the sealed block left them: representation reuse, exportable
computation, composition and length closure are banked; PROGRAM WRITING is not.

## D = 6 eligibility gate: PASSED 3/3

Registered: the oracle must beat scratch by >= 0.75 log units in >= 2 of 3
worlds, or `D = 6` is reported UNINTERPRETABLE. Measured 2.02 / 2.16 / 2.04.
Depth 6 is a real test of the writer, not a degraded executor.

## Scorekeeping against the registered predictions

**Ours, prediction 1 (ENUM beats OPT on cost "at least ~2x, probably more", and
matches or beats it on quality): CONFIRMED, magnitude badly underestimated.**
ENUM was 26-28x cheaper in device-seconds (0.298-0.354 s against 7.66-9.94 s)
and better in 2/3 worlds on quality (0.00424 vs 0.00516; 0.00350 vs 0.00422;
tie in w1). The registered reframing follows: the opponent a writer must beat
at `D = 3` is exhaustive search, not gradient search.

**Ours, prediction 2 (QUALITY WITHOUT AMORTIZATION as the modal `D = 3`
outcome): WRONG, and exactly inverted.** We predicted the recognizer would
reach the oracle gap easily and fail on cost. It failed on quality (best gap
+0.32 against <= 0.15) and passed the cost clause with enormous room.

**Ours, prediction 3 (`OPT` degrades at `D = 6` because gradients now search a
2.99M-point space): WRONG, 3/3.** OPT MATCHED OR BEAT THE ORACLE at depth 6
(-0.21 / -0.04 / -0.15) having sat at parity at depth 3 (+0.00 / +0.05 / -0.18).
Route optimization does not degrade with program-space size in this domain.

**Ours, prediction 4 (SYNTHESIS near even at `D = 6`): WRONG.** The quality
clause failed by more at `D = 6` (+0.42 / +1.51 / +0.31) than at `D = 3`.

**Review 80's (`C_find(REC) << C_find(OPT)` with `L_REC ~ L_O`, which would
license learned program synthesis): HALF CONFIRMED.** The cost half is true by
orders of magnitude; the quality half is false in every cell.

**Amendment 1's drift-law forecast: SUCCESSFUL, the first out-of-sample use.**
Predicted `e_6 ~ e_4 exp(2b) ~ 0.019` from the sealed `b = 0.581` before any
`D = 6` cell ran; measured 0.01752 / 0.00962 / 0.01430. World 0 within 8%; all
three at or below the forecast, so the forecast was conservative in the safe
direction. Review 78 proposed the drift curve as a computable compositional
horizon and it now has one confirmed predictive use.

## The finding the registered cost clause does not capture, reported per the plan

The clause counts `C_find` only, with `C_amortize` "reported beside it". Beside
it, the writer never pays for itself:

    setting  C_amortize (s/task)   the search it replaces (s/task)
    D = 3    7.6 / 9.1 / 11.6      ENUM 0.30-0.35
    D = 6    33.7 / 36.7 / 34.0    OPT  16.2-17.8

In 6 of 6 cells, TRAINING the writer costs more than simply searching, and at
`D = 6` it costs about twice as much. So the honest one-line reading is not
"cheap but worse" - it is that at this scale amortization is a net loss even
before quality is considered, and the recognizer is second-best at a premium.

## Non-vacuity, disclosed: one cell FAILS its registered check

The plan required the recognizer's training loss to decrease materially before
its numbers may be read. `D = 6` world 1 does NOT: 14.917 -> 14.706 against a
uniform-prior loss of 14.909. Its REC arm is therefore not a writer at all but
"draw k programs from a near-uniform prior, keep the best on support", and its
+1.90/+1.62/+1.51 gaps are labeled vacuous rather than counted as evidence about
recognizers. The scorer did not enforce this clause; it is enforced here by
hand. THE VERDICT IS UNCHANGED: the rule is 2 of 3, and worlds 0 and 2 trained
(14.93 -> 0.00 and 14.94 -> 0.25) and fail the quality clause on their own.

## Secondary, never decisive

Exact-route agreement with the teacher was 0.00-0.33 at `D = 3` and 0.00 in all
three `D = 6` worlds, while the same recognizers' functional NMSE improved
monotonically in `k`. Consistent with E0.1/E2/E8: the learned vocabulary uses a
different gauge, and route agreement remains uninformative about use.

# Review 81 (2026-08-27): terminology correction, and a claim we were understating

Filed `reviews/reviewer-feedback-81.txt`, indexed. Read after E5. No E5 number
changes; this entry records a TERMINOLOGY split and one correction to the
reviewer, both registered before E6 is drafted.

## The split, adopted

    SEARCH-BASED PROGRAM SYNTHESIS   demonstrated
    AMORTIZED PROGRAM WRITER          not demonstrated

`EXPORT_BRANCH_PROGRAM.md`'s contract said route optimization "is not SYNTHESIS
until a compact program variable exists or the search space is explicitly
program-structured". Both conditions are now MET and have been since E3: the
program variable is a length-`D` vector of integer slot indices, the search
space IS that variable, and E3 confirmed bitwise in 30/30 sealed worlds that a
task's solution IS the discrete sequence. The clause was written when neither
held; it is satisfied now, so the word is earned for the SEARCH sense only.

Verified on the artifact rather than accepted from the narrative:
`DiscreteLibraryLearner._coefficients` returns a ONE-HOT route whenever the
module is not in training mode, and `adapt_cell` calls `.eval()` before scoring.
Every OPT and R number this project has reported is therefore the execution of a
hard discrete program; the relaxation is the search medium, never the answer.

## Correction to review 81: this is SEALED, not developmental

Review 81 labels search-based synthesis "demonstrated developmentally". That
understates it. The sealed block's **R** arm IS this procedure -
`EXPORT_CONFIRMATION_PLAN.md` line 43, routes "inferred from 128 support
examples, 2,000 Adam steps at lr 0.01" - and it confirmed on unseen triples,
unseen adjacent pairs, unseen operator POSITIONS and unseen program lengths
across seeds 800-829. So the licensed sentence is:

> On 30 sealed worlds, gradient search over a frozen learned vocabulary
> recovered discrete programs that solved structurally novel tasks, including
> programs using operators in positions those operators never occupied.

This is a REINTERPRETATION of confirmed sealed clauses under corrected
terminology, not a new claim and not new evidence: no threshold, estimand or
artifact changes, and nothing here re-opens seeds 800-829. Registered as such so
that the upgrade is dated and traceable rather than appearing silently in prose.

## What is NOT licensed by it

Findability was measured only at `slots = 12`, `D in {3, 6}`, on this operator
family, with 128 support examples and a fixed 2,000-step budget. "Smoothly
searchable" is a hypothesis about the language, not a measured property, until
`C_find(D, K)` is swept - which is why review 81's search-scaling audit is
registered below as a real rung rather than an aside.

## Registered before E6 is drafted

- **E6 must not assume search savings.** The law
  `H s_desc + beta H s_search > D*(A)` is SUSPENDED as a single equation.
  `V_desc`, `V_find` and `V_exec` are measured independently; any combination
  requires a bits-per-second exchange rate frozen in advance. This is the V5.1
  lesson restated (measure what an intervention did to COST and to UTILITY
  separately) plus a new one: never let a free coefficient carry a verdict.
- **E6 becomes MACRO invention over the language**, not neural-primitive birth,
  with the exact crossover `H* = D(M) / s_desc` as its registered estimand.
- **Registered prediction, ours, before E6 exists.** Given E5, we predict
  `Delta D > 0` with `Delta C_find ~ 0` - the macro will pay in description and
  NOT in search, because search is already cheap. We further predict macros will
  not make search materially HARDER (no crossing of the description-optimal and
  search-optimal languages) at reachable `D`; we register that second half at
  low confidence and would treat its falsification as the more interesting
  outcome.
- **Registered prediction, ours, for the search-scaling audit.** `D_search` does
  NOT bind before `D_execute`. From the sealed drift `b = 0.581`, execution
  error grows as `e_D ~ e_4 exp(b(D-4))`, reaching the scratch band (~0.05-0.13)
  near `D ~ 8-9`; we predict route optimization still matches or beats the
  oracle at every depth where the executor remains eligible. If true, the
  binding horizon of this substrate is EXECUTION FIDELITY, not search - which
  reverses the assumption the project has carried since V1.

# CORRECTION to E5 (2026-08-27, found while building E5.1): the S arm was mislabeled

Found by reading E5's arm construction against E1's and E8's before reusing it.
E1 and E8 build the scratch arm with `scratch_model(config, "discrete", 7717)`,
a genuinely fresh model. **E5 built it with `copy.deepcopy(model)`** - the
TRAINED library, then unfroze it. That is not scratch; it is E1's FINE-TUNE
arm (`F`) wearing the label `S`. The error is in
`src/row/experiments/audit_e5_synthesizer.py` and affects only E5.

## Measured impact, rather than asserted

Three depth-6 world-0 programs, both constructions, same tasks and budget:

    program              E5 "S" (trained start)   true scratch
    (4,5,2,4,5,1)              0.10859               0.10047
    (1,0,3,1,4,0)              0.09791               0.10035
    (1,3,1,1,1,0)              0.18562               0.10188
    geomean                    0.1268                0.1009

True scratch is BETTER by ~0.23 log units, so the corrected `D = 6` eligibility
margins are approximately 1.79 / 1.93 / 1.81 rather than the reported
2.02 / 2.16 / 2.04 - still far above the registered `>= 0.75`, in 3/3 worlds.

## What changes, and what does not

**The E5 verdict is untouched.** `AMORTIZATION WITHOUT QUALITY` rests on the
oracle gap (quality) and on ENUM/OPT costs (cost); the `S` column enters only
the `D = 6` eligibility gate, which still passes 3/3 under the correction. No
recorded E5 decision flips, and the report is left in place rather than rebuilt,
with this entry as its correction of record.

**What must be re-read:** every E5 `S` number is a fine-tuning number, and the
E5 report's `S` column is NOT comparable to E1's or E8's `S` column. Anywhere
E5's scratch band was quoted (including the `e_D` forecast's "scratch band of
~0.05-0.13"), the correct band is the true-scratch one.

**E5.1 uses `scratch_model`,** matching E1 and E8, and its plan says so
explicitly in the arm table.

## The transferable rule

REUSING AN ARM IS REUSING A CONSTRUCTION, NOT A NAME. Three modules in this
branch have an arm called `S`; two build it fresh and one copied the trained
model, and the label made them look identical at every call site. Before reusing
a baseline across modules, diff how it is CONSTRUCTED, not what it is called -
the same discipline already required for scorer CLI arguments (H35) and for
protocol fingerprints.

Incidental observation worth keeping: fine-tuning a trained 12-slot library on a
single task's 128 support examples was WORSE than training from scratch in 1 of
3 probes (0.186 against 0.102), so a trained start is not automatically an
advantage at this scale.
