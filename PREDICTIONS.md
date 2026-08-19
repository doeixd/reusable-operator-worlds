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
