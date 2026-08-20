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
