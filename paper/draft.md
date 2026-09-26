# When Does Abstraction Pay? Measuring the Value of Reusable Computation in Neural Learners

*Draft v0.9.1 — extended with the FOURTH sealed block (seeds 400-429;
section 7.6): a preregistered negative that a learned library does not
need lifecycle machinery at this scale, plus a retention law that is
birth run forwards. The third block (seeds 300-329, section 7.5) is
promotion. Earlier blocks contribute the recurrence law and its
parameter replication (seeds 100-129, 200-229) and the read/write
dissociation. Development and sealed results are labeled throughout
and the four sealed blocks are distinguished by seed range; all
numbers trace to fingerprint-validated artifacts in the public
repository, with the first two blocks archived as release
v2.0-confirmation. A later V5 structural-audit correction is recorded in
Sections 9 and 10.1 but is outside the four-block evidentiary scope of
this draft. Revision history in PROGRESS.md.*

## Abstract

We ask when a neural learner should represent computation as reusable
structure rather than task-specific parameters. We introduce Reusable
Operator Worlds (ROW), a benchmark in which the amount of latent
computational recurrence across tasks is continuously controllable and
directly measurable, and the learner is scored by cumulative prequential
(predict-before-update) cost over a 64-task lifetime. In 30 sealed
confirmatory worlds, run under a protocol pre-specified in the public
repository before any sealed world was generated, a reusable operator
basis consistently beats a compute-matched dense learner under high
recurrence and consistently loses under low recurrence (30/30 worlds on
all three pre-specified outcomes; Holm-adjusted p <= 5.6e-9). The paired
advantage is approximately linear in measured functional recurrence
(R^2 = 0.935), crossing zero near recurrence r = 0.50. A SECOND sealed
block of 30 worlds, run against parameter intervals frozen in advance,
replicates the law's coefficients — slope 6,194 nats per unit recurrence
(interval 4,000-7,500), crossing 0.45-0.48 (intervals 0.40-0.60 and
0.42-0.58), R^2 = 0.926 — upgrading the claim from a reproduced
phenomenon to a measured law with replicated parameters. On the same
sealed worlds, a learner permitted per-task low-rank corrections to a
shared parent allocates specialization inversely to each primitive's
ground-truth recurrence in 30/30 mixed-recurrence worlds (sign test
p = 1.9e-9) and beats the better fixed architecture everywhere on
prediction cost — while losing everywhere under literal two-part
description-length accounting, a replicated dissociation showing that a
learner can READ the economics of sharing almost perfectly before it can
WRITE them compactly. A THIRD sealed block of 30 worlds closes that gap in
a constructed setting: when a learner's operator library is saturated and
the environment then introduces recurring computation the library cannot
express, an explicit promotion operator creates a new shared object,
migrates the repeated task-local computation into it, and reduces total
retained description length by 63.3% while IMPROVING prediction (+1,174
nats) and making held-out related tasks cheaper to acquire (+0.0031
32-shot NMSE) — 30/30 on all four, with parameters matching development to
within 0.7%. This is the first learner in the program to win prediction and
description length simultaneously, and it does so with capacity it created
rather than capacity it was given. A FOURTH sealed block (seeds 400-429)
asks whether that library then needs lifecycle machinery: it does not,
at this scale. Local private compression beats factorization in 30/30
worlds; the one structural edit that pays is retention, and it pays
when expected remaining reuse repays the abstraction's code cost
(crossing 18.0 returning tasks against a development-derived prediction
of 17.1). The result is
conditional on representational alignment, and substantially so: when the
learner's operator family matches the environment's (residual tanh), the
exact-reuse advantage is large; changing only the learner's activation
family imposes an approximately additive penalty that consumes the
attainable return, leaving parity at exact reuse. The supportable claim
is therefore:
**when a learner's representational vocabulary can efficiently express
the environment's recurring computation, the value of using that
vocabulary is linear in measured recurrence.** Development-stage
mechanistic analyses further show that lower lifetime learning cost
appears before identifiable, recomposable primitives do: **statistical
reuse and structural abstraction are distinct phenomena** — and the
representational form tracks recurrence through three regimes:
task-specific solutions at weak recurrence, a continuous operator
manifold at partial recurrence (where a slotless hypernetwork beats
explicit operator slots), and crystallized, recomposable primitives only
near exact recurrence. The transfer
advantage is acquired over the lifetime rather than present at
initialization, and the effect survives changes in task order, replay
budget, initialization, task-code capacity, batch size, and quantization.
These results provide a controlled measurement of the economics of neural
abstraction: when shared computation pays, on what its value depends, and
what kind of reuse a learner actually acquires. ROW is deliberately small
(the generative programs are known exactly), and no claim in this paper
concerns scale.

## 1. Introduction

Whether neural learners should share computation across tasks is usually
argued by benchmark anecdote: modular architectures win some transfer
suites and lose others; negative transfer appears and disappears with
tuning. The underlying question — under what conditions does a reusable
internal representation reduce the cost of future learning? — is rarely
posed where the true amount of reusable structure is known, adjustable,
and measurable.

**The economic framing.** Let C_R denote the additional representational
and inference cost a learner incurs by forcing tasks through shared
computation, and let S(r) denote the expected reduction in future
prediction cost available at functional recurrence r. Sharing is favorable
when S(r) > C_R. Empirically we measure the paired lifetime difference

    Delta(r) = L_Dense(r) - L_Reuse(r)

on identical worlds and find, to good approximation, Delta(r) = a·r + b
with a > 0 and b < 0: a linear return on recurrence offset by an
empirical sharing penalty. The intercept b is an empirical quantity, not
a separately identified representational cost — it aggregates whatever
imposing the reusable substrate costs under the present protocol,
including any inductive-bias mismatch, task-code inference difficulty,
and optimization overhead. The zero crossing r* = -b/a is simply where
the expected savings equal that aggregate price. The experiment, in this
framing, asks how Delta changes as ground-truth functional recurrence
changes — an intervention on r, not a comparison at whatever task
relatedness a dataset happens to contain.

**Two kinds of reuse.** We will need a distinction that the results force
on us. Call it *statistical reuse* when shared parameters exploit
overlapping structure to lower online prediction cost — sharing that pays
during the lifetime. Call it *structural abstraction* when the learner has
acquired stable computational objects that can be removed from their
training context and recomposed: identifiable operators that support
transfer to unseen compositions. Our central mechanistic finding is that
these are different phenomena with different onsets: sharing pays
economically well before anything recomposable exists.

**Why a small synthetic world.** ROW hides a library of six randomly
generated neural operators; every task is an opaque length-3 composition;
a knob rho interpolates per-task operators between exact sharing and task
independence, with realized functional recurrence measured on probes. The
learner never sees routes, primitive identities, or rho. Because the
generative programs are known, we can run an oracle positive control,
verify task difficulty is flat over the lifetime, and test whether learned
operators correspond to the hidden ones. The smallness is the instrument:
none of our claims concern scale, and the criterion we test is scale-free.

**Contributions.**
1. A benchmark and validity-control battery for lifetime reuse economics,
   with a development/confirmation seed firewall and a confirmatory
   protocol pre-specified in the public repository before sealed-world
   generation.
2. A confirmed regime map: dense task-specific learning wins when latent
   recurrence is weak; an explicit reusable basis wins when it is strong;
   and the paired effect is linear in measured recurrence (R^2 = 0.935),
   the apparent threshold in configured coordinates being largely
   coordinate distortion. The magnitude of this economics is strongly
   conditioned on representational alignment (a family-mismatch control
   reduces the exact-reuse advantage to approximate parity, via an
   approximately additive penalty), and we state every claim under that
   condition.
3. Parameter replication: a second sealed block of 30 worlds, tested
   against intervals frozen in advance, replicates the law's slope,
   crossing, and linearity — interval misses were pre-committed to count
   as failures even with passing signs, and none occurred.
4. The statistical-reuse / structural-abstraction dissociation, extended
   to a three-regime account of representational form (task-specific ->
   continuous manifold -> crystallized primitives) supported by four
   independent instruments, including a slotless hypernetwork that beats
   explicit operator slots at partial recurrence and loses at exact
   recurrence.
5. Selective sharing, confirmed sealed: a shared-parent learner with
   penalized per-task low-rank corrections allocates specialization
   inversely to per-primitive ground-truth recurrence (30/30 sealed
   mixed worlds) and beats the better fixed architecture on prediction
   cost everywhere — while losing everywhere under two-part
   description-length accounting. Reading the economics is demonstrated;
   writing them compactly is the characterized open problem.
6. A measured resource frontier across five substrate families under
   actual int8 quantization, and a set of instructive failures reported
   in full: two consolidation-gate designs (one firing inversely to
   structure, one barely firing), a within-lifetime amortized compiler
   that loses to plain gradient descent while its self-test confirms it
   learns the program distribution, and the falsification — timestamped
   before the sealed block — of the hypothesis that soft mixtures
   approximate a Bayesian route posterior.
7. One falsified secondary hypothesis and two weakened ones from V1,
   reported alongside.

## 2. Reusable Operator Worlds

**Worlds.** Each primitive is P_k(z) = tanh(z + alpha · U_k tanh(V_k z +
b_k)) with spectral-normalized U, V (state dimension 16, rank 8,
alpha = 0.35). A world samples 6 primitives and 64 unique length-3
programs; task IDs are random tokens; train (128/task) and evaluation
(256/task) sets are fixed at generation. The reuse continuum mixes
per-task perturbations into each primitive with weight sqrt(1 - rho^2),
re-normalized; measured pairwise residual-function correlation on probes
validates the endpoints (~0 at rho = 0, 1 at rho = 1) and serves as the
explanatory coordinate r throughout (the configured-rho-to-r map is
strongly nonlinear).

**Protocol.** For each task in sequence: evaluate zero-shot; stream 128
examples, scoring each before updating (so the measure directly rewards
reductions in the amount of new evidence required for future prediction);
update with one current plus one replay example (batch ablation in
Section 6); evaluate at fixed support counts. Paired models receive
identical worlds, orders, examples, replay draws, and evaluation sets.
Likelihood is Gaussian with fixed sigma = 0.1; a fixed-precision
quantized-target form gives an equivalent coding interpretation.

**Validity controls (all passed before any model comparison).** Per-task
scratch models show no difficulty trend in task index; outputs neither
saturate nor vary pathologically; scrambling the opaque IDs leaves
training bit-exact after relabeling; and a true-route oracle with learned
operator slots shows strong lifetime transfer (late-life zero-shot NMSE
0.0016-0.0019; unseen compositions solved essentially from the library),
establishing the hidden structure is exploitable in principle.

## 3. Learners, hypotheses, and pre-specification

All learners expose the same interface (opaque task ID in, prediction
out; per-task state initialized identically); none receives routes,
primitive identities, or rho.

- **Dense-C**: task embedding into three task-conditioned residual state
  blocks, width chosen to match the continuous model's inference
  multiply-adds (6,144 vs 6,528). Parameter-matched, width-128, and
  24-dimensional-task-code variants serve as capacity controls.
- **Continuous basis**: 8 learned operators of the teacher's residual
  form — with independently initialized, learnable residual scale and
  configurable activation (the family-alignment question is treated
  head-on in Section 6.1) — mixed by per-task, per-step softmax codes.
  Task identity influences computation only through the mixture.
- **Discrete library**: 12 slots, relaxed routing annealed to hard argmax
  evaluation. **Hypernetwork**: task codes generate low-rank operators
  directly — a continuous operator manifold with no explicit slots.
  **Shared-parent + residual**: library operators plus rank-2 penalized
  per-task corrections (Section 7).

Both primary architectures received the same staged learning-rate grid on
development worlds (seeds 0-9); configurations were then frozen. The
confirmatory protocol — worlds 100-129, six rho values, three primary
outcomes, world-level exclusion rule, Holm correction — was committed to
the public repository (CONFIRMATION_PLAN.md) before any sealed world was
generated or inspected; the git history provides the verifiable
timestamp. We describe this as pre-specified sealed confirmation.

## 4. Confirmed result: the economics of recurrence

360 paired lifetimes ran on the sealed worlds with zero failures and zero
exclusions.

**P1 — exact-reuse advantage.** At rho = 1.0, Continuous beats Dense-C in
30/30 worlds; mean paired advantage +3,204 nats (median +3,177; +0.39
nats per online example). Exact sign test p = 1.9e-9.

**P2 — recurrence dependence.** The per-world slope of the paired effect
against measured recurrence is positive in 30/30 worlds; mean +5,715 nats
per unit recurrence; p = 1.9e-9.

**P3 — within-world reversal.** In 30/30 worlds, Dense-C wins at rho = 0
and Continuous wins at rho = 1; p = 1.9e-9. All three survive Holm
correction (adjusted p <= 5.6e-9).

**The dose-response is the result (Figure 2).** We fit the pooled
regression over all 180 world-condition cells,
Delta_{w,rho} = b0 + b1·r_{w,rho} + e_{w,rho}, obtaining

    Delta(r) ~= 5,716·r - 2,625        (R^2 = 0.935, pooled)

versus R^2 = 0.642 against configured rho. The regression is descriptive;
the pre-specified per-world sign and slope tests above remain the
inferential results (each world contributes one slope, so world-level
clustering is respected where it matters), and a mixed-effects
sensitivity analysis appears in the appendix. The apparent "threshold" in
configured coordinates is largely coordinate distortion from the
nonlinear rho-to-r map: the underlying phenomenon is a smooth linear
return on recurrence with a fixed offset — sharing has a price, and
recurrence pays it down at a measurable rate. The zero crossing falls
near r = 0.50 (per-world interpolated crossings: r* = 0.499 +/- 0.050;
configured rho* = 0.835 +/- 0.023). We caution against reading anything
into the round value: r* = -b/a where b aggregates unidentified costs of
this substrate, protocol, likelihood, and lifetime length, so the
crossing location is a property of this experimental economy, not a
universal constant — the truncated-lifetime analysis (appendix) already
shows it drifting with lifetime length before saturating. The sign
pattern is unanimous:
Continuous wins 0/30 worlds at every configured rho <= 0.75 and 30/30 at
rho >= 0.9 — 180/180 cells matching the development-stage prediction. We
use "crossover," never "phase transition"; on the evidence, the crossing
is an ordinary zero of a linear function, which is the more elegant and
more falsifiable claim.

**Parameter replication (second sealed block, seeds 200-229).** The
strongest test we could construct: before generating a further 30 sealed
worlds, we froze quantitative intervals for the law's parameters, with
interval misses pre-committed to count as replication failures even if
all sign tests passed. The second block ran 360 paired lifetimes with
zero exclusions and landed inside every interval: slope 6,194 nats per
unit recurrence (frozen interval 4,000-7,500; first block 5,716); pooled
zero crossing 0.450 (interval 0.40-0.60) and per-world crossing mean
0.483 (interval 0.42-0.58), with all 30 worlds crossing; R^2 = 0.926 in
measured-recurrence coordinates with a +0.262 margin over configured
coordinates (thresholds 0.85 and +0.15); within-world sign reversal
30/30 (p = 1.9e-9). Across the two independent sealed blocks the law's
coefficients agree to within a few percent. We also report, from genuine
32- and 128-task lifetimes at the bracketing recurrence values
(development), that the crossing is stationary across a 4x range of
lifetime lengths — per-task effects are approximately constant — so the
boundary is a representational-bias sign flip rather than an
amortization threshold, with amortization movement confined to very
early lifetime.

## 5. What does the reusable learner actually acquire?

(Development worlds 0-9; labeled development-stage throughout.)

### 5.1 The advantage is absent early and acquired over the lifetime

At lifetime checkpoints we freeze shared parameters and adapt fresh task
codes on held-out unseen compositions. After 8 tasks the architectures
are statistically indistinguishable at 32-shot novel adaptation
(Continuous 0.0228 vs Dense-C 0.0230 mean NMSE; Continuous better in only
4/10 worlds). After 64 tasks Continuous leads in 10/10 worlds (0.00343 vs
0.00645). We also compute forward transfer with an explicit baseline:

    FT_tau = L(fresh learner, same architecture, task tau alone)
             - L(lifetime learner, task tau)

so that FT_tau > 0 means accumulated experience made task tau cheaper
than learning it in isolation. FT is positive for Continuous on 97.5% of
tasks, with a positive slope against task index in 10/10 worlds. The
measurable transfer advantage is absent early and acquired over the
lifetime (our earliest checkpoint is 8 tasks, not initialization); the
architecture supplies the capacity for it, and experience supplies the
advantage.

### 5.2 Statistical reuse is not structural abstraction

Define operationally:

    R_stat   = L_Dense - L_Reuse                (does sharing pay online?)
    R_struct = (frozen-library recomposition gain, operator recovery
                vs an untrained baseline)       (is there an abstraction?)

The central mechanistic finding:

    R_stat > 0   does not imply   R_struct > 0

1. At rho = 0.9, Continuous wins lifetime cost in every paired world but
   does not reliably win frozen-library novel transfer; the transfer
   advantage appears only at rho = 1.0. (The same pattern appears in the
   confirmatory secondaries.)
2. Operator recovery, measured against an untrained-basis baseline of
   0.0087 normalized distance (Figure 4): at rho <= 0.5 trained operators
   sit FARTHER from the shared primitives than untrained ones
   (0.0115-0.0121) — below the crossover, training moves the basis away
   from shared structure; rho = 0.75 matches baseline (0.0083); recovery
   first appears at rho = 0.9 (0.0048) and crystallizes at rho = 1.0
   (0.0017). The location of recovery onset and the location of the
   performance crossover coincide on the recurrence axis. We flag this
   explicitly as a coincidence of locations observed at six recurrence
   values on ten development worlds — suggestive, pre-specified for
   targeted testing in V2, and not a demonstrated causal link.

So a representation can pay economically before it crystallizes into
identifiable, recomposable computational objects. Statistical reuse
begins near r ~ 0.5; under the present learner and benchmark, evidence
for structural abstraction appears only near exact recurrence (whether
approximate reusable abstractions can form under other learners or
richer worlds is exactly the open question this distinction makes
askable). Figure 4 puts both response curves — and operator recovery —
on one recurrence axis. We note that an earlier, more attractive hypothesis — that
recompositional transfer improves before it amortizes — was proposed on
one development world and falsified on replication; the dissociation
above, with the opposite ordering, is what survived.

### 5.3 Three regimes of representational form

Assembling the instruments — paired lifetime losses, the hypernetwork
gap, operator recovery against the untrained baseline, and frozen-
library transfer — yields one account: representational form tracks
recurrence through three regimes. At weak recurrence, task-specific
solutions win. At partial recurrence, the best representation is a
CONTINUOUS OPERATOR MANIFOLD: the slotless hypernetwork closes its gap
to the explicit basis in 3/3 worlds at rho = 0.9 and beats it outright
in 2/3, while operator recovery sits at or below the untrained baseline
— useful sharing without identifiable parts. Only near exact recurrence
do explicit slots win, recovery crystallize, and frozen recomposition
work. Discreteness, on this evidence, is not the substrate of reuse but
an endpoint that emerges where recurrence is exact enough to make named
parts economical.

### 5.4 The mixtures are not beliefs about programs (a pre-sealed
###     falsification)

An attractive mechanism would unify everything: soft mixtures win
because they approximate a Bayesian posterior over discrete routes. We
tested it before the second sealed block existed, and it is false: the
trained mixture weights are uncorrelated with exact route-posterior
marginals over the same basis (mean Spearman -0.03 across worlds 0-2;
chance-level sign rate), and miscalibrated by three orders of magnitude
(mixture entropy ~1.4 nats versus posterior marginal entropy ~0.001).
Meanwhile the exact posterior itself — a deliberately advantaged bound
computed over the frozen learned library — beats the continuous learner
by ~3,900 nats, with its MAP route agreeing with the online learner's
hard routes on only 25/64 tasks while outperforming both. The corrected
picture: the continuous learner solves tasks in the basis's continuous
function space with distributed, compensatory solutions and never does
route inference at all; route-committed representations are better when
found, but gradient descent on mixture codes does not find them. This
falsification is timestamped in the public record ahead of the sealed
data it might otherwise have been suspected of accommodating.

## 6. Alternative explanations

### 6.1 Family alignment (the circularity question)

The teacher and the continuous learner share a functional family
(residual bottleneck blocks), so the strongest objection to this study is
that it measures inductive-bias match rather than reuse. In stages:
removing all parameter coupling (independently initialized, learnable
residual scale) leaves the world-0 advantage essentially unchanged
(+4,446 nats); doubling teacher rank against fixed learner rank preserves
most of it (+3,296); and changing the learner's activation family (GELU
vs tanh) is the control that bites. Running the mismatched learner
across the recurrence range (worlds 0-2) gives paired
Dense-minus-Continuous effects of:

    rho    tanh (mean)      GELU (mean)     mismatch penalty
    0.75   -1,597 (0/3)     -3,632 (0/3)    +2,035
    0.90   +1,298 (3/3)     -1,583 (0/3)    +2,881
    1.00   +3,492 (3/3)     +134   (1/3)    +3,359

The mismatch acts as an approximately additive penalty that grows mildly
with recurrence: the return-on-recurrence slope survives at roughly 74%
of the aligned slope, but the line is shifted down far enough that the
zero crossing moves to the edge of attainable recurrence — at exact
reuse the mismatched learner reaches parity with Dense-C, not advantage.
Two conclusions follow. First, alignment is a cost term, not a gate: the
effect neither vanishes nor stops rising with recurrence under
mismatch. Second, and precisely: **this experiment does not show that
generic neural learners spontaneously discover arbitrary reusable
computations. It shows that when a learner's representational vocabulary
can efficiently express the recurrent structure of its environment, the
economic value of using that vocabulary is predictable from functional
recurrence — and when it cannot, the sharing penalty grows to consume
the attainable return.** In the economic notation Delta = f(r, A, C),
alignment A enters approximately additively at this scale. Whether the
learner can acquire the vocabulary itself is the V2 program.

### 6.2 Capacity, codes, and the manifold alternative

The generic hypernetwork — a continuous operator manifold with no
explicit slots — beats Dense-C but loses to Continuous in 10/10
development worlds (mean margins ~1,791 and ~1,907 nats respectively).
This is a mechanistic narrowing, not just a capacity control: the
advantage is not explained solely by constraining tasks to a
low-dimensional operator manifold — explicit shared operator atoms
provide additional benefit in this setting, yielding the ordering
Dense < continuous task-conditioned operator manifold < explicit
reusable operator basis for lifetime learning under exact reuse.
That ordering is recurrence-dependent, and its inversion is itself
evidence: at rho = 0.9 the hypernetwork's gap to Continuous shrinks in
3/3 worlds and the manifold outright beats the explicit basis in 2/3
(-1,496 and -943 nats; +917 in the third), while losing in all worlds at
rho = 1.0 (+2,820/+1,120/+2,102). The slotless manifold is the better
substrate at partial recurrence; explicit slots pay only where
identifiable primitives exist — an independent-instrument confirmation
of the statistical/structural account in Section 5.2, whose recovery
analysis found crystallized primitives only at exact recurrence.
Task-code dimension is immaterial (Dense-24 vs Dense-32: 46 nats). A
width-128 dense variant does not close the gap.

### 6.3 Nuisance battery

All at 10/10 development worlds (Figure 6): reverse task order +3,456
[CI 3,145-3,791]; no replay +3,129 [2,511-3,743]; canonical replay
+3,698; heavy (1:4) replay +3,350. The no-replay cell matters most: the
advantage does not depend on episodic rehearsal — shared parameters alone
carry prior-task information forward. A second independent initialization
reproduces 10/10 wins on both metrics. Scrambled IDs leave training
bit-exact. Widening the update batch from 2 to 8 shrinks the worlds 0-2
advantage from +3,463 to +2,061 but preserves 3/3 wins. Evaluated int8
quantization changes NMSE by at most 1.4e-4 (mean ~1e-6).

### 6.4 The route-inference bottleneck (discrete substrate)

The discrete library learns the right structure — 92.2% exact recovery of
explained routes, primitive distance 0.00229, 11/12 slots — yet pays a
large online cost. Part is optimization artifact: per-task temperature
annealing (vs global) recovers ~8,800 nats; a ~25k-nat gap to Continuous
remains. Representation learning and program inference are separable
problems, and inference is the discrete learner's bottleneck; soft
mixtures implicitly carry hypothesis uncertainty that hard routing
discards. The route space here is small enough to test this directly:
an exact Bayesian posterior over all 1,728 routes of the frozen final
library — a deliberately advantaged bound, since online learners
trained their libraries concurrently — scores -174,844 prequential on
world 0, beating not only the online discrete learner (by ~28,700 nats)
but Continuous itself (by ~3,900). At this bound the discrete deficit
is inference cost in its entirety, and the mechanism is visible in the
posterior's behavior: its maximum-a-posteriori route agrees with the
online learner's hard routes on only 25/64 tasks while outperforming
both, because several routes are behaviorally near-equivalent and the
posterior averages over exactly the ambiguity that hard commitment
discards. Premature commitment, not discreteness, is the cost. (The
posterior concentrates below 0.1 nat of entropy after a median of 28.5
examples — a number the V2 consolidation program uses directly.)

## 7. Resource economics and adaptive sharing

**The frontier (Figure 5).** Evaluated int8 retention: Discrete 26,208
bits and ~768 inference multiply-adds; Continuous 29,248 / 6,528;
Hypernetwork 33,928 / 7,296; Dense-24 56,448 / 5,376; Dense-C 66,688 /
6,144. These are UPPER BOUNDS at 8 bits per scalar, not measured minimum
description lengths: the frontier was never swept below int8. A later
sweep on the shared-abstraction tensors of the promotion learner found
zero behavioral cost at 6 bits and 127 nats at 4 across an entire
library, implying those tensors are stored at roughly four to eight
times their functional information content. Whether the dense and
hypernetwork parameterizations tolerate the same depth is untested, so
the ordering above is a comparison at a common proxy rather than a
comparison of minimal codes. Retention, online-learning, and execution orderings all disagree:
storage efficiency, learning efficiency, and inference cost are distinct
objectives, and "compute-matched" in this paper means inference-forward
multiply-adds (training compute is not matched at the same ratio; no
claim depends on the distinction).

**A proof-of-concept adaptive substrate.** Define the fixed-architecture
envelope as the better of the two fixed learners at each condition,
L_env(rho) = min{ L_Dense(rho), L_Continuous(rho) } (lower prequential
loss is better throughout). A shared-parent + rank-2-residual learner —
task operators P_k + Delta_task,k under an explicit penalty — beats that
envelope at every intermediate recurrence on worlds 0-2 (L_env minus
L_shared: +9,168 / +7,458 / +3,745 mean nats at rho 0.5 / 0.75 / 0.9;
3/3 worlds on both metrics), degrades gracefully to parity (-246) at
rho = 1.0, and shows the predicted allocation signature: residual
magnitude falls monotonically with recurrence (functional ratio
0.284 -> 0.026; Figure 7). The learner measurably chooses its degree of
sharing. Three points bear on the size of these margins: the residual
learner follows the identical online protocol and update budget; its
extra capacity is entirely per-task (rank-2 residuals per step), which is
exactly what the description-length accounting below prices; and its
shared-parameter count matches the continuous basis. That accounting
reverses the result: the per-task residuals retain ~130,624 bits (9x the
Continuous task state), and under a literal two-part description-length
code — model bits added to the data code at their information value,
1 bit = ln 2 nats, not a tuned regularization weight — it loses to both
fixed architectures in all twelve world-rho cells. The break-even prices
(0.04-0.14 nats per bit) quantify how cheaply persistent storage would
have to be valued for the adaptive model to win. Its purpose here is not
to establish another winner but to demonstrate a precise point:
**adaptively choosing sharing improves predictive economics, but naive
flexibility buys this with excessive description length** — giving the V2
consolidation program an exact target (adaptive sharing plus
compression).

**Selective sharing, confirmed sealed.** The proof of concept graduated
to a pre-registered claim on mixed-recurrence worlds, where each
primitive carries its own reuse level (canonical profile 1.0 / 0.95 /
0.8 / 0.5 / 0.2 / 0.0) and neither fixed architecture can be correct.
Attribution is frozen in advance: each task-step residual is attributed
to the teacher primitive at that position of the hidden program — a
post-hoc ground-truth diagnostic that stays well-defined even where
slot-to-primitive matching fails. Development result: the per-world
Spearman correlation between per-primitive measured recurrence and mean
residual allocation is negative in 10/10 worlds (range -0.43 to -1.00).
Sealed result (seeds 200-229, decision rule and outcomes frozen before
generation): negative in **30/30 worlds** (sign test p = 1.9e-9), with
the envelope win replicating 30/30 (mean +7,192 nats over the better
fixed architecture) and the two-part-code reversal replicating 0/30 —
the reversal was itself a pre-registered expectation, so its
confirmation completes the dissociation: **a learner can read the
economics of sharing almost perfectly, allocating specialization
primitive-by-primitive against ground-truth recurrence it was never
shown, while still lacking any mechanism to encode that allocation
compactly.** Reading is solved; writing is the open problem. Unlike
the headline law's magnitude, this result is not strongly conditioned
on representational alignment: repeating the mixed-world experiment
with a family-mismatched (GELU) shared-residual learner preserves the
allocation signature in 3/3 worlds and the envelope win in 3/3 (margins
within ~15% of aligned), with the mismatch penalty roughly a third of
what mismatch costs the continuous basis — task residuals absorb family
misfit that a pure shared substrate cannot.

**Two more instructive failures.** A wake/sleep consolidation learner
that compiles soft tasks to hard routes was tested under two gate
designs with a pre-registered firing-rate shape (non-decreasing in
recurrence) and a one-re-derivation budget: the relative-quality gate
fired INVERSELY to structure (43% at zero recurrence, 0% at exact reuse
— relative bars are easiest to clear where the baseline is worst, and
single hard routes cannot match compensatory mixtures), and the
re-derived absolute gate satisfied the shape but barely fired (0-8 of 64
tasks). A follow-up analysis corrected our first mechanism reading: the
route posterior is in fact deterministic given the full task record, and
the binding criterion was the absolute quality bar — hard single routes
through a mixture-trained basis rarely substitute for the mixture even
at exact reuse, consistent with Section 5.4's finding that the basis
holds compensatory manifold solutions rather than route-composable
parts. A within-lifetime amortized compiler (a set encoder
warm-starting task codes, trained on the lifetime's own solved tasks)
lost to plain gradient descent in every world at both recurrence
extremes; its pre-registered self-test nonetheless passed — augmenting
compiler training with fantasy tasks sampled from the learner's own
library helps three times more where the library matches the world
(+1,554 vs +512 nats), so the compiler learns the program distribution
and is sample-starved rather than wrong. Both negatives sharpen the same
design lesson the successes suggest: compression and inference decisions
must be based on functional equivalence, not component identity, and
amortization belongs at cross-world scale.

**A characterized negative, briefly.** L0-style presence gating on the
12-slot library, tuned over a two-stage grid with a selection rule frozen
in advance, never produced a compact sufficient library: pruning pressure
is a cliff (nothing, or collapse to one slot), and the one intermediate
regime found (7 active slots; the teacher has 6) fails novel-composition
sufficiency — with the degradation tracking gate pressure, not slot
count. Penalizing structure during acquisition damages generality before
it induces useful compression, motivating learn-first, consolidate-later.
Full grids in the appendix.

## 7.5 Promotion: creating shared computation (third sealed block)

The dissociation above — allocation solved, compression unsolved — is the
gap this section closes, in a setting deliberately built to contain the
phenomenon. The construction matters as much as the result, because three
earlier designs failed and each failure eliminated a plausible criterion.

**The world.** A six-slot operator library is saturated on the
environment's six base primitives for 16 tasks. The library is then frozen
and the environment begins issuing tasks that call a SEVENTH primitive,
drawn independently, which no mixture of the frozen six can express. Two
hidden task groups use two different such primitives. Group membership is
never exposed. The learner can only respond through its task-local
low-rank innovations, so recurring computation accumulates there.

**Why the earlier designs failed.** When the library had spare capacity,
the learner absorbed the recurring structure into unused slots and
addressed it through its routing code — causally confirmed (substituting
the wrong group's mean route costs more than the right one's; single-slot
ablation damages the groups differentially). Nothing reached the residual
channel. When families were perturbations of existing primitives rather
than new ones, the residuals became recoverable but not load-bearing:
replacing them with group means barely beat deleting them outright, so
"promotion" was compression of a channel that did not matter. And
averaging members' PARAMETERS destroyed the shared function entirely,
because rank-2 innovations are gauge-equivalent under rotations of their
factors; a parameter mean captured 11.9% of behavioral value where a
functional fit captured 53.4%.

**The validity gate that survived.** We therefore define promotable
structure operationally, by a four-way leave-one-out comparison in which
the candidate abstraction is fitted WITHOUT the task it is substituted
into: private residual, family abstraction, single global abstraction, and
zero. Three properties must hold together — the private computation must
be load-bearing (zeroing it costs), compressible into one shared function,
and family-specific (beating a global abstraction). On the frozen world:
private 0.0127, family 0.0179, global 0.0205, zero 0.0224, giving 51.3%
family capture against 22.0% global, a 29.3-point advantage, with an
on-trajectory probe agreeing with a common-domain probe.

**The operator.** At fixed consolidation points the learner clusters its
task innovations functionally, fits one rank-2 abstraction per cluster by
gradient descent on behavioral distance, and accepts only if the
substitution preserves its own predictions on a probe set disjoint from
the one used to propose and fit, shortens the two-part code, and beats a
single global abstraction on members it was not fitted on. Accepted
members retire their private copies and keep a reference. New tasks may
select any library entry or none from their own early examples.

**Result.** Against the identical learner with promotion disabled, on 30
sealed worlds under intervals registered in advance: total retained
description length falls 63.3% (task bits down, shared bits up, total
down, 30/30; conservative, since shared state proved 6-14x more
compressible per task than private state, and promotion moves
description from private into shared); lifetime prediction improves by 1,174 nats (30/30); held-out
future tasks from the same hidden families are cheaper to acquire, 32-shot
NMSE improving by 0.0031 (30/30). Development and sealed parameters agree
to within 0.7%.

**What promotion does not do.** It does not refuse. In structureless
control worlds it builds a smaller library (3.0 abstractions against 6.2,
with 1.80x fewer tasks reusing) rather than declining to build one, and
the pre-registered absolute refusal criterion is therefore falsified; only
the graded contrast holds. At the two-part exchange rate a private rank-2
innovation costs far more than the prediction it buys, so any
behavior-preserving grouping pays and retrospective compression alone
cannot discriminate. Only the prospective test — must this abstraction
beat a global one on tasks it never saw — produces separation. Finally,
the promoted abstraction is not the environment's seventh primitive; it is
the best shared explanation of how this learner compensated for a
computation it could not express.

## 7.6 When a library does NOT need a lifecycle (fourth sealed block)

Having shown that abstractions can be born, we asked whether they need
managing. The premise of a lifecycle — that successful birth creates a
maintenance problem — turns out to be false at this scale, and the
negative was preregistered rather than discovered after the fact.

**Development.** An opportunity census, run with the learner frozen and
no operator implemented, scored every candidate structural edit on an
ambition ladder, KEEP < COMPRESS < SHARE/FACTORIZE < CREATE/FORK, with
each rung required to beat the cheapest simpler edit **at the same
resource budget**. Across families F in {2,4,8} and lifetimes N in
{64,128,200}, local private requantization beat shared factorization in
15/15 regime-worlds, and the margin widened with library size rather
than closing. Deduplication had nothing to remove: at a
contribution-relative tolerance no abstraction substituted for another,
and compacting the library was net negative. The reason is measurable —
the abstractions are 4-8x overparameterized, carrying zero behavioral
cost at 6 bits per scalar and 127 nats at 4 — so every structural edit
competes against a large reservoir of purely numerical slack that no
restructuring is needed to harvest.

**Sealed block (seeds 400-429), scored against a plan frozen and hashed
before the worlds were generated.** COMPRESS beat FACTORIZE in 30/30
worlds with a mean margin of 1,362 nats (registered interval
1,000-4,000); no world showed a factorization win at any realized
library size at or below 16; FORK paid, with an actual refit charged one
abstraction, in 2/30 worlds against a registered allowance of 2 — a pass
at the boundary, and we state it as "FORK pays rarely, within the
preregistered bound" rather than as never.

**The exception, and it is a mechanism rather than an absence.**
Retention pays, and it pays for the reason abstraction birth pays.
Deleting an abstraction imposes a real reacquisition cost when its
regime returns, and the decision obeys

    RETAIN A  iff  H_R * s_bar  >  lambda * D(A)

where `H_R` is the number of post-return uses and `s_bar` the per-use
saving. On sealed worlds, under a controlled counterfactual in which the
library is frozen at the gap so that deletion saves exactly one
abstraction's code, `s_bar` is constant at 61.0 nats per use across
horizons (coefficient of variation 0.3%), and the crossing falls at
`H_R = 18.0`. The prediction derived from development data alone was
17.1. Notably, the governing variable is NOT dormancy length: the
per-use saving is flat across gaps, and three earlier world designs
failed precisely because they varied dormancy rather than expected
remaining reuse.

**What this adds to the birth result.** V3's amortization criterion —
enough repeated use to repay a code cost — reappears one level up,
deciding whether a learned abstraction is worth carrying rather than
whether it is worth creating. Retention needs no theory of its own. The
boundary is that in a library which keeps evolving the carry term
becomes endogenous: when deletion merely triggers a replacement
promotion it saves nothing, and a per-object retention rule stops being
well-posed. Lifecycle decisions are sequential, not per-object.

**A methodological result we consider the most portable.** Every
apparent structural win in this block dissolved once it was scored
against the right counterfactual. A shared parameterization beat
full-precision atoms and lost to the same atoms at equal bits; a
retirement rule looked selective until value was measured against the
alternative the learner would actually have rather than against
ablation; a compaction gate passed until its tolerance was normalized
against the contribution it was licensing the loss of. We therefore
state as a standing rule: **never claim value for a structural edit
unless it beats the cheapest simpler edit at the same resource budget**,
and **measure value against the cheapest legitimate alternative, never
against removal**.

## 8. Related work

**Controlled models of task relatedness.** The closest methodological
relatives study transfer with parametric control over how related tasks
are. Gerace et al. (2022) analyze a solvable model of synthetic
correlated datasets, characterizing when transferring a learned feature
map from source to target helps as a function of dataset correlation;
Mohamud and Brink (2025) extend the correlation-controlled empirical
line to pre-trained model reuse. Multi-task theory predicts benefits that scale with shared
structure (Baxter, 2000; Maurer et al., 2016; Tripuraneni et al., 2020),
and task-grouping and negative-transfer studies document that sharing
can hurt when relatedness is low (Zamir et al., 2018; Standley et al.,
2020; Wu et al., 2020; Zhang et al., 2023). ROW differs from all of
these on three axes at once: the controlled quantity is functional
recurrence among latent computational operators rather than input or
feature correlation; the outcome is lifetime prequential learning cost
in a sequential, online setting rather than terminal generalization from
one transfer; and the study measures not only whether sharing pays but
whether the shared computations themselves are recovered.

**Controlled studies of modularity.** Mittal et al. (2022) ask, on
synthetic rule-based data with a ground-truth-modular oracle, whether
end-to-end training achieves the specialization that modular
architectures permit, and find that it generally does not. Our
route-inference result is consistent (the discrete learner recovers the
library yet pays an online inference tax), and ROW adds the recurrence
intervention: rather than fixing a modular world and asking whether
modules specialize, we vary how modular the world is and measure when
modular representation becomes economical.

**Modular, reusable, and compositional continual learning.** Modular
meta-learning (Alet et al., 2018), neural module networks (Andreas et
al., 2016), RIMs (Goyal et al., 2021), and modular continual learning
with explicit reuse decisions (Veniat et al., 2021; Ostapenko et al.,
2021) build substrates for reuse; Mendez and Eaton (2021; 2023) develop
and survey lifelong learning of compositional structures, which shares
our premise that accumulated components should make later tasks cheaper.
Mixture-of-experts and soft merging (Shazeer et al., 2017; Muqeeth et
al., 2023) study routing trainability; SMEAR in particular anticipates
our finding that hard-routing failure is not evidence against reusable
computation. Hypernetworks (Ha et al., 2017) and attention as a
hypernetwork (Schug et al., 2024) motivate our continuous-manifold
control. Kirsch et al. (2018) learn to decompose computation into
modules end to end. What this literature generally lacks, and ROW
supplies, is intervention on the environment: prior work compares
methods at fixed or naturally occurring task relatedness, whereas ROW
manipulates ground-truth functional recurrence and measures the
resulting sign and magnitude of the representation preference.

**Library learning and program induction.** The wake/sleep library-
learning lineage (DreamCoder; Ellis et al., 2021) is the closest
program-level relative, and recent work extends compositional vocabulary
learning to neural components (Shan et al., 2025). We differ in
criterion (lifetime prequential cost rather than task solve-rate), in
operating over learned neural operators whose correspondence to ground
truth is measurable, and in the negative-control discipline our V2
consolidation program inherits: a compressor must decline to compress
structureless worlds.

**Prequential evaluation.** Our headline metric operationalizes the
prequential principle (Dawid, 1984) and prequential MDL for neural
networks (Bornschein et al., 2022) as a lifetime learning-cost measure
with fixed likelihood, extending its use from model comparison to a
controlled intervention study of representation choice.

**Compositional generalization.** Studies reporting models that succeed
in distribution while failing recomposition (Lake & Baroni, 2018; Hupkes
et al., 2020) are consistent with our statistical/structural
dissociation; ROW provides a controlled dose-response version of that
observation, with the abstraction's presence checked against known
generative operators rather than inferred from behavior alone.

The novelty claim we defend is therefore: we have not found prior work
that experimentally intervenes on ground-truth functional recurrence
among latent computational operators, measures the sign and magnitude of
the optimal sharing preference by lifetime prequential cost, and
simultaneously tests whether the shared latent computations themselves
are recovered.


## 9. Limitations and what this paper does not claim

- No claim that these results transfer to large-scale or natural-data
  learning; the criterion is the bet, this paper is its controlled test.
- No claim that discrete or modular architectures are superior in general
  — the finding is a regime map, and dense learning wins a large region.
- No claim of relevance to ARC-AGI or any external benchmark.
- No claim that generic learners discover arbitrary factorizations
  irrespective of structural prior (Section 6.1 states the precise,
  conditional claim).
- No conflation of lower lifetime cost with compositional transfer: they
  dissociate (Section 5.2), and "learned reusable primitives" is claimed
  only where transfer and operator recovery both support it.
- No universality claims: expressivity is deliberately closed off (fixed
  program length) so learnability and encoding cost can be measured
  without expressivity confounds.
- Additional limitations: one functional family of teachers; length-3
  programs; 64-task lifetimes; Gaussian likelihood with fixed sigma;
  single-machine scale throughout.
- A later V5 diagnostic initially understated higher-order structure by
  comparing task functions on unaligned inputs and incompletely restoring
  model state. Correcting the audit changed effective-operator capture
  from 0.190 to 0.762 and full-population unexplained variance from 0.707
  to 0.491. Those V5 diagnostics are not evidence for the four sealed
  claims above; we report the correction because it reverses the proposed
  mechanism for the learned library's failure.
- Lifetime-length dependence: a truncated-lifetime analysis (appendix)
  reads the crossover at 16, 32, and 64 tasks from the per-example logs;
  the mean crossing moves from configured rho 0.869 to 0.822 between 16
  and 32 tasks and then saturates (0.826 at 64; 8/10 development worlds
  decline overall). Amortization is therefore a demonstrated
  early-lifetime component of the crossover, not its whole mechanism,
  and we do not claim a confirmed lifetime-length law.
- A later development-only control-flow substrate audit sharpens the
  family-alignment limitation. Adding an orthogonal state rotation makes
  iteration and branching genuinely necessary, but a learner with the same
  rotated operator family fails the registered learnability gate in all three
  development worlds (final NMSE 1.61-1.76 versus 0.012-0.017 on the ordinary
  substrate). The exact maps remain orthogonal and routing does not collapse;
  the current online protocol fails to acquire them. Thus matching the
  representational family is necessary but not sufficient for learnability, and
  this paper makes no claim that its library-learning result extends to the
  stronger operator substrates required for control flow. This is a
  development boundary, not an additional sealed claim.
- A subsequent frozen, development-only localization audit narrows that boundary
  without removing it. Exact constructive representation, Householder-Adam, and
  projected-rotation Adam fits passed all 18 primitives. The original ancillary
  LBFGS arm used an unregistered penalty and is excluded from this inference.
  With true routes fixed, joint library training passed its registered rule in
  two of three worlds (final median task NMSE `0.00625` and `0.00538`) but
  remained poor in world 0 (`0.72207`). The licensed classification is therefore
  route inference or online interference under this protocol, not a defect in
  representability or isolated Adam findability. The audit does not separate
  those two causes, and its one large world-level failure precludes a stability
  claim.
- A frozen correction subsequently reran the ancillary H-LBFGS arm without the
  unregistered penalty. It passed all 18 isolated primitives (median query NMSE
  `7.93e-5`). This corrected artifact, not the original penalized endpoints,
  supports the LBFGS findability result; it does not change the primary
  route-inference-or-online-interference classification.
- Later development work separated those causes, still without a sealed claim.
  - **Offline:** a supplied length curriculum (length-1, then length-2, then
    length-3 tasks, carrying only the shared library) forms the rotated
    library without oracle routes at two initializations. That library
    executes unseen programs.
  - **Online:** under the full online protocol, the same staged schedule
    reaches the registered export margin in nearly every development world but
    the terminal-error threshold only in some. Three preregistered runs across
    ten development worlds and three learner seeds reached it in six worlds
    (1/3, then 3/3, then 2/4; labels `SO2_FAILS`, `SO3_FAILS`, `SO4_FAILS`).
    Within one world the replay stream alone moves terminal error by up to an
    order of magnitude, so online learnability here is world-dependent rather
    than established or refuted. The non-staged online learner fails
    everywhere.
  - **Where the loss lives:** a descriptive census places it after acquisition,
    in the final stage. Recently learned tasks lose accuracy while earlier-stage
    computation largely survives.
  - **Candidate rescue:** one exploratory development world recovers with a
    halved stage-3 shared learning rate or more diverse replay storage.
  - The online learnability of this substrate therefore remains an open
    development question, now localized to post-acquisition consolidation in the
    final stage. It is not a claim of this paper. See the dated development
    addenda.


On promotion specifically (section 7.5): the world is constructed to
contain the phenomenon. A saturated library, a genuinely inexpressible new
primitive, a fixed program position, and two hidden groups are all design
choices, and three earlier designs produced worlds in which promotion
degenerated into routing, quantization, or deletion. The claim is that
promotion works when recurring structure is load-bearing, compressible,
and family-specific; whether those conditions arise unaided in natural
learners is untested here. Promotion is also compute-expensive — the
search cost is logged but not charged in the objective — so a learner that
saves storage while spending far more search would be scored as a success
by our accounting and should not be.

## 10. Discussion: toward learned abstraction economics

Sharing is not inherently good. It is an investment: a learner pays a
representational price by forcing tasks through shared computation, and
the return on that investment is determined — linearly, in our worlds,
with coefficients that now replicate across two independent sealed
blocks — by how much genuinely reusable computation the environment
contains. Even when the investment pays predictively, the learner may
not yet have discovered a clean abstraction: economic benefit precedes
recomposable structure, and only near exact recurrence do lifetime
efficiency, recomposition, and identifiable primitive recovery align.

The second sealed block adds the study's sharpest asymmetry. A learner
can be shown to READ these economics almost perfectly — allocating
per-task specialization against per-primitive ground-truth recurrence
it was never told, in every one of 30 sealed worlds — while every
mechanism we built for WRITING the result compactly failed, each for an
identified reason: gradient descent optimizes prediction and never sees
description length; identity-based instruments (route entropy, slot
matching) fail in representations where many implementations are
behaviorally equivalent; and within-lifetime data is too scarce to
amortize learned inference. Allocation solved, inference solved only
exactly, compression unsolved, representational form set by recurrence:
that four-line map, rather than any single winner, is what these two
sealed blocks establish. The successor program targets the compression
cell — whether recurring specialization can be promoted into shared,
named structure that wins prediction and description length at once —
with the objective carrying description length in the gradient rather
than discovering it in the accounting.

### 10.1 Development-stage outlook: the economy over representational transformations

The third sealed block closed the compression cell by showing that a
saturated learner can create a new shared abstraction (PROMOTE) and win
prediction and description length simultaneously. The natural follow-up
-- whether a learned library then needs lifecycle operations (MERGE,
DELETE, FACTORIZE, RETAIN) -- was first pursued as V4 on development
worlds, where the original premise failed, and then as V4R. V4R's
sealed block (seeds 400-429, section 7.6) confirmed the development
picture: local compression dominates restructuring, and the one
structural edit that pays is retention, obeying the same amortization
law as birth. The findings below remain the development account of
*why*; the sealed numbers live in 7.6.

The V4 development program falsified its original premise. The
assumption that successful abstraction birth implies a library maintenance
problem did not hold at this scale: V3's 4-6 abstractions for two hidden
families were mostly behaviorally distinct once substitutability was
measured relative to each abstraction's own contribution rather than
total output variance; apparent low-dimensional functional geometry
existed but representing it explicitly as a shared parameterized family
lost to matched-budget private compression in essentially every tested
cell; and the dominant slack was numerical (abstractions stored at ~8
bits/scalar had a behavioral coding frontier of ~1-2 bits/scalar), not
structural. A simple COMPRESS operation -- lower precision, lower rank,
pruning -- beat every cross-abstraction restructuring opportunity across
the reachable census, establishing that local coding efficiency should
be exhausted before global structural complexity is introduced. The
cumulative lesson is constitutional: every sharing or restructuring
claim requires a matched-budget non-sharing alternative, and a structural
operation earns credit only relative to the cheapest behaviorally
legitimate counterfactual the learner would actually take -- not raw
ablation.

The one positive V4 development result is a retention law that was not
fitted. Retaining an abstraction obeys the same amortization logic as
its birth: RETAIN(A) iff H_R * s_bar > lambda*D(A), where H_R is expected
remaining reuse, s_bar is mean per-reuse savings, and D(A) is the
abstraction's serialized cost. The predicted critical horizon
H_R* = lambda*D(A)/s_bar ~= 1098/64.1 ~= 17.1 returning tasks matched the
empirical crossing of 17.9 (within ~5%). Promotion and retention are
therefore the same economic decision viewed in opposite temporal
directions (birth: N_observed * s > D(A); retention: N_future * s >
D(A)), and dormancy length was not the relevant variable -- per-return
savings were flat across gaps, and only expected remaining reuse
mattered.

Two complications qualify the simple law. In a self-modifying library,
deleting an abstraction can trigger a replacement promotion that saves
zero bits, so an abstraction's value is conditional on library state:
V(A | L_t, H_t, pi), not V(A) alone. And because today's edit changes
tomorrow's representation search, library operations interact through the
future learning trajectory -- making the eventual problem sequential
structural decision-making rather than per-object garbage collection.
The development census was otherwise negative: FACTORIZE, RETIRE, and
FORK had no oracle advantage at current scale, and even tripling lifetime
and growing the library toward ~16 abstractions did not produce a
factorization crossing. Small stationary neural libraries may simply not
need sophisticated lifecycle machinery; architecture complexity itself
must be amortized. The pressures that should eventually create a
lifecycle economy -- scale, meta-recurrence, nonstationarity,
reacquisition cost, retrieval cost -- now name the axes the next
benchmark must vary.

These development findings reframe the ultimate goal. The system should
not be a neural library with a fixed checklist of lifecycle operators;
it should ask, at every point, what the cheapest available
representation is -- KEEP, COMPRESS, PROMOTE, FACTORIZE, RETAIN, FORK --
with each operation earning its existence economically. At every level
the same law appears: pay a fixed representation cost when expected
repeated savings exceed it (N * s > C), whether the unit is tasks
amortizing an abstraction, abstractions amortizing a schema, or
sequences amortizing a macro. The final system this program points
toward is a self-refactoring neural computational system that
continually asks where information should live, what form it should
take, whether it is worth naming, how precisely it should be encoded,
and whether keeping that representation will make future learning
cheaper. V4 did not deliver MERGE or DELETE; it delivered the economic
rules those operations will eventually have to obey. V4R confirmed
that those rules, not a lifecycle operator suite, are the result at
this scale.

Subsequent V5 experiments, outside the four-block scope of this draft,
made the amortization law quantitative and found a teacher-level schema
economy, while matched-budget FACTORIZE over the learned promoted library
lost in all six tested cells. A first audit attributed that gap to wake
learning failing to preserve higher-order structure. That interpretation
was wrong: the audit evaluated different tasks on unaligned states,
omitted retirement state, and used an incomplete rollout. With those
defects corrected, ordinary wake's effective task-conditioned operators
capture 0.762 of the family structure, and the full innovation population
leaves 0.491 of a teacher operator unexplained, versus 0.695 for the best
single innovation. Yet the promoted library still leaves 0.921
unexplained and FACTORIZE remains 0/6. The structure is therefore present
but distributed across routes, residuals, and tasks; PROMOTE's extraction
of one private residual at a time is the failing unit. This reverses the
outlook from "change the wake objective to create missing structure" to
"discover reusable objects by functionally refactoring the population."
The original prospective V6 comparisons remain void because the hook used
future lifetime labels and its controls and acquisition metric were not
matched. In a repaired three-world development run, the preregistered H35
pressure optimum was not observed: one- and two-step pressure produced mixed,
non-replicating effects, while eight-step pressure increased current lifetime
loss and made related futures more expensive in all three worlds. This is
exploratory evidence that the tested prospective objective becomes harmful at
high pressure, not a confirmatory V6 result or evidence for a beneficial
interior optimum. A separate 45-cell plasticity-allocation matrix found a
graded shift in representation form: increasing free shared capacity reduced
explicit library size from 7.0 to 3.7 and increased accessible geometry in
every paired arm-world endpoint, but not monotonically at intermediate
settings. No allocation produced reliable prospective fertility, and
FACTORIZE won 0/12 scoreable prospective cells (0/34 overall). V6 therefore
closes in development without a confirmatory block: plasticity influences
where recurrence is stored, but the current objective never makes that
structure economical for future acquisition. A subsequently frozen
adaptation-geometry audit localized the deficit. With the representation fixed
and 128 support examples, 2,000-step Adam reached scaled query endpoints of
0.01814 for ordinary versus 0.02976 for prospective, a 64% increase; all six
paired tasks and all three world means favored ordinary. Adam at a second
learning rate and LBFGS independently converged to nearly identical arm-specific
endpoints. Thus the tested pressure damaged the representational opportunity
available through fresh task code and private residual, rather than merely
miscalibrating the original 40-step optimizer. This is an operational
finite-budget localization, not a proof of global impossibility. Together the
results sharpen `geometry != fertility`: compressing common structure does not
ensure cheap coordinates for learning novel variation, motivating an explicit
shared-schema/fast-argument/private-innovation factorization.

### 7.7 Parameterized primitives: made online, not mined afterwards (fifth sealed block)

The factorization was then tested in a development ladder and a fifth
sealed block (seeds 700-729; `H39_CONFIRMATION_PLAN.md`,
`reports/h39_confirmation.json`). The ladder first established a negative
that turned out to be the result's foundation: a post-hoc linear schema
fitted to the finished ordinary learner's own residual objects cannot
express a member of an unseen family — at rank 8 the alpha-only endpoint
is 3.5x the free-residual cost, and at the full affine span of everything
the learner stored it is still 2.2-3.1x (`reports/h39_census.json`). A
schema formed jointly online over the same residual channel did no
better (4.2x), because the residual carries about 2% of family
computation once promotion has retired most family tasks into
references. Moving the argument into a routed basis operator,
P(alpha) = tanh(z + a (U_0 + sum_k alpha_k U_k) tanh(Vz + b)), changed
every endpoint: the alpha-only cost fell monotonically with the argument
dimension K in all three development worlds (3.37 -> 1.56 from K = 2 to
16), a matched-budget control with the U_k frozen at random init stayed
at ~3.1 and was essentially unused, a single slot saturated at K ~ 16,
and two slots at K = 32 reached 1.46 (`reports/h39c_ksweep.json`,
`reports/h39d_capacity.json`).

The sealed block compared three paired arms on 30 unseen worlds: the
ordinary learner O, the two-slot K = 32 learner M, and G, identical to M
with its argument directions frozen at initialization. All five
preregistered estimands passed: learned directions beat frozen ones
(mean log-ratio gap 0.886, CI [0.717, 1.054]); the geometric-mean
alpha-only cost relative to O's full-residual fit was 1.636 with CI
[1.495, 1.7997] against a registered ceiling of 1.8 — confirmed at its
bound and not beyond it; present-task cost fell by 1,545 nats on average
(CI [-1,649, -1,435]) and in every world; the full schema-plus-innovation
interface beat O on the future task by 18.5% (ratio 0.815, CI [0.764,
0.866], 28/30 worlds); and zeroing the arguments raised family-task NMSE
2.9-4.3x in all 30 worlds. Two facts are recorded with the verdict: the
E2 interval clears its bound by 0.0003, and the E2 rule was amended
before any sealed cell was read (from a fraction-of-worlds rule, which
would have read 0.433 and CONFIRMED-RELATIVE, to the geometric-mean
interval, on a reviewer's argument that the 1.5x threshold was an
inherited convenience with no theoretical standing).

Two audits on the sealed artifacts closed the obvious objections
(`reports/h39_confirmation_followups.json`). Post-hoc extraction from the
realized task-object populations costs 2.59x (O) and 6.33x (M's own
residuals, which the argument channel has drained of family structure)
against 1.64x for the online arguments — the difference is WHEN the
coordinates were formed, not which learner. And removing the arguments
and re-optimizing route and residual at matched budget still costs
2.01x: the channel is used and not substitutable.

What the block licenses is existence and use, not discovery. The
architecture — two parameterized slots, K = 32 — was supplied. And a
baseline measurement taken before the discovery experiments showed that
M's two slots are not two abstractions: routing over them has conditional
entropy 0.92-0.95 of a possible 1.0 bit and agreement with the teacher
families at chance, because at full recurrence the generator places all
four families on one rank-2 manifold. The confirmed representation is a
single distributed continuous coordinate system, P(alpha) with alpha in
R^64, not a pair of operators. Forcing discrete commitment on that
continuous family (H47 B1) cost 10-15% in alpha-only acquisition and
7-25% of the present gain, less when the commitment came late; an
arbitrary partition imposed from the start forfeited 36-69% of the
present gain while barely touching the future, the innovation channel
repairing what the wrong ontology broke.

A generator extension with two orthogonal family subspaces (G = 1
reproducing the original bitwise) then asked whether real membership is
worth representing. At K = 32 it was not: the learner absorbed both
groups into one channel, paid an inconsistent +0.27 / +0.10 / -0.03 log
units on alpha-only acquisition, and beat the told-membership oracle on
present cost by ~500 nats per world (`reports/h47_b2_gate.json`).
Narrowing the channel changed that: at K <= 8 the oracle wins the future
channels in every world (alpha-only by 0.12-0.52, full interface by
0.08-0.26 log units) while its present cost shrinks tenfold without
quite crossing zero (`reports/h48b_width.json`). Discrete identity
therefore starts to pay for future acquisition below a width threshold
between K = 8 and 16 — a resource boundary, not a truth boundary — and
the label-free learner still does not find the groups at widths where it
pays for ignoring them (ARI ~0 at K = 2). That is, finally, a real
discovery problem; the registered next question (H49) is whether any
quantity computable from experienced tasks alone prefers the fertile
partition, since at K = 4 the grouping has future value but no
present-objective value and the learner's non-discovery is rational
under the objective it was trained on.

The answer was no, and then a stronger no. A census of every
retrospective signal available to the learner (leave-one-out
reacquisition cost, a two-part description proxy, own-versus-other slot
substitutability) found that all three discriminate the true grouping on
a representation that was ORGANIZED around it during its lifetime, and
none discriminate it on the representation ordinary learning actually
produces (`reports/h49_discoverability.json`). Here, the detectable value
of a structure was a property of the representation that holds it, not of
the task population alone. The obvious repair — propose a grouping, reorganize the
finished representation under it, then score — failed as well: 4,096
budget-matched migration steps under the true partition, with every
task-local variable and both argument matrices free, recovered
approximately none of the organized representation's separation
(recovery fraction -0.09 to +0.09) and none of its advantage on unseen
family members, while a structureless control with the identical budget
beat every partition by 0.39-0.58 log units
(`reports/h50_reorganization.json`). Worse, migration under ANY
partition manufactured the surface signature of organization —
substitutability rose from ~0 to 0.24-0.66 regardless of which grouping
was imposed — without the discrimination or the fertility. A metric that
the intervention itself inflates cannot certify the intervention.

One further attempt closed the line. If organization must be developmental,
perhaps several candidate organizations can develop side by side and be
selected between afterwards. Six were co-formed inside a single lifetime at two
sharing depths, scored by the same instrument. Neither depth separated the true
grouping from plausible wrong ones, and the failure was not that the candidates
merged: measured with their externally supplied policies neutralised, they had
diverged to 10-44% of the distance between independently trained
representations, and remained indistinguishable anyway. The accounting was
equally discouraging - concurrent formation cost 77-86% of six independent
lifetimes in device-seconds, and at the one sharing depth where state was
genuinely amortised (5.3x) a candidate's only private state was per-task, hence
absent for an unseen task, so differential future value was impossible by
construction rather than merely unobserved.

A separate line then asked the question all of this had been circling: are
these objects usable outside the lifetime that made them? Freezing a discrete
library and asking it to execute teacher programs the lifetime never trained on
costs nothing measurable - query error 0.0019-0.0076 against a trained-task loss
of 0.0038-0.0073, where a from-scratch learner given the same adaptation budget
reaches 0.031-0.044 and a library transplanted from an incompatible world
reaches 0.048-0.073. Inferring the program from support examples alone recovers
essentially the whole gap between scratch and the teacher's own program
(0.98-1.04 of it). Held-out programs containing an adjacent operator pair that
never occurred in training behave like the rest, so what the library holds does
not depend on having seen a particular adjacency. Two boundaries belong with
that result: the inference is offline over a finished library and is not the
online routing problem an earlier block found hard, and unfreezing the library
on a single task's support drives its training objective to zero while
generalising worse, so finetuning here is overfitting rather than repair.

That result would be compatible with a duller explanation - that twelve
tanh-residual operators simply span this function class, so any trained basis
would serve - and the controls used there (an untrained library, and one
transplanted from an incompatible world) cannot separate it, because both vary
the library rather than the property the earlier blocks established as causal.
Repeating the measurement across worlds that differ only in recurrence does
separate it. The export margin over a from-scratch learner is 1.82-2.67 log
units at measured recurrence 1.0, 0.49-0.62 at 0.64, and 0.03/-0.02/0.03 at
zero: a library formed where nothing recurs is worth nothing on a held-out task,
with the same architecture, operator budget and adaptation budget throughout.
Export is a consequence of what the world made available to learn.

The sharper question is whether these objects are operators or
position-specialised variants, and answering it requires building the lifetime
rather than sampling it: three operator-position placements were withheld from
training entirely, and held-out programs then placed those operators exactly
where they had never appeared. Performance is indistinguishable from held-out
programs using familiar placements - oracle error 0.0031-0.0040 against
0.0029-0.0035, with margins over a from-scratch learner of 2.13-2.67 against
2.25-2.69 - and the withheld set included the final program position, where a
distributional penalty would have been largest. A learned operator therefore
retains its semantics in a position it never occupied. The whole pattern was then banked
in a sixth sealed block of thirty untouched worlds, against a protocol frozen and
hashed before any of them existed, with the components registered independently so
that a mechanistic miss could not erase a behavioural result. Every clause
confirmed: each task's solution is literally its three-integer program, bitwise, in
thirty of thirty worlds; that program costs 13.75 bits against a vocabulary whose
description amortizes after 4.02 tasks and which is 14-15x cheaper than coding each
task privately; a consistent relabelling of library and program leaves behaviour
bitwise unchanged while wrong programs of identical length collapse by 2.2-2.5 log
units; the frozen vocabulary executes programs it never trained on, composes on
unseen triples, unseen adjacent pairs and unseen operator positions, and remains
executable one call beyond the horizon it was trained at. Two features of the
replication are worth separating. The behavioural margins shrank slightly, as one
expects. The description-length quantities did not: across thirty untouched worlds
the two-part advantage spans 0.10 log units and the amortization point sits at
4.02 with a range of 0.43, essentially on top of three development worlds - a
description-length statistic of a substrate appears far more stable across worlds
than any performance margin measured on it. Two details qualify what
that means. Inferring the program from support examples BEATS executing the
teacher's own program through the functionally matched objects, in every world
of that stratum; and the matching itself is barely determined, several
assignments scoring within 0.02 of the best. Which object corresponds to which
teacher operation is not settled, and does not need to be for the objects to
compose.

The lesson this block adds to the earlier ones is the most consequential
in the paper: the useful abstraction was not latent in the finished
solutions waiting to be extracted; the learner had to maintain an
explicit intensional channel while it learned. An abstraction, on this
evidence, compresses what recurs while exposing cheap coordinates for
what may vary — A(alpha) + eps, not A — and whether such a coordinate
system should be split into named types is decided by capacity
economics rather than by the world's latent ontology. The corollary,
earned by the two negative rungs above, is that representation formation
is PATH-DEPENDENT: two learners can reach the same present cost on the
same task stream and leave behind states with very different future
abstraction options, and the difference is not recoverable afterwards at
any fraction of the formation budget we could afford. That suggests a
third property beside present cheapness and description length —
reorganizability, the expected cost of migrating a representation into a
plausible alternative organization — and a design target in which
learning produces editable computational objects rather than only weight
changes.

We then tested the obvious repair on the representation itself. Holding
the migration operator and the scorer fixed and varying only what wake
leaves behind, three alternative representations were measured against
the same endpoint — the migration budget at which the true grouping
becomes distinguishable. One initialized each task's local state from a
provenance trace snapshotted when the task completed; one let each task
re-acquire itself from a learned mixture of the traces of the tasks
grouped with it; one stored every task's innovation in a separately
addressable component basis learned over the lifetime, at matched present
cost (its lifetime loss differs from ordinary by 0.09-0.19%). None of the
three separated at any scored budget, recovery of the organized
representation's margin stayed within [-0.12, +0.19] of zero, and a
structureless control beat every candidate grouping in every world by
0.33-0.54 log units (`reports/h51_reorganizability.json`). So the
negative is not that ordinary learning discards task-local evidence, and
not that it stores innovation densely: whatever the organized
representation has, it is neither preserved state nor storage format.

One of those arms is worth stating on its own, because it separates two
things that are easy to conflate. The provenance-recombination channel
was genuinely valuable — it cut absolute reacquisition cost 18-38% below
ordinary — and it was actively uninformative about grouping, the wrong
partition scoring better in all three worlds. A representational addition
can be worth its cost and carry no evidence about which structure
produced the data; usefulness and informativeness are different
properties, and only the second is what a discovery procedure can run
on.

Why should anyone working at scale care about a 16-dimensional world?
Because the criterion is scale-free and the phenomena it isolates have
scaled analogues that are currently argued by anecdote: mixture-of-
experts routing is a bet that language has high-recurrence structure
worth sharing; adapter libraries are per-task residuals on a shared
parent; model merging is consolidation; and "does fine-tuning transfer?"
is the statistical-vs-structural question. ROW's contribution is not
that its numbers transfer — they will not — but that it shows these
economics are measurable at all, and supplies the falsifiable shapes
(linear return on recurrence, an alignment-dependent magnitude, sharing
without abstraction) that scaled versions of the question can be tested
against.

Reusable computation is an investment whose return is set by the
recurrence structure of experience. ROW makes that return measurable. The
next question — pre-specified in the public V2 protocol — is whether a
learner can estimate those economics itself: deciding what to share and
what to specialize (our adaptive substrate shows the benefit and names
the cost), maintaining uncertainty over programs rather than committing
prematurely, and choosing when accumulated experience justifies compiling
a recurring computation into a reusable abstraction — including declining
to compile when the world contains nothing worth abstracting.

## Reproducibility and provenance

Implementation and experimental execution were substantially
agent-assisted under human direction. All artifacts, plans, seeds, model
states, analysis scripts, and the complete agent/reviewer correspondence
are provided for reproducibility and audit: every run writes a
fingerprint-validated artifact directory (config, metrics, model weights,
world programs, seeds, git commit, environment); worlds derive from
explicit seed sequences; and the development/confirmation firewall, the
frozen confirmatory plan, and a falsified hypothesis are all in the
public record. The resulting record is intended to make all primary
claims independently auditable from artifacts rather than author
recollection. (Execution-cost details appear in the repository README.)


## References

- Alet, F., Lozano-Perez, T., & Kaelbling, L. P. (2018). Modular
  meta-learning. *CoRL*. arXiv:1806.10166.
- Andreas, J., Rohrbach, M., Darrell, T., & Klein, D. (2016). Neural
  module networks. *CVPR*. arXiv:1511.02799.
- Baxter, J. (2000). A model of inductive bias learning. *JAIR*, 12,
  149-198.
- Bornschein, J., Li, Y., & Hutter, M. (2022). Sequential learning of
  neural networks for prequential MDL. arXiv:2210.07931.
- Dawid, A. P. (1984). Present position and potential developments: Some
  personal views: Statistical theory: The prequential approach. *JRSS A*,
  147(2), 278-292.
- Ellis, K., Wong, C., Nye, M., Sable-Meyer, M., Morales, L., Hewitt, L.,
  Cary, L., Solar-Lezama, A., & Tenenbaum, J. B. (2021). DreamCoder:
  Bootstrapping inductive program synthesis with wake-sleep library
  learning. *PLDI*. arXiv:2006.08381.
- Gerace, F., Saglietti, L., Sarao Mannelli, S., Saxe, A., & Zdeborova,
  L. (2022). Probing transfer learning with a model of synthetic
  correlated datasets. *Machine Learning: Science and Technology*, 3(1),
  015030. arXiv:2106.05418.
- Goyal, A., Lamb, A., Hoffmann, J., Sodhani, S., Levine, S., Bengio,
  Y., & Scholkopf, B. (2021). Recurrent independent mechanisms. *ICLR*.
  arXiv:1909.10893.
- Ha, D., Dai, A., & Le, Q. V. (2017). HyperNetworks. *ICLR*.
  arXiv:1609.09106.
- Hupkes, D., Dankers, V., Mul, M., & Bruni, E. (2020). Compositionality
  decomposed: How do neural networks generalise? *JAIR*, 67, 757-795.
- Kirsch, L., Kunze, J., & Barber, D. (2018). Modular networks: Learning
  to decompose neural computation. *NeurIPS*. arXiv:1811.05249.
- Lake, B. M., & Baroni, M. (2018). Generalization without systematicity:
  On the compositional skills of sequence-to-sequence recurrent networks.
  *ICML*. arXiv:1711.00350.
- Mohamud, J. H., & Brink, W. (2025). An empirical study of task and
  feature correlations in the reuse of pre-trained models.
  arXiv:2506.01975.
- Maurer, A., Pontil, M., & Romera-Paredes, B. (2016). The benefit of
  multitask representation learning. *JMLR*, 17(81), 1-32.
- Mendez, J. A., & Eaton, E. (2021). Lifelong learning of compositional
  structures. *ICLR*. arXiv:2007.07732.
- Mendez, J. A., & Eaton, E. (2023). How to reuse and compose knowledge
  for a lifetime of tasks: A survey on continual learning and functional
  composition. *TMLR*. arXiv:2207.07730.
- Mittal, S., Bengio, Y., & Lajoie, G. (2022). Is a modular architecture
  enough? *NeurIPS*. arXiv:2206.02713.
- Muqeeth, M., Liu, H., & Raffel, C. (2023). Soft merging of experts with
  adaptive routing (SMEAR). arXiv:2306.03745.
- Ostapenko, O., Rodriguez, P., Caccia, M., & Charlin, L. (2021).
  Continual learning via local module composition. *NeurIPS*.
  arXiv:2111.07736.
- Schug, S., Kobayashi, S., Simsek, Y., et al. (2024). Attention as a
  hypernetwork. arXiv:2406.05816.
- Shan, H., Minni, S., & Duncker, L. (2025). Separating the what and how
  of compositional computation to enable reuse and continual learning.
  arXiv:2510.20709.
- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton,
  G., & Dean, J. (2017). Outrageously large neural networks: The
  sparsely-gated mixture-of-experts layer. *ICLR*. arXiv:1701.06538.
- Standley, T., Zamir, A., Chen, D., Guibas, L., Malik, J., & Savarese,
  S. (2020). Which tasks should be learned together in multi-task
  learning? *ICML*. arXiv:1905.07553.
- Tripuraneni, N., Jordan, M. I., & Jin, C. (2020). On the theory of
  transfer learning: The importance of task diversity. *NeurIPS*.
  arXiv:2006.11650.
- Wu, S., Zhang, H., & Re, C. (2020). Understanding and improving
  information transfer in multi-task learning. *ICLR*. arXiv:2005.00944.
- Zamir, A., Sax, A., Shen, W., Guibas, L., Malik, J., & Savarese, S.
  (2018). Taskonomy: Disentangling task transfer learning. *CVPR*.
- Zhang, W., Deng, L., Zhang, L., & Wu, D. (2023). A survey on negative
  transfer. *IEEE/CAA Journal of Automatica Sinica*, 10(2), 305-329.
  arXiv:2009.00909.

---
*Figures: (1) regime map, per-world sealed traces, both coordinates;
(2) linear dose-response with pooled fit — now overlaying BOTH sealed
blocks with the frozen intervals drawn; (3) checkpoint divergence —
indistinguishable at 8 tasks, ~2x at 64; (4) two response curves —
lifetime advantage, frozen recomposition advantage, and operator
recovery vs untrained baseline on one measured-recurrence axis; (5)
resource frontier; (6) robustness forest; (7) adaptive substrate:
envelope win in nats, loss in bits, allocation signature — extended
with the 30-world sealed allocation replication; (8, new) per-primitive
allocation vs ground-truth recurrence, sealed worlds, with per-world
Spearman distribution; (9, new) the two consolidation gates' firing
rates across recurrence against the pre-registered shape. Appendices:
MDL gating grids; batch, initialization, and lifetime-length tables;
mixed-effects sensitivity; gate designs and re-derivation record; both
confirmatory analyses exactly as pre-specified; sealed artifacts at
release v2.0-confirmation.*

# Development provenance addendum: SO1 restart (2026-09-09)

The SO1 rotated-substrate acquisition dose-response experiment has no accepted
result. Its September 8 launch failed during a six-worker batch; four successful
worker returns were lost when two other workers failed memory allocation.
The preserved launch evidence and separately frozen SO1_RESTART_AMENDMENT.md
document the failure and instrument repairs. No scientific threshold or
prediction was changed. The corrected conditional learned-route arm shares its
oracle counterpart's minibatch draws. The original oracle anchor streams remain
independent of Stage D, making their gate a joint implementation/resampling
check. Neither a failed launch nor a failed anchor licenses an acquisition
verdict or changes the earlier accepted findings.

# Development provenance addendum: SO1 anchor gate (2026-09-09)

The restarted SO1 run passed its host and bitwise pool gates, then failed its
registered anchor: the corner cell at Stage D's low budget reproduced world 0
but differed by 0.094 and 0.160 median NMSE on worlds 1 and 2 (tolerance 0.02),
while the high-budget corner agreed within 0.0033 on all worlds. SO1 therefore
stopped before its dose-response grid, as preregistered. The gate cannot say
whether the difference is numerical or due to the independent sampling streams;
no acquisition verdict is drawn and the Stage D findings are unchanged.

# Development provenance addendum: SO1 anchor diagnostic (2026-09-10)

A frozen diagnostic separated the causes of the SO1 anchor failure. The
original implementation reproduced Stage D exactly; the faster implementation
matched it on identical sampling streams to within 6e-7 after 8,192 updates,
and a 1e-7 perturbation of initial parameters did not grow. The anchor failure
was therefore due to resampling: at Stage D's low budget, five independent
minibatch streams produced terminal medians spanning up to 0.68 on one world.
SO1 is licensed to relaunch once with a matched-stream anchor; the resampling
spread is disclosed beside its low-budget comparisons.

# Development provenance addendum: SO1 acquisition dose-response (2026-09-11)

With routes supplied by an oracle, the rotated 12-slot learner acquired its
joint library (median query NMSE <= 0.05 in 2 of 3 development worlds) at
131,072 example-gradients with batch 2 and at 262,144 with batch 64. At every
equal-gradient budget the batch-2 learner (more updates, fewer distinct tasks
per batch) did better in all three worlds. With learned routes at those same
budgets the learner failed in all worlds (0.92-1.16). One world failed every
budget with either routing. Each cell is a single sampling stream, and the
batch-2 dose curve is non-monotone in two worlds; the registered
classification is that the substrate is acquirable but the route writer does
not acquire it, so the next question is the routing mechanism, not budget.

# Development provenance addendum: SO1R route-only inference (2026-09-11)

To localize SO1's learned-route failure, the libraries acquired with oracle
routes were frozen and only routes were inferred from each task's support
data. On the four libraries that met the threshold, exhaustive search chose
the exact oracle route for every task, and the learner's own gradient
relaxation reached the same query quality (0.005-0.009). Routes are therefore
recoverable on a correct library; the failure is in acquiring library and
routes together. On the one world whose libraries fail even with oracle
routes, search still recovered the oracle route, while gradient route
inference was markedly weaker.

# Development provenance addendum: J0 library-quality census (2026-09-11)

The route-only instrument was applied to all 30 libraries SO1 formed with
oracle routes, spanning median query NMSE from 1.26 to 0.003. The learner's
gradient route inference matched exhaustive search exactly on every library
at or below 0.47 and fell well behind it on nearly every library at or above
0.62, recovering the oracle route for as few as 5% of tasks, whereas search
still recovered it for at least 58%. The registered rank-correlation test
passed narrowly; the underlying pattern is a threshold in library quality.
Early in joint training every library is on the poor side of that threshold,
which is consistent with the hypothesis that joint acquisition fails because
gradient routing cannot work until the library is already good.

# Development provenance addendum: J1 search-in-the-loop (2026-09-11)

Replacing gradient routing with periodic exhaustive route search on the
current library (hard routes, 64 re-search rounds) did not form the rotated
library at SO1's budget: median query NMSE 0.76-0.77, against 0.005 with
oracle routes, 0.92-0.96 with learned soft routes and 0.97-0.98 for a sham
that permutes the searched routes across tasks. The route assignment froze
after three to five rounds on every world: the first search, on a random
library, made an arbitrary commitment that the library then fitted and that
remained optimal thereafter. Search removes the gradient weakness identified
in J0 but not the need for an informed first commitment.

# Development result: staged formation of the strong substrate (J1c, 2026-09-12)

A length curriculum acquires the rotated operator library that joint training
cannot. Three stages - 60 single-operator tasks, 64 length-2 tasks, then the
canonical 64-task length-3 world - sharing only the learned library and never
revealing a teacher route, reach median query NMSE 0.0062 / 0.0051 / 0.0072 on
development worlds 0 / 1 / 2, against 0.961 / 0.954 / 0.919 for the same
learner trained on the length-3 world alone at the same total budget, and
against the 0.05 threshold that oracle-supplied routes reach. A control that
runs the same stages and then re-initializes the library before the final
stage reaches only ~1.0, so the effect is transfer rather than compute. Five
to six of the six operation-to-slot pairings formed in stage 1 still carry the
final stage's traffic, with their functions moving by 0.06-0.21 on a common
probe while a freshly initialized library's differ by about 1.3: the structure
is extended, not rebuilt. Notably the one world that fails even with oracle
routes (0.624) passes under the curriculum, so a fixed oracle assignment
bounds only formation under that assignment. The claim is offline and
development-only; the online protocol remains untested.

# Development result: staged formation replicates (J1c-R, 2026-09-12)

The length-curriculum result was repeated at a second learner initialization
(seed 3001) with a matched non-staged control at the same seed. Staged
formation reached median query NMSE 0.0047 / 0.0060 / 0.0049 on development
worlds 0 / 1 / 2, within a factor of 1.2 of the first initialization, while
the non-staged control reached 0.916 / 0.944 / 0.974 and solved no task below
threshold. Five to six of the six operation-to-slot pairings formed in the
first stage again carried the final stage's traffic. The effect is therefore
not an artifact of one initialization, though it remains offline and
development-scale.

# Development result: the curriculum-formed library exports (J2A, 2026-09-14)

The libraries formed by the length curriculum were frozen and asked to execute
64 length-3 teacher programs drawn from the 152 compositions never used in
training, with routes chosen from each task's support examples by exhaustive
search. All six staged libraries (two initializations x three worlds) executed
every unseen program below the 0.05 threshold, at 0.95-1.12 times their own
trained-task loss. The two failure controls - the non-staged learner and a
library-reset arm - solved none of the 64. On every audited library the
learner's own gradient route inference matched exhaustive search exactly,
consistent with the earlier finding that routing difficulty tracks library
quality. The curriculum therefore forms a reusable vocabulary rather than a
fit to its training tasks, offline and at development scale.

# Prospective research: coordinate reuse and predictive autonomy (2026-09-14)

The next conceptual extension separates learning a reusable representation,
testing the autonomy of its transition law, and measuring its economic value.
The starting points are the learned coarse-grainings of
[McSharry et al. (NeurIPS 2024)](https://papers.nips.cc/paper_files/paper/2024/file/d8398f4da88975e2a9c62ecaa5ba267b-Paper-Conference.pdf)
and the distinct closure notions of
[Rosas et al. (arXiv:2402.09090v2)](https://arxiv.org/html/2402.09090v2).
Our proposed operationalization, not a result from either source, is recorded
in [the H28 successor research plan](../H28_CLOSURE_RESEARCH_PLAN.md).

H28's shared computation with cheap context adapters first requires a
coordinate-transfer and amortization test. Invertible coordinate changes alone
cannot establish nontrivial coarse-graining: the full state remains recoverable.
A separate nuisance-state construction would test whether a learned interface
discards distinctions while retaining task fidelity and an invariant transition
law. Calibrated macro-only versus augmented predictive probes would measure
residual dependence on microstate and realization, with functional interventions
where supported. Finite probe gaps are predictive diagnostics, not exact mutual
information or proofs of general computational closure.

The branch also proposes a parts-sensitivity control for componentwise emergence
and later tests of closure regularization and nested abstractions. Each needs
independent non-vacuity, transfer, and cost gates. A constant representation or
an expansion-only macro cannot establish autonomous computation through a
trivial closure check. A useful cached computation can still have legitimate
economic value without meeting this stronger claim. All of these proposals
remain untested in ROW; existing sealed findings, development statuses, and the
frozen SO2 protocol retain their original scope.

# H28 instrument development note (2026-09-14)

A preliminary implementation checks exact finite-state leakage examples and
the proposed analytic rotation control, with no learned encoder or probe.
A bounded metadata inventory identifies missing history/snapshot files in six
selected development terminal directories. This limits the immediate causal
promotion audit on those artifacts; it does not refute learned closure or
coordinate reuse. The record is provisional development work from uncommitted
instrument code, not additional experimental evidence. See
[the development checkpoint](../reports/h28_t0_development_20260914.md).

# H28 coordinate construction development note (2026-09-14)

An oracle-only fixture now checks exact coordinate conjugacy and its failure
controls using the existing primitive implementation. It distinguishes transport
of the whole nonlinear operator from transforming weights alone and constrains
adapters to tied linear inverses. These are provisional software/construction
checks, not evidence that a learner discovers invariant computations. No future
learning or economic endpoint was measured. A support-only adapter-discovery
pilot with a supplied oracle core is drafted but unrun. Details:
[coordinate development checkpoint](../reports/h28_coordinate_gate_development_20260914.md).

# H28 adapter pilot note (2026-09-14)

With the canonical core supplied, a restricted four-angle interface inferred
from support examples transferred to two withheld operations in two nontrivial
coordinate contexts, reaching roughly 1e-8 canonical query error versus
0.012–0.016 without an adapter. This is a provisional oracle-core opportunity
result. It does not establish that a learner discovers the core, that adapters
are economical, or that the resulting representation is closed. An initial
coordinate-frame bug was caught by the oracle control and withdrawn before the
corrected run was interpreted.

# Development result: what the curriculum costs (2026-09-14)

Descriptive arithmetic over the J1c and J1c-R reports, with no new training.
Against its matched non-staged control, staged formation uses the same
example-gradients (ratio 1.00; the plan matched them), 0.75 of the operator
applications (early stages execute shorter programs), and 0.94-1.15 of the
wall-clock time. Its price is data: 124 tasks (60 single-operator and 64
two-operator) and 47,616 examples that the target task distribution never
provides. On compute, the curriculum repays what it costs to build; whether it
pays at all is a question about task supply. In an online lifetime the
curriculum is an assumption about the arrival order of tasks. SO2 therefore
registered it as a supplied intervention rather than something the learner
discovers.

# H28 learner opportunity development note (2026-09-14)

The oracle-core adapter pilot was extended to a learner that fits its own
identity-context cores and a restricted context-shared adapter, with oracle-core,
independent observed-frame, and random-core control arms. The controls are
finite and non-vacuous: the oracle arm is near exact and the random core is far
worse. The learned state round-trips with a recorded maximum output error, and
serialized parameter bytes are reported for the shared and independent arms.
Amortized future-learning value was not measured, so the opportunity remains
unclassified. These are provisional implementation checks from development
code, not a coordinate-reuse or economic result. Details:
[learner opportunity checkpoint](../reports/h28_learner_opportunity_development_20260914.md).

# Development result: staged formation online misses the terminal criterion (SO2, 2026-09-14)

SO2 asked whether the staged formation that works offline also works under
ROW's online protocol. In that protocol tasks arrive in sequence, each example
is scored before it is trained on, and replay replaces i.i.d. resampling. It
was frozen with two amendments before any code existed. Amendment 1 made its
export margin G5R's statistic verbatim, after the first draft's log10 version
proved not comparable to G5R's threshold. Amendment 2 let the novel-composition
diagnostic report "unavailable" in a stage whose tasks use every program of
their length.

The first launch failed on a key-name error before writing any cell. A full
re-read before relaunch found a more consequential construct error. The runner
classified on END-OF-TASK error (each task's error measured right after that
task trained, while the shared library was still changing), but the plan, the
research program, and the committed independent scorer all specified the
TERMINAL model. The corrected runner scores the terminal model and adds an
exact last-task anchor: nothing trains after the last task, so its terminal and
end-of-task errors must agree. That anchor passed at every stage of every cell.
Two numbers from the failed launch had been seen before this fix, and the record
discloses them.

| world | staged terminal | staged end-of-task | G5R margin | unseen programs <= 0.05 | plain terminal |
|---|---:|---:|---:|---:|---:|
| 0 | **0.019** | 0.065 | +4.97 | 63/64 | 1.96 |
| 1 | 0.126 | 0.077 | +2.74 | 5/64 | 1.97 |
| 2 | 0.085 | 0.035 | +3.06 | 12/64 | 1.91 |

The terminal clause (median NMSE <= 0.05 in two of three worlds) held in one
world. The export clause (margin >= 0.75) held in all three, against G5R's
+0.12 to +0.29. The registered label is therefore `SO2_FAILS`, not "acquires
only". The non-staged online learner failed everywhere. One registered
prediction was wrong: the staged stream was expected to cost more prequentially,
but it cost about half as much (6.7-9.9M versus 13.9-14.2M nats) despite scoring
nearly three times the examples. The margin passing where terminal quality
failed also shows how weak G5R's scratch comparator is (geometric-mean NMSE
2.2-2.7). The 64-program export diagnostic tracks terminal quality more closely.

# Development localization: the online loss is post-acquisition and recency-weighted (2026-09-15)

A descriptive census over the frozen SO2 artifacts, with no training, localized
the failure. Its plan and code were committed before any number existed, and its
consistency guards passed.
- **Only one stage degrades.** Terminal error exceeds end-of-task error only in
  the final (length-3) stage of the two failing worlds. Every other stage in
  every world shows the opposite: tasks keep improving after they are learned.
- **The newest tasks lose most.** Within that stage the loss grows with arrival
  position (Spearman +0.56 and +0.64). Recently learned tasks reach their lowest
  error and then lose it, while terminal error is flat across positions.
- **Earlier computation survives.** Length-1 task codes run through the final
  library solve 51-54 of 60 tasks.
- **The failing worlds' libraries move twice as far** in the final stage (median
  per-slot functional drift 0.52-0.58 versus 0.25).

# Development exploration: a stage-3 consolidation setting rescues one world (SO2-P, Tier 1, 2026-09-15)

An exploratory, one-world check (world 1, one replay stream per arm) continued
stage 3 from SO2's saved stage-2 library and changed one setting per arm. It
first reproduced SO2's stage 3 bit for bit. It is not a verdict.

| stage-3 arm | terminal (tasks <= 0.05) | lost / gained after learning | drift |
|---|---:|---:|---:|
| baseline | 0.126 (6) | 21 / 3 | 0.52 |
| shared LR x1/2 | 0.018 (33) | 0 / 12 | 0.28 |
| shared LR x1/4 | 0.113 (16) | 0 / 0 | 0.23 |
| shared LR x1/10 | 0.090 (15) | 0 / 0 | 0.17 |
| 8 stored replay examples per task | 0.028 (53) | 1 / 30 | 0.35 |
| 16 stored per task | 0.090 (18) | 17 / 8 | 0.54 |

The registered triage returned LIVE, but the pattern rules out the simple
account that motivated it. Lower learning rates cut drift and eliminated the
losses, but the two lowest also eliminated the gains. The best arm has more
drift than any learning-rate arm, and the 16-example arm fails at baseline-level
drift. Acquisition is nearly identical across arms. The working hypothesis is
now a balance between destructive and consolidating library movement after
learning, not the magnitude of that movement.

A correction was appended after the result was committed.
`replay_examples_per_task` sets how many examples of each finished task the
buffer STORES; every update in every arm replays one example. The replay arms
were therefore gradient-matched, contrary to the first write-up. But the
buffer's storage and sampling share one random generator, so those arms also ran
a different replay stream. Their gain is confounded with stream luck.

# Frozen, not yet run: SO3 stage-3 consolidation (2026-09-15)

SO3 is a preregistered test of the two candidate settings. It runs on
development worlds unused by SO2 (3-5), at a fresh model seed, through all three
stages, with three replay streams per arm. The baseline's extra streams serve as
a stream-only control, and the replay setting counts only if it beats the best
baseline stream in two of three worlds. A default-off `replay_seed` option was
added to the lifetime runner, gated on reproducing SO2's stage 3 exactly.
Stages 1-2 are shared across arms, gated on matching a full recomputation. SO3
does not compute G5R's margin, so a pass would not be an SO2 pass. The plan is
frozen and hashed; no SO3 result exists at the time of writing.

# Development result: the online staged protocol passes on fresh worlds; the stage-3 rescue does not replicate (SO3, 2026-09-15)

SO3 was a preregistered test of the two stage-3 settings suggested by the
one-world exploration. It ran on development worlds unused by SO2 (3-5), at a
fresh learner seed, through all three stages, with three replay streams per
arm. The baseline's extra streams served as a stream-only control. Every
registered gate passed, including exact reproduction of SO2's stage 3 when the
new stream option is omitted. The independent scorer reproduced the label.

| world | unchanged protocol | halved stage-3 shared LR | 8 stored replay examples |
|---|---:|---:|---:|
| 3 | 0.028 | 0.026 | 0.032 |
| 4 | 0.014 | 0.019 | 0.047 |
| 5 | 0.012 | 0.021 | 0.011 |

Entries are world medians of terminal NMSE over three replay streams.

Neither candidate improved on the unchanged protocol in two of three worlds,
so the registered label is `SO3_FAILS`: the rescue did not replicate. The
larger finding is the baseline. SO2's unchanged online staged protocol passed
the terminal criterion in all three fresh worlds, with median 52-58 of 64
tasks below 0.05. Its terminal error varied by up to a factor of seven across
replay streams within a single world (0.009-0.066). SO2's two failing worlds
were single-stream measurements on a quantity with that spread.

Two registered predictions were wrong. The baseline was expected to fail
again, and the candidates were expected to lose fewer tasks after learning;
on fresh worlds almost no task was lost after learning in any arm. SO3 did not
compute G5R's held-out margin and did not register the baseline as a gate, so
it does not establish online learnability of the rotated substrate. It does
show that SO2's negative is not a stable property of the protocol. A registered
multi-stream re-test of the unchanged protocol, including the margin, is the
appropriate next gate and has not been run.

# Development result: online staged formation is world-dependent (SO4, 2026-09-16)

SO4 was the preregistered online gate for the strong (rotated) substrate, run
with its stream variance measured for the first time. It used the staged
protocol unchanged, on the four development worlds no earlier rung had touched,
at a fresh learner seed, with three replay streams per world and a non-staged
control. Its held-out margin used the earlier construction verbatim on each
world's pre-specified first stream. Every registered gate passed, including
exact reproduction of the earlier run's third stage when the new stream option
is omitted, and an independent scorer reproduced the label from the saved
models.

| world | stream terminals | world median | held-out margin | non-staged |
|---|---|---:|---:|---:|
| 6 | 2.24 / 2.17 / 0.20 | 2.17 | **-0.08** | 1.89 |
| 7 | 0.064 / 0.010 / 0.013 | **0.013** | +3.78 | 1.99 |
| 8 | 0.016 / 0.173 / 0.091 | 0.091 | +5.17 | 1.88 |
| 9 | 0.020 / 0.014 / 0.016 | **0.016** | +5.10 | 2.05 |

The registered criterion needed three of four worlds below 0.05; two passed, so
the label is `SO4_FAILS`. The margin clause passed in three of four.

Read with the two earlier runs, the picture is not a protocol that fails but one
whose success depends on the world. Across ten development worlds at three
learner seeds, the same online staged protocol reached the terminal criterion in
six: one of three, then three of three, then two of four. Within a single world,
changing only the replay sampling stream moved terminal error by up to an order
of magnitude (world 6: 0.20 to 2.24), and the stage-3 consolidation settings that
rescued one world in an exploratory check did not improve on the unchanged
protocol anywhere. World 6 is the clearest failure: its earlier stages were
already poor, its library ends no better than a from-scratch learner on held-out
programs, and it executes none of the 64 unseen programs.

Where a stream fails it does so after learning rather than during it: one world-8
stream reached 0.037 at the end of each task but 0.173 on the final model, losing
33 of 64 tasks. Where a stream succeeds, tasks keep improving after their own
training ends. The non-staged control fails in all four worlds, so staging
remains responsible for whatever is learned.

This paper therefore makes no claim that the rotated substrate is learnable
online. What the three runs establish is narrower and, we think, more useful: the
quantity such a claim depends on has a within-world spread comparable to the
threshold itself, so any single-stream, single-world measurement of it - including
the first of these three runs - cannot settle it. Development worlds 0-9 are now
spent for this protocol. The open question is what distinguishes an acquiring
world from a failing one, and it is answerable on the existing artifacts before
any further training.

# Development result: three candidate mechanisms for the world-dependence, all withdrawn (2026-09-16)

The online result above is world-dependent. Three descriptive censuses then asked
what distinguishes a world or a replay stream that acquires the substrate from one
that does not. All three ran on the twenty-four staged cells the three online runs
had already produced: no training, no new world, and each plan with its code and
tests committed before any number existed.

- **Prefix error.** Error at the end of the second curriculum stage correlates with
  the third stage's outcome across cells (Spearman +0.67), but separates failing
  from passing cells by only 1.66x, below the 2x the plan registered. Asked the
  sharper question - within a single world, does the worst-prefix stream fail? - it
  scores one of seven, worse than chance.
- **Slot duplication.** The hypothesis that a low-error library can still be a poor
  vocabulary because its slots do nearly the same thing. The measure is flat
  (+0.10, separation 1.01).
- **Route identifiability.** How sharply the second stage's support data picks out
  one route. This is the only measure that tracked the outcome (-0.51), and it
  ordered the one cell the other measures could not explain. But it mixes the
  library with its task set, and spans nearly three hundredfold across worlds. Made
  comparable - by generating targets from the library's own routes, with a gate
  confirming the generating route is always recovered - it is null and points the
  other way (+0.14). Within-world normalization drops the original correlation from
  -0.51 to -0.22, which is what one expects if most of it was scale.

Two cells refuse any monotone account on these measures: the library with the
highest intrinsic identifiability fails completely, and one of the lowest passes.

We therefore report the online result as world- and stream-dependent with no
identified mechanism, and we say so rather than offering the one surviving
correlation as an explanation. The measures that looked explanatory pooled across
worlds did not survive being made comparable, and the sequence is recorded here
because the failure mode - a statistic that carries scale rather than signal - is
the same one this project has met before, and the cheap tests that exposed it cost
minutes on artifacts that already existed.

# Development instrument check: fixed-library route inference (2026-09-17)

A descriptive preflight on the twelve existing J2A libraries checked whether
reducing support from 128 to 32 or 8 examples changed exhaustive support-only
program inference at depth three. All 96 staged-library/task pairs retained
exactly the same hard route and query error (pooled median NMSE 0.00572;
all below 0.05). These reuse 48 programs across three development worlds at
two model initializations. The 96 non-staged/reset control pairs stayed above
0.05 at every support size. The result validates the artifact anchors but
provides no observed hard-route commitment cost on the staged libraries over
this evidence range. It is not a comparison of beam, posterior or full-budget
gradient inference and does not resolve PX7. The larger inference census stays
unfrozen pending a bounded opportunity check. Raw measurements, derivation and
validation are in reports/l0d_preflight.json and reports/l0d_preflight_2026-09-17/.

# Depth-four frozen-vocabulary execution gate (2026-09-17)

On one frozen STAGED5000/world-0 vocabulary, support-only exhaustive route
search over 16 deterministic length-four programs reached query NMSE <= 0.05
on all 16 (median 0.00872). This is a scoped frozen-vocabulary usability check;
it does not establish a learned inference mechanism or a PX7 result. Beam,
posterior, commit-late and depth-five measurements remain unrun.

# Development result: on a formed vocabulary, program inference is not ambiguous (SG0, SG6, 2026-09-22)

A depth-five gate first confirmed that exhaustive support-only route search over
the frozen vocabulary is feasible and exact there too. It covers 248,832 routes
in 243 memory-bounded blocks, gives support MSE bitwise equal to the direct
evaluator, and reaches query NMSE 0.0134 on its registered task. SG0 then asked
the question a learned program proposer needs answered first: does committing
to the support-optimal route ever cost anything on the usable staged libraries?
The criterion, registered before any data existed, allowed at most two headroom
cells among 36. The result was zero:
- 574 of 576 staged tasks carry exactly zero commitment regret.
- No staged program at depths three or four has any rival route within 1% of
  its winner on support. Median identifiability is 53.9: the runner-up is about
  54 times worse.
- A 200-resample bootstrap never changed a selected route.

The same code path on control libraries, which failed to form, finds 500 of 576
routes differing, regret up to 0.663, and 142 non-trivial near-ties. So the
instrument can detect ambiguity, and it found none where the vocabulary is good.

On this substrate, then, the difficulty of inferring a program is a property of a
library that has not finished forming, not an independent problem that a
proposer could be trained to solve. A follow-up that would have correlated
inference difficulty with vocabulary quality was stopped before a plan existed.
Quality across the twelve held libraries is bimodal: staged 0.0047-0.0073,
control 1.26-1.30, a gap 30 times the wider cluster's spread. The pooled rank
correlation (-0.69) therefore re-detects cluster membership, and within each
cluster it is null and of the wrong sign. We report the amortized-proposer
branch on this substrate as closed. We make no claim about substrates whose
formed vocabularies do carry ambiguity, which these data cannot sample.

# Methods note: three adequacy gates (2026-09-22)

Many rungs in this programme failed for reasons that had one name but three
different fixes. Every plan now passes three separately checked gates before it
is frozen:
- NECESSITY: does the task require the behaviour under study? The fix is to
  change the task.
- OPPORTUNITY: could the generator produce the effect at all? The fix is to
  change the generator.
- DISCRIMINATION: could this instrument, on this sample, come out either way?
  The fix is to change the sample or the statistic.

The discrimination check runs the registered decision rule on simulated null and
effect data and requires a false-fire rate at most 5% and a detection rate at
least 80%. Reconstructions of two earlier defects fail it:
- a first-crossing "horizon" statistic, which fires on 42.5% of pure-noise
  series;
- a macro threshold of 1.5 uses, which nearly every pattern clears.

Most earlier negatives in this paper are necessity failures: the task did not
require the behaviour. They are worded as statements about the construction,
not about the behaviour (claims audit, `reports/claims_audit_a2_2026-09-24.md`).

# Development result: what makes staged formation work is the presence of single-operation tasks, not their order (N1, N1b, N1c, 2026-09-22/23)

The length curriculum (J1c) confounded two things: presenting short programs
FIRST, and presenting short programs AT ALL. Three offline rungs on development
worlds 0-2 separated them. All ran at a matched 65,536-update budget with no
route ever revealed.

| rung | arm | terminal median NMSE, worlds 0/1/2 |
|---|---|---|
| N1 | curriculum (short first) | 0.0062 / 0.0051 / 0.0073 |
| N1 | 124 anchors pooled with the length-3 tasks, no order, length never revealed | **0.0093 / 0.0101 / 0.0066** |
| N1 | pool-matched sham (same count, length-3 fillers) | 1.04 / 1.06 / 1.09 |
| N1b | 60 length-1 anchors only | **0.0175 / 0.0112 / 0.0067** |
| N1b | 64 length-2 anchors only | 1.16 / 1.12 / 1.14 |
| N1b | 8 mixed anchors | 1.09 / 1.11 / 1.04 |
| N1b | 32 mixed anchors | **0.0141 / 0.0076 / 0.0212** |
| N1c | 6 length-1 anchors covering all 6 operations | 1.04 / 1.09 / 1.05 |
| N1c | 18 length-1 anchors covering 5 of 6 operations | 0.199 / **0.0129 / 0.0259** |

Three conclusions follow for this substrate and budget.
- **Order is not the active ingredient; presence is.** Anchors pooled with the
  length-3 tasks, with no stage order at all, work as well as the curriculum. The sham, with the same pool and verifiably
  identical minibatch draws, is worse than no anchors at all.
- **Only single-operation tasks matter.** Length-2 tasks do nothing on their
  own, even though an untrained library already clusters them above chance.
  Clearing chance was the wrong threshold.
- **What counts is how many anchors, not coverage.** A post-hoc pattern in
  N1b, that passing cells covered every operation, did not survive a design
  that held count fixed. Six anchors fail with full coverage. Eighteen
  covering five operations pass in two worlds, where the library forms the
  sixth operation from compositions. In the third world the failure is local
  to tasks using the missing operation (0.025 on the other tasks, 1.13 on
  those).

An online learner needs a quota of single-operation tasks, somewhere between 6
and about 18 in a 188-task stream here. It does not need a curriculum, which
would require knowing program length.

# Development exploration: order-free anchors online (O1, Tier 1, 2026-09-24)

O1 carried the offline finding online on fresh worlds 10-12, with one replay
stream per world. It is exploratory, and its job was to decide whether the
question is live. The arms:
- one random-order online lifetime over the same 188 tasks the staged protocol
  uses (`SHUFFLED`);
- the length-1 quota plus the canonical tasks, with no length-2 tasks
  (`MIXED_L1`);
- the staged protocol itself;
- a no-anchor control.

| arm | w10 | w11 | w12 |
|---|---:|---:|---:|
| staged protocol | 0.272 | **0.0269** | **0.0188** |
| `SHUFFLED` | **0.0366** | 0.473 | **0.0184** |
| `MIXED_L1` | 0.191 | 0.149 | **0.0358** |
| no anchors | 2.01 | 1.92 | 1.91 |

The order-free stream passed as often as the curriculum, two of three, but in a
different pair of worlds. Order is not needed online, and it did not make
formation reliable either.

Without length-2 tasks, the online learner passed only one world, where offline
it passed all three. We read that as an indication, not a finding, because it
rests on one stream.

In the single-lifetime arms the canonical tasks are fitted poorly when first
seen: end-of-task error on those 64 tasks is 17-39 times the terminal error in
passing cells. They become usable only after the library forms later in the
stream. This corrects a first reading that took the end-of-task median over all
stream tasks, including the anchors (`reports/o1_online_anchor_20260924/stream_audit.json`).

The reliability question is registered as O2: worlds 13-19, three streams each,
a k-of-21 rule, and its result reported below when it exists.

# Development result: online formation is unreliable with or without a curriculum (O2, 2026-09-26)

O2 asked the reliability question directly. It ran on seven fresh worlds (13-19)
with three replay streams each and a registered rule: an arm is reliable only if
at least 19 of its 21 cells reach terminal NMSE below 0.05. At the measured
incumbent rate the rule fires by mistake at most 2%, and it detects a 95%-reliable
mechanism at least 84% of the time. Before launch, equivalence gates reproduced
O1's committed cells bitwise.

| world | `SHUFFLED` s0 / s1 / s2 | `STAGED` s0 / s1 / s2 | `MIXED_L1` s0 / s1 / s2 | `PLAIN` s0 |
|---|---|---|---|---:|
| 13 | **0.049** / **0.026** / 0.172 | 0.163 / **0.025** / **0.020** | **0.048** / 0.073 / 0.891 | 1.95 |
| 14 | 2.00 / **0.027** / 0.306 | 0.183 / 2.39 / 2.03 | 1.31 / 0.857 / 0.702 | 1.96 |
| 15 | **0.026** / 0.076 / 0.118 | **0.036** / 0.143 / **0.028** | **0.044** / 0.353 / 0.116 | 1.88 |
| 16 | 0.061 / **0.040** / **0.045** | 0.077 / 2.25 / 0.136 | 0.056 / 1.81 / **0.034** | 1.98 |
| 17 | 0.227 / **0.022** / **0.042** | **0.027** / **0.041** / **0.012** | 0.093 / 0.064 / 1.06 | 1.85 |
| 18 | 0.151 / 0.069 / 0.149 | **0.023** / **0.025** / **0.016** | 0.184 / 2.04 / 1.02 | 1.93 |
| 19 | 0.185 / 0.449 / **0.036** | 0.059 / **0.038** / **0.027** | **0.027** / 1.65 / 0.481 | 1.92 |

Passing cells: 9 of 21 for the order-free stream, 12 for the curriculum, 4 for
the stream without length-2 tasks, and 0 of 7 for the no-anchor control. None of
the three is reliable. The curriculum's 12/21 matches its historical cell rate
(0.58) on fresh worlds, so O1's apparent order-free success was one draw from the
same unreliable regime. The two main arms fail differently:
- The order-free stream almost never collapses, but often lands just above the
  threshold. Its canonical tasks are fitted late in the stream.
- The curriculum either succeeds or collapses completely.

Without length-2 tasks, online formation is worse still. Offline it was not
(N1b), so the online learner uses those tasks in a way the offline one did not.

We therefore make no claim that the rotated substrate is learnable online by any
protocol tested here. What the offline and online results together establish is
narrower. The library forms from single-operation anchors in any order. That is
reliable offline and only about 50% reliable online at this budget, so the
online deficit is one of reliability, not of possibility.

# Development exploration: the online deficit is convergence, and a replay-only sleep phase recovers much of it (O2C, 2026-09-26)

O2's order-free failures were mostly near-misses, with canonical tasks still
improving when the stream ended. O2C took the 21 saved terminal models and
added 8,192 further updates (a third of the stream's budget), under two regimes:
- updates on all of the data the learner had seen, an opportunity ceiling that
  no online learner has;
- updates on only its end-of-stream replay buffer of four examples per task,
  which an online learner does keep.

A gate reproduced every saved terminal exactly. The buffer was rebuilt from the
learner's own call sequence and matched a spy on a real lifetime item for item.

| cell | O2 terminal | + ORACLE_DATA | + REPLAY_ONLY |
|---|---:|---:|---:|
| w13 s0 | 0.049 | 0.023 | 0.034 |
| w13 s1 | 0.026 | 0.029 | 0.015 |
| w13 s2 | 0.172 | 0.021 | 0.043 |
| w14 s0 | 2.003 | 1.456 | 2.156 |
| w14 s1 | 0.027 | 0.029 | 0.029 |
| w14 s2 | 0.306 | 0.029 | 0.043 |
| w15 s0 | 0.026 | 0.021 | 0.018 |
| w15 s1 | 0.076 | 0.026 | 0.039 |
| w15 s2 | 0.118 | 0.030 | 0.128 |
| w16 s0 | 0.061 | 0.026 | 0.036 |
| w16 s1 | 0.040 | 0.026 | 0.023 |
| w16 s2 | 0.045 | 0.022 | 0.024 |
| w17 s0 | 0.227 | 0.050 | 0.064 |
| w17 s1 | 0.022 | 0.017 | 0.018 |
| w17 s2 | 0.042 | 0.029 | 0.024 |
| w18 s0 | 0.151 | 0.017 | 0.169 |
| w18 s1 | 0.069 | 0.020 | 0.032 |
| w18 s2 | 0.149 | 0.021 | 0.016 |
| w19 s0 | 0.185 | 0.033 | 0.259 |
| w19 s1 | 0.449 | 0.047 | 0.405 |
| w19 s2 | 0.036 | 0.015 | 0.017 |

With full data, 20 of 21 cells pass, and none of the nine that already passed
breaks. The one survivor had collapsed during the stream. With only the replay
buffer, 15 of 21 pass, again breaking none. So in this construction the online
deficit is mostly one of convergence: the library the stream forms is nearly
right. Much of the remaining gap is what the learner retains, not how its tasks
were ordered. These are exploratory results on development worlds already used
by O2, and a confirmatory test needs fresh worlds.

# Development exploration: how much to retain, which protocols sleep helps, and what a collapse is (O2D, O2E, O2G, 2026-09-26)

Three follow-ups on the same saved O2 models asked what a practical sleep phase
needs and what it cannot fix. All three are exploratory on development worlds
already used by O2.

**Retention (O2D).** Keeping a reservoir of 4, 16 or 64 training examples per
task for the sleep phase gave 18, 18 and 20 of 21 passing cells. At 64 per task
it matched the ceiling set by consolidating on all the data. At 4 per task the
lifetime's own replay buffer had given 15, so near-threshold cells depend on
which few examples are kept. No arm broke a cell that was already passing.

**Other protocols (O2E).** The same sleep phase repaired every near-miss of the
curriculum, which went from 12 to 18 of 21. It also repaired every near-miss of
the stream without length-2 tasks, which went from 4 to 13. It broke none, and
it rescued only one of ten collapsed runs across the three protocols. After
sleep, each protocol fails only by collapse. The protocols then differ mainly in
how often they collapse: 1, 3 and 6 of 21.

**Collapse (O2G).** The curriculum's collapses happen at a single moment: the
first length-3 tasks after a healthy length-2 stage. Re-running that last stage
from the saved stage-2 library reproduced the original collapse bitwise under
its original seed. Under two fresh seeds:
- The early disruption recurred every time from a collapse-prone library (6 of
  6), and only 3 times in 36 from the others.
- It led to collapse in 3 of those 6. Healthy libraries collapsed once in 36.

A collapse is therefore a risk carried by the library, realised about half the
time, and preceded by a disruption visible within the first few tasks. The
online learner's two failure modes thus have different remedies. A sleep phase
over retained examples repairs under-convergence. A collapse needs detection
and retry, which remains to be tested on fresh worlds.
