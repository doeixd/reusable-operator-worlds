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
produces (`reports/h49_discoverability.json`). The value of a structure
is a property of the representation that holds it, not of the task
population. The obvious repair — propose a grouping, reorganize the
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
