# When Does Abstraction Pay? Measuring the Value of Reusable Computation in Neural Learners

*Draft v0.4 — rebalances the alignment dependence into the abstract and
contributions, adds the r* caution and the recovery-coincidence hedge,
and connects the scale-free criterion to scaled analogues in the
discussion.
Development and confirmatory results labeled throughout; all numbers
trace to fingerprint-validated artifacts in the public repository.*

## Abstract

We ask when a neural learner should represent computation as reusable
structure rather than task-specific parameters. We introduce Reusable
Operator Worlds (ROW), a benchmark in which the amount of latent
computational recurrence across tasks is continuously controllable and
directly measurable, and the learner is scored by cumulative prequential
(predict-before-update) cost over a 64-task lifetime. In 30 sealed
confirmatory worlds — run under a protocol pre-specified in the public
repository before any sealed world was generated — a reusable operator
basis consistently beats a compute-matched dense learner under high
recurrence and consistently loses under low recurrence (30/30 worlds on
all three pre-specified outcomes; Holm-adjusted p <= 5.6e-9). The paired
advantage is approximately linear in measured functional recurrence
(R^2 = 0.935), crossing zero near recurrence r = 0.50. The result is
conditional on representational alignment, and substantially so: when the
learner's operator family matches the environment's (residual tanh), the
exact-reuse advantage is large; changing only the learner's activation
family collapses it roughly tenfold. The supportable claim is therefore:
**when a learner's representational vocabulary can efficiently express
the environment's recurring computation, the value of using that
vocabulary is linear in measured recurrence.** Development-stage
mechanistic analyses further show that lower lifetime learning cost
appears before identifiable, recomposable primitives do: **statistical
reuse and structural abstraction are distinct phenomena**. The transfer
advantage is acquired over the lifetime rather than present at
initialization, and the effect survives changes in task order, replay
budget, initialization, task-code capacity, batch size, and quantization.
These results provide a controlled measurement of the economics of neural
abstraction: when shared computation pays, on what its value depends, and
what kind of reuse a learner actually acquires. ROW is deliberately small
— the generative programs are known exactly — and no claim in this paper
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
   collapses it roughly tenfold), and we state every claim under that
   condition.
3. The statistical-reuse / structural-abstraction dissociation: lifetime
   economic benefit precedes — and does not imply — identifiable,
   recomposable primitives, which emerge only near exact recurrence.
4. A measured resource frontier (online learning vs retained description
   vs inference compute) across five substrate families under actual int8
   quantization, plus a proof-of-concept adaptive-sharing substrate whose
   gains and costs sharpen the target for learned consolidation.
5. One falsified secondary hypothesis and two weakened ones, reported.

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

The central mechanistic finding, displayed because it is the paper's
one-line takeaway:

    R_stat > 0   does NOT imply   R_struct > 0

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

## 6. Alternative explanations

### 6.1 Family alignment (the circularity question)

The teacher and the continuous learner share a functional family
(residual bottleneck blocks), so the strongest objection to this study is
that it measures inductive-bias match rather than reuse. In stages:
removing all parameter coupling (independently initialized, learnable
residual scale) leaves the world-0 advantage essentially unchanged
(+4,446 nats); doubling teacher rank against fixed learner rank preserves
most of it (+3,296); changing the learner's activation family (GELU vs
tanh) collapses it roughly tenfold, to +441 — still positive, but small.
The precise conclusion: **this experiment does not show that generic
neural learners spontaneously discover arbitrary reusable computations.
It shows that when a learner's representational vocabulary can
efficiently express the recurrent structure of its environment, the
economic value of using that vocabulary is predictable from functional
recurrence.** Put economically: the observed advantage is a function not
of recurrence alone but of something like Delta = f(r, A, C) — where A is
the alignment between the learner's representational vocabulary and the
environment's recurring transforms, and C the substrate's cost — and the
GELU result shows the A-dependence is large. Whether family mismatch
shifts the crossover location rather than only the magnitude is a
pre-specified follow-up (V2, H6); whether the learner can acquire the
vocabulary itself is the V2 program.

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
discards. The premature-commitment hypothesis this suggests is
pre-specified for V2 (H7), where the route space is small enough to
compute the exact posterior.

## 7. Resource economics and adaptive sharing

**The frontier (Figure 5).** Evaluated int8 retention: Discrete 26,208
bits and ~768 inference multiply-adds; Continuous 29,248 / 6,528;
Hypernetwork 33,928 / 7,296; Dense-24 56,448 / 5,376; Dense-C 66,688 /
6,144. Retention, online-learning, and execution orderings all disagree:
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
sharing. Three fairness notes accompany the large margins: the residual
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

**A characterized negative, briefly.** L0-style presence gating on the
12-slot library, tuned over a two-stage grid with a selection rule frozen
in advance, never produced a compact sufficient library: pruning pressure
is a cliff (nothing, or collapse to one slot), and the one intermediate
regime found (7 active slots; the teacher has 6) fails novel-composition
sufficiency — with the degradation tracking gate pressure, not slot
count. Penalizing structure during acquisition damages generality before
it induces useful compression, motivating learn-first, consolidate-later.
Full grids in the appendix.

## 8. Related work

**Prequential evaluation.** Our headline metric operationalizes
prequential MDL (Dawid, 1984; Bornschein, Li & Hutter, 2022) as a
lifetime learning-cost measure with fixed likelihood, extending its use
from model comparison to a controlled intervention study of
representation choice.

**Modular and reusable architectures.** Modular meta-learning (Alet et
al., 2018), neural module networks (Andreas et al., 2016), RIMs (Goyal et
al., 2021), and modular continual learning (e.g., Veniat et al., 2021;
Ostapenko et al., 2021) build substrates for reuse; mixture-of-experts
and soft merging (Shazeer et al., 2017; SMEAR, Muqeeth et al., 2023)
study routing trainability — SMEAR in particular anticipates our finding
that hard-routing failure is not evidence against reusable computation.
Hypernetworks and attention-as-hypernetwork (Ha et al., 2017; Schug et
al., 2024) motivate our continuous-manifold control. **The missing axis
in this literature, which ROW supplies, is intervention: prior work
compares methods at fixed or naturally occurring task relatedness,
whereas ROW directly manipulates ground-truth functional recurrence and
measures the resulting sign and magnitude of the representation
preference.**

**Library learning and program induction.** The wake/sleep library-
learning lineage (DreamCoder, Ellis et al., 2021; and successors) is the
closest program-level relative. We differ in criterion (lifetime
prequential cost rather than task solve-rate), in operating over learned
neural operators whose correspondence to ground truth is measurable, and
in the negative-control discipline our V2 consolidation program inherits
(a compressor must decline to compress structureless worlds).

**Transfer and task relatedness.** Negative transfer and task-similarity
analyses (e.g., Standley et al., 2020; Zamir et al., 2018) document that
sharing can hurt; multi-task representation-learning theory predicts
benefits scaling with shared structure. ROW contributes a setting where
relatedness is a knob rather than an estimate, and a second novelty:
**distinguishing performance-level sharing from recovery of the actual
reusable computational factors** — compositional-generalization studies
reporting modular successes that fail recomposition are consistent with
our statistical/structural dissociation, of which ROW provides a
controlled dose-response version.

*(Citations to be completed; the list above fixes the positioning.)*

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
- Lifetime-length dependence: a truncated-lifetime analysis (appendix)
  reads the crossover at 16, 32, and 64 tasks from the per-example logs;
  the mean crossing moves from configured rho 0.869 to 0.822 between 16
  and 32 tasks and then saturates (0.826 at 64; 8/10 development worlds
  decline overall). Amortization is therefore a demonstrated
  early-lifetime component of the crossover, not its whole mechanism,
  and we do not claim a confirmed lifetime-length law.

## 10. Discussion: toward learned abstraction economics

Sharing is not inherently good. It is an investment: a learner pays a
representational price by forcing tasks through shared computation, and
the return on that investment is determined — linearly, in our worlds —
by how much genuinely reusable computation the environment contains. Even
when the investment pays predictively, the learner may not yet have
discovered a clean abstraction: economic benefit precedes recomposable
structure, and only near exact recurrence do lifetime efficiency,
recomposition, and identifiable primitive recovery align.

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

---
*Figures: (1) regime map, per-world sealed traces, both coordinates;
(2) linear dose-response with pooled fit (confirmatory); (3) checkpoint
divergence — indistinguishable at 8 tasks, ~2x at 64; (4) two response
curves — lifetime advantage, frozen recomposition advantage, and
operator recovery vs untrained baseline on one measured-recurrence axis;
(5) resource frontier; (6) robustness forest; (7) adaptive substrate:
envelope win in nats, loss in bits, allocation signature. Appendices:
MDL gating grids; batch, initialization, and truncated-lifetime tables;
mixed-effects sensitivity for the pooled regression; confirmatory
analysis exactly as pre-specified.*
