# When Does Abstraction Pay? A Preregistered Study of Reusable Computation in Neural Lifetime Learning

*Draft v0.1 — development and confirmatory results as labeled; all numbers
trace to fingerprint-validated artifacts in the public repository.*

## Abstract

We ask when a neural learner benefits from an explicitly reusable
computational substrate, using a benchmark — Reusable Operator Worlds (ROW)
— in which the answer can be checked against ground truth. Each world hides
a small library of randomly generated neural operators; every task is a
composition of them; a continuous knob `rho` controls how much genuine
computational recurrence exists across tasks; and the learner is scored by
cumulative prequential (predict-before-update) cost over a 64-task
lifetime. On 30 sealed worlds opened only after configurations, metrics,
and analysis were frozen, all three preregistered primary outcomes passed
in 30 of 30 worlds (Holm-adjusted p <= 5.6e-9): a reusable operator basis
beats a compute-matched dense learner at exact reuse; the advantage grows
with measured functional recurrence; and the preference reverses within
every world, with dense learning better below a crossover at measured
recurrence r* = 0.499 +/- 0.050 and the reusable basis better above it.
The effect is a smooth linear dose-response in measured recurrence
(R^2 = 0.935), not a threshold. Development-stage analyses locate the
mechanism: the reusable learner's advantage at novel-composition transfer
is absent after 8 tasks and grows to ~2x by 64 (learning-to-learn is
acquired, not architectural); identifiable recovery of the hidden
primitives begins only at the performance crossover and crystallizes only
at exact reuse, dissociating statistical parameter sharing from structural,
recomposable abstraction; and the advantage is invariant to task order,
replay budget (including none), initialization, task-code size, batch
size, and quantization. We report one falsified secondary hypothesis and
two weakened ones. ROW is small by construction — its contribution is a
controlled measurement of the economics of abstraction: when sharing pays,
in what currency, and what kind of reuse is actually acquired.

## 1. Introduction

Whether neural learners should share computation across tasks is usually
argued by benchmark anecdote: modular architectures win some transfer
suites and lose others; negative transfer appears and disappears with
tuning. The underlying question — under what conditions does a reusable
internal representation reduce the cost of future learning? — is rarely
posed in a setting where the true amount of reusable structure is known
and adjustable.

We built such a setting. ROW generates worlds from a hidden library of
K = 6 random residual operators on a 16-dimensional state; each task is a
hidden length-3 composition; task identifiers are opaque; and a parameter
`rho` interpolates the per-task operators between exact sharing (rho = 1)
and task-independence (rho = 0), with the realized functional recurrence
measured directly on probe inputs. The learner never sees routes,
primitive identities, or `rho`. The primary metric is lifetime prequential
cost: every online example is scored before the model updates on it, so
the measure rewards a learner that becomes able to predict future data
from fewer examples — Chollet-style skill-acquisition efficiency, made
exact.

The deliberate smallness is the instrument, not a limitation to be
excused: because the generative programs are known, we can run an oracle
positive control, verify that task difficulty is flat over the lifetime,
measure functional recurrence rather than assuming it, and test whether
learned operators actually correspond to the hidden ones. None of our
claims concern scale; the criterion we test is scale-free (Section 7).

**Contributions.**
1. A benchmark and validity-control battery for lifetime reuse economics,
   with a development/confirmation seed firewall and a frozen,
   preregistered confirmatory protocol.
2. A confirmed regime map: dense task-specific learning wins when latent
   recurrence is weak; an explicit reusable basis wins when it is strong;
   the crossover location is reproducible across worlds to +/-3% and the
   effect is linear in measured recurrence.
3. A mechanism account: the reusable advantage is acquired over the
   lifetime from an equal start, and statistical sharing (which lowers
   lifetime cost) dissociates from structural abstraction (identifiable,
   recomposable primitives), which appears only near exact recurrence.
4. A measured resource frontier (online learning vs retained description
   vs inference compute) across five substrate families, under actual
   int8 quantization rather than parameter-count proxies.
5. A falsified secondary hypothesis and two weakened ones, reported.

## 2. The benchmark

**Worlds.** Each primitive is P_k(z) = tanh(z + alpha * U_k tanh(V_k z +
b_k)) with spectral-normalized U, V (d = 16, rank 8, alpha = 0.35). A
world samples 6 primitives and 64 unique length-3 programs; task IDs are
random tokens carrying no route information; train (128/task) and
evaluation (256/task) sets are fixed at generation. The reuse continuum
mixes per-task perturbations into each primitive with weight
sqrt(1 - rho^2) followed by re-normalization; measured pairwise
residual-function correlation on probes validates the endpoints
(~0 at rho = 0, 1 at rho = 1) and is the explanatory coordinate for all
dose-response analyses (the configured-rho-to-measured-recurrence map is
strongly nonlinear).

**Protocol.** For each task in sequence: evaluate zero-shot; stream 128
examples, scoring each before updating; update with one current plus one
replay example (a batch-size ablation appears in Section 6); evaluate at
fixed support counts. Paired models receive identical worlds, orders,
examples, replay draws, and evaluation sets. Likelihood is Gaussian with
fixed sigma = 0.1; a fixed-precision quantized-target form yields an
equivalent coding interpretation.

**Validity controls (all passed before any model comparison).** Scratch
models trained per-task show no difficulty trend in task index; outputs
neither saturate nor vary pathologically; opaque-ID scrambling leaves
training bit-exact after relabeling; and a true-route oracle with learned
operator slots shows strong lifetime transfer (late-life zero-shot NMSE
0.0016-0.0019; unseen compositions solved essentially from the library),
establishing that the hidden structure is exploitable in principle.

## 3. Models

All learners expose the same interface (opaque task ID in, prediction
out; per-task state initialized identically by policy) and none receives
routes, primitive identities, or `rho`.

- **Dense-C**: task embedding concatenated into three task-conditioned
  residual state blocks; width chosen to match the continuous model's
  inference multiply-adds (6,144 vs 6,528). Dense-P (parameter-matched),
  a width-128 variant, and a 24-dimensional task-code variant serve as
  capacity controls.
- **Continuous basis**: 8 learned operators of the teacher's residual
  form (with independently initialized, learnable residual scale and
  configurable activation — Section 5.3); each task learns per-step
  softmax mixture codes. Task identity influences computation only
  through the mixture — operators never see the task ID.
- **Discrete library**: 12 operator slots, relaxed routing annealed to
  hard argmax evaluation.
- **Hypernetwork**: task codes generate low-rank operators directly — a
  continuous operator manifold with no explicit slots.
- **Shared-parent + residual**: library operators plus rank-2 per-task
  residual corrections with an explicit penalty (Section 6.3).

Symmetric tuning: both primary architectures received the same staged
learning-rate grid on development worlds; configurations were frozen
before confirmation.

## 4. Confirmed results (sealed worlds 100-129)

The confirmatory protocol, primary outcomes, exclusion rules, and
reporting rules were frozen (CONFIRMATION_PLAN.md) before any sealed
world was generated or inspected. 360 paired lifetimes ran with zero
failures and zero exclusions.

**P1 — exact-reuse advantage.** At rho = 1.0, Continuous beats Dense-C in
30/30 worlds; mean paired advantage +3,204 nats (median +3,177), i.e.
+0.39 nats per online example. Exact sign test p = 1.9e-9.

**P2 — recurrence dependence.** The per-world slope of the paired effect
against measured recurrence is positive in 30/30 worlds; mean +5,715 nats
per unit recurrence; p = 1.9e-9.

**P3 — within-world reversal.** In 30/30 worlds, Dense-C wins at rho = 0
and Continuous wins at rho = 1; p = 1.9e-9. All three outcomes survive
Holm correction (adjusted p <= 5.6e-9).

**The regime map.** Continuous wins 0/30 worlds at every configured
rho <= 0.75 (mean effects -2,370 to -2,397 nats; -1,443 at 0.75) and
30/30 at rho >= 0.9 (+1,172 at 0.9; +3,204 at 1.0) — 180/180 paired cells
matching the development-stage prediction. The crossover sits at
configured rho* = 0.835 +/- 0.023; in measured-recurrence coordinates,
r* = 0.499 +/- 0.050.

**Linearity.** Against measured recurrence the mean effect is
approximately linear (R^2 = 0.935; slope +5,716 nats per unit; intercept
-2,625) versus R^2 = 0.642 against configured rho. The apparent
"threshold" in configured coordinates is a coordinate artifact of the
nonlinear rho-to-recurrence map: the underlying phenomenon is a smooth
dose-response with a sign flip, and we use the word crossover, not phase
transition, throughout.

## 5. Mechanism (development worlds 0-9; labeled development-stage)

### 5.1 The advantage is acquired, not architectural

At lifetime checkpoints, shared parameters are frozen and fresh task
codes are adapted on held-out unseen compositions. After 8 tasks the two
architectures are statistically indistinguishable at 32-shot novel
adaptation (Continuous 0.0228 vs Dense-C 0.0230 mean NMSE; Continuous
better in only 4/10 worlds). After 64 tasks Continuous leads in 10/10
worlds (0.00343 vs 0.00645). Explicit forward transfer (fresh-task minus
lifetime prequential cost) is positive for Continuous on 97.5% of tasks
and its slope against task index is positive in 10/10 worlds: the
reusable learner becomes a better learner as experience accumulates,
from an equal start.

### 5.2 Statistical sharing is not structural abstraction

Two development findings dissociate "sharing that lowers lifetime cost"
from "abstraction that supports recomposition":

1. At rho = 0.9 Continuous wins lifetime cost in every paired world but
   does not reliably win frozen-library novel transfer; the transfer
   advantage appears only at rho = 1.0. (This pattern reproduces in the
   confirmatory secondaries.)
2. Post-hoc operator recovery, measured against an untrained-basis
   baseline of 0.0087 normalized distance: trained operators at
   rho <= 0.5 sit FARTHER from the shared primitives than untrained ones
   (0.0115-0.0121) — below the crossover, training moves the basis away
   from shared structure; rho = 0.75 matches the baseline (0.0083);
   recovery first appears at rho = 0.9 (0.0048) and crystallizes at
   rho = 1.0 (0.0017). Recovery onset coincides with the performance
   crossover.

We therefore distinguish statistical reuse (shared parameters exploit
overlapping structure during online learning) from structural reuse
(identifiable operators that can be frozen and recomposed). The former
pays from r ~ 0.5; the latter requires near-exact recurrence. We note an
earlier, more attractive hypothesis — that transfer improves before it
amortizes — was proposed on one development world and FALSIFIED on
replication; the dissociation above is what survived.

### 5.3 The circularity question

The teacher and the continuous learner share a functional family
(residual bottleneck blocks), so the strongest objection to this study is
that the result measures inductive-bias match rather than reuse. We
addressed it in stages. Removing all parameter coupling (independently
initialized, learnable residual scale) leaves the world-0 advantage
essentially unchanged (+4,446 nats). Doubling the teacher's rank against
a fixed learner rank preserves most of it (+3,296). Changing the
learner's activation family (GELU vs the teacher's tanh) collapses it
roughly tenfold, to +441 — still positive, but small. Prior alignment is
therefore a large term in the economics: what V1 demonstrates is that a
learner whose representational family is broadly aligned with the world's
computation can discover and exploit reuse, with the advantage scaled by
that alignment. Whether family mismatch shifts the crossover location
rather than only its magnitude is a preregistered V2 question (H6), and
we do not claim family-agnostic abstraction discovery anywhere in this
paper.

### 5.4 The route-inference bottleneck

The discrete library learns the right structure — 92.2% exact recovery
of explained routes, primitive distance 0.00229, 11/12 slots used — yet
pays a large online cost. Part is optimization artifact: replacing global
temperature annealing (which forces late tasks to infer routes through a
nearly-hard softmax) with per-task annealing recovers ~8,800 nats; the
remainder (~25k nats vs Continuous on world 0) persists. Representation
learning and program inference are separable problems, and inference —
not the library — is the discrete learner's bottleneck. Soft mixtures
implicitly carry hypothesis uncertainty that hard routing discards; the
premature-commitment hypothesis this suggests is preregistered for V2
(H7), where the route space is small enough to compute the exact
posterior.

## 6. Robustness and the resource frontier

### 6.1 Nuisance battery (all 10 development worlds, all 10/10)

Reverse task order +3,456 [CI 3,145-3,791]; no replay +3,129
[2,511-3,743]; canonical replay +3,698 [3,233-4,177]; heavy (1:4) replay
+3,350 [2,959-3,766]. The no-replay cell matters most: the advantage
does not depend on episodic rehearsal — shared parameters alone carry
prior-task information forward. A second independent model
initialization reproduces 10/10 wins on both metrics. Scrambled task IDs
leave training bit-exact. Widening the update batch from 2 to the spec's
suggested 8 shrinks the world 0-2 advantage from +3,463 to +2,061 but
preserves 3/3 wins. Task-code dimension is immaterial (Dense-24 vs
Dense-32: 46 nats).

### 6.2 The frontier

Evaluated int8 retention (degradation <= 1.4e-4 NMSE worst-case,
~1e-6 mean): Discrete 26,208 bits and ~768 inference multiply-adds;
Continuous 29,248 bits / 6,528 MACs; Hypernetwork 33,928; Dense-24
56,448; Dense-C 66,688 / 6,144 MACs. The retention ordering differs from
the lifetime-cost ordering (Hypernetwork beats Dense-C but loses to
Continuous in 10/10 worlds — continuous operator structure helps, and
explicit reusable slots help more), and both differ from inference cost:
storage efficiency, online learning efficiency, and execution cost are
genuinely distinct objectives. Compute matching in this paper means
inference-forward multiply-adds; training-time compute is not matched at
the same ratio (backpropagation touches all basis operators), and no
claim depends on the distinction.

### 6.3 Choosing how much to share — and what it costs

A shared-parent + rank-2-residual learner (task operators
P_k + Delta_task,k under an explicit penalty) beats the ENVELOPE of both
fixed architectures at every intermediate recurrence on worlds 0-2 —
+9,168 / +7,458 / +3,745 mean nats at rho 0.5 / 0.75 / 0.9, 3/3 worlds
on both metrics — degrading gracefully to parity (-246) at rho = 1.0,
with the predicted allocation signature: mean residual magnitude falls
monotonically with recurrence (functional ratio 0.284 -> 0.026). The
learner measurably chooses its degree of sharing. But the accounting
matters: its per-task residuals retain ~130,624 bits (9x the Continuous
task-state footprint), and under a two-part MDL code (bits priced at
ln 2 nats) it loses to both fixed architectures in all twelve
world-rho cells; break-even prices are 0.04-0.14 nats/bit. This is a
prediction-cost win with the predicted allocation behavior — not a
code-length win at this residual budget. We report both currencies and
claim only the former.

### 6.4 A characterized negative: MDL presence gating

L0-style presence gates on a 12-slot library, tuned over an 8-cell
two-stage grid with a selection rule frozen in advance, never produced a
compact sufficient library: five cells prune nothing; the strongest
penalty collapses to one slot; the bisection finds a 7-slot library
(teacher has 6) at negligible lifetime-accuracy cost whose novel-
composition transfer nonetheless fails the frozen sufficiency limit —
and the transfer degradation tracks gate pressure, not slot count (an
11-slot cell fails equally). Penalizing during learning damages
compositional generality before pruning bites; compressing after
evidence accumulates (consolidation, V2) is the natural alternative this
motivates.

## 7. What this paper does not claim

- No claim that these results transfer to large-scale or natural-data
  learning; the criterion is the bet, this paper is its controlled test.
- No claim that discrete or modular architectures are superior in
  general — the finding is a regime map, and dense learning wins a large
  region of it.
- No claim of relevance to ARC-AGI or any external benchmark.
- No claim that the learner discovers arbitrary factorizations
  irrespective of structural prior (Section 5.3).
- No conflation of lower lifetime cost with compositional transfer: they
  dissociate (Section 5.2), and "learned reusable primitives" is claimed
  only where transfer and operator recovery both support it.
- No universality claims: expressivity is deliberately closed off (fixed
  program length) so that learnability and encoding cost — the actual
  subjects — can be measured without expressivity confounds.

## 8. Related work

Prequential MDL as a learning-quality measure follows Bornschein, Li &
Hutter (2022). Modular meta-learning (Alet et al., 2018), soft expert
merging (SMEAR; Muqeeth et al., 2023), and attention-as-hypernetwork
(Schug et al., 2024) motivate the continuous-basis and hypernetwork
baselines; SMEAR in particular anticipates our finding that hard routing
failure is not evidence against reusable computation. The wake/sleep
library-learning lineage (DreamCoder; Ellis et al., 2021) is the closest
program-level relative: we differ in criterion (lifetime code-length
rather than task solve-rate), in operating over learned neural operators
with measurable ground-truth correspondence, and in the negative-control
discipline the V2 consolidation program inherits (a compressor must
refuse to compress structureless worlds). Compositional-generalization
studies reporting modular architectures that succeed in-distribution
while failing recomposition are consistent with our statistical/
structural dissociation, which provides a controlled dose-response
version of that observation.

## 9. Provenance

This is agent-executed research with a human principal investigator: the
implementation, experiments, and analyses were carried out by large-
language-model agents under human direction, from a specification frozen
before any code existed to preregistered confirmation, within
approximately one day on a single consumer machine, with an external
reviewer model in the loop throughout. We expect — and welcome — extra
scrutiny: every run writes fingerprint-validated artifacts (config,
metrics, model weights, world programs, seeds, git commit, environment);
worlds derive from explicit seed sequences; the development/confirmation
firewall, the frozen analysis plan, the full reviewer correspondence,
and a falsified hypothesis are all in the public record. The work is
designed to be checkable without trusting its authors, human or
otherwise.

## 10. Conclusion and the V2 program

In a world whose latent computational recurrence is known and tunable,
the value of a reusable neural substrate is neither mythical nor
universal: it is a measurable, linear function of how much genuinely
recurrent computation exists, with a sharply reproducible crossover
below which specialization wins. The advantage, where it exists, is
acquired over the lifetime from an equal start; and the sharing that
lowers lifetime cost is not yet the abstraction that supports
recomposition, which emerges only near exact recurrence. The preregistered
V2 program (public in the repository) asks whether a learner can operate
these economics itself: exact program posteriors and premature-commitment
tests, consolidation that must decline to compile structureless worlds,
mixed-recurrence worlds where the correct factorization is heterogeneous,
and within-lifetime amortized program inference.

---
*Figures to be generated from reports/: (1) regime map with per-world
traces and both coordinate systems; (2) effect vs measured recurrence
with linear fit (confirmatory); (3) checkpoint divergence (equal at 8,
2x at 64); (4) recovery-onset vs crossover alignment with untrained
baseline; (5) resource frontier; (6) robustness forest plot; (7)
shared-residual envelope with allocation signature and both-currency
accounting.*
