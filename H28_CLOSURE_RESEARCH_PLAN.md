# H28 successor: discovery, closure, and the economics of abstraction

Status: DRAFT RESEARCH PROGRAM, 2026-09-14. Hypotheses CL1-CL8 below are
prospective working hypotheses, not frozen predictions or results. This is an
additive operationalization of H28 in `row_v5_experimental_spec.md`, not an
amendment to any frozen experiment. No experiment, new seed band, or lifetime
is launched or authorized by this document. Numerical acceptance margins and
confirmatory sample sizes remain to be calibrated on development instruments
and frozen in separate executable plans.

The V1 source-of-truth question remains whether reuse reduces lifetime learning
cost. This branch adds separate tests of what is being reused: a coordinate
change, a predictive coarse-graining, or a computation that transfers across
realizations. It does not redefine previous results as failures for lacking a
new diagnostic, reopen V5, or rename the completed V6/V6R branches.

# Motivation and source boundary

[McSharry et al., NeurIPS 2024, Learning diverse causally emergent
representations from time series data](https://papers.nips.cc/paper_files/paper/2024/file/d8398f4da88975e2a9c62ecaa5ba267b-Paper-Conference.pdf)
learn an encoder with predictive and componentwise MI critics. Their criterion
`Psi = I(V_t; V_next) - sum_j I(X_t^j; V_next)` is sufficient, not necessary,
for emergence under their chosen decomposition. They also evaluate an adjusted
criterion for redundant components. The encoder is not an explicit hierarchy:
multiple diverse features are not nested levels. Residual MLPs, stronger critics,
critic pretraining, and post-hoc critic refitting address optimization issues;
they do not certify closure. The synthetic encoder widths `[256,256,1]` and
five critic steps per encoder step are reference settings, not ROW defaults.

[Rosas et al., Software in the natural world, arXiv:2402.09090v2](https://arxiv.org/html/2402.09090v2)
distinguish informational, causal, and computational closure, using histories,
causal states, and coarse-grainings of computational processes. Their
informational criterion concerns the micro past and macro future conditional on
the macro past, across future horizons. A one-step probe comparison is a
restricted diagnostic, not the general theorem. The paper's hierarchy describes
nested computational structures; it does not supply a tested recursive ROW
learning algorithm.

The experiment designs, analytic control, loss accounting, and hypotheses below
are ROW proposals motivated by these sources and the PI's synthesis. Neither
paper establishes their success in ROW. In particular, we do not import a claim
that a narrow real-valued latent must discard information, that critic failure
proves independence, or that a profitable module must be dynamically autonomous.

# Relationship to the current program

H28 asks whether the same computation can be reused under different coordinates
with cheap adapters. Its mandatory adapter-complexity constraint still applies.
Its status is independent of H20. The current J1c/J1c-R and J2A records provide
development evidence for formation and export, not for learned coarse-graining
or cross-realization autonomy. SO2 has its own frozen plan and amendments; this
branch changes none of its cells, metrics, thresholds, or acceptance rules.

Candidate starting points are existing development V3/V5 promoted artifacts
and J1c/J1c-R libraries with their controls. Their suitability is an audit
question, not assumed. Spent sealed artifacts are not a convenient source of
new development transitions. Existing reviewed results may motivate hypotheses;
fresh confirmatory evidence requires a separately allocated untouched band.

The immediate output is a cheap eligibility/instrument report. A hierarchy
learner, new lifetime, and closure regularization all remain downstream.

# Claims to keep separate

Report a vector, never a single abstraction score:

| Axis | Evidence required | Insufficient by itself |
|---|---|---|
| Discovery | Candidate and interfaces learned from permitted observations; supplied structure disclosed | An oracle encoder or supplied teacher route |
| Functional fidelity | Held-out transition and task accuracy on common on-trajectory states | Similar weights or a decodable label |
| Predictive autonomy | Calibrated micro-leakage equivalence with non-vacuity controls | A nonsignificant probe difference |
| Cross-realization transfer | A frozen shared law with new interfaces, common support, and limited adapter cost | Invertible recoding of a known function |
| Economic value | Paired future learning savings and complete representation/acquisition accounting | Lower terminal error or fewer retained parameters |

Closure and economic value need not move together. A closed computation may be
too rare to amortize. A useful feature may rely on realization-specific detail.
Discovery is another empirical property, not implied by either outcome.

# State, action, and target contract

In a frozen feed-forward ROW artifact, `t` means an operation boundary along an
executed program, not the index of an arriving training example. The proposed
controlled process is

    X_(t+1) = F_W(X_t, U_t),      Z_t = E_W(X_t).

`X_t` must include all causally relevant execution state for that boundary:
activation, persistent workspace if any, and applicable private/context state.
Weights are fixed during this audit. If a study instead concerns learning
dynamics, weights, optimizer state, replay state, and random state become part
of a different registered process; an activation-only audit cannot make that
claim. Use explicit exogenous action schedules or condition on the complete
permitted inputs so a hidden policy is not mistaken for micro leakage.

`U_t` denotes the operation/input exposed by the registered interface. A learned
slot identifier can be used if it is actually available to that learner. It
cannot silently be replaced by a teacher primitive ID. Teacher programs,
canonical states, and primitive alignment belong only to isolated oracle and
analysis paths. If the needed action is unobservable, the study must register
action inference or a whole-task transition instead of leaking the answer.

`W` denotes realization/context within a paired canonical world; the canonical
world seed is a separate replication unit. A known context tag may select an
adapter only in a disclosed-context arm. An unknown-context arm must infer and
price it. Giving W to a diagnostic probe does not grant it to the learner.

Freeze E, the target definition, and all normalizations before fitting probes.
The target is the same next coarse state for every probe. A static promoted
weight tensor or slot label is not automatically a dynamical macrostate.

# Two constructions, with different scientific roles

## H28-C: coordinate reuse and adapter economics

For a canonical operator family `A_u`, define

    x = R_i z,    F_(i,u)(x) = R_i A_u(R_i^-1 x).

Start with identity and permutation, then orthogonal transforms, then bounded
condition-number dense invertible transforms. Transform the entire operator,
including nonlinearities and residual paths; rotating only its weights need
not produce a conjugate function. Draw canonical trajectories once and map the
same states, actions, examples, and noise to every realization. A separate
versioned generator/config fingerprint leaves canonical WorldConfig untouched.

The oracle `E_i = R_i^-1` satisfies `E_i F_(i,u) = A_u E_i` by construction.
This is a harness identity, not evidence of emergence: an invertible E retains
the whole state and makes micro-leakage zero trivially. The empirical questions
are whether the learner finds a reusable shared law, whether it transfers to
new coordinates, and whether its interfaces are cheaper than independent
implementations. General `F_i = D_i A E_i` implies the commuting identity only
when `E_i D_i` is identity on the reachable canonical image (or the appropriate
weaker identity on A's outputs); arbitrary learned E/D do not ensure it.

Arms: independent implementations; exact sharing without adapters; shared core
plus restricted adapters; and a supplied inverse-coordinate oracle. Use identity
coordinates as the exact-sharing positive control. Test lack of sharing on
non-conjugate, difficulty-matched operation families. An adapter-only or
fixed/random-core ablation tests whether interfaces absorb the computation;
negative-control separability must be established before fitting learners.

Adapters start linear, invertible, shared across operations within a context,
and incapable of conditioning on task/primitive identities. Register their
parameter count, precision, inverse representation, conditioning bound, fitting
budget, and actual serialized cost. A larger nonlinear adapter is a new arm,
not a rescue after failure. Sweep measured adapter description cost separately
from numerical conditioning; dense matrices are not more complex merely because
they have a larger condition number.

For fresh realizations freeze the core and fit only the interface from support
data; for fresh compositions infer permitted codes/routes from support only.
Use disjoint coordinate draws and composition sets, and include their crossed
holdout if claiming both kinds of generalization simultaneously. Compare learned
cores only up to the admissible common gauge, fitted on proposal data; a shared
rotation of all latent coordinates is not an error.

Score commuting error, pairwise interface substitution, task error, adapter bits,
and future acquisition cost. Normalize geometry in common canonical units or
with a registered transported metric so dense scaling cannot manufacture an
advantage. Comparing log densities across coordinate systems also requires
the matching Jacobian/noise/quantization convention. Within a paired comparison
all arms predict the same target in the same units.

## H28-Q: nontrivial coarse-graining and autonomy

Use a separately fingerprinted controlled process with state `(z,n)`, where n
contains nuisance distinctions, and observations `x = R_i(z,n)`. The positive
construction has

    z_next = A_u(z, epsilon),    n_next = B_(i,u)(z,n,eta).

The nuisance may depend on z but cannot affect its future conditional on z and
u. The oracle coarse-graining recovers z and discards n. Construct a paired
leaky alternative by adding a calibrated nuisance-dependent term to z_next,
and a world-dependent alternative by changing the abstract law across i. Match
scales/difficulty and use the same exogenous draws where possible. These are
instrument/testbed controls, not an assertion that canonical ROW already
contains a nontrivial closed quotient.

A learned small latent must preserve task-relevant behavior while discarding
real distinctions. Register precision or channel capacity, predictive gain over
a constant/no-state baseline, held-out task fidelity, and distinct microstates
in each tested coarse class. Dimension alone does not prove information loss.
A full-state identity and a constant encoder must fail the combined abstraction
gate for opposite reasons. Restrict conclusions to the tested coarse-graining,
distribution, action set, and horizons.

For causal robustness, compare next coarse outputs after admissible nuisance
interventions holding z and u fixed, within the registered support. In finite
controls enumerate the conditional transition laws. In learned continuous
representations approximate matching may confound this test; failed overlap or
uncontrolled changes to Z yield INCONCLUSIVE, not a causal claim. Observational
world leakage does not by itself establish interventional autonomy.

# Predictive leakage instrument

Fit macro, micro-augmented, world-augmented, and jointly augmented probes:

    q_M(Y | Z,U),     q_X(Y | Z,U,X),
    q_W(Y | Z,U,W),   q_XW(Y | Z,U,X,W),
    Y = registered next coarse-state target.

Report `Delta_X = L_M - L_X`, `Delta_W = L_M - L_W`, and
`Delta_XW = L_M - L_XW` as paired held-out log-loss differences. Include the
joint probe because X and W may carry information jointly that neither adds
alone. Do not clamp negative empirical gaps to zero.

For well-defined targets and Bayes-optimal log-loss predictors, Delta_X equals
`I(X;Y | Z,U)`. In finite models it also contains the difference between the
probes' approximation and optimization errors. It is neither an automatic MI
estimate nor a bound in a known direction. A common fixed-variance Gaussian
score tests mean-prediction adequacy, not equality of full conditional laws.
Use a finite target alphabet with proper log scores, or a registered stochastic
observation model and calibrated density family, for distributional claims.
Deterministic continuous relations can have singular densities/infinite MI;
do not feed those into a nominally finite MI comparison without a measurement
convention. Freeze bins/noise/precision from development calibration and check
its sensitivity. Keep these diagnostic nats separate from task-target nats.

Probe training must be capable of detecting the predeclared material leakage.
Use nested predictors able to reproduce q_M by ignoring added inputs, a
capacity-matched macro control, independent restarts, equal tuning opportunities,
and a stronger post-hoc audit probe trained without access to evaluation labels.
Freeze all selection on training/validation data. Hold out whole trajectories
and tasks, not adjacent transitions randomly assigned to opposite splits.
Encoders, candidate selection, gauge fitting, and probe calibration never see
the final evaluation trajectories. Use paired canonical states across artifacts
and the same fitted target mapping when comparing pre/post promotion; do not
subtract losses for incompatible learned targets.

One-step tests are primary only for the registered controlled Markov interface.
Add prespecified history and multi-step/action-sequence probes when that
assumption is unsupported. Passing finite horizons cannot certify all-history
computational closure in the source paper's sense.

World invariance requires overlap: if Z already identifies W, adding W cannot
help even when the laws differ. Use balanced common canonical-state/action
distributions across realizations, held-out coordinates, and direct law
comparisons on that support. A learned representation with disjoint support
cannot pass a universal-law claim on a zero world-probe gap alone. A categorical
world probe fitted on familiar contexts measures residual heterogeneity; it
does not itself predict unseen world categories.

## Calibration and decision rules to freeze before a scored study

The executable plan must name `epsilon_X`, `epsilon_W`, `epsilon_XW`, commuting
and task-fidelity tolerances, non-vacuity thresholds, horizons, and the smallest
material injected leakage. Calibrate detection on closed/leaky/world-dependent,
constant/full-state, temporally shuffled, weak-probe, and collapsed-trajectory
controls. The full-state control is a trivial-closure anchor, never a discovery
success. Do not choose tolerances from the target audit's results.

Use paired differences aggregated by independent world; tasks, transitions,
coordinate copies, and initialization repeats are not independent worlds.
Prespecify interval construction, multiplicity control across gaps/horizons,
sample size, and stopping rules. A development sample of a few worlds is not
enough to assert precision by resampling millions of transitions.

- INSTRUMENT_INVALID: reconstruction, controls, target scoring, or provenance
  fails. Do not interpret a target result.
- LEAKAGE_DETECTED: a calibrated gap's lower confidence bound exceeds its
  material tolerance on a valid instrument.
- APPROXIMATELY_CLOSED_ON_TESTED_DOMAIN: all required gaps' upper bounds are
  below their tolerances, fidelity and non-vacuity pass, common support exists,
  and registered functional/intervention checks pass where claimed.
- INCONCLUSIVE: neither equivalence nor material leakage is established,
  including insufficient power or untestable overlap.

Missing eligible artifacts gives AUDIT_UNAVAILABLE. Lack of a representational
or economic opportunity gives NO_OPPORTUNITY. Neither is a null result for a
trained method. H28-C uses the transfer/economics classification and labels its
closure test a harness identity; it cannot earn the nontrivial H28-Q claim.

# Coordinate sensitivity: a cheap exact control before neural critics

The following is our proposed analytic control, not a new empirical discovery.
Let U be a stationary unit-variance Gaussian AR(1) process with coefficient r,
`0 < |r| < 1`, and N independent unit-variance temporally white Gaussian noise.
Take `X=(U,N)` and `V=U`. Then `Psi=0`: the first microscopic component contains
all the predictive information in V. Rotate by 45 degrees:

    X'_1 = (U+N)/sqrt(2),    X'_2 = (U-N)/sqrt(2).

Transport the encoder so V remains exactly U. Each coordinate's squared
correlation with V_next becomes `r^2/2`, giving

    Psi_rotated = -0.5 log(1-r^2) + log(1-r^2/2) > 0.

The macro transition, joint predictive information, and nuisance closure remain
unchanged. This demonstrates dependence on the chosen parts, not an error in
the paper's stated parts-relative criterion. Coordinate permutations should
leave the sum unchanged; general mixing need not. For finite/discrete states,
a generic real-valued projection can even encode the whole finite state, so
Gaussian rotation intuition cannot be assumed for every ROW distribution.

Transport a frozen candidate encoder first to isolate metric sensitivity; a
second, separately labeled retraining experiment measures discoverability under
new coordinates. If using SMILE, distinguish Psi from the adjusted Psi_A,
refit independent critics, and check against the analytic control. A difference
of estimated MI terms is not automatically a reliable emergence certificate.

An optional comparator is `J_k = sup_rank(P)=k I(PX; V_next)`: the best linear
rank-k predictor captures accessible information rather than individual
coordinate information. The ideal unrestricted row-space family is invariant
under invertible linear changes of X; finite optimizers, regularization,
precision, and whitening can break that property. `I(V;V_next)-J_k` is a new
diagnostic, not the paper's PID criterion or a test of closure. Estimate it only
after the analytic gate, with fixed k, held-out evaluation, and priced search.

# Prospective hypotheses and falsifiers

These IDs extend H28 without renumbering existing H hypotheses. CL1 and CL2 are
instrument/mathematical expectations; CL3-CL8 are unsettled empirical claims.
No numerical posterior odds or success probabilities are invented here.

| ID | Working hypothesis | Decisive counterevidence or restriction |
|---|---|---|
| CL1 | The probe protocol separates planted closed and materially leaky quotients and rejects trivial encoders as abstractions. | Missed injected leakage, false equivalence with weak probes, or a constant/full-state encoder passing the combined gate invalidates the instrument. |
| CL2 | Coordinatewise Psi changes under some invertible mixes while the transported law and closure remain fixed. | Failure to reproduce the analytic control is an instrument failure; lack of change for a particular ROW representation is not a refutation of the analytic existence statement. |
| CL3 (H28 economics) | At sufficient recurrence and cheap adapters, learned shared-core reuse transfers and repays its full cost; expensive adapters reduce that advantage. | Failure on held-out realizations/compositions or no net saving at the registered reuse/cost envelope rejects that cell's claim. A full three-phase transition is not assumed. |
| CL4 (PROMOTE, tentative) | A learned promoted interface reduces predictive micro leakage at matched fidelity relative to its eligible pre-promotion/control interface. | No material paired reduction, worsened fidelity, or a gap explained by changed target geometry rejects the operational claim. Terminal checkpoints alone cannot establish that PROMOTE caused it. |
| CL5 (nontrivial autonomy) | A learned bottleneck recovers the planted quotient and remains approximately closed across new realizations and nuisance interventions. | Detectable micro/world/joint leakage or failure of non-vacuity/transfer defeats the combined claim; oracle success alone establishes opportunity only. |
| CL6 (dissociation) | Closure and economic value can dissociate: recurrence changes amortization while a fixed transition law's closure remains stable. | Failure to keep laws/measurement fixed invalidates the contrast. The sign of a trained method's payoff remains empirical even if planted existence controls show dissociation. |
| CL7 (regularization, deferred) | Adding calibrated closure pressure improves held-out realization transfer and future learning cost at matched total resources. | No material gain over capacity/compute-matched predictive and bottleneck controls, or new collapse, rejects benefit at the registered budget. |
| CL8 (hierarchy, deferred) | A second learned coarse-graining adds transferable computation and net value beyond a matched flat latent and definitional caching. | Duplicate/constant levels, loss of lower-level fidelity, or no incremental savings after full hierarchy cost defeats the claim. |

# Staged experiment sequence and stop gates

## T0-A: artifact eligibility, no learner training

Inventory development checkpoints, full state reconstruction, immutable hashes,
legal on-trajectory traces, and candidate encoders/interfaces. Reproduce the
existing artifact's accepted score before analyzing new transitions. Identify
whether matched pre/post promotion checkpoints and an unchanged target exist.
If they do not, report that a causal PROMOTE audit is unavailable; use a
descriptive terminal audit only where meaningful. Do not reconstruct nonexistent
historical model states from terminal weights. A future causal ablation needs
new paired development runs under its own plan.

Existing library slots may be operations with full-state inputs, not learned
coarse-grainings. If every candidate is identity-like or constant, stop this
artifact branch. J2A export success is not permission to label a slot a closed
macrovariable. T0-A's deliverable is an eligibility manifest and the cheapest
live question, including NO_OPPORTUNITY/AUDIT_UNAVAILABLE when appropriate.

## T0-B: instrument and construction checks

Use analytic Gaussian/finite-state controls for CL1/CL2, a short conjugacy
identity check for H28-C, and oracle quotient/leak controls for H28-Q. Fit only
tiny probes if necessary. No full lifetime and no heavy per-coordinate critic
suite. A planned Tier 0 census that grows beyond minutes is explicitly re-scoped
as Tier 1 instead of being called cheap. Stop on a failed control before asking
whether existing promoted models have closure.

## T1-A: frozen-artifact diagnostic, conditional on eligibility

One development world, immutable model, bounded transition set, fixed small
probe family. CL4 is exploratory; use matched promotion controls only if the
artifact contract supports them. A before/after association cannot establish
causality without a matched no-PROMOTE intervention. This stage measures
detectability and runtime; a small-budget null cannot close the larger question.

## T1-B: learned H28-C and then H28-Q opportunity pilots

Start H28-C with identity and one nontrivial coordinate realization, the
restricted shared model, independent reference, and decisive oracle/sham
controls. Price adapter acquisition and check the future-task opportunity.
Advance to a small H28-Q pilot only if nontrivial autonomy is the live question
and the quotient instrument passes. Keep pilots within roughly one hour each,
exploratory only, and serialize with other local experiments. Do not begin a
wide transform/architecture/reuse grid before its anchors justify it.

## T2: separate preregistrations, conditional on successful instruments

Freeze coordinate-transfer/economics and quotient-autonomy plans separately:
exact cells and seeds, full configs, estimands, calibrated tolerances, independent
replication units, budget, scorers, and gate-first early-stop rules. Any sealed
study needs a new untouched seed allocation and protected plan before access.
Development results remain development results even when their protocol was
frozen. All Tier 2 work uses clean committed code, `check_prereg.py`,
`check_invalid.py`, paired output validation, and the repository's restart test,
one-writer, host-reserve, detached-launch, logs, exit, and artifact rules.

Before every launch perform the prescribed correctness/performance pass and time
real updates. Record before/after timing; apply bitwise-preserving wins only
after a short equivalence check. Version non-bitwise changes at a plan boundary.
One local full lifetime at a time. No heavy diagnostics run beside SO2. Relaunch
must validate and reuse completed cells and restart only unfinished cells.

## Later: discovery objectives and nested abstractions

Only after diagnostic calibration compare (i) task prediction plus complexity,
(ii) the same finite-capacity bottleneck without closure pressure, (iii) closure
pressure, and (iv) an optional parts-relative emergence objective. Match tuning,
examples, model capacity, and total probe/critic training resources; report a
resource frontier if exact compute matching is impossible. Evaluate the frozen
encoder with independent stronger critics/probes. Online auxiliary targets may
use only already available transitions, never query labels or future examples.

A regularized learner may minimize a registered combination of task loss,
description cost, and calibrated Delta_X/Delta_W/Delta_XW surrogates. The encoder
can game weak predictors or collapse its targets, so independently scored
fidelity, information capacity, and transfer remain binding gates. All weights
are chosen on development validation, not by final closure scores.

For CL8, form `X -> Z1 -> Z2` with a genuinely nested encoder. Parallel diverse
features or depth in an MLP do not count as levels. Each level needs its own
non-vacuity, interface, action semantics, horizon, and incremental cost tests;
register a fixed macro-duration/action block before temporal coarse-graining.
Compare a matched flat bottleneck, random coarse-graining, and expansion-only
macro. Two empirical approximate-closure passes do not prove all-level exact
closure; test composed rollout errors and cross-realization substitution.

E6 definitional macros retain their naming/coding interpretation: expansion
equivalence is automatic and cannot certify autonomy. New claims of autonomous
hierarchical computation require the H28-C/H28-Q evidence appropriate to them;
ordinary useful caches remain legitimate economic objects. Existing execution,
control-flow, and synthesis gates in the post-E6 program still apply.

# Economic accounting and interpretation

For a fixed future stream and common task-target quantization, report

    S_future = L_future(independent) - L_future(shared),
    Delta_B = B_shared - B_independent,
    G_total = L_all(independent) - L_all(shared) - ln(2) * Delta_B.

`L_all` includes scored acquisition/calibration stages when those are part of
the supplied stream. A positive G_total is a coding-economy comparison only
under an actual common coding convention; a retained int8 proxy stays a proxy.
Use a fixed two-part/prequential accounting rule and do not charge the same
parameter description twice. Extra task supply, examples, gradient work,
operator applications, search, rejected candidates, probe/critic training,
wall-clock time, and peak memory are separately reported resources. They are
not silently converted into nats. Compare arms on identical supplied data;
also report the curriculum/interface data price against the original stream.

The shared representation includes the core, every adapter/inverse if separately
stored, task codes, route tables, macro definitions, precision metadata, and any
retained discovery machinery. Diagnostic-only probe bits are not inference
storage; their fitting cost still belongs to research/discovery accounting.
Full future savings must repay setup under the registered reuse distribution,
not just improve one terminal score. Short-horizon negative net benefit alongside
closed dynamics is informative, not a failed definition of abstraction.

For CL6, cross a fixed closed/leaky transition construction with low/high reuse
while holding the evaluation state/action distribution fixed. Reuse changes
task arrival frequencies and the payoff horizon, not the closure target. First
establish that the closed low-reuse arm can have negative net value after setup;
then test whether a deliberately incomplete, substrate-dependent predictor can
still be profitable on its declared distribution. Do not assume all four
closure/value combinations occur for the learned method, or infer robustness
under nuisance shift from a profitable in-distribution shortcut.

# Required next artifacts and unresolved choices

First produce the T0-A eligibility manifest and T0-B calibration design under a
small dedicated plan. Then fix: the execution-state/coarse-target definition,
whether a meaningful existing encoder exists, allowable actions/context tags,
artifact hashes and legal transitions, finite measurement convention, adapter
coding family, support/query/coordinate splits, common-support interventions,
probe family and detectable margins, cost frontier, and uncertainty design.

These are explicit prerequisites, not blanks an implementer may fill after
reading outcomes. No Tier 2 hypothesis in this document is ready to score yet.
Append outcomes and corrections to the prediction ledger, progress, learnings,
and paper; retain invalid, unresolved, negative, and exploratory statuses.
