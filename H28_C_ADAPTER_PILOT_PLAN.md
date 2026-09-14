# H28-C first adapter-discovery pilot

Status: DRAFT, DEVELOPMENT CHECK RUN, NOT FROZEN, 2026-09-14. A bounded exploratory successor to
H28_C_COORDINATE_GATE_PLAN.md. Freeze and protect this plan before running the
pilot from committed code, after SO2's commit/resume coordination is resolved.
No sealed seeds or confirmatory inference. This pilot supplies an oracle core;
it cannot establish joint core discovery, autonomous coarse-graining, or an
economic advantage over independently learned implementations.

# Question

Can an interface inferred from input-output support examples of some operations
transfer to other operations expressed in the same coordinates? This tests
adapter identifiability before undertaking shared-core learning.

# Fixed construction and information boundary

Use the six existing development-seed-0 primitives, d=16/rank=8/alpha=.35, from
the coordinate gate. Supply their canonical weights and the operation identity
to all applicable arms and label this ORACLE-CORE throughout. Supply no true R,
inverse, coordinate angles, or paired canonical states to the fitted-adapter
arm. The observation/context tag is supplied and permits one adapter per context.

Use `SeedSequence([0,2821,stream])`, a new fixture namespace. Three contexts:
identity and two independent products of four Givens rotations on the fixed
planes (0,1), (2,3), (4,5), (6,7). The two sets of true angles come from stream
1, independently uniform in [-0.7,0.7] radians. Disclose the restricted family
and planes; infer the four angles. Inverse = transpose, shared over operations.
This tests a cheap structured orthogonal family, not arbitrary dense adapters.

For each context, supply 16 observed input-output support examples for operations
0-3. Use 32 independent query examples for each operation 0-5. Draw canonical
inputs once from streams 2 (support) and 3 (query), then transport the paired
draws to every context. No canonical inputs are given to fitting. Query results
are read only after all fitting and restart choices are final. The canonical
implementations of operations 4-5 are known to the oracle-core arm, but their
observed-coordinate input-output examples are withheld from fitting.

# Arms and fitting budget

- ORACLE-ADAPTER: the true R, an existence/serialization anchor.
- FITTED-ADAPTER: infer the four angles from support pairs only.
- NO-ADAPTER: the same supplied core directly in the observed coordinates.
- IDENTITY-CORE: a tied inverse adapter around identity. Its predictions do not
  depend on the angles; run no redundant angle search for this arm.

FITTED uses deterministic coordinate search on observed support MSE, equally
weighted over the four support operations and their examples. Three starts:
all angles 0, all +0.4, all -0.4. Step levels, in order:
0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625,
0.00078125. Four full sweeps per level, coordinates in index order. At each
coordinate compare the current angle and its plus/minus step, clamped to
[-pi/2,pi/2]; select support MSE only, preferring the current value on ties,
then minus before plus. Cache the current score. Never stop based on query.

There are at most 321 objective evaluations per start and 963 per context
(one initial score plus 10 levels x 4 sweeps x 4 coordinates x 2 proposals).
Charge duplicate clamped candidates if evaluated; any cache saving is logged.
Select the final restart by support MSE, ties by fixed start order. The core
never changes. No optimizer/hyperparameter tuning on these pilot queries.

# Measurements and exploratory decisions

Report support loss and canonical energy-normalized query error separately for
support operations 0-3 and withheld operations 4-5, per context and arm. Report
each operation, not just a pooled mean. Use the coordinate gate's common-target
normalization and fixed draws. Log objective evaluations, wall time, four-angle
payload, supplied core/plane metadata, and any additional data supply.

Instrument failure: non-finite values, changed core, support/query overlap,
query-dependent fitting/restart selection, more than the registered budget, or
oracle error >1e-10. Check exact identity before any nontrivial context.

The following are pilot triage rules, not confirmatory thresholds. If the
NO-ADAPTER mean withheld-operation error is <=1e-4 in a nontrivial context,
report that context's opportunity as too small; do not tune it to create a gap.
Call the restricted adapter question live for a larger plan if, in both
nontrivial contexts, FITTED reduces mean withheld-operation error by at least
10x against NO-ADAPTER and reaches <=0.01. Otherwise retain the continuous
errors and classify the pilot as unresolved/unsuccessful at this search budget.
A small pilot null does not refute full H28 or authorize an enlarged search.

# Validation and launch contract

Before launch test that changing query labels cannot change fitted angles or
restart choice, oracle parameters remain unchanged, the support budget is
bounded, the same R serves every operation, and save/reload preserves the
chosen interface and predictions. An independent scorer reconstructs every
reported error from saved angles, fixture arrays, and frozen core.

Time one real support-objective evaluation and a tiny search prefix before
scheduling; projected runtime must stay within the Tier 1 envelope. New source
must not change existing SO2 imports. Keep one writer per context, save durable
completed contexts and the full protocol/source/config/seed hashes, and restart
unfinished contexts from initialization. If runtime exceeds minutes, use the
repository's detached launcher, restart test, logs/status/exit records and host
preconditions. Protect the plan before any scored pilot data is generated.

Only a validated pilot can motivate the next shared-core learner plan. That
next plan must remove the oracle core, add matched independent implementations
and random-core controls, charge acquisition and representation cost, and test
unseen coordinate families/compositions. No such claim is made by this pilot.
