# H28-C learner/core opportunity gate

Status: DRAFT, DEVELOPMENT CHECK RUN, NOT FROZEN, 2026-09-14. This follows the provisional oracle-core
adapter result but is not frozen. It asks whether a shared core and coordinate
interface can be learned together. It is a Tier 1 development study, not a
confirmatory H28 result, and uses no sealed seeds.

# Question

When the canonical operator core is no longer supplied, can a learner discover
a shared core plus a context-shared coordinate adapter, and transfer it to
operations or coordinate contexts withheld from fitting? The oracle pilot
established only that the restricted interface is identifiable when the core is
known.

# Construction and splits

Reuse the H28-C oracle fixture family only after regenerating it from this plan
under a new namespace `SeedSequence([0, 2822, ...])`. Use six canonical
nonlinear primitives, d=16/rank=8, alpha=.35, identity plus two four-angle
Givens contexts and the same fixed planes. Never reuse the oracle pilot's
support/query arrays. Context angles remain hidden from learned arms. The
operation identity is supplied; primitive weights and routes are not.

Each context has 64 support examples and 128 query examples per operation. Train
the six core operators on identity-context examples for operations 0–5. Fit the
context-1 adapter jointly with that core using operations 0–3 only; operations
4–5 are withheld from adapter fitting but their core operators were learned in
identity coordinates. Hold context 2 out of core and context-1 adapter fitting;
freeze the learned core and infer only its adapter from 16 fresh support
examples per operation, then score all six operations on 128 fresh queries.
Query labels never select initialization, optimizer, stopping, or adapter family.

Identity's adapter is fixed. Context adapters are tied across all operations and
restricted to four-angle Givens matrices. The identity-context core is the fixed
reference gauge; no post-hoc per-operation transformation is permitted.

# Arms

- `SHARED_CORE_ADAPTER`: learn one core per operation in canonical coordinates
  and the context-1 adapter, then freeze the core before fitting context 2.
- `INDEPENDENT`: one observed-coordinate operator per context and operation,
  with matched scalar count and the same data split.
- `SHARED_NO_ADAPTER`: one core in observed coordinates, no context adapter.
- `RANDOM_CORE_ADAPTER`: fixed random core with trainable adapters.
- `ORACLE_CORE_ADAPTER`: supplied-core positive anchor under the new namespace.

Independent arms cannot receive unrestricted dense matrices as hidden extra
capacity. Match parameters, optimizer evaluations, and support data; report a
frontier if exact matching is impossible. A shared failure with an oracle pass
localizes core acquisition; failure of both is an opportunity or optimization
failure, not evidence against H28.

# Learner and budget

Use a small fixed torch MLP primitive with the teacher's input/output dimension
and rank bottleneck, initialized from a deterministic model seed. The shared arm
has one parameter set per operation in canonical coordinates and four trainable
adapter angles for context 1. Optimize support MSE only with Adam; select its
learning rate and steps on separate calibration data, then freeze them before
this gate. Query labels never enter calibration. Start with one model seed and
add two fixed seeds only if the instrument passes.

Log trainable/frozen scalars, optimizer steps, evaluations, wall time, peak
memory, and serialized core/adapter bytes. Save a durable result per arm/context
and restart unfinished arms from initialization; never resume a partial optimizer.

# Endpoints and triage

Primary endpoints are canonical energy-normalized query MSE for withheld
operations 4–5 in contexts 0/1 and all operations in held-out context 2.
Secondary endpoints are support MSE, adapter angle error as analysis,
commuting error on common canonical states, and random-core performance.
Report every operation/context before pooled means.

Before endpoint scores require oracle error <=1e-10, finite outputs, unchanged
frozen core after context-2 fitting, disjoint support/query data, identical
targets and operation labels across arms, and independent saved-state
reconstruction. The opportunity is LIVE only if oracle and non-vacuity controls
pass. A learner is `ACQUIRES` at this envelope if it beats shared-no-adapter by
10x on both crossed endpoints, reaches <=0.05 mean query error, and remains
within 2x of independent-arm parameter bytes. Otherwise report
`NO_ACQUISITION` or `INCONCLUSIVE` with the failed gate. These are development
triage rules, not H28 confirmation thresholds.

# Integrity and next steps

Teacher weights, true angles, and canonical states never reach learned-arm
fitting. The oracle arm remains explicitly labeled. A later economic study must
charge core discovery, adapter acquisition, storage, and future learning; this
pilot cannot establish amortization.

Freeze this document, code, seeds, learning budget, and scorer before running.
Perform the performance pass and restart dry run, use one writer and detached
logs/status/exit records, and keep this study away from SO2. If oracle or a
non-vacuity control fails, stop before learner cells.
