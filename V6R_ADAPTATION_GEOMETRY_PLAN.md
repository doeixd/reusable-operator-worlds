# V6R adaptation-geometry localization plan

Status: frozen before writing or running the V6R audit. This is a development
diagnostic over existing artifacts, not a confirmation block. It opens no new
world seeds and cannot rescue H30 or H35.

# Question

V6 observed higher acquisition cost under the prospective representation but
did not localize the deficit. The audit distinguishes three possibilities:

1. **Representational opportunity loss:** even high-data, high-budget fitting of
   the allowed task-local state cannot match the ordinary representation.
2. **Sparse identifiability loss:** both representations have comparable
   high-data capacity, but the prospective representation remains worse from
   one example even under high-budget optimization.
3. **Optimizer/findability loss:** comparable solutions remain available from
   one example, but the registered finite-step adaptor cannot reach them.

The terms are operational. No finite optimizer is called a global oracle.

# Frozen inputs

- Artifacts: `artifacts/v6_clean/{ordinary,prospective}/world_{0,1,2}/lifecycle`.
- Worlds: development seeds 0, 1, and 2 only.
- Tasks: the same two `novel_family_tasks` per world used by the corrected V6
  fertility scorer. These tasks never enter lifetime training or prospective
  gradients.
- Config: `configs/v5_h72.yaml`, four families, 16 tasks per family,
  `r_meta=1`, rank-2 meta-subspace, 12 operator slots.
- Representation: frozen completely. Only the fresh task code and private
  task residual may move, exactly as in the standardized fertility adaptor.
- Query set: the fixed 256 evaluation examples. Query labels never select an
  optimizer, restart, checkpoint, stopping time, or fitted state.

# Anchor

Before the new endpoints are read, the 40-step Adam adaptor at learning rate
0.05 and support `k=1` must reproduce `reports/v6_clean_fertility.json` to
absolute tolerance `1e-12` for ordinary and prospective related-task costs.
Failure stops the audit without a report.

# Fitting protocols

Every arm/task/protocol begins from the checkpoint's canonical fresh-task
initialization. All optimizers minimize support MSE only. Query MSE is evaluated
at fixed registered checkpoints and after fitting; it never enters an optimizer
closure.

## S0: registered standardized adaptor

- Support: `k=1`.
- Adam, learning rate 0.05, 40 updates.
- Records the complete query-loss trajectory and endpoint.

## B1: primary high-budget adaptor

- Supports: `k=1` and `k=128` (all available training examples).
- Adam, learning rate 0.01, 2,000 updates from the canonical initialization.
- Fixed query checkpoints: updates 0, 1, 2, 4, 8, 16, 40, 100, 250, 500,
  1,000, and 2,000. Checkpoints are reported, never selected.
- The update-2,000 query endpoint is the primary localization endpoint.

## B2: optimizer robustness

Two fixed alternatives, reported separately:

- Adam, learning rate 0.05, 2,000 updates from the canonical initialization.
- PyTorch LBFGS, learning rate 1.0, `max_iter=500`, `history_size=100`,
  `line_search_fn="strong_wolfe"`, from the canonical initialization.

The scientific classification requires B1 and at least one B2 optimizer to
agree. No per-task or per-arm best-of-query envelope is permitted.

## B3: restart sensitivity (secondary)

At `k=1`, run Adam learning rate 0.01 for 2,000 updates from three deterministic
perturbations of the canonical local state. Perturbation scale is `1e-3` per
scalar. Seeds use NumPy `SeedSequence([61037, world, task_index, restart])`.
The restart with the lowest final **support** loss is selected before its query
loss is read. B3 cannot determine the primary classification; it tests whether
the canonical initialization alone creates the result.

# Metrics

For every arm, world, task, support, and optimizer:

- initial and final support MSE;
- query MSE at each registered checkpoint;
- final query endpoint in both raw MSE and the V6 scaled currency
  `MSE / (2 * 0.1^2)`;
- local-parameter displacement from initialization;
- number of optimizer evaluations and finite/non-finite status.

Pair ordinary and prospective within the identical world and future task.
Define endpoint gap

    G(k, optimizer) = C_prospective - C_ordinary

so positive values mean the prospective representation is worse.

# Operational equivalence and replicated harm

For a protocol, call endpoints **operationally equivalent** only when:

- mean absolute paired task gap is at most 10% of the ordinary mean endpoint;
- at least five of six paired tasks have absolute gaps at most 20% of their
  ordinary endpoints; and
- no world-mean absolute gap exceeds 20% of that world's ordinary endpoint.

Call prospective **replicated worse** only when all three world-mean gaps are
positive, the mean world gap exceeds its population standard deviation, and
the mean gap exceeds 10% of the ordinary mean endpoint.

These are descriptive development gates, not significance tests.

# Decision tree

Apply the tree to B1, requiring agreement from at least one B2 optimizer:

1. **Representational opportunity loss** if `k=128` is replicated worse.
2. **Sparse identifiability loss** if `k=128` is operationally equivalent but
   `k=1` is replicated worse.
3. **Optimizer/findability loss** if `k=128` and `k=1` are operationally
   equivalent under high-budget fitting, while S0 reproduces the registered
   prospective deficit. Require at least 80% reduction in the mean S0 gap.
4. Otherwise: **unresolved/mixed**.

The result may support more than one contributing mechanism only if the gates
explicitly overlap; narrative inspection cannot override the tree.

# Anti-fooling guards

- Same support/query arrays, initialization, and optimizer protocol across arms.
- Support-only optimization and support-only restart selection.
- Fixed checkpoints and budgets; no convergence stopping based on query data.
- Complete lifecycle reconstruction, including retirement and references.
- Fresh probe IDs are removed after every fit.
- No mutation of checkpoint models across tasks or protocols; each fit uses a
  newly loaded or deep-copied frozen representation.
- Fail closed on missing artifacts, anchor mismatch, non-finite primary cells,
  or fewer than six paired future tasks.
- Report every cell, including optimizer failures; no optimizer or task is
  excluded after query inspection.

# Scope after the result

- Opportunity loss motivates an explicitly factorized
  shared-schema/fast-argument/private-innovation architecture.
- Sparse identifiability loss motivates an argument channel or statistical
  interface, not merely more optimizer steps.
- Optimizer/findability loss motivates representation-updater co-design.
- An unresolved result motivates direct local conditioning measurements, not a
  new prospective-pressure sweep.

No H34, H36, confirmatory V6 seeds, or successor architecture is authorized by
this plan.
