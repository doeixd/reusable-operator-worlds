# Audit scope

This audit maps `neural_library_learning_v1_experimental_spec.md` to the current
implementation and evidence as of commit `7e66639`. Confirmatory worlds 100–129
remain sealed. “Complete” means implemented and checked on the stated
development scope; it does not turn a development result into a confirmatory
claim.

# Complete core benchmarks

- Deterministic random residual teacher primitives, 64 unique length-three
  programs, opaque randomized task IDs, and fixed train/evaluation arrays are
  implemented.
- Exact reuse and the six-point reuse continuum are implemented with measured
  functional recurrence. The continuum is complete on development worlds 0–9.
- Online examples are scored before update. Paired models receive identical
  worlds, task orders, examples, replay policy, and evaluation arrays.
- Scratch difficulty, uniform output scale, and the leak-free oracle positive
  control pass.
- Dense-P, Dense-C, Continuous, hard Discrete, and a generic low-rank
  hypernetwork control are implemented. Dense-24 closes the task-state-size
  sensitivity.
- Cumulative prequential Gaussian log loss, examples-to-criterion, frozen-library
  novel composition, functional matching, route recovery, specialization, and
  forward multiply-add accounting are implemented.
- The required exact-reuse and reuse-sweep figures exist and have been visually
  inspected.

# Complete development conclusions

- Continuous beats Dense-C on exact-reuse lifetime loss across all ten
  development worlds and becomes progressively better at fresh-composition
  adaptation over the lifetime.
- The causal effect reverses consistently: Dense-C wins through configured
  `rho=0.75`; Continuous wins at `rho=0.9` and `rho=1.0` in all ten worlds.
- The generic hypernetwork beats Dense-C but loses to Continuous on lifetime loss
  in every development world. This separates the benefit of a continuous
  operator manifold from the additional benefit of an explicit reusable slot
  basis.
- Hard Discrete learns good operators and routes but pays high online route
  inference cost. Per-task annealing reduces, but does not close, that gap.
- Alpha leakage, teacher-rank mismatch, activation mismatch, fixed identity, and
  dense task-code dimension have been investigated.

# Protocol validity checks complete

- Opaque task-ID reassignment is exactly invariant for selected Continuous and
  Dense-C runs.
- The effective batch-size discrepancy is explicitly ablated on worlds 0–2.
  Batch 8 improves both models and preserves Continuous's advantage, while
  narrowing it because Dense-C benefits more. The confirmation batch and any
  retuning remain a freeze decision rather than an untested obligation.

# Open diagnostic obligations

- Add world-level median and bootstrap intervals for the ten-world reports and a
  task-level model-by-lifetime analysis for examples-to-criterion. Existing
  three-world intervals are only exploratory.

# Open model investigation

- Implement the shared-parent plus task residual control for intermediate reuse.
  This is the remaining fixed-topology copy-on-write test in the V1
  implementation order and directly probes share-versus-specialize economics.
- MDL presence-gate pruning is conditional on deciding that the hard discrete
  route learner “works” well enough to justify pruning. The current evidence
  shows strong final operator/route recovery but poor online route inference;
  this decision must be made explicitly before adding Model 4.

# Explicitly outside V1

- Prospective macro regimes are reserved for Benchmark C after V1.
- Dynamic fork/merge/delete, self-refactoring, prospective abstraction, and the
  full train/inference/peak-capacity objective are Phase II.
- The strong self-refactoring hypothesis is explicitly not required for V1.

# Confirmation gate status

The gate remains closed. Oracle transfer, high-reuse advantage, reuse dependence,
novel-composition improvement, broad capacity controls, learned-model task-order
robustness, replay/forgetting separation, and the two-initialization pilot pass.
A clean checkout artifact-to-report rehearsal is still required by
`EXPERIMENT_PLAN.md` before opening worlds 100–129, and the remaining diagnostic
obligations below should be resolved or explicitly deferred.

# Execution order

1. Implement the shared-parent plus residual control at intermediate reuse.
2. Freeze batch protocol, tuning, and statistical summaries.
3. Run a clean artifact/plotting rehearsal.
4. Re-audit the gate before touching confirmatory worlds.

# Closed during this audit

- Current learnable-alpha int8 evaluation is complete. Continuous,
  Hypernetwork, Dense-24, and Dense-C were behaviorally evaluated on all ten
  development worlds, and current per-task Discrete was checked on world 0. The
  tracked result is `reports/retention/current-retention.json`.
- Reverse-order and replay-ratio robustness are complete on worlds 0–9.
  Continuous wins lifetime loss in all paired worlds at replay ratios 0, 1, and
  4 and under reverse order. The tracked result is
  `reports/robustness/robustness.json`.
- The two-initialization pilot is complete. Both model-seed pairs reproduce
  lifetime loss and novel-transfer wins in all ten development worlds. The
  tracked result is
  `reports/model_initializations/model-initializations.json`.
- Same-architecture fresh-task forward transfer is complete on worlds 0–9.
  Both models benefit from lifetime experience, but Continuous acquires more
  transfer in every world and the gain rises with task index. The tracked result
  is `reports/forward_transfer/forward-transfer.json`.
- Checkpoint true-route operator analysis is complete on worlds 0–9 for
  Continuous and per-task-annealed Discrete. Primitive matching and program
  performance through matched slots improve from 8 to 64 tasks in every world;
  Continuous's learned mixtures also reveal that a forced one-slot teacher route
  is not an oracle upper bound. The tracked result is
  `reports/operator_checkpoints/operator-checkpoints.json`.
- Explicit scrambled-ID invariance passes exactly for selected Continuous and
  Dense-C world-0 runs. Reassigned IDs are disjoint, while normalized metric
  rows, summaries, and final tensors are identical. The tracked result is
  `reports/scrambled_ids/scrambled-ids.json`.
- The batch-size sensitivity is complete on development worlds 0–2. A paired
  target batch 8 improves both models and preserves Continuous lifetime and
  novel-composition wins 3/3, although Dense-C's larger improvement narrows the
  architecture effect. The tracked result is
  `reports/batch_sizes/batch-sizes.json`.
- The free measured-recurrence and truncated-lifetime bridge analyses are
  complete across all ten development worlds. The reuse threshold drops from
  16 to 32 tasks and then plateaus; measured recurrence smooths the mean effect
  curve but does not tighten per-world crossover alignment. The tracked result
  is `reports/rho_bridge/rho-bridge.json`.
