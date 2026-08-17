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

# Open protocol obligations

- Run learned-model reverse task order on paired development worlds. Only the
  scratch interpretation and oracle reverse-order control are currently complete.
- Run replay ratios 0, 1, and 4 for the selected Continuous and Dense-C models.
  This is required to separate forward transfer from reduced forgetting.
- Run a second model initialization for the ten-world pilot. Current development
  replication uses one model seed per architecture.
- Implement explicit scrambled-ID invariance rather than relying only on opaque,
  independently generated IDs.
- Resolve or ablate the effective update batch of one current plus one replay
  example, which differs from the suggested batch size eight.

# Open diagnostic obligations

- Implement same-architecture fresh-task baselines and report explicit forward
  transfer `scratch_task_loss - lifetime_task_loss` against task index, reuse,
  prior primitive exposure, and task similarity.
- At multiple lifetime checkpoints, evaluate current learned operators under a
  teacher-route diagnostic. Existing final functional matching identifies good
  operators, but does not yet isolate route inference throughout learning.
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
novel-composition improvement, and broad capacity controls pass. Learned-model
task-order robustness, replay/forgetting separation, and the second model
initialization remain unresolved. A clean checkout artifact-to-report rehearsal
is also required by `EXPERIMENT_PLAN.md` before opening worlds 100–129.

# Execution order

1. Add resumable reverse-order and replay-ablation sweeps.
2. Run the second model initialization on the ten development worlds.
3. Implement fresh-task forward-transfer and checkpoint true-route diagnostics.
4. Implement the shared-parent plus residual control at intermediate reuse.
5. Freeze statistical summaries and run a clean artifact/plotting rehearsal.
6. Re-audit the gate before touching confirmatory worlds.

# Closed during this audit

- Current learnable-alpha int8 evaluation is complete. Continuous,
  Hypernetwork, Dense-24, and Dense-C were behaviorally evaluated on all ten
  development worlds, and current per-task Discrete was checked on world 0. The
  tracked result is `reports/retention/current-retention.json`.
