# Project intent

Reusable Operator Worlds (ROW) tests whether reusable computational substrates
reduce lifetime/prequential learning cost on tasks generated from recurring
hidden neural operations.

# Working conventions

- Treat `neural_library_learning_v1_experimental_spec.md` as the research source
  of truth.
- Preserve paired comparisons: models in a comparison must receive identical
  worlds, task orders, examples, replay budgets, and evaluation sets.
- Score online examples before updating on them.
- Never expose hidden programs or primitive identities to non-oracle models.
- Keep experiment outputs reproducible from explicit world and model seeds.
- Commit completed, verified milestones as work progresses; do not leave finished
  implementation checkpoints only in the working tree.
- Use H1 Markdown headings for section beginnings in this file and
  `PROGRESS.md`.

# Commands

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m row.experiments.scratch_difficulty --config configs/v1.yaml
```

# Implementation learnings

- Python's built-in `hash()` is process-dependent and must not derive experiment
  seeds. ROW uses NumPy `SeedSequence` with explicit integer components.
- Primitive `U` and `V` matrices are independently spectral-normalized so random
  teachers have comparable operation scale.
- Training and evaluation examples are generated once per world from independent
  deterministic random streams; model code must consume those fixed arrays.
- Opaque task IDs are random tokens and are deliberately decoupled from program
  order and primitive IDs.
- Generate opaque IDs from fixed-length random bytes with an explicit uniqueness
  check. NumPy's `choice(..., replace=False)` cannot index the full `uint64`
  range on platforms whose sampling path uses signed indices.
- PyTorch 2.13 provides a native Python 3.14 wheel in this environment and is the
  autograd/runtime dependency for oracle and learned reusable models. The NumPy
  teacher remains the deterministic data-generation source of truth.
- World seed 0 produced tightly matched task output variance (approximately
  0.213–0.228) and no output saturation. The initial scratch optimizer produced
  a flat final-error trend but did not reach NMSE 0.05 in 128 examples; tune the
  scratch optimization control before using that threshold as a validity gate.
- The tuned NumPy scratch control uses learning rate 0.001 and four minibatch
  updates per arriving example. Across world seeds 0–2, final-NMSE/task-index
  correlations were -0.060, 0.002, and -0.039; every task reached NMSE 0.1 at
  support 64. NMSE 0.05 remained fully censored, so scratch validity currently
  rests on the flat 0.1 curve and flat continuous final error, not that stricter
  threshold.
- The oracle's random initialization is already below NMSE 0.1, so that threshold
  cannot demonstrate oracle transfer. Use NMSE 0.05, NMSE 0.02, zero-shot NMSE,
  and prequential cost for the positive-control claim.
- Oracle seed 0 reduced first-quarter to last-quarter zero-shot NMSE from 0.0345
  to 0.00192. Mean examples to NMSE 0.05 fell from 18 to 0, and examples to 0.02
  fell from 74.5 to 0. The unseen-composition zero-shot NMSE was 0.00215.
- The oracle's learned slots recovered the teacher primitives with a one-to-one
  mean normalized functional distance of 0.000539. Because the true route fixes
  slot identity, exact index alignment is expected; functional distance is the
  substantive operator-quality check.
- Reverse task order and two additional world seeds reproduced zero-example late
  performance at NMSE 0.05 and 0.02, so the oracle effect is not specific to the
  seed-0 curriculum.
- Dense baselines must mirror the teacher's three state-update/tanh stages. A
  conventional hidden MLP with only one output tanh begins at a severe geometric
  disadvantage. The fair dense architecture uses three task-conditioned
  state-space residual blocks with zero-initialized deltas.
- For `d=16`, a three-block Dense-P with hidden width 11 has 2,193 shared scalars,
  closely matching the eight-slot continuous basis's 2,112. Dense-C width 32
  approximately matches its matrix-multiply count and has 6,288 shared scalars.
- Continuous task-code LR 0.005 left routes diffuse (mean maximum coefficient
  0.216) and underperformed Dense-P. Raising only task-code LR to 0.05 increased
  the mean maximum coefficient to 0.466, improved one-to-one primitive distance
  from 0.0151 to 0.00349, and reversed the seed-0 comparison. Route inference is
  therefore an optimization bottleneck, not a cosmetic hyperparameter.
- In the seed-0 pilot, tuned continuous prequential NLL was -165,119 versus
  -155,993 for Dense-P, -161,984 for Dense-C, and -163,775 for width-128 dense.
  Tuned continuous novel-composition NMSE improved from 0.0453 to 0.00803 using
  32 code-only examples. Treat this as exploratory until paired-world replication.
- Tuned continuous beat Dense-P on all three exploratory worlds. Paired
  dense-minus-continuous prequential NLL differences were 9,126, 4,940, and
  5,746 nats (mean 6,604; three-world bootstrap interval 4,940–9,126). This is a
  replicated pilot, not a confirmatory estimate.
- Continuous also improved 32-shot frozen-library novel NMSE over Dense-P by
  0.0302, 0.0151, and 0.0114 across the three worlds.
- The simple 8-bit retained-code proxy is 29,184 bits for continuous versus
  33,928 for Dense-P. Actual symmetric per-tensor quantized evaluation changed
  mean final NMSE by at most about 2.2e-5 in the pilot artifacts.
- Parameter matching does not imply compute matching: the continuous basis uses
  an estimated 6,528 matrix/mixture multiply-adds per prediction versus 2,112 for
  Dense-P. Dense-C replication is required before making a compute-controlled
  claim.
- Dense-C replication is complete. Tuned continuous beat Dense-C on all three
  worlds by 3,135, 675, and 1,176 prequential nats (mean 1,662) at approximately
  matched multiply-add counts (6,528 continuous versus 6,144 Dense-C).
- Dense-C retains 66,688 8-bit proxy bits versus continuous's 29,184 because
  compute matching requires width 32 and 6,288 shared dense scalars. Continuous
  also improved 32-shot novel NMSE over Dense-C by 0.0210, 0.00944, and 0.00558.
- The seed-0 hard discrete library used 11/12 slots without collapse, recovered
  92.2% of teacher routes exactly (96.4% per position), and matched primitives
  at mean normalized distance 0.00229. Its weak prequential NLL (-134,784) is
  therefore a route-inference cost, not a failure to learn the operator library.
- Hard discrete zero-shot NMSE worsened from 0.103 in the first quarter to 0.160
  in the last because an unseen task's tie-broken default route becomes less
  generic as operators specialize. Nevertheless, examples to NMSE 0.02 fell
  from 128.9 to 25.3 and 32-shot novel hard-route adaptation reached 0.00413.
- Hardened discrete retention stores exact categorical route indices, not
  quantized training logits. Quantize only operator weights and reconstruct
  routes losslessly when validating the 26,112-bit proxy. This yields mean NMSE
  degradation 1.10e-5; quantizing logits directly is the wrong artifact model.
- Hardened discrete inference is only about 768 multiply-adds, versus 6,528 for
  the continuous mixture, exposing a strong storage/inference versus online
  learning-cost tradeoff.
- `rho` is implemented through task-specific correlated teacher parameters,
  followed by spectral renormalization of `U` and `V`. At `rho=0`, measured
  pairwise residual-function correlation is approximately zero; at `rho=1`, it
  is exactly one. Novel tasks receive independently derived correlated libraries.
- On development world 0, the causal endpoint reverses: Continuous beats Dense-C
  by 3,135 Gaussian-log-loss units at `rho=1`, while Dense-C beats Continuous by
  6,714 at `rho=0`. This is the first direct evidence that the advantage depends
  on latent computational recurrence.
- Checkpoint probes must operate on deep-copied models so code-only novel-task
  adaptation cannot alter lifetime training. Use four fixed novel programs at
  8, 16, 32, and 64 tasks.
- On exact-reuse world 0, Continuous mean 32-shot novel NMSE improved from 0.0298
  after 8 tasks to 0.00467 after 64; Dense-C improved from 0.0367 to 0.0159.
  This directly supports progressive learning-to-learn, beyond total loss alone.
- Call the density-based metric cumulative prequential Gaussian log loss. A
  fixed target precision of 1/256 adds the common density-to-mass term for a
  defensible quantized-target coding interpretation without changing model
  differences.
- Development worlds are 0–9. Confirmatory worlds 100–129 must remain untouched
  until symmetric tuning and all analysis rules are frozen in `EXPERIMENT_PLAN.md`.
- Symmetric stage-one tuning materially improved both models. Dense-C's best
  development-0–2 setting is global/task LR 0.001/0.05 with mean Gaussian log
  loss -168,768; its prior 0.001/0.005 setting scored -164,842. Continuous's
  best is 0.003/0.05 at -171,866; its prior 0.001/0.05 setting scored -166,505.
- After equal tuning, Continuous leads Dense-C by 3,098 mean Gaussian-log-loss
  units on worlds 0–2 and has better mean novel 32-shot NMSE (0.00237 versus
  0.00486). This supersedes both the untuned Dense-C comparison and the temporary
  observation that tuned Dense-C beat the old Continuous configuration.
- Stage-two finalists are Continuous 0.003/0.05 and 0.001/0.05, and Dense-C
  0.001/0.05 and 0.003/0.05, evaluated on development worlds 3–9.
- Learner operators no longer read teacher `alpha`. They initialize an independent
  learnable scale at 0.2. On world 0, decoupled Continuous scores -170,967 versus
  tuned Dense-C -166,521, so the earlier win is not caused by leaking alpha 0.35.
- Teacher-rank mismatch (teacher 16, learner 8) preserves a 3,296 log-loss
  advantage and better novel adaptation for Continuous. GELU learner mismatch
  preserves only a 441 advantage, showing that family alignment materially helps
  but does not wholly explain the seed-0 result.
- Adding a fixed identity candidate does not help Continuous on seed 0
  (-170,404 versus -170,967 without identity). The inability to skip a step is
  therefore not the source of the primary gain in this control.
- Per-task temperature re-annealing improves hard discrete log loss from -137,321
  to -146,146 and novel zero-shot NMSE from 0.150 to 0.0436 under the leak-free
  setup. Global annealing exaggerated route-inference difficulty, but per-task
  discrete still substantially trails Continuous.
- The leak-free oracle re-gate passes: early/late zero-shot NMSE is 0.0288/0.00159,
  early/late examples to NMSE 0.02 is 72.8/0, unseen-composition zero-shot NMSE
  is 0.00175, and primitive matching distance is 0.000484.
- Sparse world-0 `rho` controls show the crossover is above 0.75. At rho 0.5
  (measured residual correlation 0.064), Dense-C beats Continuous by 2,159 total
  Gaussian-log-loss units; at rho 0.75 (correlation 0.317), it wins by 1,615.
  Continuous already has slightly better 32-shot novel adaptation at both points.
- The complete leak-free seed-0 curve reverses between `rho=0.75` and `rho=0.9`.
  Dense-C-minus-Continuous cumulative-loss differences are -2,390, -2,280,
  -2,159, -1,615, +2,371, and +4,446 at configured rho 0, 0.25, 0.5,
  0.75, 0.9, and 1.0. Linear interpolation gives a descriptive crossing near
  configured rho 0.811, corresponding to measured residual correlation 0.454.
  This is a one-world development diagnostic, not a population estimate.
- Measured functional recurrence is strongly nonlinear in configured `rho` after
  spectral renormalization: the six seed-0 correlations are approximately
  -0.001, 0.003, 0.064, 0.317, 0.654, and 1.0. Plot and model effects against
  both the intervention and this measured explanatory variable.
- Report total, per-online-example, and per-target-scalar Gaussian log loss.
  Compute accounting distinguishes training-forward all-slot evaluation from
  hardened inference; it excludes backward and optimizer operations.
- Learned alpha scalars use a no-weight-decay optimizer group. `forward_tasks`
  groups samples by task ID before forwarding, preserving gradients and enabling
  useful batching when replay contains repeated tasks.

# Standing scientific doubts

- Reusable learners retain a favorable three-stage residual structural prior even
  after alpha/rank/activation mismatch controls. A generic low-rank hypernetwork
  remains necessary.
- The stage-one optimizer grid predates the learnable-alpha architecture change;
  its winners are useful development settings but must be revalidated before
  freezing confirmation.
- The effective online update batch is one current plus one replay example, not
  the suggested batch size eight. This is symmetric but must be disclosed and
  later ablated.
- Dense-C matches inference multiply-adds only. Continuous training backpropagates
  through every basis operator and uses materially more training compute.
- The initial n=3 bootstrap intervals are not inferentially meaningful. Public
  reporting should show paired world deltas directly until world count is larger.
- Identity-slot results are one world only; the negative ablation is not a
  population claim.
- Confirmatory worlds 100–129 remain sealed. Do not inspect them until these
  doubts and the development `rho` curve are resolved.
- The apparent seed-0 crossover must be replicated across development worlds
  before choosing a frozen confirmatory grid or interpreting 0.811 as a phase
  boundary.
