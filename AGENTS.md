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
