# Current milestone

Milestone 001–002: deterministic teacher world and scratch-task difficulty
control.

# Completed

- Initialized the Git repository and Python package.
- Implemented deterministic random residual primitives.
- Implemented programs, opaque task IDs, unique task generation, and fixed
  train/evaluation datasets.
- Added world validity diagnostics for output scale and saturation.
- Added a NumPy scratch residual MLP and examples-to-criterion experiment.
- Added unit tests for determinism, uniqueness, opacity, shapes, and metrics.
- Passed the initial test suite and smoke experiment.
- Ran the 64-task seed-0 scratch control: median final NMSE 0.0742, final-NMSE
  task-index correlation -0.0963, output variance range 0.2129–0.2284, and zero
  saturation.
- Prepared the verified foundation as the initial Git checkpoint.

# In progress

- Tune or strengthen the scratch learner so examples-to-criterion at NMSE 0.05
  is not fully censored.

# Next

- Repeat the scratch control across three exploratory worlds and inspect both
  forward and reverse task orders.
- Implement the oracle compositor and verify declining late-life learning cost.
- Implement matched dense and continuous reusable-basis learners.

# Decisions

- V1 starts with exact reuse (`rho = 1.0`). The reuse continuum follows only
  after the oracle and at least one learned reusable substrate work.
- Fixed arrays are part of a generated `World`, preventing accidental data-stream
  differences between paired models.
