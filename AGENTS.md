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
- The current environment has NumPy and PyYAML but no PyTorch. The benchmark
  foundation and scratch control therefore use NumPy; later neural baselines may
  add an accelerator dependency after environment compatibility is confirmed.
- World seed 0 produced tightly matched task output variance (approximately
  0.213–0.228) and no output saturation. The initial scratch optimizer produced
  a flat final-error trend but did not reach NMSE 0.05 in 128 examples; tune the
  scratch optimization control before using that threshold as a validity gate.
