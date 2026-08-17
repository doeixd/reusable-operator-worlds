# Reusable Operator Worlds

This repository implements the experiments described in
`neural_library_learning_v1_experimental_spec.md`.

# Current scope

The first milestone provides deterministic hidden primitives, unique programs,
opaque task IDs, fixed train/evaluation data, benchmark diagnostics, and a
scratch-task difficulty experiment.

# Quick start

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m row.experiments.scratch_difficulty --config configs/v1.yaml
python -m row.experiments.oracle_lifetime --config configs/v1.yaml --output artifacts/oracle
python -m row.experiments.learned_lifetime --config configs/v1.yaml --model dense --output artifacts/dense
python -m row.experiments.learned_lifetime --config configs/v1.yaml --model continuous --output artifacts/continuous
```

Scratch controls accept `--world-seed`, `--learning-rate`,
`--updates-per-example`, and `--output` overrides for small optimization and
replication sweeps without editing the canonical configuration.
