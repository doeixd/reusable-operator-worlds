# G5R diagnosis H-LBFGS protocol correction

Status: FROZEN after the original diagnosis endpoints were observed but before
any corrected H-LBFGS endpoint exists. Development worlds 0-2 only. No sealed
world is generated, loaded, or summarized.

# Discrepancy

`G5R_DIAGNOSIS_PLAN.md` registered H-LBFGS with learning rate 1.0,
`max_iter=500`, strong-Wolfe line search, gradient tolerance `1e-9`, and change
tolerance `1e-12`. The committed scorer at `3b3820a` additionally added
`0.5 * 0.0001 * sum(theta^2)` to the LBFGS closure. Weight decay was registered
for H-Adam and Q-Adam, but not H-LBFGS. The 18 original H-LBFGS cells are
therefore `INVALID_PROTOCOL_ARM`; their observed numerical pass cannot be cited
as a preregistered H-LBFGS result.

The discrepancy does not alter the registered diagnosis path or its primary
classification. H-Adam passed independently and was the only condition that
authorized Stage C. Stage C then passed its 2/3-world gate. Those arms did not
use the invalid LBFGS closure and are preserved unchanged in
`reports/rotated_g5r_diagnosis.json`.

# Corrected arm

Rerun only H-LBFGS with the original plan's data loss and no penalty term. Keep
every other registered element unchanged:

- canonical `configs/v1.yaml`;
- development worlds 0, 1, and 2, all six primitives;
- the same determinant-matched slot assignment and model seeds;
- the same 512 training and 2,048 query examples derived from
  `SeedSequence([1700, world, primitive, stream])`;
- PyTorch LBFGS at learning rate 1.0, `max_iter=500`, strong-Wolfe line search,
  gradient tolerance `1e-9`, and change tolerance `1e-12`;
- query NMSE threshold `<=0.02` per cell and the original arm gate of at least
  5/6 primitives in at least 2/3 worlds.

Write `reports/rotated_g5r_diagnosis_lbfgs_correction.json` atomically with the
frozen correction-plan path and hash, git commit, protocol constants, all 18
cell endpoints, finite/tensor checks, per-world counts, and the arm gate.

# Interpretation and stopping rule

This correction can restore or reject only the ancillary H-LBFGS evidence:

- corrected arm passes: `H_LBFGS_CORRECTED_PASS`;
- corrected arm fails: `H_LBFGS_CORRECTED_FAIL`.

Neither outcome changes
`ROUTE_INFERENCE_OR_ONLINE_INTERFERENCE_FAILURE`, because that classification
depends on the already-valid H-Adam and Stage-C gates. Do not rerun, tune, or
reinterpret any unaffected arm. Preserve the original report and append this
correction to the progress, predictions, learnings, and paper records.

# Acceptance checks

Accept the corrected arm only after exit code 0, exactly 18 cells, all expected
world/primitive keys, all-finite metrics and terminal tensors, report freshness,
the frozen code commit, unit tests, `git diff --check`, `tools/check_prereg.py`,
and `tools/check_invalid.py` pass.
