# H28-C coordinate construction checkpoint

Status: PROVISIONAL DEVELOPMENT CHECK, not an accepted research result. Built
and checked the oracle fixture in
[H28_C_COORDINATE_GATE_PLAN.md](../H28_C_COORDINATE_GATE_PLAN.md).
The [machine record](h28_coordinate_gate_development_20260914.json) fingerprints
the fixture arrays, teacher source, plan, tests, implementation and runtime.
Source HEAD a31a397; new files were uncommitted. No learner was fitted.

# Construction checks

The fixture uses six canonical development-seed-0 primitives, two fixed
three-operation programs, independent 16/32 support/query inputs, and four
coordinate realizations. There are 48 tested split/context/program/step
boundaries. Each operator comparison uses the same canonical on-trajectory
states; rollout error is reported separately. All eleven new tests pass.

Maximum canonical energy-normalized discrepancy across each context's tested
boundaries (not centered ROW NMSE, not learned task performance):

| Context | Correct conjugacy | Weight-only transformation | Missing adapter |
|---|---:|---:|---:|
| Identity | 0 | 0 | 0 |
| Permutation | 0 | 2.08e-33 | 0.02525 |
| Orthogonal | 5.06e-31 | 0.12531 | 0.14512 |
| Dense, condition number 2 | 4.91e-31 | 0.19522 | 0.21040 |

These are numerical harness observations from known oracle functions. They
confirm that outer tanh must be transported with the rest of the computation:
rotating only U/V is insufficient under general mixing. The oracle's maximum
composed-rollout discrepancy is 1.65e-30. The identity-core sham stays identity
under tied inverse adapters and differs from the non-identity target in every
tested boundary. That does not exclude all wrong/random cores; their gate remains
owed before a learned-model comparison.

The derivative at zero agrees with independent central differences to maximum
absolute error 3.86e-11. A0 has trace 15.89741 and 1.1*A0 has trace 17.48715;
trace invariance under linear conjugacy therefore excludes that scaled negative
from the declared adapter family. This is not a difficulty-matched negative
world and says nothing about unrestricted nonlinear adapters.

# Costs and limits

The six known canonical operators have 12,720 float64 array-payload bytes. A
stored dense 16x16 coordinate matrix has 2,048; its inverse is derived. These
counts exclude executable and metadata costs and are not a coding-economy
result. In particular, the exact transformed operator cannot generally be
priced as an ordinary untransformed Primitive merely because dimensions match.

No artifact was loaded, no full world or sealed seed was generated, no torch
was imported, and no parameter updates, learned adaptation, future savings,
or nontrivial closure were measured. Exact conjugacy is a construction identity.

# Performance and next step

Pre-run timing: ten full fixture checks took 0.18123 s. The recorded check took
0.05769 s excluding startup. No speedup was applied; inverses are calculated
once per context/split and all arms reuse the same teacher/input draws. No
training-update timing applies. Existing SO2 source and its partial report were
left untouched.

Commands:

```powershell
python -m unittest discover -s tests -p test_h28_coordinate_gate.py -v
python tools/h28_coordinate_gate.py --development-check --output reports/h28_coordinate_gate_development_20260914.json
```

The output command refuses overwrite. A future development check must use a
new path and retain its provisional status.

Prepared [H28_C_ADAPTER_PILOT_PLAN.md](../H28_C_ADAPTER_PILOT_PLAN.md): infer a
four-angle interface from support examples of four operations, then test two
operations withheld from interface fitting. The core is supplied explicitly as
an oracle, and all restart selection is support-only. This pilot is drafted,
not launched. Commit/resume coordination with SO2 and a protected executable
plan remain prerequisites for a scientific pilot launch. Jointly learning the
core and measuring economic value remain subsequent questions.
