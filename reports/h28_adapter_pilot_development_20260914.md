# H28-C adapter-discovery pilot: provisional development result

Status: PROVISIONAL DEVELOPMENT CHECK, not a confirmatory result and not
evidence of joint core discovery, autonomous closure, or economic value. The
pilot uses a supplied oracle core and a restricted four-angle orthogonal adapter
family. The machine-readable record is
[h28_adapter_pilot_development_20260914_r4.json](h28_adapter_pilot_development_20260914_r4.json).

# Construction and information boundary

The pilot uses six seed-0 canonical primitives, identity plus two independent
four-angle Givens contexts, 16 support examples for operations 0–3, and 32 query
examples for operations 0–5. The adapter is inferred from support pairs only;
query data cannot select angles or restarts. The core is fixed and explicitly
identified as ORACLE-CORE. The true angles and canonical inputs are not supplied
to the fitted arm.

The initial r2 output is retained but withdrawn: fitting and query prediction
passed canonical inputs into an observed-coordinate function. The oracle error
was therefore nonzero, exposing a construction failure before interpretation.
After correcting both support and query paths to map inputs into observed
coordinates, r3 was rerun from the same fixture namespace; r4 repeats the same
cell after the plan's status hash was updated. No r2 number is used.

# Results

| Context | Fitted adapter, withheld ops 4–5 | No adapter, withheld ops 4–5 | Oracle adapter |
|---|---:|---:|---:|
| Identity | 0 | 0 | 0 |
| Context 1 | 1.04e-8 | 0.01594 | <5.8e-33 |
| Context 2 | 8.82e-9 | 0.01230 | <5.7e-33 |

Values are mean canonical energy-normalized query MSE over operations 4–5;
they are not ROW NMSE or prequential nats. The fitted adapter is selected from
support MSE among three fixed starts and a fixed coordinate-search budget. It
uses one shared adapter across all six operations in each context. Both
nontrivial contexts satisfy the pilot triage rule (at least 10x reduction and
<=0.01); this is an oracle-core interface result at one structured family and
one development seed, not a general H28 result.

The fitted angles are within the registered coordinate-search resolution of the
planted angles. Support operations and withheld operations both transfer, but
the withheld pair is the relevant generalization endpoint. The identity anchor,
finite adapter budget, core immutability, query exclusion, and no-economic-value
flags are recorded in the JSON. Runtime was 0.408 seconds; no speedup or model
training was used.

# Interpretation and limits

This establishes that the restricted adapter family is identifiable from a small
support set when the shared computation is supplied. It does not show that a
learner can discover the shared core, that arbitrary dense coordinates are
recoverable, that adapter storage amortizes, or that the representation is a
nontrivial coarse-graining. The fitted core is an oracle, and the two nontrivial
contexts are generated from the same known family. A future learner pilot needs
matched independent and random-core arms, held-out coordinate families, complete
adapter accounting, and a separate future-learning endpoint.

# Validation

```powershell
python -m unittest discover -s tests -p 'test_h28_adapter_pilot.py' -v
python tools/h28_adapter_pilot.py --development-check --output reports/h28_adapter_pilot_development_20260914_r3.json
```

The output command refuses overwrite. This result was generated from uncommitted
development code and remains provisional. SO2's partial report and source were
not modified.
