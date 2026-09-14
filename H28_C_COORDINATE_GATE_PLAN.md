# H28-C T0: coordinate construction and adapter restriction gate

Status: DEVELOPMENT CONSTRUCTION CHECK, 2026-09-14, specified before executing
the new fixture. Not frozen for experimental scoring. Extends
H28_T0_INSTRUMENT_PLAN.md after its exact controls and metadata census. Outputs
remain provisional until the branch can be committed without affecting SO2's
resume contract. No learned-method result or economic verdict is produced.

# Question and bounded construction

Can the proposed conjugacy harness distinguish a correct coordinate interface
from missing/incorrect adapters, and does its adapter restriction exclude an
identity core that hides the task in a free decoder?

Use the existing `row.world.Primitive.random` implementation, development seed
0, d=16, rank=8, six operations, alpha=0.35. Generate no World or sealed seed.
Apply two fixed three-operation programs (0,1,2) and (5,3,1) to independent
support/query input draws (16/32 examples), using
`SeedSequence([0,2820,stream])`: stream 1 support, 2 query, 3 coordinates.
No fitting, routing, or label selection occurs. All primitive/program knowledge
is confined to these explicitly oracle construction controls.

Four realizations: identity, a fixed cyclic coordinate permutation, a QR-derived
orthogonal matrix, and an independent orthogonal pair with singular values
geometrically spaced from 1 to 2 (condition number 2). Use one R per realization
for all six operations. Decode with R and encode with its inverse; column-vector
convention `x=Rz`, implemented consistently for row-batched arrays.

The realized operation is `F(x)=R A(R^-1 x)`, including the outer tanh. The
naive weight-only version uses transformed U and V but leaves the outer tanh in
the observed coordinates. It must agree for identity/permutation and can fail
under general mixing. Do not normalize transformed weights again.

# Checks, all structural rather than scientific thresholds

Use numerical tolerance 1e-10 in canonical normalized squared error (denominator
mean target squared norm). Fail closed if target energy is <=1e-12 or any value
is non-finite. This is an energy-normalized discrepancy, not ROW's centered NMSE.

- At each operation boundary compare every implementation on the same mapped
  canonical on-trajectory input. Map predictions back to canonical units before
  measuring error; never compare functions on their different rollout states.
- Separately compare composed oracle rollouts with mapped canonical rollouts.
  This measures accumulated execution error, not a common-state operator audit.
- Oracle conjugacy, inverse round trip, and identity/permutation weight-only
  controls must agree within tolerance.
- Missing adapters and weight-only transformations in orthogonal/dense contexts
  must differ detectably (> tolerance) on this fixture. If not, report a failed
  opportunity control; do not tune the fixture after seeing it without a new
  version and disclosure. Do not interpret this tolerance as meaningful task loss.
- An identity core with tied inverse adapters must always execute identity.
  Compare it to the non-identity target to expose the adapter-only sham. The
  stronger random-core family test remains owed before a learned-model study.
- A negative context with operation `1.1*A` has Jacobian trace at zero scaled by
  1.1. Trace at zero is invariant under every invertible linear conjugacy, so a
  nonzero difference proves this particular operation cannot be a linear
  recoding of A. Check the analytic Jacobian against central differences and
  verify the trace witness. This is an instrument negative, NOT a difficulty-
  matched negative world or a claim about nonlinear adapters.

# Accounting limits

Record actual float64 array payload bytes for the six canonical operations
(U,V,b,alpha) and each stored R. Its inverse is derived and not separately
retained. Payload bytes exclude executable/metadata headers, so they are not a
total description length. Report adapter/core payload ratio only.

Do not price independent exact observed-coordinate implementations as if they
were ordinary canonical Primitive instances: outer tanh breaks that assumption.
Do not call deduplicating known oracle arrays learned economic value. A future
H28-C learner plan must supply matched-capacity independent implementations,
charge all adapters and acquisition, use held-out coordinates/compositions, and
measure future prequential loss. No break-even point is inferred here.

# Validation and resource contract

Use NumPy and existing teacher code only; import no torch or learner modules.
No existing Python file is edited. Tests cover deterministic pairing, stream
separation, common-state comparisons, full-nonlinearity conjugacy, primitive
order, inverse/condition guards, sham failure, derivative witness, and correct
payload counting. Hash the plan, utility/tests, teacher implementation, fixture
arrays and source revision; record Python/NumPy versions and dirty state.
Reject overwriting any output and check for an existing path before work.

Run a short timing pass on real fixture evaluations before the development
check; no training-update timing applies. Record any applied speedup and its
equivalence check. Stop well within minutes, without a lifetime, parameter sweep,
artifact loading, or competing long-running process. All outputs are explicitly
unaccepted development checks. Neither a conjugacy identity nor a negative
fixture proves that a learner can discover the shared operator.

# Next gate

If the construction works, specify a small support-only adapter-discovery pilot
with a supplied frozen core clearly labeled as an oracle arm. Inferring that
interface is separate from jointly learning a shared core. Before either run,
resolve SO2's commit/resume coordination and freeze the executable plan, budgets,
controls, held-out splits, and interpretation. This T0 check does not open
learned quotient autonomy, regularization, or hierarchical discovery.
