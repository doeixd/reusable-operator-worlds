# L0d depth-four execution gate

Status: DRAFT, 2026-09-17. Tier 0 development instrument work; no PX7 verdict.
This gate follows the completed sparse-evidence gate, which found no route-choice
opportunity at depth three. It asks the separate question whether one frozen
staged vocabulary remains usable when composed at depth four.

# Fixed construction

- Source: `STAGED5000`, development world 0, stage-three frozen library from
  the J2A artifacts. No training, new world, or new model parameters.
- Programs: 16 distinct length-four programs drawn deterministically from the
  teacher primitive alphabet with `SeedSequence([2704, 0, 4])`. Their support
  and query arrays use the same deterministic streams as J2A's held-out-task
  generator, with 128 support examples and the existing evaluation size.
- Search: support-only exhaustive ENUM over `12**4 = 20,736` slot routes. Query
  targets are never used to select a route. Score the chosen route on query
  data. The terminal route tensor is materialized one task at a time; its
  float32 lower bound is `128 * 12**4 * 16 * 4 = 169,869,312` bytes.
- Diagnostic only: map teacher primitive IDs through the stage-three traffic
  map and report its query score. This assignment is not an eligibility gate
  and cannot exclude a successful support-only route.

# Anchors and gates

Reload and hash-check the J2A source, stage-three model, and parent report.
Verify the variable-depth executor at depth three reproduces the prior hard
and soft forwards bitwise. Verify the dynamic depth-four all-route evaluator
against direct hard execution on selected routes. Require finite support/query
metrics, 16 distinct programs, one completed durable cell, unchanged frozen
library parameters, and an independent scorer with matching protocol hashes.

Report per task: program, support-selected route and support MSE, query NMSE,
diagnostic mapped route/query NMSE, route agreement between those diagnostics,
and execution seconds. Report medians and counts separately. The result is
classified descriptively as `SEARCH_USABLE` when all 16 ENUM query NMSE values
are at most 0.05, `MIXED` when some pass, and `SEARCH_NOT_USABLE` when none pass.
This classification describes this library and depth only; it is not PX7.

# Memory and restart discipline

Never allocate all tasks' route tensors together. The runner is single-process,
one Torch thread, and writes one hashed cell plus run/status/exit/error logs.
Relaunch reuses only a complete matching cell. A changed plan, code, source
hash or commit fails closed. Run preregistration and invalid-artifact checks
before the full invocation. Archive the small operational records with the
report. The gate is expected to be minutes, not hours; if runtime or memory is
larger than measured, stop and report the gate as blocked rather than lowering
the reserve.

No later beam, posterior, commit-late, or depth-five claim is licensed by this
gate. Those mechanisms require a separate frozen protocol if this execution
gate passes.
