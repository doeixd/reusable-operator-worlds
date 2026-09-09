# SO1 restart audit (2026-09-09)

No SO1 scientific result exists. The September 8 attempt failed with two
MemoryErrors, four unsaved completions and 24 undispatched jobs. Its launcher,
log and empty report are preserved in reports/so1_launch_failure_20260908.
SO1_RESTART_AMENDMENT.md was frozen at c56e7f6 and protected at e253614 before
the repaired implementation. The original plan and predictions are preserved.

# Repairs and checks

| Defect | Repair | Verification |
| --- | --- | --- |
| Pool discarded successes if any worker failed | Immediate parent callback plus atomic worker-owned cell records | Successful callback survives later failure; actual trained artifact resumes without retraining |
| Parent interruption could lose all aggregate state | Per-cell results plus durable run manifest retain original start time | Independent artifact reload and corruption/cross-protocol rejection tests |
| Single-writer discipline was only assumed | OS locks for output, batch, launcher and individual cells | Second writer refused; lock released for subsequent owner |
| Fingerprint omitted world/config/input fields | Full resolved configs, software versions and input SHA-256 under protocol hash | Changed teacher alpha, model rank and evaluation precision each change hash |
| No reconstructible terminal model | Tensor-only state_dict, all codes, temperature and requires_grad state | Exact per-task reload and independently computed NMSE |
| Anchor checked after 30 cells | Six anchors first; failure prohibits other 24 cells | Driver integration test with a deliberately failing anchor |
| Conditional learned arm sampled new batches | Separate identity from sampling index; reuse paired oracle stream | Driver test compares every learned/oracle job's stream, budget, world and resolved config |
| No independent scorer or dose monotonicity | Independent terminal/checkpoint scores, envelope, pairs, persistence, dose monotonicity, predictions and ladder | Positive, negative, nonmonotone, incomplete and failed-instrument controls |
| RSS/free RAM treated as sufficient | Record Windows commit headroom, private memory, startup peaks, page-file use and other tenants | Real calibration required at launch commit before scientific grid |

The six anchors retain original SO1 streams, which differ from Stage D. They
therefore jointly test numerical implementation and resampling stability, not
the fast-versus-sequential numerical change alone. A failed anchor cannot be
interpreted as a scientific acquisition verdict or blamed on one of those two
causes without a separately registered diagnostic.

# Memory investigation

During repair testing, the pre-existing dispatch test's synthetic free-memory
probe launched four real Python/torch workers. OpenMP failed allocation and the
pool correctly raised BatchFailed. The arithmetic fixture now uses threads;
the actual scientific gate still uses ProcessPoolExecutor and compares all
scientific fields bitwise on nine reduced-update fast-family cells.

A subsequent Windows sample reported physical availability 7,035,424,768 bytes
but commit availability only 2,728,488,960 bytes, against a commitment limit of
47,013,195,776 bytes. The probing torch process alone held 735,285,248 private
bytes while resident size was 211,312,640 bytes. Thus RSS and page-file occupancy
were inadequate proxies for allocation headroom. These are current operational
observations; the September 8 log did not measure commitment, so its exact
historical limiting resource remains unresolved.

# Validation and restart command

Full suite: 261 tests passed in 42.428 seconds. check_prereg.py protects 44 frozen
files; check_invalid.py passes; git diff --check passes. The tests establish
implementation behavior, not a passed scientific or memory gate.

From clean committed code, run `python -u tools/run_so1_restart.py`. It performs
the minute-long host precondition, the actual fast-family equivalence/memory
gate at this commit, a second host precondition, the anchor-first grid, and
independent scoring after a complete run or an anchor failure. Operational
records live under artifacts/so1_restart; the report lives at
reports/so1_budget_bracket.json. --calibrate-only stops after the gate;
--reuse-gate permits an already-passed gate only at the same commit/protocol.
Launch long runs detached with hidden windows and durable stdout/stderr logs.
Do not advance HEAD while a scientific batch is running.

The restart's two-worker cap does not override its physical AND commitment
reserve checks. A failed host precondition means no gate/scientific cells are
launched; free resources and rerun the same command. Do not lower a threshold
or reinterpret a failed gate to keep the research narrative moving.
