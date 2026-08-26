# E1-R: does export depend on the world having recurrent structure?

Status: DRAFT (freeze commit in `tools/check_prereg.py` before any code or
lifetime). A registered CONTROL rung under `EXPORT_BRANCH_PROGRAM.md`
(Amendments 1-2); it may not alter that document's decision tree, E1's verdict,
or the terminology contract. Development worlds 0-2. No sealed seeds.

# Why this rung exists

E1 passed both halves — a frozen discrete library executes held-out teacher
programs at or below its own trained-task loss, and support-only inference finds
the route. But **E1 ran only at exact reuse.** Its controls vary the LIBRARY
(scratch, and a library from an incompatible world); neither varies the property
this whole project treats as causal:

> Is export a consequence of the world having RECURRENT structure, or would a
> library trained on anything export equally well?

If a library formed in a world with no recurrence exports just as well, then E1
measured the architecture's function-space coverage rather than learned reusable
structure, and E2's meaning changes accordingly. This is the V1 question
(reuse pays only above `rho* ~ 0.83`) asked of exportability.

# The design decision this rung turned on

At `rho = 1` every task shares the base teacher library, so an unseen PROGRAM is
the natural held-out object — E1's construction. At `rho < 1`,
`_task_library()` derives task-specific primitives from the base library plus
task-indexed noise scaled by `sqrt(1 - rho^2)`, so the same program index means a
DIFFERENT operator for a different task. A held-out "program" is therefore not a
comparable object across `rho`, and testing one would confound a change of
target with a change of recurrence.

The held-out object is instead a **held-out TASK**, and the protocol is uniform:

- generate the world with `tasks = 76` from the same seed; tasks 0-63 are
  BITWISE the trained world's tasks (verified: identical programs, task IDs and
  examples), and tasks 64-75 are the tasks that world would have produced had it
  been longer;
- the lifetime trains on the 64 as before; the 12 remaining are the test set;
- at `rho = 1` this reduces exactly to E1's held-out-program protocol, because
  every task's library IS the base library.

The `rho = 1` cells therefore double as a REPRODUCTION CHECK of E1 under a
protocol built independently of it.

# Conditions and arms

`rho` in {0.0, 0.9, 1.0} x worlds {0, 1, 2}. The `rho = 1` lifetimes already
exist (`artifacts/e1_disc/world_*`); six new discrete lifetimes are run at
`rho = 0.0` and `rho = 0.9` from committed code at `configs/v1.yaml`. The grid
brackets V1's measured crossing (`rho* ~ 0.83`) with one point far below and one
just above.

Arms per held-out task, interface E1-P throughout (frozen library, route only;
DISC has no residual channel):

| arm | library | route | budget |
|---|---|---|---|
| **O** oracle | frozen | teacher program through a functional assignment matched against THAT TASK's teacher library | none |
| **R** inference | frozen | inferred from the task's 128 support examples | 2,000 Adam steps, lr 0.01 |
| **S** scratch | fresh random init, trainable | inferred | same budget |

Disclosed: `O`'s assignment is fitted per held-out task against that task's own
teacher primitives, which is strictly more teacher information than E1's single
global assignment used. This is necessary for comparability across `rho` (at
`rho = 1` the per-task assignment IS the global one) and makes `O` an oracle
CEILING, never evidence on its own. `R` uses no teacher information at any
`rho` and is the arm the verdict reads.

# Endpoint and decision rule

Per world and `rho`, on geometric-mean query NMSE over the 12 held-out tasks:

    M_R(rho) = log L_S - log L_R        export margin, inference arm  <- PRIMARY
    M_O(rho) = log L_S - log L_O        export margin, oracle ceiling

Registered readings:

- **RECURRENCE-DEPENDENT (expected).** `M_R(1.0) >= +1.0` and
  `M_R(0.0) <= +0.3`, in >= 2 of 3 worlds, with `M_R(0.9)` between them.
  Export is then a consequence of learned recurrent structure, E1's result is
  about reuse rather than about architecture, and E2 proceeds as planned.
- **RECURRENCE-INDEPENDENT.** `M_R(0.0) >= +1.0` in >= 2 of 3 worlds. Then E1
  measured function-space coverage, not learned reuse. E1's verdict is NOT
  withdrawn — it stands as measured — but its INTERPRETATION is amended in the
  ledger, and E2 must add a `rho = 0` arm to every claim before any composition
  language is used.
- **INTERMEDIATE.** Anything else: reported as a curve with no verdict, and E2
  inherits the `rho = 0` arm as a precaution.

Non-vacuity, required before reading anything:

- the `rho = 1` cells reproduce E1's numbers within 0.15 log units (same
  substrate, independently built protocol);
- tasks 0-63 of the 76-task world are verified identical to the trained world's;
- `R` and `S` each reduce their own training objective by > 1% (mode-consistent,
  per E1 Amendment 2);
- measured functional recurrence is reported per `rho` from the world itself, so
  the x-axis is the MEASURED quantity and not only the configured knob (V1's
  coordinate lesson).

# Registered predictions

Ours. `M_R(1.0)` reproduces E1 at roughly +1.8 to +2.8. `M_R(0.0)` near zero:
with independent per-task teachers there is no shared operator set to carry, so
a frozen library should be worth no more than a fresh one — and `S` itself may
be poor, so we predict the ABSOLUTE losses at `rho = 0` to be high for both arms
and the MARGIN to vanish, which is the quantity the rule reads. `M_R(0.9)` above
+1.0 but below the `rho = 1` value, since 0.9 sits just above V1's crossing.
Confidence is highest on `rho = 1` (a reproduction) and lowest on `rho = 0.9`.

If `rho = 0` exports anyway, our reading is that twelve tanh-residual operators
span enough of this function class that any trained basis serves, which would
make the `d = 16, rank = 8` operator budget — not the learning — the thing E1
measured.

# Cost

Six discrete lifetimes (~15 min each, pool of 3). Scoring: 3 `rho` x 3 worlds x
12 tasks x 2 adapting arms x 2,000 steps, plus the free oracle arm — comparable
to E1's, a few hours in the background, resumable through a
protocol-fingerprinted per-cell cache.

# Explicitly out of scope

Any change to E1's verdict; E2's constructed world; the mixture substrate (it
failed the E1.0 gate); depth generalization; program recognizers.
