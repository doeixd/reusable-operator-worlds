# L0d next gate: can sparse evidence change a usable program choice?

Status: DRAFT, 2026-09-17, informed by the completed preflight. Not frozen or
implemented. Tier 0 development instrument work, no PX7 verdict.

# Reason for this smaller step

The preflight at launch 4a6aa47 validated all twelve frozen libraries and 576
support measurements. Across the six staged libraries, all 96 library/task
pairs chose the identical hard route with 128, 32 and 8 examples; all query
errors were identical and below 0.05. The failed control libraries changed
routes but stayed above 0.05. That does not establish an inference failure in
a usable vocabulary. Source: reports/l0d_preflight.json and its archived
descriptive summary. These are development observations, not PX7 verdicts.

Before comparing early and delayed commitments, find out whether committing
with fewer than eight examples can select a worse program in a fixed usable
library. The earlier draft's depth/beam/posterior grid remains deferred.

# Bounded proposed measurement

- One existing library: STAGED5000, development world 0. First sixteen J2A
  held-out programs, exactly as in preflight; no new teacher, world or program.
- Fresh support-only exhaustive enumeration on nested prefixes of 1, 2 and 4
  examples, compared with each task's existing 128-example route and query
  score. Recompute the 128 anchor exactly before accepting a measurement.
- Query arrays remain unchanged and never influence route choice. Verify
  source hashes and that the library parameters never change.
- Report the hard route, best/second-best support MSE gap, query NMSE, route
  change from 128, and signed query error difference from 128 for every task.
  Preserve favorable, tied and unfavorable differences. Do not filter tasks
  by their sparse-support outcome or pool them as independent worlds.
- Count examples AND observed target scalars (16 scalars per example). One
  noiseless vector example can still supply considerable route information.
- The only new scientific computation is this one grid. No iterative search
  for a support subset, noise setting, observation mask or near-duplicate slot
  that makes the desired outcome appear.

# How the observation changes the next plan

If sparse choices differ and increase query error, quantify the observed size
and incidence before specifying an early-commit experiment. A faithful future
comparison would lock the sparse-evidence route while a delayed arm may use
later evidence; it must report the evidence schedule and charge both costs.
It would concern commitment as evidence arrives, not averaging internal states
at different execution depths. It does not yet license a posterior or beam
claim, and a one-world pilot cannot supply a general verdict.

If every sparse route equals its 128-example anchor, there is no observed
route-choice opportunity on this fixed depth-three grid. Stop this evidence-
size sweep; do not manufacture a positive by repeatedly changing the setting.
A depth-four proposal then needs a separate plan and bounded-memory executor
equivalence gate. If routes differ but query errors do not worsen, record that
distinction: route ambiguity need not have predictive cost.

Whether or not this small gate finds an opportunity, the broader PX7 mechanism
comparison needs its own precise protocol. Joint program prediction averaging
must not be replaced by averaging intermediate states through nonlinear
operators; temperature units, search heuristics, denominators and cost
accounting still require definitions before a full census.

# Operations

Implement and commit the runner, checker and gates before measurement. Use one
process and a new artifact/report path; never overwrite preflight. Timestamp,
hash and atomically save the output, and run the preregistration and invalid-
artifact checks. Time a short real invocation before assigning a runtime. If
the command will be long, give it to the PI per the standing instruction.
