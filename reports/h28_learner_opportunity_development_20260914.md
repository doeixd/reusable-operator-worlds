# H28-C learner opportunity development check

Status: **PROVISIONAL_DEVELOPMENT_CHECK**. This is not an accepted scientific
result and does not freeze or launch the H28-C learner gate.

The corrected harness learned six canonical operators on identity-context data,
jointly fit one four-angle context-1 adapter from operations 0–3, froze the
core, and fit a context-2 adapter from fresh support. Query labels were not
used for fitting. Mean canonical query NMSE was approximately `3.05e-5` on
identity, `3.06e-5` on context 1, and `3.09e-5` on context 2. The paired
shared-no-adapter means were approximately `3.05e-5`, `1.57e-2`, and
`1.19e-2`, respectively. Per-operation rows are in the JSON artifact.

The result is deliberately incomplete despite the controls now being present.
The oracle anchor, independent arm, and random-core control all run in the
same development harness and provide finite, non-vacuous comparisons. The
learned state also round-trips with a recorded maximum output error, and
serialized parameter bytes are reported for shared and independent arms.
Economic value was not measured, so the opportunity still cannot be classified
under the drafted plan. The result shows only that this small implementation
can fit the restricted shared-adapter construction under a development seed.

An earlier implementation passed canonical inputs to an observed-coordinate
function and was withdrawn before interpretation. The current harness maps
both support and query inputs through the observed frame and records the
correction in its input hashes. The JSON record is the canonical corrected run:
`h28_learner_opportunity_development_20260914_r6.json`.

The next action is to add and validate the missing oracle, independent, and
random-core arms, then run the registered non-vacuity, reconstruction, paired
cost, and restart checks before freezing any learner result.
