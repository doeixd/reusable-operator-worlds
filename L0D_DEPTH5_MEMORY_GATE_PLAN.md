# L0d depth-five memory gate

Status: DRAFT, 2026-09-17. Tier 0 implementation and feasibility check; no
PX7 verdict. The depth-four gate passed one frozen library, but its plan does
not license a depth-five result. A full depth-five ENUM tensor would occupy
about 1.90 GiB before temporaries, so first test a bounded route-block
implementation.

# Fixed check

Use the same STAGED5000/world-0 stage-three library and one deterministic
length-five program generated from `SeedSequence([2705, 0, 5])`. Use 128 fixed
support examples and the existing evaluation size. No training, new world or
new model parameters. Search all `12**5 = 248,832` routes in blocks of 1,024.
Keep only the best route and loss per block, then the global best; never retain
all route outputs. Record process peak RSS when available, block count, bytes
of the largest block's terminal output, runtime and selected-route query NMSE.

# Equivalence gates

On a small depth-three input, compare the chunked evaluator's complete route
loss vector to the existing all-route evaluator within `1e-6` absolute/relative
tolerance (batching changes reduction order), including identical argmin route
index. On depth five, compare selected-route support loss to direct hard
execution within `1e-5`,
require finite metrics, unchanged library hashes and exactly 243 blocks. The
report is classified `MEMORY_SAFE_SEARCH` only when all checks pass; it is a
feasibility result, not a claim about depth-five generalization or PX7.

The gate must fail closed if the plan, source, implementation or commit changes
on resume. Archive its operational records and run the independent scorer.
Do not expand to more programs or libraries unless this gate passes and a new
protocol is frozen.
