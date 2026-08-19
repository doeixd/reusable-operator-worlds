# V3 Confirmation Plan — PROMOTE on sealed seeds 300-329

**Status: FROZEN 2026-08-19. Once this file is committed and its hash is
recorded in `tools/check_prereg.py`, it must never be edited.** Sealed
seeds 300-329 have not been generated, inspected, or summarized at the
time of writing, and no promoter parameter may change after the freeze.

Inherits V1/V2 sealed discipline: parameter INTERVALS, not just signs;
interval misses are failures even when signs pass; one registered
re-derivation per gate; outcomes appended to `PREDICTIONS.md` whichever
way they land.

---

# 1. What is being confirmed

H11, the Abstraction Promotion hypothesis, in its three mandatory parts,
plus the refusal requirement. Development (worlds 0-9, frozen testbed)
gave a unanimous result on all three; the sealed block asks whether the
PARAMETERS replicate on worlds nobody has looked at.

# 2. Frozen configuration

World (`row.task_group_world.TaskGroupSpec`): 2 hidden task-group
families, `eta = 0.9`, canonical rho profile, `family_onset = 16`,
`new_primitive_families = True` at a fixed program position, 8 held-out
future tasks. Learner: the frozen H9 shared-residual configuration with
`operator_slots = 6`, basis frozen at task 16.

PROMOTE (`row.models.PromotingSharedResidualLearner`): sleeps after tasks
24, 32, 48, 64; candidate partitions by k-means over k in {2, 3, 4};
minimum cluster 3; abstractions fitted FUNCTIONALLY (300 Adam steps at
lr 0.02) on `Z_proposal`; acceptance by behavioral substitutability at
`epsilon = 0.02` on the disjoint `Z_validation`; selection charge
log2(M); V_transfer requires beating the single global abstraction on
held-out members; library reuse offered to every task at example 8 with
"none" always available. `lambda = ln 2`, `mu = 0`.

Comparator: the identical learner with PROMOTE disabled
(`--model shared_residual`), same world, same seeds, same data.

# 3. Registered outcomes and intervals

Each is scored over the 30 sealed worlds. Development means are given for
reference; the intervals are deliberately wider than development spread,
in the V2 style (V2 registered [4,000, 7,500] for a development slope of
~5,700).

**O1 — H11.2 causal comparison, both currencies.**
Two-part gain J(unpromoted) - J(promoted) at lambda = ln 2.
Development: mean +55,697, range [+46,127, +64,840], 10/10 positive.
REGISTERED: positive in at least 27 of 30 worlds AND mean in
[+35,000, +75,000] nats.

**O2 — H11.2 prediction is not sacrificed.**
Lifetime loss gain L(unpromoted) - L(promoted).
Development: mean +1,350, range [+761, +2,513], 10/10 positive.
REGISTERED: positive in at least 25 of 30 worlds AND mean in
[+400, +2,800] nats. A negative mean falsifies the "wins both
currencies" claim even if O1 passes, and must be reported as such.

**O3 — H11.1 migration, the three-sign pattern.**
Per world: D_task falls, D_shared rises, D_total falls.
Development: 10/10 on all three, D_total reduction mean 63.6%, range
[52.6%, 74.3%].
REGISTERED: the three-sign pattern in at least 28 of 30 worlds AND mean
D_total reduction in [0.40, 0.80].

**O4 — H11.3 prospective value.**
Held-out future block, 32-shot NMSE improvement (unpromoted minus
promoted) and adaptation-nats improvement.
Development: 32-shot mean +0.00310, range [+0.00172, +0.00438], 10/10;
adaptation nats mean +16, 10/10.
REGISTERED: 32-shot improvement positive in at least 26 of 30 worlds AND
mean in [+0.0010, +0.0055].

**O5 — Refusal, reported as a ratio rather than an absolute.**
Structured worlds against matched structureless controls (`eta = 0`,
same seeds, same everything else). Development, ten worlds each:
structured promotes 5.3 [3, 7] and refuses 4.2, with 31.8 of 40 tasks
reusing [22, 39]; the control promotes 2.9 [1, 4] and refuses 6.3, with
12.8 of 40 reusing [0, 24]. Reuse ratio 2.48x, library ratio 1.83x, and
the promote/refuse balance inverts between conditions.
REGISTERED: mean tasks-reusing ratio structured/control at least 1.4,
AND mean library size strictly larger in structured worlds, AND the
structured promote/refuse ratio above 1 with the control's below 1.
NOTE, registered in advance: the promoter does NOT refuse outright in
structureless controls — it promotes a smaller library there. The
hypothesis as originally written demanded no promotion at all; that
stricter form is ALREADY FALSIFIED on development and the sealed block
tests only the graded form above. This is recorded here so the weaker
criterion cannot be mistaken later for the one that was registered.

# 4. Analysis rules

- Paired per-world deltas reported directly; world-level means with
  bootstrap intervals; exact binomial sign tests for the counts.
- No world is excluded for any reason. A failed or crashed cell is rerun
  identically and the incident logged.
- Teacher family labels are used ONLY for post-hoc scoring, never by any
  learner and never in the promotion criterion.
- The four-way substitutability audit (private / family / global / zero,
  leave-one-out) is reported for the sealed block as a world-validity
  check, not as an outcome.
- Compute is logged, not charged (`mu = 0`); fitting cost is recorded
  separately as C_search for later V4 economics.

# 5. Single surrendered control

Per rung discipline, one control is surrendered to keep the block
affordable: the drifting-family and regime-change worlds (V3 spec 2.3)
are NOT run on sealed seeds. They remain development-only, and any claim
about observable instability is reported as development evidence.

# 6. Execution

Structured arm: 30 seeds x 2 models (promoting, shared_residual) = 60
lifetimes, which supply O1-O4. Control arm: 30 seeds x promoting only =
30 lifetimes, which supply O5 (the control needs no unpromoted
comparator because O5 is a reuse and library-size contrast). 90
lifetimes total in a resumable detached driver at 4 jobs, followed by
the future-block audit on the structured arm.
