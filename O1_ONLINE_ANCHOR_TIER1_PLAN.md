# O1 Tier 1: does an order-free anchor supply make ONLINE formation work?

Status: DRAFT, 2026-09-23. **Tier 1, EXPLORATORY: produces no verdict.** Its job
is to decide whether the online anchor question is live and to size a registered
Tier 2 run. Development band 2 (worlds 10-19, allocated by the PI 2026-09-23);
this run uses worlds 10-12 only.

# Why

The offline line established what formation needs on the rotated substrate: a
quota of single-operation tasks, not their order (N1), not length-2 tasks (N1b),
not one per operation (N1c). The online staged protocol that SO2-SO4 carried
online relies on ORDER - three sequential lifetimes of length 1, 2, then 3 - and
passed in only 6 of 10 development worlds, world- and stream-dependently. If
order is not the active ingredient offline, an online stream that simply MIXES
single-operation tasks in may do what the ordered protocol could not reliably
do, and needs no knowledge of program length.

# Arms, as constructions

All at model seed 5000, replay stream 0 (the canonical `seed + 1`), scored the
same way: the TERMINAL model on the canonical 64 length-3 tasks, excluding every
anchor task and the novel-composition probe task.

| arm | construction | lifetimes |
|---|---|---|
| `STAGED` | SO2's protocol verbatim: `audit_so2_online_gate.run_arm('STAGED', w)` - 60 length-1, then 64 length-2, then 64 length-3, library carried | 3 |
| `SHUFFLED` | the same 188 tasks in ONE online lifetime, in a registered random order (seed `[1920, w]`) | 1 |
| `MIXED_L1` | the 60 length-1 and the 64 canonical length-3 tasks in ONE lifetime, random order (seed `[1921, w]`); no length-2 | 1 |
| `PLAIN` | SO2's protocol verbatim: `run_arm('PLAIN', w)`, the 64 length-3 tasks alone | 1 |

`STAGED` against `SHUFFLED` is the online order test: the same task multiset,
the same per-task online budget, the same code path, only the order differs.
`MIXED_L1` asks whether the length-1 quota alone suffices online, as it does
offline. The single-lifetime arms use `PlannedDepthRotatedLearner`, which at
uniform depth reproduces the committed online lifetime BITWISE - verified in the
feasibility gate below, through the real `learned_lifetime.run`.

**Differences that are part of the construction, stated rather than hidden.** A
single lifetime anneals its routing temperature over the whole stream and keeps
one replay buffer; `STAGED` anneals per stage and starts a fresh buffer at each
stage. That is what "order-free" means for this learner, and it is recorded
beside the result.

# Feasibility gate, passed before this plan (Tier 0)

On world 10 at scale 16, through `learned_lifetime.run`:

- **G1, uniform-depth equivalence:** the planned-depth learner and the committed
  fast learner give bitwise-identical initial libraries, terminal libraries and
  prequential loss.
- **G2, mixed stream:** 64 length-3 and 8 length-1 tasks in random order run end
  to end; every route has its planned length; prequential loss is finite; every
  diagnostic survives. The one extra task code is the novel-composition probe,
  which the scorer excludes.

# Necessity

**Target behaviour.** Formation of the rotated substrate ONLINE, from a stream.

**The arm that refuses it.** `PLAIN`: length-3 tasks only, no single-operation
task anywhere. Its committed SO2 values are 1.957 / 1.967 / 1.910 on worlds
0-2, against a 0.05 usability threshold missed by a factor of ~40 - the scale is
that threshold, not output variance.

**Impostors scored.** Task count and per-task budget: `SHUFFLED` holds both fixed
against `STAGED`, so a difference cannot come from compute. The length-2 tasks:
`MIXED_L1` removes them.

**Difficulty band, and its direction.** Terminal median NMSE, higher-is-harder,
band `(0.0005, 0.50)` on the arms under test; `PLAIN` sits above it by
construction and is the floor.

# Discriminating power

**The rule, as it will be applied.** A single-lifetime arm is LIVE when its
terminal median is `< 0.05` in at least **2 of 3** worlds. Denominator 3.

**Null and effect samplers, from committed values.** Null: the arm behaves like
the online floor, `U(1.910, 1.967)` (SO2 `PLAIN`). Effect: it behaves like a
passing online staged cell, `U(0.0088, 0.0400)` - the full range of SO3's seven
passing `BASE` streams (a draft of this plan cited 0.013-0.028 from memory; the
committed report says 0.0088-0.0400, and the upper end sits close to the
threshold, which is what the detection rate has to survive).
Intermediate: `U(0.05, 0.13)`, the range SO2's failing staged cells occupied.
2,000 draws, seed 11; rates in `reports/o1_tier1_design/rates.json`.

| regime | rate the rule fires |
|---|---|
| null | **false-fire 0.0000** |
| effect | **detection 1.0000** |
| intermediate | 0.0000 |

**What the checks do not cover.** One replay stream per cell. SO3 and SO4 showed
the stream alone moves an online cell by up to an order of magnitude, so a Tier 1
pass or fail on one stream is an indication, never a verdict - which is why this
is Tier 1. Three worlds cannot establish reliability, the property the online
line actually failed on; that is the Tier 2 question.

# What it decides

- **LIVE**: `SHUFFLED` or `MIXED_L1` passes in 2 of 3 worlds. A registered Tier 2
  follows on worlds 13-19 with three replay streams, testing RELIABILITY paired
  against `STAGED`.
- **NOT LIVE**: neither passes. The order-free online anchor route is recorded as
  not viable at this budget, and the online line stays closed.

Recorded either way, with the paired `STAGED` values beside them.

# Cost

12 cells; measured SO2 cells: `STAGED` ~14.5 min, `PLAIN` ~3 min; the
single-lifetime arms fall between. About 55 min as a pool of 3 with the parent
as sole writer. Precondition 8 GiB free, failing closed. `status.json` at launch.
