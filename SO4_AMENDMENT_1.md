# SO4 Amendment 1 (2026-09-15): the stream sub-clause must be able to fail

Status: FROZEN at this commit, before any SO4 cell or result exists (only the
unrun runner, scorer and tests have been drafted). It supplements, without
rewriting, `SO4_B2_RETEST_PLAN.md` (eba13b0).

# Defect

The plan's terminal clause is `W = median over 3 streams of M <= 0.05` in at
least 3 of 4 worlds. Its stream sub-clause is "in every world counted as
terminal-passing, at least 2 of its 3 streams have `M <= 0.05`".

With exactly three streams these are the same condition: the median of three
values is at most `t` if and only if at least two of them are. The sub-clause
therefore can never fail on its own, and `SO4_STREAM_FRAGILE` (terminal clause
holds, sub-clause fails) can never be returned. A registered label that cannot
fire is an estimand that cannot come out any other way. The error was found
while writing the classification tests, before any SO4 code ran.

# Correction

The stream sub-clause is replaced by one that can fail and says what the plan
meant: "a pass carried by lucky streams is not a pass".

**Stream sub-clause (amended):** in every world counted as terminal-passing
(`W <= 0.05`), no stream has `M > 0.10`, twice the threshold. A world whose
median passes while one stream collapses is terminal-passing but not
stream-robust.

Sizing, from existing results only (no SO4 data exists):
- SO3 BASE's worst stream in any world was 0.066. Its other failing cell was
  0.052. BASE-like variation therefore does not trip the clause.
- SO3's collapsed STORE_8 cell (world 4, stream 1: 0.122) and SO2's world-1
  single stream (0.126) would both trip it.
- The bound separates the within-world spread measured for the unchanged
  protocol from the collapses seen in failing cells. It does not re-use any
  quantity SO4 will measure.

**Classification ladder**, otherwise unchanged:
- `SO4_PASSES`: `W <= 0.05` in at least 3/4 worlds, AND no stream above 0.10 in
  any terminal-passing world, AND margin `>= 0.75` in at least 3/4 worlds.
- `SO4_ACQUIRES_ONLY`: the first two clauses hold; the margin misses in 2 or
  more worlds.
- `SO4_STREAM_FRAGILE`: `W <= 0.05` in at least 3/4 worlds, but some
  terminal-passing world has a stream above 0.10.
- `SO4_FAILS`: otherwise.

The registered prediction "stream sub-clause holds in every terminal-passing
world, given the terminal clause: 0.60" now refers to the amended sub-clause.
The probability is unchanged, and the change is disclosed here.

Everything else stands as frozen: worlds, seeds, streams, arms, margin
construction and coverage, gates, the other predictions, consequences, and
cost.
