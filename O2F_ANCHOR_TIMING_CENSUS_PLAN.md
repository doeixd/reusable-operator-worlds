# O2F: does EARLY anchor supply predict which order-free streams form? (Tier 0 census)

Status: **REGISTERED 2026-09-25, before any O2 cell finished.** It could not be
committed then, because O2 was in flight and its runner forbids commits. The
registration evidence is this file's sha256 in the timestamped OptMem working
log (entry of 2026-09-25 ~21:05Z), written while `status.json` showed 0 of 70
cells done. It is committed with O2's result. Observational census over O2's
frozen artifacts: no training and no new world.

**Runs only if O2's primary label is not `ORDER_FREE_RELIABLE`.** If order-free
formation is reliable, there is no failure to explain and this census is moot.

# Why

O1 and O2 run the order-free stream as one random permutation per (world,
stream). A permutation that happens to put few single-operation tasks early
leaves the library forming from length-3 tasks at the start of the lifetime,
while routing temperature is still high. J1 showed that early uninformed
commitments can lock in. If that is why order-free streams fail when they do,
the fix is an online SUPPLY SCHEDULE, a quota of anchors early, rather than a
larger quota. If early dose does not predict, timing is not the lever, and the
next rung looks elsewhere (world identity, replay draws).

# Predictor, outcome, unit

- **Predictor `E`:** the number of length-1 tasks among the first 32 stream
  positions. It is a pure function of the registered order seed, and was computed
  for all 42 cells BEFORE any outcome existed:
  `reports/o2_design/o2f_predictors.json`.
  - `SHUFFLED` values range 9-14, `MIXED_L1` 11-19.
  - The within-world spread is about 2-5.
  - It is graded, not bimodal (the SG6 check).
- **Outcome:** O2's registered cell outcome, terminal median `< 0.05`, with
  non-finite counted as failing.
- **Unit:** the 42 single-lifetime cells (`SHUFFLED` and `MIXED_L1`, 7 worlds x
  3 streams), analysed WITHIN (arm, world) groups, so world identity cannot
  masquerade as timing.

A stream changes both order and replay draws. Replay luck is independent of
`E`, so it adds noise, not bias.

# Decision rule (registered)

A group is INFORMATIVE if its three streams contain at least one pass and one
failure, and the mean `E` of passing and failing streams differ. Let `n` be the
number of informative groups (denominator: informative groups, at most 14) and
`a` the number in which the passing streams have the higher mean `E`.

| condition | label |
|---|---|
| `n < 5` | `UNMEASURABLE_WORLD_DRIVEN`: outcomes do not vary enough within worlds to test timing. This is a finding about worlds, not a null about timing. |
| `n >= 5`, `a/n >= 0.85` | `EARLY_DOSE_PREDICTS` |
| `n >= 5`, `a/n <= 0.60` | `NO_TIMING_EFFECT` |
| otherwise | `INCONCLUSIVE` |

The four labels partition every possible outcome.

# Necessity

The necessity gate does not apply in its usual form: this is an observational
census over frozen artifacts and trains nothing. The reason is stated rather
than skipped. It asks what O2's streams already did. The refusal comparison is
built into the rule: within the same world, the stream that received fewer
early anchors is the one that "refused" early supply. The impostors scored are
world identity (removed by within-world analysis) and replay luck (independent
noise). The difficulty band, NMSE higher-is-harder, is O2's registered one.

# Discriminating power

**Samplers.** The actual `E` values of the 14 groups are used, not idealised
ones. Each group draws a world log-odds `a_w ~ Normal(m, sd)`. Each stream then
passes with `sigmoid(a_w + b (E - mean_group E))`.
- **Null:** `b = 0`, at `(m, sd)` = (0.4, 1.5), (0.4, 0.8), (0.4, 2.5) and
  (1.5, 1.5). `m = 0.4` is the incumbent's ~0.58 cell pass rate.
- **Effect:** `b = 1.0`, one extra early anchor adding one log-odds unit.

4,000 draws each, seed 11.

| regime | fires `EARLY_DOSE_PREDICTS` | `UNMEASURABLE` |
|---|---|---|
| null (0.4, 1.5) | **false-fire 0.029** | 0.09 |
| null (0.4, 0.8) | false-fire 0.027 | 0.01 |
| null (0.4, 2.5) | false-fire 0.023 | 0.37 |
| null (1.5, 1.5) | false-fire 0.021 | 0.27 |
| effect `b = 1.0` | **detection 0.863** | 0.00 |
| effect `b = 0.75` | detection 0.708 | 0.01 |

False-fire is at most 5% under every null, and detection is at least 80% at
`b = 1.0`. **What the checks do not cover:** only a STRONG timing effect is
detectable. At `b = 0.75`, detection is 71%, so a `NO_TIMING_EFFECT` or
`INCONCLUSIVE` label does not exclude a moderate one. One predictor is
registered, and no others will be tested for a verdict; positional coverage
(when the sixth operation first appears) is recorded descriptively only, because
N1c found count, not coverage, operative offline. The census cannot separate
timing from anything else that co-varies with a permutation.

# Outputs

Per cell: `E`, outcome, and the descriptive coverage position. Per group:
informative flag and direction. Totals `n` and `a`, and the label. The report is
`reports/o2f_anchor_timing.json`, from a scorer committed before it is run.
