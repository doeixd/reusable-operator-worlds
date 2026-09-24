# A2 claims audit: necessity failure versus genuine refutation (2026-09-24)

IMPLEMENTATION AUDIT (`FOLLOWUP_AUDITS.md` item A2): it is read-only, runs no
experiment, and changes no number. The site-level reading was done by a
read-only subagent. The parent checked each MISWORDED item against its source
(including `PREDICTIONS.md` E6D at lines 7016-7071, E6.2 at line 7294 and the
H48b table) before applying anything.

**Disposition, applied in the same commit:**
- **M1, M2, M3, M5, M6, M8 and M10** reworded in `README.md`. M2 uses
  "K <= 8 per slot here", following the H48b table, instead of the proposed
  "between K = 8 and 16".
- **M4** reworded in `paper/draft.md`.
- **M9, M11 and M12** reworded in `RESEARCH_STATUS.md`. So was the uncertain
  item 1 (E7, scoped to "this testbed's corpus").
- **M7 and M13** appended to `PREDICTIONS.md` as a correction, not edited in
  place.
- The SO1 row in `README.md` now points to SO1R (uncertain item 4).
- **Not applied, left optional:** the scope tightening on genuine refutations
  (paper 610, 1305-1308; README 147-148) and the class-ambiguity notes.

---
# A2 claims audit: necessity failure versus genuine refutation

Executed 2026-09-24, read-only, against `FOLLOWUP_AUDITS.md` section A2. Scope files:
`PREDICTIONS.md`, `paper/draft.md`, `README.md`, `RESEARCH_STATUS.md`. Gate
definitions from `DESIGN_ADEQUACY.md` and the `AGENTS.md` entries "NECESSITY IS A
THIRD GATE", "OPPORTUNITY AND DISCRIMINATION ARE TWO GATES" and the terminology
contract. No file in the repository was edited. No Python was run.

Classes:
- **NEC**: NECESSITY FAILURE. The task did not require the behaviour, so the result is about the construction.
- **OPP**: OPPORTUNITY FAILURE. The generator or architecture could not produce the effect.
- **DISC**: DISCRIMINATION FAILURE. The data or instrument could not distinguish, so the result is unmeasurable here.
- **GEN**: GENUINE REFUTATION. The behaviour was available, required and measurable, and it did not pay.

The wording column judges whether each sentence matches its class. For NEC, OPP
and DISC, a sentence is MISWORDED when it states a general claim about the
behaviour instead of a claim about this construction. For GEN, this pass flags
only understatement, because the audit brief is to "reword only what is
misclassified". Overreach on genuine refutations appears separately in section 4
and is not counted as MISWORDED.

## 1. Table of negative claims

| # | result | class | justification (one line) | locations | wording |
|---|---|---|---|---|---|
| 1 | V1 secondary: transfer improves before lifetime cost | GEN | Replicated on 10 worlds; transfer was measurable and did not lead | paper 363-366; AGENTS learnings | OK |
| 2 | Soft mixtures approximate a Bayesian route posterior | GEN | Exact posterior computed; no correlation (Spearman -0.03) | paper 173-175, 384-403 | OK |
| 3 | MDL presence gating never gives a compact sufficient library | GEN | Grid tuned under a frozen rule; the teacher has 6 slots, so compaction was available | paper 603-611 | OK (optional scope, see section 4) |
| 4 | Consolidation gates and the within-lifetime amortized compiler | GEN | Both built and scored; the compiler's self-test passed, so the construction was sound | paper 170-173, 577-601 | OK |
| 5 | V2: the shared-residual learner loses the two-part code 30/30 | GEN | Sealed; the pre-registered reversal was confirmed | README 24; paper 41-44, 536-547, 562-568, 945-956 | OK |
| 6 | P-2026-08-18-A: variational coding does not win the two-part cell | GEN | Falsified 0/3; the ledger already appends a narrowing to "a fixed topology, this Gaussian implementation" | PRED 236-275 | OK |
| 7 | P-2026-08-18-D: residuals do not cluster by family | OPP | Two spare basis slots absorbed the structure, so the residual channel never received it | PRED 180-234 | OK ("does not hold in this testbed") |
| 8 | V3: absolute refusal in structureless worlds is falsified | NEC | An abstraction over noise still pays at the two-part rate (PRED 608-614), so the task never required refusal | PRED 175-178; paper 672-683 | OK (the paper gives the economic reason) |
| 9 | V4.1: compaction/fragmentation retracted and not supported | OPP | The abstractions are mutually distinct, so there is no redundancy to compact | PRED 277-327; paper 699-701, 978-982 | OK ("here", "at this scale") |
| 10 | V4.2: factorization fails the matched-bit compression null | NEC | COMPRESS is the cheaper impostor; the atoms are 4-8x overparameterized | PRED 329-363, 693-741; paper 982-990 | OK |
| 11 | Value filter at promotion time fails | GEN | Implemented and run; the value estimate available at birth is structurally minimal | PRED 490-539 | OK |
| 12 | V4 premise and V4R sealed: lifecycle machinery not needed; FORK/FACTORIZE do not pay | NEC | Local compression slack is the impostor that every structural edit loses to | PRED 667-691; README 26, 114-116; paper 3-6, 54-60, 685-714, 1017-1022, 1040-1043; RS 630 | OK (qualified "at this scale" throughout) |
| 13 | Single-abstraction retention "no demonstrated net value" | DISC | A mid-lifetime deletion is unpaired; the endpoint J could not distinguish the options; later superseded | PRED 790-833, 835-870 | OK (superseded in the ledger) |
| 14 | V5.1: H* proportional to D(A) falsified | OPP | Residual rank moves cost and utility together, so D is not an independent knob | PRED 1339-1378; RS none | OK ("in this substrate") |
| 15 | H20b / sealed C3: learned library does not realize the schema economy | GEN | Measured on promoted atoms and survived review 55; attributed to PROMOTE's unit | PRED 1903-1995, 2293-2307, 3332-3339; paper 1045-1060 | OK |
| 16 | H21 fails at r_meta 0.9 and 0.0 | OPP | A rank-2 schema cannot express members outside its subspace; capacity limit by construction | PRED 1997-2034 | OK |
| 17 | H27 spectral mechanism falsified; sealed C4 sign clause fails | GEN | Mechanism measured and absent; the gap itself was unstable | PRED 2036-2072, 2309-2351 | OK |
| 18 | Sealed C2 15% clause | DISC | The clause has no denominator and crossings are unreachable at F=12 | PRED 2114-2147 | OK ("UNRESOLVED") |
| 19 | "Post-hoc refactoring cannot be the remedy" / population span dead | n/a (withdrawn) | Instrument defect: unaligned coordinates. Narrowed at PRED 2805 and retracted at 3236 | PRED 2705-2767, 2805-2826, 3236-3248, 3369-3390 | OK (retracted in ledger) |
| 20 | V6: H30/H31/H33/H35 not supported; prospective pressure harms | GEN | The objective was implemented (after review 55) and measurable, and harmful at the tested pressures | PRED 3446-3516, 3692-3807; paper 1061-1089; RS 631 | OK (PRED 3469 overreaches but 3507-3516 scopes it in the same entry) |
| 21 | Over-alignment mechanism refuted | GEN | Predicted drops in discrimination and code sensitivity were absent | PRED 3608-3690 | OK ("refutation stands on an absence") |
| 22 | H39 census C0 negative: post-hoc schema cannot express unseen family | OPP | The ordinary learner's residual span lacks the directions at any rank | PRED 4115-4169; paper 1095-1104; README 125-127 | OK ("neither supported nor falsified") |
| 23 | H47 B1: imposing discrete commitment on a continuous family | NEC | No membership exists in that world (PRED 4596-4605) | PRED 4585-4700; paper 1148-1153; README 132-133 | OK in PRED/paper; README folded into #24 |
| 24 | H47 B2: learner ignores two orthogonal groups | NEC | At K=32 x 2 one channel absorbs both subspaces (`capacity_forces_structure`) | PRED 4702-4754; paper 1155-1160; README 133-135; DESIGN_ADEQUACY 49-51, 83-90 | PRED/paper OK; **README MISWORDED** |
| 25 | H48b: discrete identity never pays for present cost | NEC | At K=4 grouping has future value but no present value, so non-discovery is rational | PRED 4756-4815; paper 1161-1173; README 136-138 | PRED/paper OK; **README MISWORDED** |
| 26 | H49: retrospective signals discriminate only on the organized representation | NEC | The label-free objective never rewards the partition (necessity, per DESIGN_ADEQUACY) | PRED 4865-4916; paper 1175-1183; README 139-140 | PRED OK; **README and paper 1182-1183 MISWORDED** |
| 27 | H50: bounded migration recovers ~0% of separation | GEN | Operator matched, instrument validated on L_4, sham controlled | PRED 4949-5009; paper 1183-1195; README 141-143 | OK ("within the tested operator") |
| 28 | H51: preserved state/decomposable basis do not buy reorganizability | GEN | Balance-gated arms, unchanged instrument, no separation | PRED 5051-5129; paper 1292-1318; README 144-148 | OK (optional scope, see section 4) |
| 29 | H53: co-formation neither cheap nor discriminating | GEN | Heads verifiably distinct; the L1 future impossibility is correctly called "by construction" | PRED 5445-5523; paper 1197-1209; README 158-167 | OK ("frontier, if it exists, lies deeper") |
| 30 | E0.1/E1.0: MIX fails route expressibility | OPP | A mixture's solution is not a route, so MIX rows are uninterpretable as export failures | PRED 5329-5355 | OK |
| 31 | E5: writer misses quality; amortizer never pays | NEC | Incumbent search does not degrade (exponent 0.068), so there is nothing for a writer to sell. The quality miss is a real miss for this recognizer | PRED 6217-6299; README 37, 180-198, 214-217; RS 634 | OK ("at this scale", "in 6 of 6 cells") |
| 32 | E5.1: no horizon located ("SEARCH BINDS FIRST") | DISC | A first-crossing rule fires on noise (42.5%) | PRED 6430-6460; README 228-235 | OK |
| 33 | E6D: retrospective code cannot refuse a dead pattern | OPP | A step-function generator puts no signal of non-continuation in the observed data; corrected the same day (PRED 7016-7071) | PRED 6933-7071; README 304-315 | PRED OK (corrected); **README MISWORDED** |
| 34 | E6E: registered prospective estimator gives NO IMPROVEMENT | GEN | The argmax estimator was testable on a decaying case and failed; case C correctly unscoreable | PRED 7073-7162; README 337-348 | OK |
| 35 | E6.2: compilation fails at matched budget and does not pay | NEC | A definitional macro is the cheaper impostor, and the generator has no recurring subroutine structure (PROGRESS 4110-4113) | PRED 7253-7344; README 38, 363-406; RS 336-339, 582 | Licensed-claim text OK; **PRED 7294, README 38, README 391 MISWORDED** |
| 36 | E6 line summary "macros cannot be timed or compiled" | NEC/OPP | E6D timing failure was an impossible control; E6F timed with a gate; compilation is #35 | RS 582 | **MISWORDED** |
| 37 | E7 census: parameterized-macro rationale refuted | NEC | No parameterized family beyond the learner's own unplanted-structure null in this testbed's corpus | RS 336-338, 582-583 | OK (borderline, see section 4) |
| 38 | Loop census / iteration world spec | NEC | Straight-line generator; contractive operators make iteration count unidentifiable | RS 318-322, 580-581 | OK (the correction is stated) |
| 39 | G5 / G5R: rotated substrate not learnable online | GEN | Registered protocol, matched family, fails 0/3; scoped as not a global impossibility | PRED 7346-7423; README 40-41; paper 869-895; RS 401 | OK |
| 40 | SO0: no budget axis separable | DISC | Existing cells do not control the axes | README 42 | OK |
| 41 | SO1: learned routes fail at both envelopes | GEN | Oracle arm passes at the same budget | PRED 7941ff; README 43; paper 1484-1493 | OK (note: "wall is the route writer" is superseded by SO1R, see section 4) |
| 42 | J1 / CF4 refuted | GEN | Search in the loop ran and locked in; scoped "as registered, at SO1's envelope" | PRED 8114-8158; README 46; paper 1520-1530; RS 403, 643 | OK |
| 43 | SO2/SO3/SO4 online staged formation FAILS; SO2-P rescue does not replicate | GEN | Registered criteria missed; worded as world-dependent, not refuted | PRED 8372ff, 8662ff, 8744ff; README 50, 111-113; paper 900-918, 1667-1843; RS 385-409 | OK |
| 44 | Mechanism hunt: three candidates withdrawn | DISC | Measures are scale-incomparable or null once made comparable | PRED 8877ff; README 51; paper 1845-1879; RS 411-463 | OK ("no identified mechanism") |
| 45 | L0d x4: no route ambiguity on usable vocabulary | NEC | The task never required inference under ambiguity (TOO_EASY) | PRED 8917-8986; README 59; paper 1881-1902; RS 513-579 | Scope OK; **RS 566 label MISWORDED** (low priority) |
| 46 | SG0 NO-HEADROOM / sub-triage SATURATED; PX7 retired; proposer branch closed | NEC | Staged identifiability 53.9 is TOO_EASY; controls in band | PRED 9117-9236; README 60-61; RS 71-122, 669-680 | PRED/RS OK; **README 60 MISWORDED** |
| 47 | SG6 closed before opening | DISC | Bimodal axis, effective n = 2 | PRED 9238-9273; README 62, 70; RS 124-153 | PRED/README OK; **RS 150-153 label MISWORDED** |
| 48 | RF0b semantic-motif ceiling unresolved | DISC | One world's strong cell failed its permutation null | README 39 | OK |
| 49 | N1b: length-2 anchors do nothing (NB2 not supported) | GEN | Available, measured, did not suffice at this budget | PRED 9346-9377; RS 11-12, 43-47 | OK (scoped to the offline line) |
| 50 | N1c: NB4 refuted; NC1/NC3 not supported | GEN | Design separated the count/coverage confound | PRED 9400-9435; RS 52-58 | OK (explicit scope) |
| 51 | C4 plasticity (SO2-P LR sweep) did not improve anywhere | GEN | Swept 2x/4x/10x; untested ratio named | RS 330-335 | OK |
| 52 | H29 restructuring term structurally zero | OPP | Promoted abstractions carry `requires_grad=False` | PRED 2149-2187 | OK |
| 53 | Aggregate label "N rungs failed for absence of opportunity" | mixed (mostly NEC) | Pre-2026-09-22 label for what DESIGN_ADEQUACY now splits three ways | PRED 9075-9080; README 72-75; RS 150 | README OK (quoted and explained); PRED appended note proposed; RS covered by #47 |

## 2. Proposed changes

### MISWORDED items

**M1. README.md:132-135 (H47, NEC)**
Current: "**H47** — imposing discrete membership on a continuous manifold is a cost, not a gain; on a world with two genuinely orthogonal family subspaces the learner rationally absorbs both into one channel and beats the told-membership oracle on present cost."
Proposed: "**H47** — on this substrate, imposing discrete membership on a continuous manifold was a cost, not a gain; and on a world with two genuinely orthogonal family subspaces, at two slots of K = 32 argument directions, the learner rationally absorbs both into one channel and beats the told-membership oracle on present cost — at that capacity the task did not require the split (H48b narrows the channel)."

**M2. README.md:136-138 (H48b, NEC)**
Current: "**H48b** — identity pays for *future* acquisition only below a channel-width threshold; never for present cost."
Proposed: "**H48b** — identity pays for *future* acquisition only below a channel-width threshold (between K = 8 and 16 per slot), and for present cost at no tested width (K = 2-32) on this world."

**M3. README.md:139-140 (H49, NEC)**
Current: "**H49** — discriminating retrospective signals exist **only** on a representation already organized around the true structure."
Proposed: "**H49** — on the two-subspace world, the retrospective signals we tested discriminate the true structure **only** on a representation already organized around it; the label-free objective never rewarded that organization."

**M4. paper/draft.md:1182-1183 (H49, NEC)**
Current: "The value of a structure is a property of the representation that holds it, not of the task population."
Proposed: "Here, the detectable value of a structure was a property of the representation that holds it, not of the task population alone."

**M5. README.md:38 (E6/E6.2 row, NEC)**
Current (clause): "compiling a macro into one operator works only at ~4× slot capacity and never pays."
Proposed: "compiling a macro into one operator works only at ~4× slot capacity and pays at no tested capacity (0/3 worlds) on this testbed, where a definitional macro already supplies the abbreviation."

**M6. README.md:389-391 (E6.2 boxed aphorism, NEC)**
Current: "\boxed{\text{the capacity that makes compilation correct is the capacity that makes it not worth doing}}"
Proposed: "\boxed{\text{on this substrate, the capacity that made compilation correct made it not worth doing}}"

**M7. PREDICTIONS.md:7294-7295 (E6.2, NEC). APPEND, do not edit**
Current: "**The capacity that makes compilation correct is the capacity that makes it uneconomic.**"
Appended correction text:

> # CORRECTION (2026-09-24, A2 claims audit) to E6.2's headline sentence
>
> The E6.2 entry states, in bold, "The capacity that makes compilation correct is
> the capacity that makes it uneconomic." Under `DESIGN_ADEQUACY.md` E6.2 is a
> NECESSITY failure. The task never required compilation, because a definitional
> macro (paying after 7.44 uses) is the cheaper impostor. The testbed also has no
> recurring subroutine structure (PROGRESS correction of 2026-08-31). The sentence
> is therefore a statement about this construction and not a law of compilation.
> The licensed reading is the entry's own "What this licenses" paragraph, scoped
> to this substrate: *on this testbed, the capacity that made compilation correct
> (about 4x a library slot) made it uneconomic.* No number, verdict or
> threshold changes.

**M8. README.md:304-313 (E6D, OPP)**
Current heading: "**The economics picks the right macro and cannot refuse a dead one (E6D).**"
Current sentence (311-313): "It is specifically blind to **non-continuation**: \(H_{\text{eff}}\) counts uses that *have* happened, and the decision needs uses that *will*."
Proposed heading: "**The economics picks the right macro, and cannot refuse one that stops without warning (E6D).**"
Proposed replacement for 311-313: "A same-day correction narrowed what this shows. The planted pattern was uniform until it stopped, so the observed corpus held no signal of non-continuation and *no* past-only rule could have refused. The control could not separate 'the rule lacks a criterion' from 'the task is impossible'. E6E and E6F therefore used a pattern that decays inside the observed window."
(This matches PREDICTIONS 7036-7051, which the README does not reflect.)

**M9. RESEARCH_STATUS.md:582 (E6 line summary, NEC/OPP)**
Current: "the E6 macro line (macros pay but cannot be timed or compiled);"
Proposed: "the E6 macro line (on this testbed macros pay; retrospective accounting alone cannot time creation but a gated criterion can (E6F); compilation works only at ~4x slot capacity and does not pay there);"

**M10. README.md:60 (SG0, NEC)**
Current (clause): "**In force as registered:** the amortized-proposer branch is CLOSED,"
Proposed: "**In force as registered:** the amortized-proposer branch *on this substrate* is CLOSED,"
(This matches RS 89-90 and PRED 9075.)

**M11. RESEARCH_STATUS.md:150-153 (SG6 label, DISC)**
Current: "**Tenth rung to fail for absence of opportunity, and the FIRST to fail before a plan existed.** The other nine cost a branch each; this cost one command. Recorded in AGENTS.md: the opportunity gate is what decides whether a plan gets written, not a step inside one."
Proposed: "**Tenth rung to fail an adequacy gate, and the FIRST to fail before a plan existed. Its gate was DISCRIMINATION (the sample has no graded axis), not opportunity or necessity.** The other nine cost a branch each; this cost one command. Recorded in AGENTS.md: the adequacy gates decide whether a plan gets written, not a step inside one."

**M12. RESEARCH_STATUS.md:566-567 (L0d label, NEC). Low priority**
Current: "**Four consecutive L0d opportunity gates have now failed to produce route ambiguity**"
Proposed: "**Four consecutive L0d gates (called opportunity gates in their plans; NECESSITY failures under `DESIGN_ADEQUACY.md`) have now failed to produce route ambiguity**"

**M13. PREDICTIONS.md:9079-9080 (aggregate label). APPEND. Optional, low priority**
Current: "eight rungs have already failed for absence of opportunity, and the pattern is the finding."
Appended note text (this can go in the same appended block as M7):

> Relabelling, not a new result: the "rungs failed for absence of opportunity"
> count recorded in the SG1-SG4 consequence (2026-09-22) predates
> `DESIGN_ADEQUACY.md`. Under its three gates most of those rungs failed
> NECESSITY (the task did not require the behaviour). SG6 failed DISCRIMINATION.
> E9 and the 2026-08-31 census premise failed OPPORTUNITY. None is a refutation
> of the behaviour it studied.

### Nothing proposed for
- PREDICTIONS H47 B2, H48b and H49 entries. They are scoped in-entry ("at this argument capacity", "PRESENT pays at no K", "on this world").
- E6D's ledger entry. It is already corrected at 7016-7071.

## 3. Counts by class (53 rows)

| class | rows |
|---|---|
| GENUINE REFUTATION | 22 (#1-6, 11, 15, 17, 20, 21, 27, 28, 29, 34, 39, 41, 42, 43, 49, 50, 51) |
| NECESSITY FAILURE | 14 (#8, 10, 12, 23, 24, 25, 26, 31, 35, 36, 37, 38, 45, 46; #36 is shared NEC/OPP) |
| OPPORTUNITY FAILURE | 8 (#7, 9, 14, 16, 22, 30, 33, 52) |
| DISCRIMINATION FAILURE | 7 (#13, 18, 32, 40, 44, 47, 48) |
| not classified | #19 (withdrawn for an instrument defect), #53 (aggregate label) |

Wording: 13 MISWORDED items (M1-M13) across 10 results (#24, 25, 26, 33, 35, 36, 45, 46, 47, 53, with #26 and #35 each worded wrongly in more than one place). M1-M10 generalize a construction-bound negative. M11-M13 are gate-label fixes (old "opportunity" label). By file:
- README: 7 (M1, M2, M3, M5, M6, M8, M10; M5 and M6 are the same result)
- paper: 1 (M4)
- RESEARCH_STATUS: 3 (M9, M11, M12)
- PREDICTIONS: 2 appended corrections (M7, M13)

No genuine refutation was found hedged into mush. The ledger's genuine negatives (V6/H30-H35, H50, H53, SO2-SO4, J1) are stated plainly with their scope.

## 4. What was checked, and uncertain items

**Checked.**
- `FOLLOWUP_AUDITS.md` A2 and `DESIGN_ADEQUACY.md` in full.
- `README.md` and `RESEARCH_STATUS.md` in full.
- `paper/draft.md` in full (1-1903).
- `PREDICTIONS.md`: every verdict, FAILS, REFUTED, NOT SUPPORTED, WITHDRAWN, RETRACTED, closure and correction entry located by the heading index. That covers 1-1125, 1339-1378, 1900-2460, 2700-2850, 3025-3830, 4115-4170, 4585-5530, 6217-6530, 6933-7425, 8114-8160 and 8917-9451.
- `PROGRESS.md` 3933-3972 and 4064-4160, for the E7 census and the loop-census correction, which the scope files cite.
- Skipped: purely positive or registration-only ledger sections (V5 hypotheses, H39 pilot ladder, export confirmation, E6A/E6B, J1c, J2A).

**Uncertain or optional, not counted as MISWORDED.**
1. **E7 (RS 336-338).** "E7 refuted the parameterized-macro rationale against the learner's own unplanted-structure null." The object refuted is a specific rationale (notes/e7-sketch §2) about this corpus, so the sentence does not strictly overreach. Adding "in this testbed's corpus" would guard against it being read as refuting parameterized macros in general. The PI's auto-memory index also carries "parameterized M(alpha) rationale REFUTED by E7" and "macros pay but cannot be timed or compiled". Both are outside A2's scope and should be updated by the parent if M9 is adopted.
2. **Genuine-refutation overreach, optional scope tightening.**
   - paper 610: "Penalizing structure during acquisition damages generality before it induces useful compression". This rests on one gate family on one library. Suggest "Here, penalizing ...".
   - paper 1305-1308: "whatever the organized representation has, it is neither preserved state nor storage format".
   - README 147-148: "useful retained information is not information about how to reorganize".
   All three generalize from the specific arms tested. The paper's own 1314-1318 wording ("usefulness and informativeness are different properties") is the correct form.
3. **Class ambiguities.**
   - E6D could be OPP or DISC. The generator's step function removes the signal, so I chose OPP, and the fix (graded detectability) is a generator change.
   - The loop census could be NEC (DESIGN_ADEQUACY's list) or OPP (PROGRESS: "measured the absence of iteration in the world").
   - V3 absolute refusal: I classed it NEC on the strength of the V4R retraction that noise abstractions pay. It could be read as GEN against a registered criterion.
   - E5 mixes NEC (amortization; flat incumbent) with a real quality miss for this recognizer. I classed it NEC by DESIGN_ADEQUACY's own use of E5.1.
   - H50 and H51: DESIGN_ADEQUACY does not list them. README 152-156 attributes the sham wins to "the objective that fails to value structure", which is a necessity-style reading. I kept GEN because the instrument was validated on L_4 and the arms were matched.
4. **Superseded interpretation, not a class error.** README 43 (SO1): "The wall is the route writer, not compute." The next row, SO1R (README 44), relocates the wall to co-formation. A reader of row 43 alone gets the older reading. Consider appending "(relocated by SO1R)".
5. **paper 497**: "Premature commitment, not discreteness, is the cost." This concerns the V1-era online discrete learner, a different setting from SG0's frozen staged vocabulary, where commitment is never wrong. It is not contradictory, but a reader could conflate the two. Not flagged.
6. **H47 B1 in README.** "is a cost, not a gain" matches only the L_arb arm's registered J COST. The B1 label was MIXED with sub-tolerance hardening effects. M1's "was a cost" keeps the sentence. A stricter version would be "was a cost to the present, and at most a small one to the future".
