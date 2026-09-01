# EDA Findings — r59, six-formulation deterministic bake-off

**Runner:** `r59_eda_joint_player_formulations.jl` · **Machine:** `archpc` · **Date:** 2026-09-01
**Artifacts:** `r59_formulation_results.csv`, `r59_coverage.csv`
**Compute:** local closed-form ridge only. No MCMC, nothing on `mcmc-beast`, no betting simulation.

---

## Executive summary

**Lineup RAPM alone does not clear a held-out gate on the deployment target. Lineup RAPM *combined
with* production wealth does — and it is the only structure in the six that does.**

On the decision-bearing 710-fixture Scottish League One/Two window (24/25 + 25/26), only
`m12_joint_hybrid_synergy` (R² = **+0.0196**) and `m13_joint_composite` (**+0.0164**) beat the
held-out mean. Every pure-lineup arm sits at or below zero (`m09` −0.0030, `m10` −0.0031,
`m11` −0.0123), reproducing the earlier `current_development/player_lineup_dynamics/` null.

This is a genuinely new result rather than a restatement: the previous EDA never tested lineup
ratings *stacked with* a squad-value covariate, and the stack is where the signal appears.

**Recommendation: run the 40-fold grid, but treat `m12` as the hypothesis under test and
`m09`/`m10`/`m11` as its falsification controls.** Do not drop `m13` — it is the cheapest available
test of whether distance adds anything, and the answer so far is "essentially nothing".

---

## What this evidence is, and what it is not

Each candidate's log-rate structure was mapped onto one ridge design; see the header of
`l59_eda_loader.jl` for the full contract. Three simplifications bound every claim below:

1. **Identity link, not `log λ`.** The ridge is linear in the rate. It preserves the *ordering* of
   a supremacy signal, not its scale.
2. **Static team state.** `m05`'s `TimeDecayDynamics(180 d)` becomes static attack/defence dummies.
   This is a strictly weaker control than the production component, and it is the single biggest
   reason not to read `m05`'s scope-C number as a verdict on the production `m05`.
3. **No Gamma arm.** The two-arm observation has no ridge analogue. Scoring official SofaScore-xG
   supremacy as a second target is the compensation, not a reproduction.

So this ranks formulations and detects dead signal. It does not estimate what the production models
will score.

## Filtration

RAPM vectors, their internal standardisations, and every ridge coefficient are fit on the history
block only; target fixtures are scored and never fitted.

| Scope | Tiers | Split | History | Held out |
|---|---|---|---|---|
| A · England + Scotland | 1, 2, 3, 84, 54, 55, 56, 57 | chronological 80/20 | 9,952 | 2,489 |
| B · Scotland | 54, 55, 56, 57 | fit ≤ 23/24, score 24/25 + 25/26 | 2,727 | 1,461 |
| C · Scottish Lower | 56, 57 | fit ≤ 23/24, score 24/25 + 25/26 | 1,260 | **710** |

Official SofaScore xG comes from the independent cached r92 pull, zero-filled placeholders excluded.
Tiers 56/57 carry no official xG, so scope C has a goal-supremacy row only.

---

## Scope C — the decision-bearing 710-match target

Held-out goal supremacy, ranked by R².

| Formulation | Pearson r | Spearman ρ | MAE | R² |
|---|---:|---:|---:|---:|
| `m12_joint_hybrid_synergy` | **0.1419** | **0.1279** | **1.3828** | **+0.0196** |
| `m13_joint_composite` | 0.1342 | 0.1221 | 1.3852 | +0.0164 |
| `m09_joint_player_shots_outfield` | 0.0379 | 0.0415 | 1.3917 | −0.0030 |
| `m10_joint_player_shots_bench` | 0.0372 | 0.0389 | 1.3922 | −0.0031 |
| `m11_joint_player_pxg_bench` | 0.0087 | 0.0053 | 1.3993 | −0.0123 |
| `m05_joint_production_wealth` | 0.0835 | 0.0717 | 1.4430 | −0.0422 |

Two things are worth separating carefully.

**The synergy is real and it is not just the wealth column.** `m05` here is *wealth plus static
team dummies* and lands at −0.042; `m10` is *lineup ratings alone* and lands at −0.003. Adding the
same wealth column to `m10` produces `m12` at **+0.020**. The combination beats either part, which
is exactly the shape hypothesis H4 predicts. Its Pearson r (0.142) is nearly four times the best
pure-lineup arm's (0.038).

**`m05`'s last place is an artefact of the proxy, not a finding.** 1,260 history fixtures across
~30 teams is a thin base for ~60 static team parameters, and the proxy cannot decay old form the
way `TimeDecayDynamics(180 d)` does. `m05` scored **+0.056** on scope B and **+0.047** on scope A,
where the same dummies have three to eight times the data. Read scope C's `m05` row as "the static
proxy overfits on 1,260 matches", and let the MCMC grid decide the real control.

---

## Scope A — England + Scotland, 12,441 matches

| Formulation | Goals r | Goals R² | xG r | xG R² |
|---|---:|---:|---:|---:|
| `m13_joint_composite` | 0.2600 | **0.0675** | 0.3432 | **0.1172** |
| `m12_joint_hybrid_synergy` | 0.2591 | 0.0670 | 0.3432 | 0.1171 |
| `m05_joint_production_wealth` | 0.2485 | 0.0473 | 0.3219 | 0.0291 |
| `m11_joint_player_pxg_bench` | 0.1628 | 0.0177 | 0.2480 | 0.0603 |
| `m10_joint_player_shots_bench` | 0.1439 | 0.0164 | 0.2256 | 0.0496 |
| `m09_joint_player_shots_outfield` | 0.1432 | 0.0161 | 0.2250 | 0.0494 |

The ordering is identical on both targets and matches scope C: the hybrids lead, team-state-plus-wealth
is second on goals, and pure lineups trail. Against official xG the hybrids quadruple `m05`'s
R² (0.117 vs 0.029), which is the cleanest signal in the whole run that lineup and wealth are
measuring different things.

## Scope B — Scotland 54–57, 1,461 held-out fixtures

| Formulation | Goals r | Goals R² | xG r | xG R² |
|---|---:|---:|---:|---:|
| `m05_joint_production_wealth` | **0.2785** | **0.0564** | −0.0493 | −0.1412 |
| `m12_joint_hybrid_synergy` | 0.1877 | 0.0284 | 0.2394 | 0.0129 |
| `m13_joint_composite` | 0.1863 | 0.0271 | 0.2384 | 0.0118 |
| `m11_joint_player_pxg_bench` | 0.1755 | 0.0222 | **0.2714** | **0.0159** |
| `m10_joint_player_shots_bench` | 0.1642 | 0.0176 | 0.2394 | 0.0129 |
| `m09_joint_player_shots_outfield` | 0.1614 | 0.0167 | 0.2376 | 0.0132 |

Scope B is the one scope where the static-dummy control wins on goals, and it also fails hardest on
xG (−0.141, with a *negative* correlation). Pooling four tiers with sharply different scoring levels
rewards a per-team level parameter on goals and punishes it on a per-shot-quality target. This is a
scope artefact and neither number should be carried forward.

---

## Hypotheses

| | Claim | Verdict |
|---|---|---|
| **H1** | Lineups beat team state | **Not supported as stated.** `m09`/`m10` lose to `m05` on scopes A and B and only tie it on C. Lineups are not a replacement for team state. |
| **H2** | Bench depth is a small positive | **Supported, negligibly.** `m10` − `m09` is +0.0003 R² on A, +0.0009 on B, −0.0001 on C. `w_bench = 0.10` is doing almost nothing, which is the right amount for a term this weak. |
| **H3** | Shot volume beats shot quality | **Refuted on the broad scopes, supported on the target.** `m11` (pxG) beats `m10` (shots) on A (0.0177 vs 0.0164) and B (0.0222 vs 0.0176), but loses on C (−0.0123 vs −0.0031). Since C is the deployment window, keep shots as the primary — but H3's stated reasoning was wrong. |
| **H4** | Wealth is complementary | **Supported, and it is the headline.** `m12` − `m10` is +0.051 R² on A, +0.011 on B, and +0.023 on C. It is the only effect in this study large enough to matter. |
| **H5** | Travel is small but real | **Not supported.** `m13` − `m12` is +0.0004 on A, −0.0013 on B, −0.0032 on C. Distance is within noise and slightly negative where the catalog is complete. |

---

## Coverage — the checks that stop a null being mistaken for a fair test

| Scope | Shots-RAPM rated | Wealth present | Wealth on target | Distance fallback | Distance sd |
|---|---:|---:|---:|---:|---:|
| A | 100.0% | 81.9% | 86.3% | 81.2% | 0.409 |
| B | 100.0% | 46.8% | 48.6% | 45.1% | 0.695 |
| C | 99.9% | 99.6% | 100.0% | **0.0%** | 0.918 |

Three consequences:

1. **The distance result is only trustworthy on scope C.** 81% of scope-A fixtures and 45% of
   scope-B fixtures fall back to the deterministic 45-mile value, because the stadium catalog is
   Scottish. Scope C has full coverage and the widest spread — and it is where `m13` loses to `m12`.
   That is the row H5 should be judged on.
2. **`m12` and `m10` are *exactly* identical on scope B's xG target** (r, ρ, MAE, R² all equal to
   the last digit). The wealth column is exactly zero on every xG-scorable scope-B training fixture,
   so its ridge coefficient is zero and the two designs collapse. Scope B's xG rows contain no
   information about wealth; they are not evidence against it.
3. **Scope C is the only scope where all four signals are simultaneously well covered.** That is a
   second, independent reason to weight it most heavily.

---

## What to do with the 40-fold grid

1. **Run all six.** The grid is the test, and `m09`/`m10`/`m11` are the controls that make `m12`'s
   result falsifiable. Dropping them would leave the headline unfalsifiable.
2. **`m12_joint_hybrid_synergy` is the hypothesis under test.** It leads on the target scope, leads
   on both scope-A targets, and is second on scope B. If it does not lead on out-of-sample LogLoss
   after MCMC, the synergy was a ridge artefact and should be reported as such.
3. **Expect `m05` to do better than it did here.** Its ridge proxy is the least faithful of the six.
   Do not pre-commit to a story in which the control loses.
4. **`m13` is a cheap null test, not a contender.** If it again fails to separate from `m12` after
   MCMC, retire `DistanceCovariate` from this arm rather than re-tuning its prior.
5. **The effect sizes are small in absolute terms.** R² ≈ 0.02 on goal supremacy is a weak signal by
   any standard. Nothing here justifies staking money; it justifies spending MCMC.

## Reproduction

```bash
julia --project -t 8 experiments/scottish_lower/06_joint_player_lineup_fusion/r59_eda_joint_player_formulations.jl
```

Requires two local caches produced by earlier streams:
`current_development/scottish_lower/l50_english_store.jls` and
`current_development/scottish_lower/l92_pxg_validation_pull.jls`.
