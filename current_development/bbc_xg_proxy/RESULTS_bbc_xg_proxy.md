# RESULTS — bbc xG-proxy, WP0–WP2 (2026-07-17)

Scope of this session: data QA → proxy training with tier-transfer gate →
forward-informativeness test on League 1/2. Bayesian integration deferred (WP3+).

## WP0 — data quality: PASS

- 4,188/4,394 bbc matches join sofascore cleanly; all 206 failures are matches
  absent from sofascore entirely (191 kickoff-2020, 15 kickoff-2026). No id drift.
- bbc↔sofascore agreement on Prem/Champ (both sources): cor 0.91–0.98,
  exact-match 84–95% per stat. `filled=true` rows are backfilled-but-decent
  (SoT cor 0.992), noisier for shotsBlocked (0.86).
- Distributions benchmark-consistent in all four tiers.

## WP1 — proxy model: m2/gamma frozen, transfer gate PASS

Training: 1,454 team-match rows (Prem 25/26 + Champ 23/24–25/26).
Winner **m2**: Gamma GLM, log link —
`xg ~ sot + soff + sblock + corners + poss + woodwork + fouls_opp + is_home + sqrt(sot) + sqrt(soff) + sot×poss`

| gate | result |
|---|---|
| season-blocked CV (pooled OOS) | R² 0.442, MAE 0.464, Spearman 0.715 |
| vs SoT-only scaler | R² 0.381, Spearman 0.635 — m2 clearly better |
| transfer Prem→Champ (downward, mimics L1/2 use) | R² 0.408 (in-sample 0.496), MAE 0.452 (0.443), Spearman 0.701 (0.707) — **rank signal transfers intact** |
| calibration | deciles 1–9 clean; decile 10 over-predicts (3.02 vs 2.56) |

Artifact: `proxy_model_v1.jls`. Applied to League 1/2: 3,936/3,936 team rows
covered, proxy mean 1.276 vs actual goals mean 1.377 (sane).

## WP2 — forward informativeness on League 1/2: CONDITIONAL PASS

Expanding-window OOS (test 22/23–25/26; 2,776 team rows / 1,388 matches),
nested decayed-form sets A {goals} ⊂ B {+SoT} ⊂ C {+proxy-xG}:

| metric (hl365 / hl180) | B−A | C−B | C−A |
|---|---|---|---|
| goals Poisson log-lik (↑) | +0.0054 / +0.0052 | **+0.0002 / +0.0005** | +0.0056 / +0.0056 |
| home-win log-loss (↓) | −0.0038 / −0.0036 | **−0.0013 / −0.0018** | −0.0050 / −0.0054 |
| home-win AUC (↑) | +0.008 / +0.006 | +0.005 / +0.005 | +0.013 / +0.011 |

- C ≤ B on home-win log-loss in **all 4 test seasons × both half-lives**;
  paired per-match t = 1.22 (hl365) / 1.42 (hl180) — consistent but **not
  individually significant** at n=1,388.
- In the joint set-C Poisson, `form_pxf` is significant (z=2.01, p=0.044) while
  the SoT forms go inert (z≈0.5) — **the proxy subsumes the shots signal** and
  adds a little on top.
- On next-match goals the proxy adds ≈nothing beyond shots (+0.0002); its edge
  shows on match outcomes.

### Verdict

**Conditional green-light for WP3+.** The proxy transfers across tiers, covers
100% of League 1/2 matches, subsumes SoT, and adds a small consistent
outcome-prediction edge over goals+shots (underpowered at one league-pair scale
— the same statistical situation as the Ireland totals edge before pooling).
Two caveats carry forward:

1. This test used decayed *form aggregates*; the Bayesian engine consumes xG
   per-match as a likelihood observation — a strictly richer use. Treat WP2's
   Δ as a conservative lower bound, but do not expect Ireland-sized effects.
2. The direct-SoT-route arm (WP3b) stays essential: most of the proxy's value
   IS repackaged shots information, so the A/B must show the Gamma-route
   packaging beats a raw SoT route before graduating anything.

### Next session (WP3+)

ProxyXGFeature extractor emitting `:flat_home_xg`/`:flat_away_xg` from
`proxy_model_v1.jls` → xG Gamma route into `TeamSmileDPGoalsModel` (copy l03
`_unpack_xg` + NaN-sanitize gotcha; consider wider ν_xg prior) → A/B grid vs
`none_pois_hl365_hs2` incl. direct-SoT arm → verdict vs Stage-A numbers
(x12 0.0143, BTTS 0.0014 LogLoss diff vs Bet365 close).

---

# WP3 — Funnel cascade, Stage 1 (2026-07-21)

`Shots ~ Poisson(λ_s); SoT|Shots ~ Bin(p₁); Goals|SoT ~ Bin(p₂)` on ScottishLower
(56/57). Poisson thinning keeps the goals marginal `Poisson(λ_s·p₁·p₂)`, so all
pricing is unchanged; the funnel only enriches the observation model.

Engine `TeamFunnelDPGoalsModel` (l03) vs the `none_pois` comparator
(`TeamDPGoalsModel`), identical folds: target 25/26, 2 history seasons, biweek
folds, warmup_period 16 → 5 splits, 4 chains, 1000 warmup / 600 samples,
max_depth 8, half-life 365d. 66 OOS matches. **27/27 checks passed.**

## Convergence and posteriors

| | funnel | none_pois |
|---|---|---|
| global max R-hat | 1.0107 | 1.0111 |
| p1_raw / p2_raw R-hat | 1.0005 / 1.0013 | — |
| wall | 27.9 min | 14.4 min |

p₁ = 0.443 [0.434, 0.453], p₂ = 0.333 [0.320, 0.346] — both land on the
data-implied MLE (0.4416 / 0.3302). λ_shots 9.89, λ_goals 1.465/1.326.
δ₅₆−δ₅₇ = +0.041 (shot scale; League One shoots slightly more).

## Team-strength spread — the headline

across-team sd of posterior log λ_goals (home), 20 teams:

| model | sd | 
|---|---|
| funnel | **0.1072** |
| none_pois | 0.0595 |
| ratio | **1.80** |

The pre-registered worry was that global p₁/p₂ would COMPRESS team strength
(shot-rate spread is proportionally tighter than goal-rate spread). The opposite
happened, because shrinkage dominates: at ~1.4 goals/match the goals-only model
cannot separate teams and the hierarchical prior pulls them to the mean, while
7× the count volume lets the posterior resolve them. The funnel RESOLVES team
strength rather than compressing it.

## LogLoss vs the Bet365 fair close (negative = beats the close)

Family-pooled mean diff:

| model | x12 | btts | totals |
|---|---|---|---|
| funnel_pois | 0.0153 | −0.0078 | −0.0139 |
| none_pois | 0.0224 | −0.0082 | −0.0217 |
| **Δ (funnel − none)** | **−0.0071** | +0.0004 | **+0.0078** |

Per line, the funnel's gain is concentrated in `home` (0.0384 vs 0.0521) and
`away` (0.0089 vs 0.0244); its loss is concentrated in `over/under 2.5`
(−0.0126 vs −0.0269) and the `3.5/4.5` tails.

**Read:** shots pin RELATIVE strength well → 1X2 improves, closing about a third
of the gap to the close. The goal LEVEL is forced through a global conversion
constant → totals calibration degrades (though both engines still beat the close
on totals). This is exactly the trade-off the separability argument predicts.

**Caveat: n = 66 OOS matches over 5 folds. A smell test, not a verdict.**

## Structural finding

The cascade log-likelihood is additively separable in (λ_s, p₁, p₂), and α/β/HA
live only inside λ_s. Therefore **goals contribute zero gradient to team
strength** — it is fitted purely to shots. Two consequences:

- `funnel_weight` does not interpolate to `none_pois`; the meaningful dial is
  `cascade_weight` on the goals routing (Binomial `G|T` vs marginal Poisson
  `G|λ_s·p₁·p₂`), where 0 recovers `none_pois` exactly. Not yet implemented.
- Stage 2's per-team p₂ is the channel through which goals re-enter the
  goal-rate prediction.

## Performance note

The likelihood was rewritten onto sufficient statistics computed in the builder
(exact — verified as a constant offset to 1e-11). Gradient 0.83 ms / 95 params.
Fixing the intercept parameterisation (`shot_scale` offset, so `UniformInit`
starts on the right scale) moved ε from 4e-4 to 0.22; the first attempt had not
finished 1 of 20 chains in 4.5 h.

---

# WP3 — Funnel cascade, Stage 2: hierarchical per-team conversion (2026-07-21)

    logit p₁_ij = p1_μ + a₁_i + d₁_j      logit p₂_ij = p2_μ + a₂_i + d₂_j

Non-centred, zero-sum, shared σ per layer, half-Normal(0, 0.3) priors
(prior mean 0.239). Same folds/config as Stage 1; comparators loaded from disk.
8/10 checks (the two failures are mine, not the model's — see Caveats).

## VERDICT: NULL. Conversion is a league constant.

| | posterior mean | 90% CI | prior mean |
|---|---|---|---|
| σ_p1 (SoT per shot) | **0.0344** | [0.0035, 0.0743] | 0.2394 |
| σ_p2 (goal per SoT) | **0.0581** | [0.0061, 0.1253] | 0.2394 |

Both σ are pulled to **1/7 and 1/4 of their prior** — the data actively argues
against per-team conversion. In football terms, ±1sd of team finishing spans
p₂ ∈ [0.320, 0.346] around a pooled 0.333, i.e. a **±4% relative** spread.

This reproduces the Ireland hierarchical-σ null (smile l08/r17 and iso l09/r18:
τ pulled below prior, ±4% team spread, global scalar σ wins) on a completely
different quantity. Same conclusion, third time of asking.

Corroboration — team-strength spread barely moves, so the extra flexibility is
not being used:

| model | across-team sd log λ_goals(home) |
|---|---|
| funnel_hier | 0.1041 |
| funnel_pois (Stage 1) | 0.1072 |
| none_pois | 0.0595 |

Convergence is clean (max R-hat 1.0117; p1_μ 1.0021, p2_μ 1.0009, σ_p1 1.0015,
σ_p2 1.0056), and p1_μ/p2_μ still sit on the pooled MLE (0.4414/0.3363 vs
0.4416/0.3302) — so the null is a real posterior, not a fitting failure.

## LogLoss vs the Bet365 fair close (negative = beats the close)

| model | x12 | btts | totals | wall |
|---|---|---|---|---|
| funnel_hier | 0.0140 | −0.0081 | −0.0151 | 3h 40m |
| funnel_pois | 0.0153 | −0.0078 | −0.0139 | 27.9 min |
| none_pois | 0.0224 | −0.0082 | −0.0217 | 14.4 min |

hier − Stage-1 funnel: x12 −0.0013, btts −0.0003, totals −0.0012. Better on all
three, but by ~0.001 on n = 66 — indistinguishable from noise, for **7.9× the
compute**.

Crucially, hier recovers only ~15% of the totals gap to none_pois (deficit
0.0078 → 0.0066). **Per-team conversion is not the explanation for the totals
loss.**

## What this means

Teams in Scottish L1/L2 differ in shot VOLUME, not in shot quality or
finishing — at least not measurably at this sample size. That is a real
football finding and it validates Stage 1's global p₁/p₂ as the right model.
It also removes the leading hypothesis for the totals deficit, leaving
`cascade_weight` (routing a fraction of the goals likelihood through the
marginal Poisson, so goals inform λ_s directly) as the remaining lever.

## Caveats

- n = 66 OOS matches, 5 folds, one target season. Still a smell test.
- Two failed checks were harness bugs, both fixed: `all_results` typed `Any[]`
  broke `evaluate_experiments` (eval re-run standalone; numbers above are real),
  and the gradient gate was set at < 2 ms while the hierarchical engine needs
  2.89 ms over 189 params (vs 0.83 ms over 95) — a threshold miscalibration,
  not a defect.
- The 3h 40m wall is 7.9× Stage 1 for 2× the parameters: the 80 team-effect
  coordinates are poorly conditioned when their σ collapses (the funnel of a
  near-zero scale is exactly the geometry non-centred parameterisation struggles
  with). If per-team conversion is ever revisited, that needs addressing first.
