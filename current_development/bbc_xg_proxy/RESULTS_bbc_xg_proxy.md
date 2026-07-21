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
