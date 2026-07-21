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
