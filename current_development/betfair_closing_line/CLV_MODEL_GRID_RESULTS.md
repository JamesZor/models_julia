# Closing-Line-Value of the M1/M2/M3 bigChance grid — results

**Date:** 2026-06-21 · **Data:** `Data.Ireland()` betdb (`:5433`) · **Engine:** `l01_clv_eval.jl`
(unchanged) · **Runner:** `r02_clv_model_grid.jl` · **Mode:** PPD-only inference, no MCMC.

## TL;DR

> **The xG baseline (M1) leads the Betfair close best. The bigChance model (M2) is the
> *weakest* of the three — exactly on the totals/BTTS markets where bigChance was supposed
> to help.** The ordering **M1 ≳ M3 > M2** is consistent across CLV alpha, held-out
> log-loss, and filtered ROI, and holds at every pre-close horizon. **This flips the prior
> verdict** (LogLoss + 1X2 Kelly P&L on Bet365, 281 matches: *M2 > M3 ≳ M1*). CLV is the
> more robust forward metric, so the grid verdict is now: **keep xG, drop bigChance.**

## The grid

| ID | Experiment | Pillars |
|----|-----------|---------|
| M1 | `DP_Goals_Market_XG` | {goals, market, xG} (baseline) |
| M2 | `DP_Goals_Market_BigChance` | {goals, market, bigChance} |
| M3 | `DP_Goals_Market_BigChance_XG` | {goals, market, bigChance, xG} |

## Scope (Step 1 — coverage triage)

`ds.betfair_odds`: **994 matches, 1,051,125 last-traded ticks**. Markets present: `1X2,
BTTS, CorrectScore, DOUBLE_CHANCE, OverUnder`. Final target set = model-emittable ∩
Betfair-liquid ∩ gradeable = **17 selections**:

- **1X2**: home, draw, away (≈940 matches, ~12–15k ticks near close)
- **BTTS**: yes, no (≈927 matches)
- **OverUnder**: 0.5 / 1.5 / 2.5 / 3.5 / 4.5 / 5.5 (over & under)

**Excluded:** *Double-Chance* — the model emits market `"DoubleChance"` / selections
`:DC_1X,:DC_X2,:DC_12` while Betfair uses `"DOUBLE_CHANCE"` / `:dc_home_draw…`; both names
mismatch (no join) and coverage is thin (~437 matches). *CorrectScore* — no
`compute_market_probs` / `grade_selection` rule. Both are out of scope.

Tagged panels concatenate to **`grid_panel` = 78,933 rows** (26,311 each, 275 matches per
model — balanced, so the cross-model comparison is apples-to-apples).

**LOCF watch:** O/U **4.5 / 5.5** lines are LOCF-heavy near the close (`locf_frac`
0.21–0.49) and have sub-0.5 hit-rates — treat their (inflated) β as scale noise, not edge.
The core (1X2, BTTS, O/U 0.5–3.5) has `locf_frac < 0.13` at τ=−5.

## Stage 2 (HEADLINE) — CLV alpha: which model leads the close?

Pooled over all horizons & selections (`mean_clv` is model-independent by construction):

| model | β | β p-value | hit-rate | hit p-value |
|-------|------:|---------:|--------:|-----------:|
| **M1_xG** | **0.0563** | 1.7e-203 | **0.5185** | 1.9e-9 |
| M3_both | 0.0543 | 1.3e-195 | 0.5166 | 8.1e-8 |
| M2_bigChance | 0.0472 | 1.1e-142 | 0.5097 | 1.7e-3 |

`β > 0` ⇒ the model's signal predicts the direction the line moves to at close. **M1 has
the largest, most-significant CLV; M2 the smallest.**

### By horizon — M1 ≳ M3 > M2 at every horizon, peak at τ=−720

| τ (min) | β M1 | β M2 | β M3 | hit M1 | hit M2 | hit M3 |
|--------:|-----:|-----:|-----:|------:|------:|------:|
| −1440 | 0.079 | 0.071 | **0.077** | 0.533 | 0.535 | 0.539 |
| −720 | **0.114** | 0.101 | 0.112 | 0.559 | 0.559 | 0.560 |
| −360 | **0.079** | 0.064 | 0.076 | 0.543 | 0.528 | 0.542 |
| −180 | **0.070** | 0.058 | 0.068 | 0.529 | 0.522 | 0.531 |
| −90 | **0.065** | 0.056 | 0.063 | 0.541 | 0.534 | 0.540 |
| −45 | **0.042** | 0.033 | 0.040 | 0.524 | 0.512 | 0.522 |
| −20 | **0.019** | 0.015 | 0.018 | 0.498 | 0.484 | 0.490 |
| −5 | **0.005** | 0.003 | 0.004 | 0.456 | 0.447 | 0.449 |

CLV peaks ~12h out (β≈0.11, hit ~56%) and decays to ~0 by the close as the line absorbs
the same information. At τ=−5 the hit-rate dips **below** 0.5 — the model and close have
converged and the residual move is mean-reverting noise, not signal. **The tradeable CLV
window is roughly −720 to −90 minutes**, not the last few minutes.

### Where each model's CLV lives (β by selection × model, liquid core)

| selection | M1_xG | M2_bigChance | M3_both |
|-----------|------:|-------------:|--------:|
| btts_no | **0.052** | 0.040 | 0.047 |
| btts_yes | **0.052** | 0.041 | 0.047 |
| draw | 0.031 | 0.014 | **0.034** |
| over_15 | 0.024 | **0.004** | 0.026 |
| over_25 | 0.037 | 0.022 | **0.038** |
| over_35 | 0.020 | 0.011 | **0.021** |
| home / away | ~0.025 | ~0.020 | ~0.024 |

*(O/U 0.5 / 4.5 / 5.5 omitted — extreme-probability scale artifacts with hit-rate < 0.5.)*

**The damning result:** M2 (bigChance) is weakest precisely on **totals (O/U 1.5–3.5) and
BTTS** — the markets the bigChance pillar was meant to sharpen. On O/U 1.5 its CLV β
collapses to ~0. M1 and M3 are statistically tied; the xG pillar carries the edge and
adding bigChance never improves it (and used alone, degrades it).

## Stage 1 — Edge vs the Betfair fair line

Pooled (most-negative `diff_ll` = best); all three **beat** the de-vigged Betfair line:

| model | model_ll | market_ll | diff_ll [95% CI] | diff_brier |
|-------|---------:|----------:|-----------------:|-----------:|
| **M1_xG** | 0.5471 | 0.6053 | **−0.0582** [−0.076, −0.042] | +2e-5 |
| M3_both | 0.5475 | 0.6053 | −0.0578 [−0.076, −0.042] | +2e-4 |
| M2_bigChance | 0.5486 | 0.6053 | −0.0567 [−0.075, −0.040] | +6e-4 |

Same ordering M1 < M3 < M2 (M1 best). **Caveat:** `diff_brier ≈ 0` while `diff_ll` is
strongly negative ⇒ the log-loss win is concentrated in extreme-probability tails (a
vig-removal artifact on thin O/U 0.5 / 4.5 / 5.5), **not** core probability accuracy. Read
Brier as the honest tie; the *relative* model ordering is what carries.

## Stage 3 — Entry-timing filtered P&L (sanity)

Flat-stake, enter when `prob_model − prob_fair_τ > 0.02`, pooled over horizons:

| model | total bets | mean ROI |
|-------|-----------:|---------:|
| **M1_xG** | 8,791 | **0.131** |
| M3_both | 8,820 | 0.127 |
| M2_bigChance | 8,691 | 0.095 |

Even the noisy flat-stake ROI reproduces **M1 ≳ M3 > M2** on this Betfair held-out panel.

## Stage 4 — Microstructure + calibration

- **PIT vs close:** all three diverge significantly from the Betfair close (KS p ≈ 0,
  D≈0.16). Non-uniform PIT = the model takes systematic positions away from the close —
  that divergence is the *source* of the CLV, and it is largest for M1/M3 (D=0.166) vs M2
  (D=0.157, marginally closest to market), consistent with their higher β.
- **Roll (1984) spread:** estimable only on 1X2 (decimal-odds units; O/U autocovariances
  mostly positive ⇒ undefined). Microstructure sanity only, not model-discriminating.

## Verdict

On the strongest forward metric available at this sample size — **CLV against the Betfair
close** — the ranking is **M1 ≳ M3 > M2**, corroborated by held-out log-loss and filtered
ROI, stable across every pre-close horizon. This **flips** the earlier LogLoss/1X2-Kelly
ordering (M2 > M3 ≳ M1). The bigChance pillar does **not** help the model anticipate the
market, and used on its own it actively dilutes the totals/BTTS edge that xG provides.

**Recommendation:** keep the xG pillar (M1); do not adopt the bigChance pillar. If anything
is salvageable it is M3 (both), which tracks M1 within noise but never beats it — so there
is no reason to pay bigChance's extra complexity.

## Artifacts

- Runner: `current_development/betfair_closing_line/r02_clv_model_grid.jl`
- Plots (server `…/betfair_closing_line/plots_grid/`): `headline_clv_beta_by_model.png`,
  `headline_hitrate_by_model.png`, and per-model `edge_/clv_/pnl_/pit_*.png`.
