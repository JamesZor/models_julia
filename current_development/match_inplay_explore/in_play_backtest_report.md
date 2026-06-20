# In-Play Intensity Model — Backtest, Parameters & Calibration Report

*Ireland Premier Division. Model = Bayesian Poisson intensity (l03 baseline config), fit on a 75%
match split, evaluated/backtested on the held-out 25% (63 matches). Server: Ryzen 9 5950X, threads
pinned to 16 physical cores.*

## TL;DR

The model is a **good, well-calibrated fair-value estimator** of in-play scoring — but the backtest
shows **no tradeable edge**. A naive backtest prints +29% ROI; that is almost entirely a **stale-price
artifact**. Filling at a price you could *actually* get matched at (the next tick after the signal)
collapses ROI to ~0–2%, i.e. noise. The market reprices faster than the model's signal can be acted on.

---

## 1. Model parameters (posterior, baseline config)

Sampler health: **R̂ = 1.003, min ESS ≈ 2000, 4000 draws** (NUTS, `AutoReverseDiff(compile=true)`,
4 chains). Coefficients on the log scale (`exp` = multiplicative effect on the remaining-goals rate);
90% credible intervals:

| term | post. mean | 90% CI | ×rate | reading |
|---|---|---|---|---|
| log_pregame | 1.249 | [1.07, 1.43] | — | **quality backbone** (pregame λ drives in-play rate) |
| trailing | +0.327 | [0.25, 0.41] | **×1.39** | losing team scores ~39% faster |
| leading | −0.247 | [−0.35, −0.15] | **×0.78** | winning team scores ~22% slower |
| is_home | +0.156 | [0.07, 0.24] | ×1.17 | residual home edge beyond pregame λ |
| man_adv | +0.071 | [−0.13, 0.27] | ×1.07 | red-card effect — **not** distinguishable from 0 (rare) |
| t_m | −0.004 | [−0.01, 0.00] | ×1.00 | no extra time-trend once the exposure offset handles the clock |
| t_m² | ~0 | — | ×1.00 | no curvature |

**Score-state (trailing/leading) and team quality are the robust, significant effects.** Time-trend
and red cards are not identified here (the latter for lack of red-card events).

## 2. Fit / calibration

Across **15,900** held-out (bin × selection) pairs, model probabilities vs realized outcome:

| model_p bucket | n | mean model_p | actual |
|---|---|---|---|
| 0.0–0.1 | 3798 | 0.024 | 0.041 |
| 0.1–0.4 | 4221 | ~0.25 | ~0.29 |
| 0.4–0.6 | 1656 | ~0.50 | ~0.49 |
| 0.6–0.9 | 2779 | ~0.75 | ~0.69 |
| 0.9–1.0 | 3446 | 0.978 | 0.966 |

Overall, **perfectly calibrated on average**: Over/Under mean `model_p` 0.500 = actual 0.500; 1X2
0.333 = 0.333. Mild **over-extremity in the tails** (slightly too confident near 0 and 1) — a small,
fixable miscalibration (e.g. a touch of dispersion / Dixon-Coles correlation, or Platt scaling).
*(Separately, the remaining-goal count is well-calibrated by decile — see the l02 GLM report.)*

**Conclusion:** the model is right about outcomes. So any large gap to market price is about the
*price*, not the model.

## 3. Backtest

Strategy: one value bet per (match, selection) at the first bin where edge appears; back-bet EV with
**5% commission** on net winnings; **fractional (¼) Kelly** stake; markets = Over/Under 0.5–5.5 + 1X2.

### 3.1 The naive result is a trap
Filling at the latest price **as-of** the bin: **ROI +28.8%**, 556 bets, **average "edge" 32.8%**.
An average edge of a third of the price is not credible — it is the signature of an artifact.

### 3.2 It is a stale-price (lookahead) artifact
ROI rises monotonically with how stale the price is allowed to be:

| price staleness | ROI | avg edge |
|---|---|---|
| ≤1 min | +12.1% | 0.26 |
| ≤2 min | +20.6% | 0.29 |
| ≤5 min | +26.0% | 0.33 |
| ≤10 min | +28.8% | 0.33 |

A stale last-traded price can pre-date a goal that the model already conditions on → the model "knows"
something the stale quote doesn't. A **2–5% spread haircut barely moves ROI** (+20.6% → +22.7%),
*because the fake edges are far larger than any realistic spread* — another tell.

### 3.3 Realistic execution kills it
Fill instead at the **next available price after the signal** (you observe the state, then get matched):

| fill | ROI | P&L | avg edge |
|---|---|---|---|
| as-of (stale) | **+20.6%** | +4.32 | 0.29 |
| forward +0.5 min | +5.3% | +1.53 | 0.37 |
| forward +2 min | +1.6% | +0.56 | 0.41 |
| forward +5 min | +0.8% | +0.32 | 0.48 |

The edge evaporates as soon as execution is realistic. Note the *claimed* edge stays high (0.29→0.48)
while ROI → 0: the model thinks it has edge, but by the time you can trade, the market has already
moved to the (calibrated) fair price.

### 3.4 Where the fake profit concentrated (as-of, staleness 2)
Longshots dominate the illusion: price-7+ bucket showed +77% ROI at a 13% hit rate on tiny Kelly
stakes — high variance + tail over-extremity, not alpha. OU and 1X2 behaved similarly (+21% / +18%).

## 4. Verdict & next steps

- **Does it make money in-play? No demonstrated tradeable edge.** The model is a sound, calibrated
  fair-value engine, but the in-play exchange reprices faster than the signal can be executed; the
  realistic-execution ROI (~0–2% on 63 matches) is within noise and *before* the LTP-vs-back-price
  spread and liquidity constraints.
- **Honest caveats:** only 63 test matches (wide error bars); fills assume small matched size; the
  clock/score anchoring peeks at the full price series (mild timing lookahead); commission modelled,
  queue/latency not.
- **What could still be worth testing:** (a) a **live forward-test** (paper trade) — the only true
  test of speed-of-execution edge; (b) hunting **specific transient inefficiencies** (post-goal
  repricing lag) with a proper microstructure model of the order book, not last-traded prices;
  (c) fixing the mild tail over-extremity (Dixon-Coles correlation / calibration layer) to sharpen
  the fair value; (d) more data (ScottishLower / more leagues) to shrink the error bars.

## Reproduce
`l04_backtest.jl` (functions) + a runner that fits the l03 baseline (`run_sampler`, `NUTSConfig`),
extracts posterior-mean `ᾱ, β̄`, builds `finmap = make_finmap(ds)`, then:
`run_backtest(panel, bf, te_ids, finmap, ᾱ, β̄, inp; mode = :forward, lag = 0.5)` for the realistic
number, `mode = :asof, staleness = …` to reproduce the stale-price illusion.
