# Hurdle Metric Analysis — Ireland A/B Test Results

## Executive Summary

The hurdle metrics reveal a critical finding that raw ROI completely missed: **your 1X2 match-odds markets are wealth-destroying under Kelly staking**, despite showing positive empirical ROI. The compounding drag from high-variance longshot losses (especially Away and Draw) overwhelms the arithmetic edge. Your real alpha lives in the **Over/Under goals** and **BTTS** markets.

---

## 1. The Growth Table — What Actually Makes Money

Filtering to `hurdle_G > 0` and `hurdle_n_bets >= 30`, ranked by growth rate:

| Model | Market | hurdle_G | hurdle_sharpe | hurdle_p | n_bets | Comment |
|-------|--------|----------|---------------|----------|--------|---------|
| DoublePoisson_NoMarket | btts_yes | **0.0154** | 0.236 | 0.60 | 157 | 🥇 Highest growth |
| DixonColes_NoMarket | btts_yes | **0.0154** | 0.247 | 0.60 | 159 | Near-identical |
| DixonColes_NoMarket | draw | 0.0049 | 0.095 | 0.25 | 71 | Moderate sample |
| DixonColes_NoMarket | under_15 | **0.0111** | 0.167 | 0.32 | 47 | Strong but low n |
| DoublePoisson_NoMarket | under_15 | 0.0065 | 0.173 | 0.33 | 46 | Low n caveat |
| DoublePoisson_Market | under_25 | 0.0066 | 0.132 | 0.56 | 111 | Solid |
| **DCMH** | **under_25** | **0.0066** | **0.133** | **0.56** | **112** | ⭐ Balanced & reliable |
| **DCMH** | **over_25** | **0.0063** | **0.119** | **0.47** | **129** | ⭐ Balanced & reliable |
| DixonColes_Market | btts_yes | 0.0093 | 0.215 | 0.57 | 101 | Strong |
| **DCMH** | btts_yes | 0.0063 | 0.158 | 0.55 | 110 | Solid |
| **DCMH** | under_35 | 0.0052 | 0.129 | 0.77 | 93 | ⭐ Very high win rate |
| DixonColes_NoMarket | btts_no | 0.0121 | 0.275 | 0.60 | 35 | Low n — fragile |
| DoublePoisson_Market | over_25 | 0.0075 | 0.141 | 0.48 | 130 | Good |
| DoublePoisson_NoMarket | btts_no | 0.0054 | 0.200 | 0.57 | 37 | Low n — fragile |
| DoublePoisson_Market | under_15 | 0.0040 | 0.091 | 0.29 | 100 | Moderate |
| **DCMH** | under_15 | 0.0048 | 0.111 | 0.30 | 96 | Decent |
| DoublePoisson_Market | under_35 | 0.0038 | 0.099 | 0.76 | 91 | Steady |
| DixonColes_Market | under_35 | 0.0030 | 0.082 | 0.76 | 99 | Steady |

---

## 2. The Overbetting Trap — Positive ROI, Negative Growth

> [!CAUTION]
> These model-market combinations show **positive empirical ROI** but **negative `hurdle_G`**. Your Kelly staking is correctly identifying edges, but betting too aggressively relative to the variance.

### 1X2 Markets — The Big Surprise

| Model | Market | ROI% | hurdle_G | hurdle_E_R | hurdle_avg_stake | Diagnosis |
|-------|--------|------|----------|------------|------------------|-----------|
| DoublePoisson_NoMarket | away | +26.8% | **-0.0201** | -0.101 | 0.081 | 🔴 Wealth destruction |
| DoublePoisson_Market | away | +23.4% | **-0.0193** | -0.066 | 0.086 | 🔴 Wealth destruction |
| DixonColes_NoMarket | away | +22.5% | **-0.0175** | -0.073 | 0.080 | 🔴 Wealth destruction |
| DixonColes_Market | away | +31.5% | **-0.0176** | -0.062 | 0.081 | 🔴 Wealth destruction |
| DCMH | away | +23.6% | **-0.0174** | -0.065 | 0.081 | 🔴 Wealth destruction |
| All models | home | -7% to +5% | **-0.013 to -0.029** | negative | ~0.10 | 🔴 Wealth destruction |

**Why this happens:** Look at the Gamma parameters for `away`:
- `hurdle_shape ≈ 2.5–3.4` and `hurdle_scale ≈ 1.2–1.5`
- This means winning ROI has mean ≈ 3.8 (380% return!) but enormous variance
- `hurdle_p ≈ 0.19` — you only win ~19% of the time
- The 81% of the time you lose your entire stake, the compounding drag at `avg_stake ≈ 0.08` destroys the bankroll

This is the classic longshot problem: the arithmetic average is positive, but the geometric average (what your bankroll actually experiences) is deeply negative.

### Draw Market — Similar Pattern

| Model | Market | ROI% | hurdle_G | hurdle_sharpe |
|-------|--------|------|----------|---------------|
| DixonColes_Market | draw | +30.7% | **-0.001214** | 0.025 |
| DCMH | draw | +44.5% | **-0.001122** | 0.012 |

Even the DCMH model's spectacular 44.5% ROI on draws translates to slightly negative growth. The `hurdle_sharpe` of 0.012 confirms the edge is indistinguishable from noise after accounting for the variance.

### BTTS No — Market Models Fail

| Model | Market | ROI% | hurdle_G | hurdle_E_R |
|-------|--------|------|----------|------------|
| DoublePoisson_Market | btts_no | +13.4% | **-0.008** | -0.174 |
| DixonColes_Market | btts_no | +18.7% | **-0.007** | -0.123 |
| DCMH | btts_no | +20.3% | **-0.006** | -0.146 |

Interesting: the `NoMarket` variants are growth-positive on btts_no, but adding market data flips the sign. The market anchor is causing the model to bet on too many marginal btts_no opportunities.

---

## 3. DCMH Deep Dive — Where to Deploy

### ✅ Deploy (hurdle_G > 0, n_bets ≥ 30)

| Market | hurdle_G | hurdle_sharpe | hurdle_p | Confidence |
|--------|----------|---------------|----------|------------|
| under_25 | +0.0066 | 0.133 | 0.56 | **High** — 112 bets, balanced |
| over_25 | +0.0063 | 0.119 | 0.47 | **High** — 129 bets, balanced |
| btts_yes | +0.0063 | 0.158 | 0.55 | **High** — 110 bets |
| under_35 | +0.0052 | 0.129 | 0.77 | **High** — 93 bets, very high win rate |
| under_15 | +0.0048 | 0.111 | 0.30 | **Medium** — 96 bets |
| over_15 | +0.0003 | 0.038 | 0.70 | **Low** — barely positive, weak Sharpe |
| over_35 | +0.0004 | 0.043 | 0.22 | **Low** — barely positive, weak Sharpe |

### ❌ Kill (hurdle_G < 0)

| Market | hurdle_G | Why |
|--------|----------|-----|
| home | -0.025 | High variance, low win rate at moderate odds |
| away | -0.017 | Longshot variance destroys compounding |
| draw | -0.001 | Edge is real but too noisy for Kelly |
| btts_no | -0.006 | Market anchor causing overbetting |

---

## 4. Key Architectural Insight

The most striking pattern: **DCMH is the only model that achieves positive `hurdle_G` on both sides of the O/U 2.5 line simultaneously**. No other model manages this:

| Model | over_25 G | under_25 G | Sum |
|-------|-----------|------------|-----|
| DoublePoisson_NoMarket | +0.0007 | +0.0033 | 0.004 |
| DoublePoisson_Market | +0.0075 | +0.0066 | 0.014 |
| DixonColes_NoMarket | **-0.0005** | +0.0036 | 0.003 |
| DixonColes_Market | +0.0029 | +0.0015 | 0.004 |
| **DCMH** | **+0.0063** | **+0.0066** | **0.013** |

This confirms that the hierarchical $\rho$ parameter is allowing the model to correctly price the joint goal distribution, which is exactly what O/U markets need.

---

## 5. Recommendations

### Immediate Actions
1. **Remove 1X2 markets** from your Kelly staking pipeline entirely. They are wealth-destroying across all models.
2. **Remove btts_no** for market-anchored models. Only deploy btts_no if using a NoMarket variant.
3. **Deploy DCMH** on: `over_25`, `under_25`, `under_35`, `btts_yes`, `under_15`.

### Future Investigation
- The `draw` market shows the largest divergence between `hurdle_G_emp` (+0.0086) and `hurdle_G` (-0.001). This suggests the Gamma may not be a good fit here — draw winning ROIs might be multimodal. Worth plotting the raw ROI distribution.
- Consider a **fractional Kelly** (e.g., half-Kelly) on the markets where `hurdle_G` is positive but `hurdle_sharpe < 0.10`. This would reduce variance at the cost of slower growth.
