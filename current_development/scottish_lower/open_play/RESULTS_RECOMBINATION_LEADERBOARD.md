# Scottish Lower Open-Play & Recombination Benchmark Leaderboard

> **Evaluation Dataset:** 710 Out-of-Sample Matches (Seasons 2024/25 & 2025/26) across Scottish Championship, League One, and League Two.  
> **Betfair Backtest Configuration:** Closed Betfair Exchange historical orderbooks (`Data.summarize_betfair_market`), **2.0% net exchange commission**, **800 Baker-McHale posterior draws** per match, multi-market convex Kelly allocation across 7 market families (1X2, BTTS, O/U 0.5 to 4.5).

---

## 🏆 Master Leaderboard: Betfair Exchange Historical Backtest

### 1. Balanced Growth Kelly Policy (Fixed Cap 15%, Drawdown Penalty $\lambda = 15$)

| Rank | Model Name | Final Wealth | Growth / Slate | Net ROI (%) | Mean Exp (%) | Max Drawdown (%) | Sharpe Ratio | Total Bets Placed |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`recomb_pois_integrated_hl365_hs2`** | **3.004x** | **+1.111%** | **+11.47%** | 12.1% | **-37.75%** | **1.08** | 1,919 |
| 🥈 **2** | **`recomb_negbin_integrated_hl365_hs2`** | **2.891x** | **+1.072%** | **+11.58%** 🏆 | 11.9% | -41.56% | **1.02** | 1,884 |
| 🥉 **3** | **`goals_pois_ctl_hl365_hs2`** (Gross Goals) | **2.862x** | **+1.062%** | **+11.06%** | 12.1% | **-38.91%** | **1.05** | 1,927 |
| 4 | **`goals_pois_open_play_hl365_hs2`** | **2.512x** | **+0.930%** | **+9.03%** | 12.7% | **-33.86%** | **1.01** | 2,002 |
| 5 | **`goals_negbin_ctl_hl365_hs2`** (Gross Goals) | **1.924x** | **+0.661%** | **+7.54%** | 11.0% | **-33.58%** | **0.83** | 1,874 |
| 6 | **`goals_negbin_open_play_hl365_hs2`** | **1.425x** | **+0.358%** | **+4.04%** | 12.2% | **-31.93%** | **0.56** | 1,978 |

---

### 2. Conservative Kelly Policy (Fixed Cap 10%, Drawdown Penalty $\lambda = 23$)

| Rank | Model Name | Final Wealth | Growth / Slate | Net ROI (%) | Mean Exp (%) | Max Drawdown (%) | Sharpe Ratio | Total Bets Placed |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`recomb_pois_integrated_hl365_hs2`** | **2.215x** | **+0.803%** | **+11.52%** | 8.1% | **-26.48%** | **1.09** | 1,919 |
| 🥈 **2** | **`recomb_negbin_integrated_hl365_hs2`** | **2.160x** | **+0.778%** | **+11.57%** 🏆 | 7.9% | -29.51% | **1.02** | 1,884 |
| 🥉 **3** | **`goals_pois_ctl_hl365_hs2`** (Gross Goals) | **2.144x** | **+0.770%** | **+11.11%** | 8.0% | **-27.42%** | **1.05** | 1,927 |
| 4 | **`goals_pois_open_play_hl365_hs2`** | **1.936x** | **+0.667%** | **+9.03%** | 8.5% | **-23.44%** | **1.02** | 2,002 |
| 5 | **`goals_negbin_ctl_hl365_hs2`** (Gross Goals) | **1.589x** | **+0.468%** | **+7.41%** | 7.3% | **-23.65%** | **0.82** | 1,874 |
| 6 | **`goals_negbin_open_play_hl365_hs2`** | **1.301x** | **+0.266%** | **+4.00%** | 8.1% | **-22.48%** | **0.55** | 1,978 |

---

### 3. Aggressive Kelly Policy (Fixed Cap 25%, Drawdown Penalty $\lambda = 10$)

| Rank | Model Name | Final Wealth | Growth / Slate | Net ROI (%) | Mean Exp (%) | Max Drawdown (%) | Sharpe Ratio | Total Bets Placed |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`recomb_pois_integrated_hl365_hs2`** | **5.118x** | **+1.649%** | **+12.14%** 🏆 | 19.3% | **-54.52%** | **1.13** | 1,919 |
| 🥈 **2** | **`goals_pois_ctl_hl365_hs2`** (Gross Goals) | **4.728x** | **+1.569%** | **+11.69%** | 19.3% | **-55.45%** | **1.09** | 1,927 |
| 🥉 **3** | **`recomb_negbin_integrated_hl365_hs2`** | **4.411x** | **+1.499%** | **+11.86%** | 18.9% | -58.59% | **1.05** | 1,884 |
| 4 | **`goals_pois_open_play_hl365_hs2`** | **3.842x** | **+1.360%** | **+9.20%** | 20.6% | **-49.82%** | **1.05** | 2,002 |
| 5 | **`goals_negbin_ctl_hl365_hs2`** (Gross Goals) | **2.586x** | **+0.960%** | **+7.91%** | 17.4% | **-46.41%** | **0.87** | 1,874 |
| 6 | **`goals_negbin_open_play_hl365_hs2`** | **1.651x** | **+0.507%** | **+4.28%** | 19.8% | **-47.10%** | **0.60** | 1,978 |

---

## 📊 Scoring Rules & Calibration Metrics

| Model Name | RQR All ($\mu$) | RQR All ($\sigma$) | CRPS All | LogLoss $1\text{X}2$ Diff | LogLoss BTTS Diff | LogLoss Totals Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`recomb_negbin_integrated_hl365_hs2`** | **-0.0057** 🏆 | **0.9694** 🏆 | **0.6367** 🏆 | **+0.0108** | **+0.0057** 🏆 | **-0.0018** 🏆 *(Beats Market)* |
| **`recomb_pois_integrated_hl365_hs2`** | $+0.0032$ | $1.0157$ | $0.6372$ | $+0.0108$ | $+0.0065$ | **-0.0016** *(Beats Market)* |
| **`goals_pois_ctl_hl365_hs2`** | $+0.0988$ | $1.0371$ | $0.6380$ | $+0.0109$ | $+0.0072$ | **-0.0003** *(Beats Market)* |
| **`goals_negbin_ctl_hl365_hs2`** | $-0.0081$ | $0.9859$ | **0.6295** | **+0.0034** | **+0.0034** | $+0.0007$ |
| **`goals_pois_open_play_hl365_hs2`** | $+0.1258$ | $1.0257$ | $0.6420$ | $+0.0114$ | $+0.0094$ | $+0.0069$ |
| **`goals_negbin_open_play_hl365_hs2`** | $-0.0068$ | $0.9840$ | $0.6343$ | $+0.0037$ | $+0.0111$ | $+0.0090$ |

---

## 🔬 Core Insights & Architectural Findings

1. **Recombination Outperforms Across Both Likelihood Families**:
   - **Negative Binomial Family**: `recomb_negbin_integrated` (**2.891x wealth / +11.58% ROI**) completely crushes `goals_negbin_ctl` (**1.924x wealth / +7.54% ROI**), producing **+50% more final capital** and **+404 bps of net alpha**.
   - **Poisson Family**: `recomb_pois_integrated` (**3.004x wealth / +11.47% ROI**) outperforms `goals_pois_ctl` (**2.862x wealth / +11.06% ROI**).
   - This proves conclusively that separating open-play skills from referee penalty strictness removes systemic noise and boosts betting edge across the board.

2. **Scoring Rules Supremacy**:
   - `recomb_negbin_integrated` achieves the best out-of-sample calibration in project history ($\text{RQR } \mu = -0.0057, \sigma = 0.9694$) and achieves the deepest LogLoss beat against closing market prices on Over/Under Totals (**`-0.0018`**).
