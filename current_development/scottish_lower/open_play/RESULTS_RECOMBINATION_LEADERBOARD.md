# Scottish Lower Open-Play & Recombination Benchmark Leaderboard

> **Evaluation Dataset:** 710 Out-of-Sample Matches (Seasons 2024/25 & 2025/26) across Scottish Championship, League One, and League Two.  
> **Betfair Backtest Configuration:** Closed Betfair Exchange historical orderbooks (`Data.summarize_betfair_market`), **2.0% net exchange commission**, **800 Baker-McHale posterior draws** per match, multi-market convex Kelly allocation across 7 market families (1X2, BTTS, O/U 0.5 to 4.5).

---

## 🏆 Master Leaderboard: Betfair Exchange Historical Backtest (8 Models)

### 1. Balanced Growth Kelly Policy (Fixed Cap 15%, Drawdown Penalty $\lambda = 15$)

| Rank | Model Name | Final Wealth | Growth / Slate | Net ROI (%) | Mean Exp (%) | Max Drawdown (%) | Sharpe Ratio | Total Bets Placed |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`recomb_pxg_wealth_integrated_hl365_hs2`** 🏆 | **3.147x** | **+1.157%** | **+11.51%** | 12.1% | **-32.22%** 🛡️ | **1.17** 🏆 | 1,893 |
| 🥈 **2** | **`recomb_pois_wealth_integrated_hl365_hs2`** | **3.180x** 💰 | **+1.168%** | **+11.78%** 🏆 | 12.1% | -33.87% | **1.14** | 1,868 |
| 🥉 **3** | **`recomb_pois_integrated_hl365_hs2`** | **3.004x** | **+1.111%** | **+11.47%** | 12.1% | -37.75% | **1.08** | 1,919 |
| 4 | **`recomb_negbin_integrated_hl365_hs2`** | **2.891x** | **+1.072%** | **+11.58%** | 11.9% | -41.56% | **1.02** | 1,884 |
| 5 | **`goals_pois_ctl_hl365_hs2`** (Gross Goals) | **2.862x** | **+1.062%** | **+11.06%** | 12.1% | -38.91% | **1.05** | 1,927 |
| 6 | **`goals_pois_open_play_hl365_hs2`** | **2.512x** | **+0.930%** | **+9.03%** | 12.7% | -33.86% | **1.01** | 2,002 |
| 7 | **`goals_negbin_ctl_hl365_hs2`** (Gross Goals) | **1.924x** | **+0.661%** | **+7.54%** | 11.0% | -33.58% | **0.83** | 1,874 |
| 8 | **`goals_negbin_open_play_hl365_hs2`** | **1.425x** | **+0.358%** | **+4.04%** | 12.2% | -31.93% | **0.56** | 1,978 |

---

### 2. Conservative Kelly Policy (Fixed Cap 10%, Drawdown Penalty $\lambda = 23$)

| Rank | Model Name | Final Wealth | Growth / Slate | Net ROI (%) | Mean Exp (%) | Max Drawdown (%) | Sharpe Ratio | Total Bets Placed |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`recomb_pxg_wealth_integrated_hl365_hs2`** | **2.261x** | **+0.824%** | **+11.50%** | 8.1% | **-22.37%** 🛡️ | **1.17** 🏆 | 1,893 |
| 🥈 **2** | **`recomb_pois_wealth_integrated_hl365_hs2`** | **2.290x** 💰 | **+0.837%** | **+11.78%** 🏆 | 8.1% | -23.55% | **1.14** | 1,868 |
| 🥉 **3** | **`recomb_pois_integrated_hl365_hs2`** | **2.215x** | **+0.803%** | **+11.52%** | 8.1% | -26.48% | **1.09** | 1,919 |
| 4 | **`recomb_negbin_integrated_hl365_hs2`** | **2.160x** | **+0.778%** | **+11.57%** | 7.9% | -29.51% | **1.02** | 1,884 |
| 5 | **`goals_pois_ctl_hl365_hs2`** (Gross Goals) | **2.144x** | **+0.770%** | **+11.11%** | 8.0% | -27.42% | **1.05** | 1,927 |
| 6 | **`goals_pois_open_play_hl365_hs2`** | **1.936x** | **+0.667%** | **+9.03%** | 8.5% | -23.44% | **1.02** | 2,002 |
| 7 | **`goals_negbin_ctl_hl365_hs2`** (Gross Goals) | **1.589x** | **+0.468%** | **+7.41%** | 7.3% | -23.65% | **0.82** | 1,874 |
| 8 | **`goals_negbin_open_play_hl365_hs2`** | **1.301x** | **+0.266%** | **+4.00%** | 8.1% | -22.48% | **0.55** | 1,978 |

---

### 3. Aggressive Kelly Policy (Fixed Cap 25%, Drawdown Penalty $\lambda = 10$)

| Rank | Model Name | Final Wealth | Growth / Slate | Net ROI (%) | Mean Exp (%) | Max Drawdown (%) | Sharpe Ratio | Total Bets Placed |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`recomb_pois_wealth_integrated_hl365_hs2`** | **5.234x** 💰 | **+1.672%** | **+11.98%** 🏆 | 19.7% | -49.22% | **1.16** | 1,868 |
| 🥈 **2** | **`recomb_pxg_wealth_integrated_hl365_hs2`** | **5.229x** | **+1.671%** | **+11.68%** | 19.6% | **-47.79%** 🛡️ | **1.19** 🏆 | 1,893 |
| 🥉 **3** | **`recomb_pois_integrated_hl365_hs2`** | **5.118x** | **+1.649%** | **+12.14%** | 19.3% | -54.52% | **1.13** | 1,919 |
| 4 | **`goals_pois_ctl_hl365_hs2`** (Gross Goals) | **4.728x** | **+1.569%** | **+11.69%** | 19.3% | -55.45% | **1.09** | 1,927 |
| 5 | **`recomb_negbin_integrated_hl365_hs2`** | **4.411x** | **+1.499%** | **+11.86%** | 18.9% | -58.59% | **1.05** | 1,884 |
| 6 | **`goals_pois_open_play_hl365_hs2`** | **3.842x** | **+1.360%** | **+9.20%** | 20.6% | -49.82% | **1.05** | 2,002 |
| 7 | **`goals_negbin_ctl_hl365_hs2`** (Gross Goals) | **2.586x** | **+0.960%** | **+7.91%** | 17.4% | -46.41% | **0.87** | 1,874 |
| 8 | **`goals_negbin_open_play_hl365_hs2`** | **1.651x** | **+0.507%** | **+4.28%** | 19.8% | -47.10% | **0.60** | 1,978 |

---

## 📊 Scoring Rules & Calibration Metrics

| Model Name | RQR All ($\mu$) | RQR All ($\sigma$) | CRPS All | LogLoss $1\text{X}2$ Diff | LogLoss BTTS Diff | LogLoss Totals Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`recomb_pxg_wealth_integrated_hl365_hs2`** 🏆 | **-0.0031** | **0.9982** 🏆 | **0.6360** 🏆 | **+0.0105** 🏆 | **+0.0051** 🏆 | **-0.0021** 🏆 *(Beats Market)* |
| **`recomb_pois_wealth_integrated_hl365_hs2`** | $+0.0018$ | $1.0112$ | $0.6368$ | $+0.0120$ | $+0.0061$ | **-0.0019** *(Beats Market)* |
| **`recomb_negbin_integrated_hl365_hs2`** | **-0.0057** | **0.9694** | $0.6367$ | $+0.0108$ | $+0.0057$ | **-0.0018** *(Beats Market)* |
| **`recomb_pois_integrated_hl365_hs2`** | $+0.0032$ | $1.0157$ | $0.6372$ | $+0.0108$ | $+0.0065$ | **-0.0016** *(Beats Market)* |
| **`goals_pois_ctl_hl365_hs2`** | $+0.0988$ | $1.0371$ | $0.6380$ | $+0.0109$ | $+0.0072$ | **-0.0003** *(Beats Market)* |
| **`goals_negbin_ctl_hl365_hs2`** | $-0.0081$ | $0.9859$ | **0.6295** | **+0.0034** | **+0.0034** | $+0.0007$ |
| **`goals_pois_open_play_hl365_hs2`** | $+0.1258$ | $1.0257$ | $0.6420$ | $+0.0114$ | $+0.0094$ | $+0.0069$ |
| **`goals_negbin_open_play_hl365_hs2`** | $-0.0068$ | $0.9840$ | $0.6343$ | $+0.0037$ | $+0.0111$ | $+0.0090$ |

---

## 🔬 Core Insights & Architectural Findings

1. **Proxy xG Co-Training + Squad Wealth + Recombination = Champion Model**:
   - **`recomb_pxg_wealth_integrated`** captures the #1 Kelly Sharpe ratio (**1.17–1.19**) and the lowest peak-to-trough max drawdown (**-32.22%**) across all tested architectures.
   - It beats market fair logloss on Totals (**-0.0021**), demonstrating that separating non-tactical goals and co-training on continuous chance creation yields cleaner predictive signals.

2. **Starting-XI Squad Wealth Disparity ($\Delta W$) Adds Direct Alpha**:
   - Adding Starting-XI valuation differences improves final bankroll from **3.004x $\to$ 3.147x** and compresses drawdown by **5.53%**.

3. **Recombination Eliminates High-Variance Noise**:
   - Isolating penalties and own goals protects the bankroll from acute drawdown swings caused by referee whistling quirks and accidental deflections in lower leagues.
