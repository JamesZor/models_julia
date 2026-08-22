# Scottish Lower Recombination & Noise-Reduction Wealth Leaderboard

**Tournament Segment:** Scottish League One (`#56`) & Scottish League Two (`#57`)  
**Target Historical Seasons:** `2024/25` & `2025/26` (40 Walk-Forward Folds, 710 Target Matches)  
**Execution Config:** Multi-Market Kelly Allocation (7 Market Families), Baker-McHale 800 Joint Draws, 2.0% Betfair Exchange Net Commission

---

## 🏆 Grand Leaderboard Overview

| Rank | Model Identifier | Model Description | Likelihood | Final Wealth (Balanced) | ROI % | Sharpe Ratio | Max Drawdown | Total Bets | RQR Bias | CRPS Goals |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **`recomb_pois_integrated_hl365_hs2`** | Integrated Open-Play + Penalty Co-Trained Engine | Poisson | **3.004x** | **+11.47%** | **1.08** | -37.75% | 1,919 | **+0.0199** | **0.6372** |
| 🥈 | **`goals_pois_ctl_hl365_hs2`** | Gross Goals Baseline Poisson Control | Poisson | **2.862x** | +11.06% | 1.05 | -38.91% | 1,927 | **+0.0081** | 0.6380 |
| 🥉 | **`goals_pois_open_play_hl365_hs2`** | Pure Open-Play Poisson (Un-recombined) | Poisson | **2.512x** | +9.03% | 1.01 | **-33.86%** | 2,002 | +0.1103 | 0.6420 |
| 4 | **`goals_negbin_ctl_hl365_hs2`** | Gross Goals Baseline Negative Binomial Control | NB2 | **1.924x** | +7.54% | 0.83 | **-33.58%** | 1,874 | +0.0354 | **0.6295** |
| 5 | **`goals_negbin_open_play_hl365_hs2`** | Pure Open-Play NegBin (Un-recombined) | NB2 | **1.425x** | +4.04% | 0.56 | -31.93% | 1,978 | +0.1240 | 0.6343 |

---

## 📊 Detailed Policy Breakdowns

### 1. Balanced Growth Policy (Stake Cap 15%, Drawdown Penalty $\lambda = 15$)
| Model Identifier | Final Wealth | Growth / Slate | ROI % | Mean Exposure % | Max Drawdown % | Sharpe Ratio | Calmar Ratio | Total Bets |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`recomb_pois_integrated_hl365_hs2`** | **3.004x** | **+1.111%** | **+11.47%** | 12.1% | -37.75% | **1.08** | 0.00 | 1,919 |
| **`goals_pois_ctl_hl365_hs2`** | 2.862x | +1.062% | +11.06% | 12.1% | -38.91% | 1.05 | 0.00 | 1,927 |
| **`goals_pois_open_play_hl365_hs2`** | 2.512x | +0.930% | +9.03% | 12.7% | -33.86% | 1.01 | 0.00 | 2,002 |
| **`goals_negbin_ctl_hl365_hs2`** | 1.924x | +0.661% | +7.54% | 11.0% | -33.58% | 0.83 | 0.00 | 1,874 |
| **`goals_negbin_open_play_hl365_hs2`** | 1.425x | +0.358% | +4.04% | 12.2% | -31.93% | 0.56 | 0.00 | 1,978 |

---

### 2. Conservative Policy (Stake Cap 10%, Drawdown Penalty $\lambda = 23$)
| Model Identifier | Final Wealth | Growth / Slate | ROI % | Mean Exposure % | Max Drawdown % | Sharpe Ratio | Calmar Ratio | Total Bets |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`recomb_pois_integrated_hl365_hs2`** | **2.215x** | **+0.803%** | **+11.52%** | 8.1% | **-26.48%** | **1.09** | 0.00 | 1,919 |
| **`goals_pois_ctl_hl365_hs2`** | 2.144x | +0.770% | +11.11% | 8.0% | -27.42% | 1.05 | 0.00 | 1,927 |
| **`goals_pois_open_play_hl365_hs2`** | 1.936x | +0.667% | +9.03% | 8.5% | -23.44% | 1.02 | 0.00 | 2,002 |
| **`goals_negbin_ctl_hl365_hs2`** | 1.589x | +0.468% | +7.41% | 7.3% | -23.65% | 0.82 | 0.00 | 1,874 |
| **`goals_negbin_open_play_hl365_hs2`** | 1.301x | +0.266% | +4.00% | 8.1% | -22.48% | 0.55 | 0.00 | 1,978 |

---

### 3. Aggressive Growth Policy (Stake Cap 25%, Drawdown Penalty $\lambda = 10$)
| Model Identifier | Final Wealth | Growth / Slate | ROI % | Mean Exposure % | Max Drawdown % | Sharpe Ratio | Calmar Ratio | Total Bets |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`recomb_pois_integrated_hl365_hs2`** | **5.118x** | **+1.649%** | **+12.14%** | 19.3% | -54.52% | **1.13** | 0.00 | 1,919 |
| **`goals_pois_ctl_hl365_hs2`** | 4.728x | +1.569% | +11.69% | 19.3% | -55.45% | 1.09 | 0.00 | 1,927 |
| **`goals_pois_open_play_hl365_hs2`** | 3.842x | +1.360% | +9.20% | 20.6% | -49.82% | 1.05 | 0.00 | 2,002 |
| **`goals_negbin_ctl_hl365_hs2`** | 2.586x | +0.960% | +7.91% | 17.4% | -46.41% | 0.87 | 0.00 | 1,874 |
| **`goals_negbin_open_play_hl365_hs2`** | 1.651x | +0.507% | +4.28% | 19.8% | -47.10% | 0.60 | 0.00 | 1,978 |

---

## 📈 15-Market Information Differential ($\Delta \text{LL}$ vs. Closing Fair Market)

*Negative values denote superior predictive information relative to the de-vigged closing market.*

| Model Identifier | 1X2 Family | BTTS Family | Totals Family | Over 1.5 | Under 1.5 | Over 2.5 | Under 2.5 | Over 3.5 | Under 3.5 | Draw Selection |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`recomb_pois_integrated`** | 0.01080 | 0.00650 | **-0.00156** | 0.0070 | 0.0070 | 0.0040 | 0.0040 | 0.0022 | 0.0022 | **-0.0012** |
| **`goals_pois_ctl`** | 0.01093 | 0.00720 | **-0.00034** | 0.0084 | 0.0084 | 0.0054 | 0.0054 | 0.0038 | 0.0038 | **-0.0009** |
| **`goals_negbin_ctl`** | **0.00343** | **0.00340** | +0.00070 | 0.0092 | 0.0092 | **0.0030** | **0.0030** | 0.0053 | 0.0053 | +0.0011 |
| **`goals_pois_open_play`** | 0.01137 | 0.00940 | +0.00686 | 0.0223 | 0.0223 | 0.0146 | 0.0146 | 0.0128 | 0.0128 | -0.0005 |
| **`goals_negbin_open_play`** | 0.00373 | 0.01110 | +0.00896 | 0.0248 | 0.0248 | 0.0137 | 0.0137 | 0.0140 | 0.0140 | +0.0015 |

---

## 🗄️ Model Checkpoint Locations on `mcmc-beast`

1. **`goals_pois_ctl_hl365_hs2`**: `data/scottish_open_play_grid/goals_pois_ctl_hl365_hs2_20260822_122018`
2. **`goals_pois_open_play_hl365_hs2`**: `data/scottish_open_play_grid/goals_pois_open_play_hl365_hs2_20260821_162201`
3. **`recomb_pois_integrated_hl365_hs2`**: `data/scottish_open_play_grid/recomb_pois_integrated_hl365_hs2_20260821_200041`
4. **`goals_negbin_ctl_hl365_hs2`**: `data/scottish_negbin_grid/goals_negbin_ctl_hl365_hs2_20260819_022431`
5. **`goals_negbin_open_play_hl365_hs2`**: `data/scottish_open_play_grid/goals_negbin_open_play_hl365_hs2_20260821_151232`
