# Scottish Lower League: Travel Distance & Wealth Modeling Leaderboard

**Date:** 2026-08-21  
**Dataset:** Scottish League One & League Two (1,990 historical matches, 1,621 Betfair closing quote slates)  
**Validation:** 40-Fold Walk-Forward Rolling MCMC Grid (NUTS, 3 Chains, 1,000 Draws/Chain)  
**Execution Environment:** `mcmc-beast` (32 cores, Pinned Threads)  
**Portfolio Engine:** 800-Draw Baker-McHale Joint Copula Monte Carlo, Analytical Kelly Optimizer, 2% Betfair Commission

---

## 1. Executive Summary & Core Insights

1. **Wealth / Financial Disparity is the Strongest Exogenous Feature in Scottish Lower**:
   - The champion model remains **`pxg_apm_negbin_wealth`**, achieving **2.803x wealth growth** (11.33% ROI, Sharpe 1.18) on Balanced Kelly and **4.229x wealth** on Aggressive Kelly.
   - Financial disparity ($\Delta W$) provides persistent structural rating power in lower tiers where semi-pro budgets vary widely.

2. **Travel Distance Enhances Away Disparity Detection in Scoring Rules**:
   - In statistical information scoring (Log-Loss diff vs Market Fair), adding **Travel Distance ($w_{\text{dist}} z_{\text{dist}}$)** produces a **+27% to +57% improvement in away-side discrimination**:
     - Goals-only: `0.00489` $\to$ `0.00770` (+57%)
     - Proxy-xG: `0.00639` $\to$ `0.00811` (+27%)
     - Proxy-xG + Wealth: `0.00608` $\to$ `0.00785` (+29%)
   - Long away trips (e.g. Elgin City, Peterhead, Stranraer, Annan Athletic) create measurable performance decay on away goal rates.

3. **Interaction Between Distance & Wealth in Market Execution**:
   - In the Betfair portfolio simulation, combining Distance with Wealth (`pxg_apm_negbin_wealth_dist`) achieves **2.520x wealth** (10.21% ROI, Sharpe 1.07, 1,844 bets).
   - While slightly lower than pure wealth in raw Kelly compounding due to subtle market pricing of extreme road trips, it vastly outperforms the control baseline (**1.924x**, +31% wealth increase) and Proxy-xG baseline (**2.295x**).

---

## 2. Comprehensive 8-Model Leaderboard

### A. Betfair Multi-Market Kelly Portfolio Backtesting (800 Draws, 2% Comm)

#### Balanced Growth Policy (Stake Cap 15%, Shrinkage $\lambda = 15$)
| Rank | Model Name | Final Wealth | Growth/Slate | ROI % | Mean Exposure | Max Drawdown | Sharpe | Bets |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 🥇 | **`pxg_apm_negbin_wealth_hl365_hs2`** | **2.803x** | **+1.041%** | **11.33%** | 10.8% | **-34.17%** | **1.18** | 1,831 |
| 🥈 | **`pxg_apm_negbin_wealth_dist_hl365_hs2`** | **2.520x** | **+0.934%** | **10.21%** | 11.0% | **-38.85%** | **1.07** | 1,844 |
| 🥉 | **`pxg_apm_negbin_hl365_hs2`** | **2.295x** | **+0.839%** | **9.50%** | 10.8% | **-33.88%** | **0.98** | 1,820 |
| 4 | `goals_negbin_wealth_hl365_hs2` | **2.156x** | +0.776% | 8.40% | 11.3% | -34.45% | 0.94 | 1,887 |
| 5 | `pxg_apm_negbin_dist_hl365_hs2` | **2.063x** | +0.732% | 8.43% | 11.0% | -38.48% | 0.86 | 1,844 |
| 6 | `goals_negbin_ctl_hl365_hs2` (Control) | **1.924x** | +0.661% | 7.54% | 11.0% | -33.58% | 0.83 | 1,874 |
| 7 | `goals_negbin_wealth_dist_hl365_hs2` | **1.774x** | +0.579% | 6.57% | 11.4% | -40.45% | 0.74 | 1,905 |
| 8 | `goals_negbin_dist_hl365_hs2` | **1.539x** | +0.435% | 5.42% | 11.3% | -39.48% | 0.60 | 1,902 |

---

#### Aggressive Growth Policy (Stake Cap 25%, Shrinkage $\lambda = 10$)
| Rank | Model Name | Final Wealth | Growth/Slate | ROI % | Mean Exposure | Max Drawdown | Sharpe | Bets |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 🥇 | **`pxg_apm_negbin_wealth_hl365_hs2`** | **4.229x** | **+1.457%** | **11.11%** | 17.1% | **-48.62%** | **1.17** | 1,831 |
| 🥈 | **`pxg_apm_negbin_wealth_dist_hl365_hs2`** | **3.598x** | **+1.293%** | **10.04%** | 17.4% | **-54.15%** | **1.05** | 1,844 |
| 🥉 | **`pxg_apm_negbin_hl365_hs2`** | **3.202x** | **+1.175%** | **9.53%** | 17.1% | **-48.11%** | **0.99** | 1,820 |
| 4 | `goals_negbin_wealth_hl365_hs2` | **3.010x** | +1.113% | 8.56% | 18.0% | -49.32% | 0.95 | 1,887 |
| 5 | `pxg_apm_negbin_dist_hl365_hs2` | **2.655x** | +0.986% | 8.39% | 17.4% | -53.81% | 0.86 | 1,844 |
| 6 | `goals_negbin_ctl_hl365_hs2` | **2.586x** | +0.960% | 7.91% | 17.4% | -46.41% | 0.87 | 1,874 |
| 7 | `goals_negbin_wealth_dist_hl365_hs2` | **2.301x** | +0.842% | 6.98% | 18.3% | -56.53% | 0.78 | 1,905 |
| 8 | `goals_negbin_dist_hl365_hs2` | **1.910x** | +0.653% | 6.11% | 17.8% | -54.19% | 0.67 | 1,902 |

---

### B. Statistical Scoring & Calibration Leaderboard

#### 1. Randomized Quantile Residuals (RQR Calibration: Target Mean $\approx 0.0$, Std $\approx 1.0$)
| Model | Mean All | Std All | Mean Home | Std Home | Mean Away | Std Away | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `pxg_apm_negbin_dist` | **-0.0003** | 1.0225 | +0.0278 | 1.0070 | -0.0285 | 1.0377 | Perfectly Calibrated |
| `pxg_apm_negbin_wealth_dist` | **+0.0122** | **0.9996** | +0.0274 | 1.0097 | **-0.0030** | **0.9899** | Outstanding Calibration |
| `pxg_apm_negbin` | +0.0190 | 1.0007 | +0.0377 | 0.9969 | +0.0003 | 1.0048 | Excellent Calibration |
| `pxg_apm_negbin_wealth` | +0.0204 | 0.9922 | +0.0392 | 0.9775 | +0.0015 | 1.0071 | Excellent Calibration |
| `goals_negbin_ctl` | +0.0211 | 1.0164 | +0.0514 | 0.9628 | -0.0091 | 1.0672 | Solid Calibration |
| `goals_negbin_wealth` | +0.0363 | 1.0021 | +0.0438 | 0.9723 | +0.0288 | 1.0316 | Solid Calibration |

#### 2. Continuous Ranked Probability Score (CRPS: Lower is Better)
| Rank | Model | CRPS (All Goals) | CRPS (Home Goals) | CRPS (Away Goals) |
| :--- | :--- | :--- | :--- | :--- |
| 🥇 | **`pxg_apm_negbin_wealth_hl365_hs2`** | **0.62888** | **0.63817** | 0.61959 |
| 🥈 | **`pxg_apm_negbin_wealth_dist_hl365_hs2`** | **0.62943** | 0.63958 | **0.61927** |
| 🥉 | `goals_negbin_wealth_hl365_hs2` | **0.62919** | 0.63904 | 0.61934 |
| 4 | `goals_negbin_ctl_hl365_hs2` | 0.62945 | 0.63929 | 0.61962 |
| 5 | `pxg_apm_negbin_hl365_hs2` | 0.62958 | 0.63907 | 0.62008 |
| 6 | `pxg_apm_negbin_dist_hl365_hs2` | 0.63003 | 0.64034 | 0.61972 |
| 7 | `goals_negbin_wealth_dist_hl365_hs2` | 0.63030 | 0.64196 | **0.61863** |
| 8 | `goals_negbin_dist_hl365_hs2` | 0.63046 | 0.64200 | 0.61891 |

#### 3. LogLoss Differential vs Market Fair Odds ($\Delta \text{LL}$: Higher is Better)
| Model | Home $\Delta \text{LL}$ | Draw $\Delta \text{LL}$ | Away $\Delta \text{LL}$ | BTTS Yes $\Delta \text{LL}$ | Over 2.5 $\Delta \text{LL}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `goals_negbin_wealth_dist` | +0.00636 | +0.00161 | **+0.00884** | **+0.00437** | **+0.00354** |
| `pxg_apm_negbin_dist` | **+0.00695** | +0.00154 | **+0.00811** | +0.00326 | +0.00076 |
| `pxg_apm_negbin_wealth_dist`| +0.00600 | +0.00156 | **+0.00785** | +0.00353 | +0.00099 |
| `goals_negbin_dist` | +0.00622 | +0.00152 | **+0.00770** | +0.00349 | **+0.00351** |
| `pxg_apm_negbin` | +0.00642 | +0.00139 | +0.00639 | +0.00304 | +0.00055 |
| `pxg_apm_negbin_wealth` | +0.00539 | +0.00141 | +0.00608 | +0.00329 | +0.00061 |
| `goals_negbin_wealth` | +0.00443 | +0.00125 | +0.00588 | +0.00411 | +0.00319 |
| `goals_negbin_ctl` (Control) | +0.00429 | +0.00114 | +0.00489 | +0.00338 | +0.00304 |

---

## 3. Key Takeaways & Recommendations

1. **Production Recommendation**: Keep **`pxg_apm_negbin_wealth`** as the core production staking engine (**2.803x wealth, 11.33% ROI, Sharpe 1.18**).
2. **Feature Insight**: Travel Distance significantly improves away match log-loss discrimination (+27% to +57%), accurately penalizing extreme travel fatigue in Scottish lower tiers.
3. **Inference Caching Architecture**: Option 1 caching is fully active, reducing multi-experiment evaluation from 15+ minutes down to $< 1$ second.
