# Scottish Lower: Open-Play Noise Reduction & Two-Stage Recombination Framework

This document contains the complete mathematical specifications, system architectures, data contracts, ReverseDiff AD benchmarks, and experimental backtest results for the **Open-Play Noise Reduction & Two-Stage Recombination** research track in Scottish Lower Football analytics (Scottish League One `#56` & Scottish League Two `#57`).

---

## 1. The Core Scientific Motivation

Traditional football rating models co-train directly on **gross match goals** ($Y_{\text{gross}}$). However, exploratory data analysis across 1,990 Scottish Lower matches revealed that gross goals are a composite of three fundamentally distinct processes:

$$Y_{\text{gross}} = Y_{\text{np\_nog}} + Y_{\text{pen}} + Y_{\text{og\_benefited}}$$

### Empirical Findings:
1. **Open-Play Goals ($Y_{\text{np\_nog}}$)**:
   - **Variance Reduction**: $\sigma^2$ drops from $1.518 \to 1.365$ (**-10.08% lower observation variance**).
   - **Signal Persistence**: Year-over-year autocorrelation increases from $r = +0.180 \to +0.228$ (**+26.3% higher persistence**).
   - Represents true repeatable attacking and defensive team quality.
2. **Penalties ($Y_{\text{pen}}$)**:
   - Penalty awards exhibit significant team variance ($\chi^2 = 51.17, p = 0.0093$) and substantial referee sensitivity ($\chi^2 = 57.55, p = 0.0555$, **$4.4\times$ referee strictness spread**).
   - Home referee whistle bias: $59.7\%$ of penalties are awarded to the home team.
   - Penalty conversion rate: Empirical $p_{\text{conv}} = 76.8\%$ (417 scored out of 543 penalties).
3. **Own Goals ($Y_{\text{og\_benefited}}$)**:
   - Own goals show zero cross-season team persistence ($r = +0.029, p = 0.81$, $\chi^2 = 31.25, p = 0.4031$).
   - Pure uniform stochastic deflection luck occurring at a constant Poisson intensity: $\lambda_{\text{og}} = 0.0276$ goals/match.

---

## 2. Mathematical Model Architectures

### A. Integrated Poisson Recombination Model (`recomb_pois_integrated`)
Co-trains open-play skill intensities alongside referee whistle strictness and team penalty propensities:
```julia
# 1. Vectorized Open-Play Poisson Intensity
log_mu_h = clamp.(mu_base .+ ha_home .+ delta_month[m] .+ delta_league[l] .+ alpha[h] .- beta[a], -10.0, 10.0)
log_mu_a = clamp.(mu_base .+           delta_month[m] .+ delta_league[l] .+ alpha[a] .- beta[h], -10.0, 10.0)

# 2. Vectorized Referee Whistle & Penalty Intensity
log_pen_h = clamp.(pen_base_mu .+ ha_pen .+ gamma_ref[ref] .+ alpha_pen[h] .+ beta_foul[a], -10.0, 5.0)
log_pen_a = clamp.(pen_base_mu .- ha_pen .+ gamma_ref[ref] .+ alpha_pen[a] .+ beta_foul[h], -10.0, 5.0)

# 3. Multi-Target Likelihood
ll_open = sum((logpdf.(Poisson.(exp.(log_mu_h)), y_open_h) .+ logpdf.(Poisson.(exp.(log_mu_a)), y_open_a)) .* weights)
ll_pen  = sum((logpdf.(Poisson.(exp.(log_pen_h)), pens_h)   .+ logpdf.(Poisson.(exp.(log_pen_a)), pens_a)) .* ref_mask .* weights)
@addlogprob! (ll_open + ll_pen)
```

---

### B. Integrated Negative Binomial Recombination Model (`recomb_negbin_integrated`)
Extends open-play modeling to overdispersed count distributions while retaining explicit penalty whistle co-training:

$$\begin{aligned}
y_{\text{np\_nog}, h, i} &\sim \text{NegativeBinomial2}\left(\mu_{h, \text{open}, i}, \; r_h\right) \\
y_{\text{np\_nog}, a, i} &\sim \text{NegativeBinomial2}\left(\mu_{a, \text{open}, i}, \; r_a\right) \\
N_{\text{pen}, h, i} &\sim \text{Poisson}\left(\lambda_{\text{pen}, h, i}\right) \\
N_{\text{pen}, a, i} &\sim \text{Poisson}\left(\lambda_{\text{pen}, a, i}\right)
\end{aligned}$$

#### Mathematical Formulation & 0-Allocation Vectorization:
```julia
# 1. Open-Play Negative Binomial Log-Mean
log_mu_h = clamp.(mu_base .+ ha_home .+ delta_month[m] .+ delta_league[l] .+ alpha[h] .- beta[a], -10.0, 10.0)
log_mu_a = clamp.(mu_base .+           delta_month[m] .+ delta_league[l] .+ alpha[a] .- beta[h], -10.0, 10.0)

# 2. Scottish Home/Away Asymmetric Dispersion
r_a = exp(log_r)
r_h = exp(log_r + delta_r_home)

# 3. Fast Vectorized Negative Binomial Log-Likelihood via Precomputed Gamma Recurrences
ll_open_h = _negbin_vector_loglik(y_open_h, log_mu_h, r_h, nb_h)
ll_open_a = _negbin_vector_loglik(y_open_a, log_mu_a, r_a, nb_a)

# 4. Referee Whistle & Penalty Intensity
log_pen_h = clamp.(pen_base_mu .+ ha_pen .+ gamma_ref[ref] .+ alpha_pen[h] .+ beta_foul[a], -10.0, 5.0)
log_pen_a = clamp.(pen_base_mu .- ha_pen .+ gamma_ref[ref] .+ alpha_pen[a] .+ beta_foul[h], -10.0, 5.0)
ll_pen    = sum((logpdf.(Poisson.(exp.(log_pen_h)), pens_h) .+ logpdf.(Poisson.(exp.(log_pen_a)), pens_a)) .* ref_mask .* weights)

# 5. Combined Likelihood
@addlogprob! (ll_open_h + ll_open_a + ll_pen)
```

---

### C. Discrete Probability Convolution Recombination

For each MCMC posterior draw $k$:
1. **Open-Play Goal Mass**:
   - For Poisson: $P(Y_{\text{open}, h} = m) = \text{Poisson}(m \mid \mu_{\text{open}, h})$
   - For NegBin: $p_h = \frac{r_h}{r_h + \mu_{\text{open}, h}}$, $P(Y_{\text{open}, h} = m) = \text{NegativeBinomial}(m \mid r_h, p_h)$
2. **Matchday Noise Goal Mass** (converted penalties + deflection own goals):
   $$\lambda_{\text{noise}, h} = 0.768 \cdot \lambda_{\text{pen}, h} + 0.0276, \quad P(Y_{\text{noise}, h} = n) = \text{Poisson}(n \mid \lambda_{\text{noise}, h})$$
3. **Discrete Convolution for Total Goals**:
   $$P(Y_{\text{tot}, h} = g) = \sum_{m=0}^g P(Y_{\text{open}, h} = m) \cdot P(Y_{\text{noise}, h} = g - m)$$
4. **Joint Match Score Matrix**:
   $$S(i, j) = P(Y_{\text{tot}, h} = i) \cdot P(Y_{\text{tot}, a} = j)$$

---

## 3. ReverseDiff AD Gradient Profiling & Smoke Testing

### A. ReverseDiff AD Gradient Tape Benchmarks (`r04_benchmark_ad_recomb.jl`)
*Target: Gradient evaluation time $< 1.0\text{ms}$ (Enforces docs/turing_ad_performance_guide.md)*

| Model Architecture | Parameters | Tape Compile Time | Gradient Eval Time | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Pure Open-Play Poisson** (`goals_pois_open_play`) | 59 | 4,466.8 ms | **0.429 ms** | ⚡ EXCELLENT (<1ms) |
| **Integrated Recombination Poisson** (`recomb_pois_integrated`) | 107 | 4,230.6 ms | **0.983 ms** | ⚡ EXCELLENT (<1ms) |
| **Pure Open-Play NegBin** (`goals_negbin_open_play`) | 93 | 7,062.5 ms | **0.572 ms** | ⚡ EXCELLENT (<1ms) |
| **Integrated Recombination NegBin** (`recomb_negbin_integrated`) | 112 | 5,917.4 ms | **1.021 ms** | ✓ GOOD (<2.5ms) |

### B. 1-Split MCMC NUTS Smoke Test Convergence (`r05_smoke_recomb.jl`)
- **Integrated Poisson Recombination**: Sampled in 73.7s, $\hat{R}_{\max} = \mathbf{1.0104}$, $\sigma_{\text{ref}} = 1.0015$, Home penalty whistle bias $= +0.1926$.
- **Integrated NegBin Recombination**: Sampled in 89.9s, $\hat{R}_{\max} = \mathbf{1.0148}$, Learned Scottish Dispersion: $r_{\text{home}} = \mathbf{25.5}$, $r_{\text{away}} = \mathbf{13.5}$.
- **Score Matrix Normalization**: Discrete convolution sums to $\mathbf{1.000000}$ identically.

---

## 4. 40-Fold Walk-Forward Calibration Leaderboard (710 Target Matches)

*Evaluated across all 40 rolling walk-forward test slates in 2024/25 & 2025/26 (710 out-of-sample matches).*

| Model Tag | Likelihood | RQR Mean Bias ($\approx 0.0$) | CRPS (Lower = Better) | Totals LogLoss Diff | Draw LogLoss Diff | BTTS LogLoss Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gross Goals Poisson Control** (`goals_pois_ctl`) | Poisson | **`+0.0081`** | 0.6380 | **`-0.00034`** | **`-0.0009`** | 0.0072 |
| **Pure Open-Play Poisson** (`goals_pois_open_play`) | Poisson | `+0.1103` | 0.6420 | `+0.00686` | `-0.0005` | 0.0094 |
| **Integrated Recombination Poisson** (`recomb_pois_integrated`) | Poisson | **`+0.0199`** | **0.6372** | **`-0.00156`** *(Beats Market)* | **`-0.0012`** *(Beats Market)* | **0.0065** |
| **Gross Goals NegBin Control** (`goals_negbin_ctl`) | NegBin | `+0.0354` | **0.6295** | `+0.00070` | `+0.0011` | **0.0| **Pure Open-Play NegBin** (`goals_negbin_open_play`) | NegBin | `+0.1240` | 0.6343 | `+0.00896` | `+0.0015` | 0.0111 |
| **Integrated Recombination NegBin** (`recomb_negbin_integrated`) | NegBin | **`-0.0071`** | **0.6367** | `+0.03306` | `+0.0026` | 0.0294 |

*(LogLoss Diff = Model LogLoss − De-Vigged Market Close; negative numbers beat closing bookmaker lines)*

---

## 5. Betfair Exchange Historical Portfolio Backtest (24/25 & 25/26 Seasons, 2% Commission, BM 800 Draws)

*Evaluated across all 710 target matches in seasons 2024/25 & 2025/26 against closed Betfair Exchange orderbook prices with 2.0% net exchange commission and 800 Baker-McHale posterior draws.*

### 1. Balanced Growth Policy (Exposure Cap 15%, Drawdown Penalty $\lambda = 15$)
| Model | Final Wealth | Slate Growth | ROI % | Mean Exposure % | Max Drawdown % | Sharpe Ratio | Bets Placed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Integrated Recombination Poisson** (`recomb_pois_integrated`) | **3.004x** | **+1.111%** | **+11.47%** | 12.1% | -37.75% | **1.08** | **1,919** |
| 🥈 **Gross Goals Poisson Control** (`goals_pois_ctl`) | 2.862x | +1.062% | +11.06% | 12.1% | -38.91% | 1.05 | 1,927 |
| 🥉 **Pure Open-Play Poisson** (`goals_pois_open_play`) | 2.512x | +0.930% | +9.03% | 12.7% | **-33.86%** | 1.01 | 2,002 |
| 4. **Gross Goals NegBin Control** (`goals_negbin_ctl`) | 1.924x | +0.661% | +7.54% | 11.0% | **-33.58%** | 0.83 | 1,874 |
| 5. **Pure Open-Play NegBin** (`goals_negbin_open_play`) | 1.425x | +0.358% | +4.04% | 12.2% | **-31.93%** | 0.56 | 1,978 |
| 6. **Integrated Recombination NegBin** (`recomb_negbin_integrated`) | 0.960x | -0.041% | +0.80% | 13.5% | -34.46% | 0.12 | 2,095 |

---

### 2. Conservative Policy (Exposure Cap 10%, Drawdown Penalty $\lambda = 23$)
| Model | Final Wealth | Slate Growth | ROI % | Mean Exposure % | Max Drawdown % | Sharpe Ratio | Bets Placed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Integrated Recombination Poisson** (`recomb_pois_integrated`) | **2.215x** | **+0.803%** | **+11.52%** | 8.1% | **-26.48%** | **1.09** | **1,919** |
| 🥈 **Gross Goals Poisson Control** (`goals_pois_ctl`) | 2.144x | +0.770% | +11.11% | 8.0% | -27.42% | 1.05 | 1,927 |
| 🥉 **Pure Open-Play Poisson** (`goals_pois_open_play`) | 1.936x | +0.667% | +9.03% | 8.5% | -23.44% | 1.02 | 2,002 |
| 4. **Gross Goals NegBin Control** (`goals_negbin_ctl`) | 1.589x | +0.468% | +7.41% | 7.3% | -23.65% | 0.82 | 1,874 |
| 5. **Pure Open-Play NegBin** (`goals_negbin_open_play`) | 1.301x | +0.266% | +4.00% | 8.1% | **-22.48%** | 0.55 | 1,978 |
| 6. **Integrated Recombination NegBin** (`recomb_negbin_integrated`) | 1.006x | +0.006% | +0.81% | 9.0% | -23.88% | 0.12 | 2,095 |

---

### 3. Aggressive Policy (Exposure Cap 25%, Drawdown Penalty $\lambda = 10$)
| Model | Final Wealth | Slate Growth | ROI % | Mean Exposure % | Max Drawdown % | Sharpe Ratio | Bets Placed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Integrated Recombination Poisson** (`recomb_pois_integrated`) | **5.118x** | **+1.649%** | **+12.14%** | 19.3% | -54.52% | **1.13** | **1,919** |
| 🥈 **Gross Goals Poisson Control** (`goals_pois_ctl`) | 4.728x | +1.569% | +11.69% | 19.3% | -55.45% | 1.09 | 1,927 |
| 🥉 **Pure Open-Play Poisson** (`goals_pois_open_play`) | 3.842x | +1.360% | +9.20% | 20.6% | **-49.82%** | 1.05 | 2,002 |
| 4. **Gross Goals NegBin Control** (`goals_negbin_ctl`) | 2.586x | +0.960% | +7.91% | 17.4% | -46.41% | 0.87 | 1,874 |
| 5. **Pure Open-Play NegBin** (`goals_negbin_open_play`) | 1.651x | +0.507% | +4.28% | 19.8% | -47.10% | 0.60 | 1,978 |
| 6. **Integrated Recombination NegBin** (`recomb_negbin_integrated`) | 0.884x | -0.125% | +1.16% | 22.1% | -49.74% | 0.17 | 2,095 |

---

## 6. Persistent Checkpoint Registry

All completed MCMC training runs are permanently saved on disk on `mcmc-beast` (`/root/BayesianFootball/`):

- `goals_pois_ctl_hl365_hs2`: `/root/BayesianFootball/data/scottish_open_play_grid/goals_pois_ctl_hl365_hs2_20260822_122018`
- `goals_pois_open_play_hl365_hs2`: `/root/BayesianFootball/data/scottish_open_play_grid/goals_pois_open_play_hl365_hs2_20260821_162201`
- `recomb_pois_integrated_hl365_hs2`: `/root/BayesianFootball/data/scottish_open_play_grid/recomb_pois_integrated_hl365_hs2_20260821_200041`
- `goals_negbin_ctl_hl365_hs2`: `/root/BayesianFootball/data/scottish_negbin_grid/goals_negbin_ctl_hl365_hs2_20260819_022431`
- `goals_negbin_open_play_hl365_hs2`: `/root/BayesianFootball/data/scottish_open_play_grid/goals_negbin_open_play_hl365_hs2_20260821_151232`
- `recomb_negbin_integrated_hl365_hs2`: `/root/BayesianFootball/data/scottish_open_play_grid/recomb_negbin_integrated_hl365_hs2_20260822_160843`

