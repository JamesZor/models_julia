# Scottish Lower League: Two-Stage Recombination & Noise-Reduction Experiment Notes

**Tournament Segment:** Scottish League One (`#56`) & Scottish League Two (`#57`)  
**Research Focus:** Noise-Reduction Decomposition (Open Play $y_{\text{np\_nog}}$ vs. Referee Penalty Whistles vs. Own Goals) & Discrete Convolution Recombination  
**Evaluation Grid:** 40-Fold Walk-Forward Rolling MCMC Grid (1,990 historical matches, 710 target test slates in 2024/25 & 2025/26, 1,900+ Betfair trades)  
**Hardware & Environment:** `mcmc-beast` (32 cores, CPU Pinned Threads, Queued NUTS Sampler)

---

## 1. Motivation & Empirical Foundations

In low-scoring sports like football, penalties and own goals introduce substantial un-systemic noise into team ratings:
1. **The Whistle Disparity**: 
   - Referees in Scottish League One/Two vary by over $4.4\times$ in penalty awards (from 0.11 pens/game up to 0.48 pens/game).
   - Penalties have low year-over-year persistence ($r = +0.128$), yet each penalty converts at $\sim 76.8\%$ ($\approx 0.77$ xG), creating massive single-event swings in team attack/defense ratings.
2. **Own Goals are Pure Poisson Noise**:
   - Occur uniformly across all 24 clubs at a rate of 1 in 36 team-matches ($0.0276$/game, $\chi^2 = 31.25, p = 0.4031$).
3. **The Open-Play Deflation Dilemma**:
   - Training purely on non-penalty, non-own-goal open play ($y_{\text{np\_nog}}$) isolates persistent team talent ($r = +0.228$, $+26.3\%$ signal gain).
   - However, un-recombined open-play models suffer a $+11\%$ to $+12\%$ probability deflation error against closing markets, underpredicting goals and over-betting Under lines ($2,486$ to $2,559$ bets).
4. **The Recombination Solution**:
   - Co-train the open-play skill baseline while explicitly modeling referee penalty tendencies.
   - Convolve the independent probability distributions back into a single, calibrated total-score matrix:
     $$P(Y = g) = \sum_{m=0}^g P(Y_{\text{open}} = m) \cdot P(Y_{\text{noise}} = g - m)$$

---

## 2. Mathematical Formulation & Architecture

### A. Integrated Co-Trained Turing Bayesian Model (`recomb_pois_integrated`)
```julia
# 1. Open-Play Skill Intensity
log_mu_h = clamp.(mu_base .+ ha_home .+ delta_month[m] .+ alpha[h] .- beta[a], -10.0, 10.0)
log_mu_a = clamp.(mu_base .+           delta_month[m] .+ alpha[a] .- beta[h], -10.0, 10.0)

# 2. Referee Whistle & Penalty Intensity
log_pen_h = clamp.(pen_base_mu .+ ha_pen .+ gamma_ref[ref] .+ alpha_pen[h] .+ beta_foul[a], -10.0, 5.0)
log_pen_a = clamp.(pen_base_mu .- ha_pen .+ gamma_ref[ref] .+ alpha_pen[a] .+ beta_foul[h], -10.0, 5.0)

# 3. Vectorized Multi-Target Likelihood
ll_open = sum((logpdf.(Poisson.(exp.(log_mu_h)), y_open_h) .+ logpdf.(Poisson.(exp.(log_mu_a)), y_open_a)) .* weights)
ll_pen  = sum((logpdf.(Poisson.(exp.(log_pen_h)), pens_h)   .+ logpdf.(Poisson.(exp.(log_pen_a)), pens_a)) .* ref_mask .* weights)
@addlogprob! (ll_open + ll_pen)
```

### B. Integrated Negative Binomial Recombination Model (`recomb_negbin_integrated`)
Co-trains open-play skill overdispersed goal counts with referee whistle tendencies:
```julia
# 1. Open-Play Negative Binomial Log-Mean
log_mu_h = clamp.(mu_base .+ ha_home .+ delta_month[m] .+ delta_league[l] .+ alpha[h] .- beta[a], -10.0, 10.0)
log_mu_a = clamp.(mu_base .+           delta_month[m] .+ delta_league[l] .+ alpha[a] .- beta[h], -10.0, 10.0)

# 2. Scottish Home/Away Asymmetric Dispersion
r_a = exp(log_r)
r_h = exp(log_r + delta_r_home)

# 3. Fast 0-Allocation Negative Binomial Log-Likelihood via Precomputed Gamma Recurrences
ll_open_h = _negbin_vector_loglik(y_open_h, log_mu_h, r_h, nb_h)
ll_open_a = _negbin_vector_loglik(y_open_a, log_mu_a, r_a, nb_a)

# 4. Referee Whistle & Penalty Intensity
log_pen_h = clamp.(pen_base_mu .+ ha_pen .+ gamma_ref[ref] .+ alpha_pen[h] .+ beta_foul[a], -10.0, 5.0)
log_pen_a = clamp.(pen_base_mu .- ha_pen .+ gamma_ref[ref] .+ alpha_pen[a] .+ beta_foul[h], -10.0, 5.0)
ll_pen    = sum((logpdf.(Poisson.(exp.(log_pen_h)), pens_h) .+ logpdf.(Poisson.(exp.(log_pen_a)), pens_a)) .* ref_mask .* weights)

# 5. Combined Likelihood
@addlogprob! (ll_open_h + ll_open_a + ll_pen)
```

### C. Discrete Convolution Score Matrix Generation
For each MCMC posterior draw $k$:
1. Compute Negative Binomial probability mass for open play:
   $$p_h = \frac{r_h}{r_h + \mu_{\text{open}, h}}, \quad P(Y_{\text{open}, h} = m) = \text{NegativeBinomial}(m \mid r_h, p_h)$$
2. Compute Poisson probability mass for noise goals (penalties + own goals):
   $$\lambda_{\text{noise}, h} = 0.768 \cdot \lambda_{\text{pen}, h} + 0.0276, \quad P(Y_{\text{noise}, h} = n) = \text{Poisson}(n \mid \lambda_{\text{noise}, h})$$
3. Discrete Convolution for total match goals:
   $$P(Y_{\text{tot}, h} = g) = \sum_{m=0}^g P(Y_{\text{open}, h} = m) \cdot P(Y_{\text{noise}, h} = g - m)$$
4. Outer product produces the complete, normalized joint score grid:
   $$S(i, j) = P(Y_{\text{tot}, h} = i) \cdot P(Y_{\text{tot}, a} = j)$$

### D. ReverseDiff AD Gradient Tape Performance Benchmarks
| Model Architecture | Parameters | Tape Compile Time | Gradient Eval Time | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Pure Open-Play Poisson** (`goals_pois_open_play`) | 59 | 4,466.8 ms | **0.429 ms** | ⚡ EXCELLENT (<1ms) |
| **Integrated Recombination Poisson** (`recomb_pois_integrated`) | 107 | 4,230.6 ms | **0.983 ms** | ⚡ EXCELLENT (<1ms) |
| **Pure Open-Play NegBin** (`goals_negbin_open_play`) | 93 | 7,062.5 ms | **0.572 ms** | ⚡ EXCELLENT (<1ms) |
| **Integrated Recombination NegBin** (`recomb_negbin_integrated`) | 112 | 5,917.4 ms | **1.021 ms** | ✓ GOOD (<2.5ms) |

---

## 3. Persistent Experiment Artifacts Registry

All models have completed 40-fold walk-forward MCMC sampling and are persisted on `mcmc-beast` under `/root/BayesianFootball/`:

| Model Tag | Likelihood | Split Config | Runtime | Saved Artifact Path | Latents Cached | MatchBooks Cached |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| `goals_pois_ctl_hl365_hs2` | Poisson Control | 40 Folds (120 chains) | 17m 41s | `data/scottish_open_play_grid/goals_pois_ctl_hl365_hs2_20260822_122018` | `oos_latents.jls` | `cache/books_goals_pois_ctl_hl365_hs2_full_bm800.jls` |
| `goals_pois_open_play_hl365_hs2` | Open-Play Poisson | 40 Folds (120 chains) | 30m 26s | `data/scottish_open_play_grid/goals_pois_open_play_hl365_hs2_20260821_162201` | `oos_latents.jls` | `cache/books_goals_pois_open_play_hl365_hs2_full_bm800.jls` |
| `recomb_pois_integrated_hl365_hs2` | Recombination | 40 Folds (120 chains) | 3h 38m | `data/scottish_open_play_grid/recomb_pois_integrated_hl365_hs2_20260821_200041` | `oos_latents.jls` | `cache/books_recomb_pois_integrated_hl365_hs2_full_bm800.jls` |
| `goals_negbin_ctl_hl365_hs2` | NegBin Control | 40 Folds (120 chains) | 1h 34m | `data/scottish_negbin_grid/goals_negbin_ctl_hl365_hs2_20260819_022431` | `oos_latents.jls` | `cache/books_goals_negbin_ctl_hl365_hs2_full_bm800.jls` |
| `goals_negbin_open_play_hl365_hs2` | Open-Play NegBin | 40 Folds (120 chains) | 1h 58m | `data/scottish_open_play_grid/goals_negbin_open_play_hl365_hs2_20260821_151232` | `oos_latents.jls` | `cache/books_goals_negbin_open_play_hl365_hs2_full_bm800.jls` |

---

## 4. Comprehensive Evaluation Benchmarks

### A. Scoring Rules & Calibration Metrics (40 Walk-Forward Folds, 710 Matches)

| Model Architecture | Likelihood | RQR Mean Bias (Target $\approx 0.0$) | Overall CRPS (Lower = Better) | Totals LogLoss Diff vs Market | Draw LogLoss Diff vs Market | BTTS LogLoss Diff vs Market |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gross Goals Poisson Control** (`goals_pois_ctl`) | Poisson | **`+0.0081`** | 0.6380 | **`-0.00034`** | **`-0.0009`** | 0.0072 |
| **Pure Open-Play Poisson** (`goals_pois_open_play`) | Poisson | `+0.1103` | 0.6420 | `+0.00686` | `-0.0005` | 0.0094 |
| **Integrated Recombination** (`recomb_pois_integrated`) | Poisson | **`+0.0199`** | **0.6372** | **`-0.00156`** *(Beats Market)* | **`-0.0012`** *(Beats Market)* | **0.0065** |
| **Gross Goals NegBin Control** (`goals_negbin_ctl`) | NegBin | `+0.0354` | **0.6295** | `+0.00070` | `+0.0011` | **0.0034** |
| **Pure Open-Play NegBin** (`goals_negbin_open_play`) | NegBin | `+0.1240` | 0.6343 | `+0.00896` | `+0.0015` | 0.0111 |

---

### B. Betfair Exchange Historical Portfolio Benchmark (24/25 & 25/26 Seasons, 2% Commission, BM 800 Draws)
*Evaluated across all 710 target matches in seasons 24/25 & 25/26 against closed Betfair Exchange orderbook prices with 2% net commission.*

#### 1. Balanced Growth Policy (Exposure Cap 15%, Drawdown Penalty $\lambda = 15$)
| Model | Final Wealth | Slate Growth | ROI % | Mean Exposure % | Max Drawdown % | Sharpe Ratio | Bets Placed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Integrated Recombination** (`recomb_pois_integrated`) | **3.004x** | **+1.111%** | **+11.47%** | 12.1% | -37.75% | **1.08** | **1,919** |
| 🥈 **Gross Goals Poisson Control** (`goals_pois_ctl`) | 2.862x | +1.062% | +11.06% | 12.1% | -38.91% | 1.05 | 1,927 |
| 🥉 **Pure Open-Play Poisson** (`goals_pois_open_play`) | 2.512x | +0.930% | +9.03% | 12.7% | **-33.86%** | 1.01 | 2,002 |
| 4. **Gross Goals NegBin Control** (`goals_negbin_ctl`) | 1.924x | +0.661% | +7.54% | 11.0% | **-33.58%** | 0.83 | 1,874 |
| 5. **Pure Open-Play NegBin** (`goals_negbin_open_play`) | 1.425x | +0.358% | +4.04% | 12.2% | **-31.93%** | 0.56 | 1,978 |

#### 2. Conservative Policy (Exposure Cap 10%, Drawdown Penalty $\lambda = 23$)
| Model | Final Wealth | Slate Growth | ROI % | Mean Exposure % | Max Drawdown % | Sharpe Ratio | Bets Placed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Integrated Recombination** (`recomb_pois_integrated`) | **2.215x** | **+0.803%** | **+11.52%** | 8.1% | **-26.48%** | **1.09** | **1,919** |
| 🥈 **Gross Goals Poisson Control** (`goals_pois_ctl`) | 2.144x | +0.770% | +11.11% | 8.0% | -27.42% | 1.05 | 1,927 |
| 🥉 **Pure Open-Play Poisson** (`goals_pois_open_play`) | 1.936x | +0.667% | +9.03% | 8.5% | -23.44% | 1.02 | 2,002 |
| 4. **Gross Goals NegBin Control** (`goals_negbin_ctl`) | 1.589x | +0.468% | +7.41% | 7.3% | -23.65% | 0.82 | 1,874 |
| 5. **Pure Open-Play NegBin** (`goals_negbin_open_play`) | 1.301x | +0.266% | +4.00% | 8.1% | -22.48% | 0.55 | 1,978 |

#### 3. Aggressive Policy (Exposure Cap 25%, Drawdown Penalty $\lambda = 10$)
| Model | Final Wealth | Slate Growth | ROI % | Mean Exposure % | Max Drawdown % | Sharpe Ratio | Bets Placed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Integrated Recombination** (`recomb_pois_integrated`) | **5.118x** | **+1.649%** | **+12.14%** | 19.3% | -54.52% | **1.13** | **1,919** |
| 🥈 **Gross Goals Poisson Control** (`goals_pois_ctl`) | 4.728x | +1.569% | +11.69% | 19.3% | -55.45% | 1.09 | 1,927 |
| 🥉 **Pure Open-Play Poisson** (`goals_pois_open_play`) | 3.842x | +1.360% | +9.20% | 20.6% | **-49.82%** | 1.05 | 2,002 |
| 4. **Gross Goals NegBin Control** (`goals_negbin_ctl`) | 2.586x | +0.960% | +7.91% | 17.4% | -46.41% | 0.87 | 1,874 |
| 5. **Pure Open-Play NegBin** (`goals_negbin_open_play`) | 1.651x | +0.507% | +4.28% | 19.8% | -47.10% | 0.60 | 1,978 |

---

## 5. Architectural Roadmap (Phase 2)

1. **Negative Binomial Recombination (`recomb_negbin_integrated`)**:
   - Transition the open-play rate to a Negative Binomial likelihood using precomputed loggamma sufficient statistics for 0-allocation ReverseDiff AD.
2. **Frank Copula Joint Dependency Structure**:
   - Couple the recombined home and away marginal distributions via hierarchical team-pair Frank copula dependence $\theta_{\text{copula}}$.
3. **Multi-Feature Synergies**:
   - Combine Recombination + Starting-XI Squad Wealth ($\Delta W$) + Travel Distance ($d_{\text{km}}$).
