# Scottish Lower: Recombination Experiment Logs & Implementation Track

This file tracks the active implementation logs, AD performance profiling, smoke tests, and walk-forward evaluations for the Two-Stage Recombination Models.

---

## 1. Experiment Grid Registry

| Experiment Tag | Model Family | Likelihood | Recombination Branch | Concurrency / Hardware | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `goals_pois_ctl_hl365_hs2` | Baseline Control | Poisson | None (Gross goals) | 16 threads (`mcmc-beast`) | ✅ Complete (17m 41s) | Fast Poisson benchmark control |
| `goals_pois_open_play_hl365_hs2` | Open-Play Decomp | Poisson | None (Clean $y_{\text{np\_nog}}$) | 16 threads (`mcmc-beast`) | ✅ Complete (30m 26s) | Pure open-play Poisson, un-recombined |
| `recomb_pois_integrated_hl365_hs2`| Recombination | Poisson | Co-Trained Turing Engine | 16 threads (`mcmc-beast`) | ✅ Complete (3h 38m) | Integrated MCMC ref penalty convolution |
| `goals_negbin_ctl_hl365_hs2` | Baseline Control | NB2 | None (Gross goals) | 16 threads (`mcmc-beast`) | ✅ Complete (1h 34m) | Incumbent NegBin baseline control |
| `goals_negbin_open_play_hl365_hs2` | Open-Play Decomp | NB2 | None (Clean $y_{\text{np\_nog}}$) | 16 threads (`mcmc-beast`) | ✅ Complete (1h 58m) | Pure open-play NegBin, un-recombined |
| `recomb_negbin_integrated` | Recombination (Scaled)| NB2 | Co-Trained Turing Engine | 16 threads (`mcmc-beast`) | ⏳ Queued | Full Negative Binomial scaling |

---

## 2. AD Performance & Profiling Standards (`docs/turing_ad_performance_guide.md`)

All Turing engines must strictly satisfy the following criteria before launching 40-fold walk-forward sampling:

- [x] **Vectorized Broadcast Operations**: `logpdf.(Poisson.(λ), y)` wrapped in a single `TrackedArray` node.
- [x] **No Scalar Loops**: Zero for-loops inside `@model`.
- [x] **No Dynamic Conditionals**: Binary masks (`xg_mask`, `ref_mask`) used for optional/missing values.
- [x] **Zero-Copy Views for Parameters**: `view(gamma_ref, ref_indices)` to prevent intermediate array allocations.
- [x] **Continuous Numerical Bounds**: `clamp.(log_λ, -20.0, 20.0)` and `1e-6` rate floors.
- [x] **Gradient Benchmark Target**: `@belapsed ReverseDiff.gradient! < 1.0ms` on compiled gradient tape.

### ReverseDiff Benchmark Results (`r04_benchmark_ad_recomb.jl`):
| Model Engine | # Parameters | Tape Compile Time | Gradient Eval Time | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Gross Goals Poisson Control** | 59 | 2,110.5 ms | **0.471 ms** | ⚡ EXCELLENT (<1ms) |
| **Pure Open-Play Poisson** | 59 | 2,247.7 ms | **0.484 ms** | ⚡ EXCELLENT (<1ms) |
| **Integrated Open-Play + Penalty Poisson** | 163 | 4,054.4 ms | **0.839 ms** | ⚡ EXCELLENT (<1ms) |
| **Pure Open-Play NegBin** | 93 | 2,834.3 ms | **0.642 ms** | ⚡ EXCELLENT (<1ms) |

---

## 3. Grand Benchmark Evaluation (All 5 Models, 40 Folds, 15 Markets)

### A. Randomized Quantile Residuals (RQR) Calibration
*Evaluates predictive calibration (Target: Mean $\approx 0.0$, Std $\approx 1.0$).*

| Model | Folds | RQR Mean All | RQR Std All | RQR Mean Home | RQR Mean Away | Calibration Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gross Goals Poisson Control** (`goals_pois_ctl`) | 40 | **$+0.0081$** | $1.0079$ | **$+0.0039$** | $+0.0124$ | 🌟 **EXCELLENT CALIBRATION** |
| **Integrated Recombination** (`recomb_pois_integrated`) | 40 | **$+0.0199$** | $1.0265$ | $+0.0290$ | **$+0.0108$** | 🌟 **NEAR-PERFECT CALIBRATION** |
| **Gross Goals NegBin Control** (`goals_negbin_ctl`) | 40 | $+0.0354$ | $1.0004$ | $+0.0400$ | $+0.0308$ | Mild Underforecasting |
| **Pure Open-Play Poisson** (`goals_pois_open_play`) | 40 | $+0.1103$ | $1.0287$ | $+0.0989$ | $+0.1217$ | Severe Underforecasting (Missing Noise) |
| **Pure Open-Play NegBin** (`goals_negbin_open_play`) | 40 | $+0.1240$ | $1.0216$ | $+0.1422$ | $+0.1058$ | Severe Underforecasting (Missing Noise) |

---

### B. CRPS & Log Loss Evaluation (Lower / Negative vs Market Close is Better)

| Model | CRPS All | CRPS Home | CRPS Away | 1X2 LogLoss Diff | Draw LogLoss Diff | BTTS LogLoss Diff | Totals LogLoss Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gross Goals NegBin Control** | **0.6295** | **0.6393** | **0.6196** | **0.00343** | +0.0011 | **0.0034** | +0.00070 |
| **Pure Open-Play NegBin** | 0.6343 | 0.6455 | 0.6231 | 0.00373 | +0.0015 | 0.0111 | +0.00896 |
| **Integrated Recombination** | 0.6372 | 0.6475 | 0.6270 | 0.01080 | **-0.0012** | 0.0065 | **-0.00156** (Beats Market) |
| **Gross Goals Poisson Control** | 0.6380 | 0.6483 | 0.6278 | 0.01093 | **-0.0009** | 0.0072 | **-0.00034** (Beats Market) |
| **Pure Open-Play Poisson** | 0.6420 | 0.6536 | 0.6304 | 0.01137 | -0.0005 | 0.0094 | +0.00686 |

---

### C. Multi-Market Kelly Portfolio Benchmark (Full 1,621 Matches, 709 Test Slates, 800 Joint Draws)

#### Balanced Growth Policy (Exposure Cap 15%, $\lambda = 15$)
| Model | Final Wealth | Slate Growth | ROI % | Mean Exposure | Max Drawdown % | Sharpe Ratio | Bets Placed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gross Goals NegBin Control** | **0.894x** | -0.108% | -0.43% | 6.8% | **-37.64%** | -0.04 | **1,653** |
| **Pure Open-Play Poisson** | 0.852x | -0.154% | +0.05% | 11.2% | -59.90% | +0.01 | 2,559 |
| **Integrated Recombination** | 0.689x | -0.359% | -1.64% | 9.9% | -63.85% | -0.15 | **1,841** |
| **Gross Goals Poisson Control** | 0.652x | -0.411% | -2.18% | 9.9% | -63.48% | -0.20 | 1,871 |
| **Pure Open-Play NegBin** | 0.561x | -0.556% | -4.38% | 10.7% | -57.17% | -0.67 | 2,486 |

---

## 4. Key Takeaways from the 5-Model Comparison

1. **Poisson Recombination beats Gross Poisson Control**:
   - In scoring rules, `recomb_pois_integrated` beats `goals_pois_ctl` on **CRPS** ($0.6372$ vs $0.6380$), **Totals LogLoss Diff** ($-0.00156$ vs $-0.00034$), **Draw Pricing** ($-0.0012$ vs $-0.0009$), and **BTTS** ($0.0065$ vs $0.0072$).
2. **Gross Poisson vs Gross NegBin Dispersion Gap**:
   - NegBin achieves lower CRPS ($0.6295$ vs $0.6380$) because goal scoring in Scottish lower leagues exhibits slight overdispersion, dampening false-precision Kelly stakes ($1,653$ bets vs $1,871$ bets).
3. **Phase 2 Expansion Clear Direction**:
   - Combining the **Recombination Architecture** with **Negative Binomial Likelihoods** (`recomb_negbin_integrated`) and **Frank Copula** dependencies is poised to achieve the highest predictive edge.
