# Recombination, Proxy xG (pxG), & Squad Wealth Architecture

**Context & Engineering Guide for BayesianFootball.jl**  
*Target Audience: Antigravity AI Agents, Quantitative Developers, and Bayesian Modelers*

---

## 🎯 1. Executive Summary & Problem Motivation

Standard football betting models fit Poisson or Negative Binomial distributions directly to **gross match scores** (e.g., 2–1, 0–0). In lower leagues (e.g., Scottish Championship & League 1/2), gross scores are heavily contaminated by non-systemic high-variance noise:
1. **Penalty Kicks:** Penalty awards occur in $\sim 18\text{--}22\%$ of matches, convert at $\sim 76.8\%$, and are strongly influenced by individual referee whistling tendencies rather than sustained team attacking supremacy.
2. **Own Goals:** Pure accidental noise ($\sim 0.0276$ goals/match) that distorts team defensive/attacking ratings.
3. **Finishing Variance on Low Shot Counts:** Low goal tallies conceal true open-play chance creation.

### The Solution: Decomposition & Co-Training
Instead of modeling gross goals as an indivisible monolithic variable, our architecture:
1. Decomposes outcomes into **Open Play + Penalty Awards + Own Goals**.
2. Co-trains tactical open-play intensity against **Continuous Open-Play Proxy xG ($\text{pxG}$)** and **Realized Open-Play Goals** via a team finishing factor $\kappa_i$.
3. Adjusts open-play intensity dynamically via **Starting-XI Squad Market Wealth ($\Delta W$)**.
4. Recombines all latent components via **Exact Discrete Poisson Convolution** to produce the final score matrix:
$$\mu_{\text{total}, h} = \kappa_h \cdot \mu_{\text{open}, h} + q_{\text{pen}} \cdot \lambda_{\text{pen}, h} + \lambda_{\text{og}}$$

---

## 📐 2. Mathematical Formulation

### 2.1 Open-Play Tactical Intensity ($\mu_{\text{open}}$)
For match $m$ between home team $h$ and away team $a$ in month $t_{\text{month}}$ and division $t_{\text{league}}$:
$$\log \mu_{\text{open}, h} = \mu_{\text{base}} + \delta_{\text{month}}[t_{\text{month}}] + \delta_{\text{league}}[t_{\text{league}}] + \gamma_{\text{home}} + \alpha_h - \beta_a + w_{\text{wealth}} \cdot \Delta W_m$$
$$\log \mu_{\text{open}, a} = \mu_{\text{base}} + \delta_{\text{month}}[t_{\text{month}}] + \delta_{\text{league}}[t_{\text{league}}] + \alpha_a - \beta_h - w_{\text{wealth}} \cdot \Delta W_m$$
where:
- $\mu_{\text{base}} \sim \mathcal{N}(0.2, 0.2)$: Global open-play scoring intercept.
- $\delta_{\text{month}} \sim \mathcal{N}(0, \sigma_{\text{month}})$: Monthly seasonal weather/pitch effect (sum-to-zero).
- $\delta_{\text{league}} \sim \mathcal{N}(0, 0.2)$: Inter-division tempo differential (sum-to-zero).
- $\gamma_{\text{home}} \sim \mathcal{N}(0.2, 0.2)$: Structural home advantage.
- $\alpha_i, \beta_i \sim \mathcal{N}(0, \tau_{\alpha/\beta})$: Team open-play attacking and defensive ratings (zero-centered).
- $w_{\text{wealth}} \sim \text{TruncatedNormal}(0.10, 0.05, a=0.0)$: Sensitivity to Starting-XI market valuation differential $\Delta W_m = W_{\text{home}, m} - W_{\text{away}, m}$.

### 2.2 Multi-Task Likelihoods
1. **Continuous Proxy xG Likelihood ($\text{pxG}$):**
   $$\text{pxG}_{\text{open}, h} \sim \text{Gamma}\left(\nu_{\text{xg}},\, \frac{\mu_{\text{open}, h}}{\nu_{\text{xg}}}\right)$$
   $$\nu_{\text{xg}} \sim \text{TruncatedNormal}(3.5, 0.5, a=0.5)$$
2. **Realized Open-Play Goals Likelihood:**
   $$Y_{\text{open}, h} \sim \text{Poisson}(\kappa_h \cdot \mu_{\text{open}, h})$$
   $$\log \kappa_i \sim \mathcal{N}(0, 0.10) \quad (\text{Team Finishing Efficiency})$$
3. **Penalty Awards Likelihood:**
   $$N_{\text{pen}, h} \sim \text{Poisson}(\lambda_{\text{pen}, h})$$
   $$\log \lambda_{\text{pen}, h} = \mu_{\text{pen}} + \gamma_{\text{ha, pen}} + \theta_{\text{ref}}[r_m]$$
   $$\theta_{\text{ref}} \sim \mathcal{N}(0, \sigma_{\text{ref}}), \quad \sigma_{\text{ref}} \sim \text{TruncatedNormal}(0.10, 0.05, a=0.01)$$

### 2.3 Out-of-Sample Score Matrix Recombination
Because the sum of independent Poisson random variables is Poisson, total match goals follow:
$$Y_{\text{total}, h} \sim \text{Poisson}(\mu_{\text{total}, h}), \quad Y_{\text{total}, a} \sim \text{Poisson}(\mu_{\text{total}, a})$$
$$P(H=g_h, A=g_a) = \text{Poisson}(g_h \mid \mu_{\text{total}, h}) \cdot \text{Poisson}(g_a \mid \mu_{\text{total}, a}) \cdot \tau_{\text{DC}}(g_h, g_a; \rho)$$

---

## ⚡ 3. ReverseDiff Automatic Differentiation (AD) Safety Rules

To achieve sub-2ms gradient evaluations on `mcmc-beast` with ReverseDiff tape compilation:

1. **Zero-Allocation Binary Masking:**
   - Never use conditional branching (`if isfinite(pxg) ...`) inside the Turing `@model`.
   - In `OpenPlayPxGFeature`, missing proxy xG values are imputed with a constant dummy `1.0` and paired with a binary mask `mask_pxg_h ∈ {0.0, 1.0}`:
     ```julia
     ll_pxg_h = logpdf.(Gamma.(ν_xg, scale_pxg_h), pxg_open_h)
     Turing.@addlogprob! sum(ll_pxg_h .* mask_pxg_h .* match_weights)
     ```
   - This maintains a 100% static execution graph with 0 heap allocations during AD gradient sweeps.

2. **Rate Clamping:**
   - Always clamp log-rates (`clamp.(log_mu, -5.0, 4.0)`) before exponentiation to prevent gradient overflow/underflow during warm-up sampling.

3. **Thread Pinning:**
   - Always execute MCMC sampling with `using ThreadPinning; pinthreads(:cores)` to lock worker threads to physical CPU cores, preventing L1/L2 cache migration and CPU thrashing.

---

## 📊 4. Empirical Historical Benchmarks (Scottish Lower, 710 Matches, Betfair 2% Comm)

| Model Architecture | Final Wealth | ROI (%) | Kelly Sharpe | Max Drawdown | CRPS | 1X2 LogLoss Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`recomb_pxg_wealth_integrated` (Champion)** | **3.147x** | **+11.51%** | **1.17 (Rank #1)** | **-32.22% (Best)** | **0.6360 (Best)** | **0.01050 (Best)** |
| `recomb_pois_wealth_integrated` | 3.180x | +11.78% | 1.14 | -33.87% | 0.6368 | 0.01203 |
| `recomb_pois_integrated` (Base Recomb) | 3.004x | +11.47% | 1.08 | -37.75% | 0.6372 | 0.01080 |
| `goals_pois_ctl` (Gross Goals Control) | 2.862x | +11.06% | 1.05 | -38.91% | 0.6380 | 0.01093 |
| `goals_pois_open_play` (Unrecombined Open Play) | 2.512x | +9.03% | 1.01 | -33.86% | 0.6420 | 0.01137 |
| `goals_negbin_ctl` (Gross NegBin Control) | 1.924x | +7.54% | 0.83 | -33.58% | 0.6295 | 0.00343 |

---

## 🛠️ 5. Module Map & API Quick Reference

### Core Structs (`src/models/pregame/`)
- `DynamicPxGRecombModel`: Production multi-task pxG + Goals + Officiating + Wealth engine.
- `DynamicRecombinedGoalsModel`: Open-play goals decomposition + Officiating + Wealth engine.
- `EmpiricalRecombinationConfig`, `HierarchicalOfficiatingConfig`: Officiating / penalty priors.
- `LinearSquadWealthConfig`, `NoSquadWealthConfig`: Squad wealth sensitivity priors.
- `GammaPxGObservationConfig`, `NoPxGObservationConfig`: Proxy xG precision priors.

### Core Features (`src/features/`)
- `OpenPlayGoalsFeature`: Extracts non-penalty, non-own-goal match scores from `ds.incidents`.
- `OpenPlayPxGFeature`: Extracts zonal open-play shot xG excluding penalty attempts.
- `SquadWealthFeature`: Computes standardized starting-XI valuation differential $\Delta W$.
- `RefereeOfficiatingFeature`: Tracks match referee assignments.

### Prediction Kernel (`src/predictions/`)
- `Predictions.compute_score_matrix(model::DynamicPxGRecombModel, params; max_goals=12)`: Executes discrete Poisson convolution returning a normalized `ScoreMatrix{Float64}` satisfying $\sum M = 1.000000$.

### Unit Tests (`test/`)
- `test/recombination_tests.jl`: 83 comprehensive assertions covering configurations, feature extraction, Turing model generation, and score matrix invariants.
