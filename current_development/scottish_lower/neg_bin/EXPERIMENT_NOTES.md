# Scottish Lower Leagues: Robust Negative Binomial Goals Investigation

## 1. Executive Summary & Problem Formulation

### Motivation & Empirical Reality
In lower leagues (Scottish League One #56 and League Two #57), match dynamics exhibit higher outcome volatility, defensive collapses, and blowout margins than top-flight divisions. Under a standard Poisson likelihood, the variance of goals is strictly constrained to equal the expected rate:
$$\text{Var}(G) = \mathbb{E}[G]$$

When goals are **overdispersed** ($\text{Var}(G) > \mathbb{E}[G]$), the standard Poisson model suffers from two structural pathologies:
1. **Underestimating Clean Sheets ($G = 0$):** In empirical Scottish lower data, away teams fail to score in $29.90\%$ of matches, but a Poisson model with $\mu = 1.288$ predicts only $27.58\%$ ($2.32\%$ deficit).
2. **Underestimating High-Scoring Blowouts ($G \ge 4$):** Poisson fails to capture fat right tails, underestimating multi-goal routs and inflating under-market confidence.

### The Solution: Robust Negative Binomial (NB2) Parameterization
We decouple the expected scoring intensity $\mu$ from the outcome variance using the **Robust Negative Binomial** formulation:
$$\mathbb{E}[G] = \mu, \quad \text{Var}(G) = \mu + \frac{\mu^2}{r}$$
where $r > 0$ is the dispersion (shape) parameter. As $r \to \infty$, the distribution smoothly recovers the Poisson distribution.

---

## 2. Statistical Exploratory Data Analysis & Overdispersion Tests

Results computed over the complete Scottish Lower historical match dataset ($N = 1,990$ matches across Scottish League One and League Two):

### A. Summary Moments & Dispersion Index ($\text{VMR} = \sigma^2 / \mu$)

| Market Pillar | Sample Size | Mean ($\mu$) | Variance ($\sigma^2$) | Dispersion Index ($\text{VMR}$) | Statistical Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Home Goals** | $1,990$ | $1.4633$ | $1.5580$ | **`1.0647`** | **Overdispersed ($+6.5\%$ excess variance)** |
| **Away Goals** | $1,990$ | $1.2879$ | $1.4641$ | **`1.1367`** | **Strongly Overdispersed ($+13.7\%$ excess variance)** |
| **Total Goals** | $1,990$ | $2.7513$ | $2.6133$ | $0.9499$ | Slight negative correlation |

#### Breakdown by Division:
- **Scottish League One (#56, N=995):**
  - Home Goals: $\mu = 1.486, \sigma^2 = 1.634 \implies \mathbf{\text{VMR} = 1.100}$
  - Away Goals: $\mu = 1.324, \sigma^2 = 1.497 \implies \mathbf{\text{VMR} = 1.131}$
- **Scottish League Two (#57, N=995):**
  - Home Goals: $\mu = 1.440, \sigma^2 = 1.482 \implies \mathbf{\text{VMR} = 1.029}$
  - Away Goals: $\mu = 1.252, \sigma^2 = 1.430 \implies \mathbf{\text{VMR} = 1.142}$

---

### B. Formal Hypothesis Tests

1. **Dean-Lawless (1989) Lagrange Multiplier Score Test:**
   - **Home Goals:** $T = 2.045$, $p = 0.0204$ $\implies$ **Reject $H_0$ Poisson ($p < 0.05$)**
   - **Away Goals:** $T = 4.312$, $p = 8.08 \times 10^{-6}$ $\implies$ **Reject $H_0$ Poisson ($p < 0.0001$)**
2. **Cameron-Trivedi (1990) Auxiliary Regression Test:**
   - **Home Goals:** $\hat{\alpha} = +0.0442$ ($t = 2.05, p = 0.020$)
   - **Away Goals:** $\hat{\alpha} = +0.1062$ ($t = 4.31, p < 0.0001$)

---

### C. Empirical vs Theoretical Probability Frequency (Clean Sheets vs Blowouts)

#### Away Goals Calibration:
| Goals ($k$) | Empirical % | Poisson (%) | Robust NegBin (%) | NegBin vs Poisson Lift |
| :---: | :---: | :---: | :---: | :---: |
| **0 (Clean Sheet)** | **`29.90%`** | $27.58\%$ | **`29.90%`** 🎯 | **$+2.32\%$ (Zero error)** |
| **1** | $34.32\%$ | $35.53\%$ | $33.88\%$ | $-1.64\%$ |
| **2** | $20.45\%$ | $22.88\%$ | $21.23\%$ | $-1.65\%$ |
| **3** | $9.75\%$ | $9.82\%$ | $9.72\%$ | $-0.10\%$ |
| **4** | $4.02\%$ | $3.16\%$ | $3.63\%$ | $+0.47\%$ |
| **5** | $1.16\%$ | $0.81\%$ | $1.17\%$ | $+0.36\%$ |
| **6+** | $0.40\%$ | $0.21\%$ | $0.46\%$ | $+0.24\%$ |

---

## 3. Codebase Architectural Review

### Existing Negative Binomial Infrastructure
1. **Distribution (`src/MyDistributions/negative_binomial.jl`):**
   - `RobustNegativeBinomial(r, μ)`:
     $$\log p(k \mid r, \mu) = \log\Gamma(k + r) - \log\Gamma(k + 1) - \log\Gamma(r) + r (\log r - \log(r+\mu)) + k (\log \mu - \log(r+\mu))$$
   - Numerically stable under AD, avoiding $p = r/(r+\mu)$ boundaries.
2. **Dispersion Component (`src/models/pregame/components/dispersion.jl`):**
   - `HomeAwayDispersion()`: Samples `disp.log_r ~ Normal(3.1, 0.4)` and `disp.δ_r_home ~ Normal(0.0, 0.5)`.
   - Generates $r_h = \exp(\text{clamp}(\log r + \delta_r, -10, 10))$ and $r_a = \exp(\text{clamp}(\log r, -10, 10))$.
3. **Turing AD Performance Guide Compliance (`docs/turing_ad_performance_guide.md`):**
   - Must use vectorised broadcasting: `logpdf.(RobustNegativeBinomial.(r_h, λ_goals_h), home_goals)`.
   - Precompute sufficient statistics and constants outside Turing `@model`.
   - Binary masks for unobserved/missing xG or time-decay splits.

---

## 4. Phase-by-Phase Implementation Plan

```
current_development/scottish_lower/
├── neg_bin/
│   ├── l01_negbin_engines.jl            <- NegBin Model Structs, Turing @models, Extractors, Preds
│   ├── r00_eda_overdispersion.jl        <- Statistical Dispersion & Score Tests
│   ├── r01_smoke_negbin.jl              <- 1-Split Fast Smoke Test (NUTS, R-hat, speed check)
│   ├── r02_grid_negbin.jl               <- 40-Fold MCMC Grid on mcmc-beast (16 pinned threads)
│   ├── r03_eval_negbin.jl               <- LogLoss, RQR, GLMEdge, Betfair Portfolio Backtest
│   └── EXPERIMENT_NOTES.md              <- Full Experiment Log & Comparative Analysis
├── portfolio/                           <- Betfair & Bet365 Kelly Portfolio backtests
├── proxy_xg/                            <- Proxy xG shot regression & RAPM engines
└── wealth/                              <- Transfermarkt starting-XI valuation models
```

### Planned Model Variants to Compare:
1. **`goals_negbin_ctl_hl365_hs2`** vs **`funnel_apm_ctl_hl365_hs2`** (Baseline Goals-only control).
2. **`pxg_apm_negbin_hl365_hs2`** vs **`pxg_apm_hl365_hs2`** (Arm A: Proxy xG Gamma + RAPM + Goals NegBin).
3. **`funnel_pxg_apm_negbin_hl365_hs2`** vs **`funnel_pxg_apm_hl365_hs2`** (Arm B: Shots Volume Poisson + Proxy xG Quality Gamma + RAPM + Goals NegBin).
