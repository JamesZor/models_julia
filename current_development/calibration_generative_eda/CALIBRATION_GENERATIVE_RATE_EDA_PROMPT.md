# Work Package Specification: Generative Rate Calibration EDA & Layer-2 Overhaul

**Target Location:** `current_development/calibration_generative_eda/` (Prototype) ➔ `src/Calibration/` (Graduation)  
**Assigned Subagent:** Claude Code (Opus) via tmux session `claude_calibration_eda`  
**Supervising Coordinator:** Antigravity (AGY)  
**Target League Data:** Scottish Lower (Tournaments 56 & 57, Seasons 24/25 + 25/26, 40-fold walk-forward, 710 matches, 2,899 market quotes)  
**Target MCMC Runs in `mcmc_experiments`:** `m12_joint_hybrid_synergy` (hybrid lineup RAPM + wealth) and `m05_joint_production_wealth` (team-level control)  
**Date:** September 2026  

---

## 1. Executive Context & Architectural Problem

### 1.1 The Incoherence of Current `src/Calibration`
The current `src/Calibration` module implements univariate, selection-level probability modifications (e.g. `BasicLogitShift` in `shift_models/basic_logit.jl`). While simple, shifting probabilities post-PPD breaks fundamental mathematical axioms of sports pricing:
1. **Internal Arbitrage / Loss of Coherence:** Shifting $P(\text{Home})$, $P(\text{Draw})$, $P(\text{Away})$ and $P(\text{Over 2.5})$ independently produces a probability board that no longer sums to a valid bivariate scoreline matrix $\mathbf{S}$. Derivative markets become internally contradictory (e.g., $P(\text{Over 2.5}) + P(\text{Under 2.5}) \ne 1.0$, or $P(\text{0-0}) > P(\text{Under 0.5})$).
2. **Post-Allocator Trust Distortion:** Using scalar trust multipliers (e.g., `FlatTrust(0.25)`) inside `src/Portfolio` fails when the Boyd-Busseti drawdown budget (`SlateDrawdown`) binds, because the drawdown constraint is homogeneous of degree 0 in stakes.

### 1.2 The Generative Discovery (`current_development/orderbook_layer2/research_questions_explore/`)
In `notes_rqs_01.md` and `rqs_001_multi_class_softmax_pooling.jl`, a mathematically coherent approach was discovered: **Log-Linear Pooling at the Generative Intensity Level ($\lambda_h, \lambda_a$)**:
1. Invert the closing market odds back into generative rates $(\lambda_{\text{mkt\_h}}, \lambda_{\text{mkt\_a}})$ via Nelder-Mead on `DoublePoissonMarketFeature`.
2. Apply a log-linear pool (weighted geometric mean) directly to all posterior MCMC draws of $\lambda$.
3. Compute the full $12 \times 12$ bivariate score tensor from the shifted draws, guaranteeing 100% derivative market coherence.

### 1.3 The Generalization Dilemma (Ireland vs Scottish Lower)
* **Ireland Premier (314 matches):** An **Inverse Dynamic Gaussian** weight with parameters $(w_{\text{base}}=0.25, \sigma=0.25)$ produced massive out-of-sample LPD gains (+145 bps on extreme edges) and +44% Sortino ratio. The inverse weighting philosophy: trust the market on small noise edges ($\Delta < 2\%$), but retain high model conviction on large structural disagreements ($\Delta > 5\%$).
* **Scottish Lower Out-of-Domain Test:** The *exact same* $(0.25, 0.25)$ setting **degraded** return, ROI, and Sharpe, and increased max drawdown. Why? Setting $w_{\text{base}} = 0.25$ crushed small genuine model edges below the 2% Betfair commission threshold.
* **Core Research Question:** Is Generative Rate Calibration fundamentally viable for Scottish Lower when properly swept? Does an Inverse Gaussian, a Standard Gaussian (optimizer's curse shrinkage), or a Static Geometric Pool dominate? And crucially: **does rate calibration alter the multi-tier trust vector law (`CanonicalScottishLowerTrust`), or can it rescue previously discarded markets (like Over 2.5)?**

---

## 2. Mathematical Formulation

### Step 1: Market Rate Inversion via Nelder-Mead
For each target match, extract the closing de-vigged fair probabilities across available liquid lines (1X2, Over/Under 1.5, 2.5, 3.5). Solve for the bivariate Poisson rates $\theta^* = (\log \lambda_{\text{mkt\_h}}, \log \lambda_{\text{mkt\_a}})$ minimizing sum of squared errors:
$$\min_{\theta} \sum_{k \in \text{Markets}} \left( P_{\text{implied}}(k; \theta) - P_{\text{fair\_close}}(k) \right)^2$$
$$\lambda_{\text{mkt\_h}} = \exp(\theta_1^*), \quad \lambda_{\text{mkt\_a}} = \exp(\theta_2^*)$$

### Step 2: Discrepancy Metric & Weight Formulations
Compute scale-invariant log-rate discrepancies using the central tendency (median) of the MCMC posterior:
$$\Delta_h = \log\left(\text{median}(\lambda_{\text{model\_h}})\right) - \log \lambda_{\text{mkt\_h}}$$
$$\Delta_a = \log\left(\text{median}(\lambda_{\text{model\_a}})\right) - \log \lambda_{\text{mkt\_a}}$$

Evaluate **three competing functional forms** for weight $w(\Delta) \in [0, 1]$:

1. **Inverse Dynamic Gaussian (Aggressive on Disagreement, Blends on Noise):**
   $$w_{\text{inv}}(\Delta) = w_{\text{base}} + (1.0 - w_{\text{base}}) \left( 1.0 - \exp\left( -\frac{\Delta^2}{2\sigma^2} \right) \right)$$
   * Behavior: At $\Delta = 0$, $w = w_{\text{base}}$. As $|\Delta| \to \infty$, $w \to 1.0$ (full model conviction on major edges).

2. **Standard Dynamic Gaussian (Conservative / Optimizer's Curse Shrinkage):**
   $$w_{\text{std}}(\Delta) = w_{\text{base}} + (w_{\text{max}} - w_{\text{base}}) \exp\left( -\frac{\Delta^2}{2\sigma^2} \right)$$
   * Behavior: At $\Delta = 0$, $w = w_{\text{max}}$. As $|\Delta| \to \infty$, $w \to w_{\text{base}}$ (shrinks extreme claims back towards market consensus).

3. **Static Geometric Pool (Constant Weight Baseline):**
   $$w_{\text{static}}(\Delta) = w_{\text{const}}, \quad \forall \Delta$$

### Step 3: Posterior Geometric Mean Shift (All Posterior Draws)
For each posterior draw $d \in \{1, \dots, D\}$ (where $D = 1,000 \times 4 = 4,000$ draws):
$$\log \lambda_{\text{shifted\_h}}^{(d)} = w_h \log \lambda_{\text{model\_h}}^{(d)} + (1 - w_h) \log \lambda_{\text{mkt\_h}}$$
$$\log \lambda_{\text{shifted\_a}}^{(d)} = w_a \log \lambda_{\text{model\_a}}^{(d)} + (1 - w_a) \log \lambda_{\text{mkt\_a}}$$

### Step 4: Coherent Score Tensor & Derivative Market Pricing
Construct the shifted $12 \times 12 \times D$ score matrix $\mathbf{S}$:
$$\mathbf{S}_{i+1, j+1, d} = \text{Pois}(i; \lambda_{\text{shifted\_h}}^{(d)}) \cdot \text{Pois}(j; \lambda_{\text{shifted\_a}}^{(d)}) \cdot \tau(i, j, \rho)$$
Derive all market probabilities across all draws:
* $P(\text{Home}) = \sum_{i > j} \mathbf{S}_{i+1, j+1}$, $P(\text{Draw}) = \sum_{i = j} \mathbf{S}_{i+1, j+1}$, $P(\text{Away}) = \sum_{i < j} \mathbf{S}_{i+1, j+1}$
* $P(\text{Over } K.5) = \sum_{i+j > K} \mathbf{S}_{i+1, j+1}$, $P(\text{Under } K.5) = 1.0 - P(\text{Over } K.5)$
* $P(\text{BTTS Yes}) = \sum_{i \ge 1, j \ge 1} \mathbf{S}_{i+1, j+1}$

---

## 3. Work Package Execution Plan

### Phase 1: Prototype Implementation in `current_development/calibration_generative_eda/`
Create the prototype suite following repository conventions (`lXX_*.jl` for loaders, `rXX_*.jl` for execution runners):

1. **Loader: `l01_generative_calibrator.jl`**:
   - Structs: `GenerativeCalibrationSpec` parameterizing functional form (`:inverse_gaussian`, `:standard_gaussian`, `:static_geometric`), $w_{\text{base}}$, $\sigma$, $w_{\text{max}}$.
   - Market Inversion: Robust Nelder-Mead extraction on Betfair closing odds with fallback checks.
   - Draw Shifter: Zero-alloc / vectorized log-linear posterior draw transformation.
   - Coherent Pricing: In-place score matrix regeneration producing calibrated `OddsView` / prediction frames.

2. **Runner 1: `r01_sweep_rate_calibration.jl` (Diagnostic & Proper Score Sweep)**:
   - Load out-of-sample predictions from `mcmc_experiments` across all 40 folds of Scottish Lower (Seasons 24/25 & 25/26, 710 matches):
     - Primary: `m12_joint_hybrid_synergy`
     - Baseline: `m05_joint_production_wealth`
   - Execute 2D hyperparameter surface grid:
     - $w_{\text{base}} \in [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]$
     - $\sigma \in [0.15, 0.25, 0.35, 0.50, 0.75, 1.00]$
     - Across all 3 functional forms (Inverse, Standard, Static).
   - Evaluate proper scoring rules:
     - Multi-class LogLoss (1X2)
     - Expected Calibration Error (ECE)
     - CRPS and Brier score across totals (0.5, 1.5, 2.5, 3.5) and BTTS
     - Edge Stratified LPD: Small edges ($|\Delta p| < 0.02$) vs Large edges ($|\Delta p| > 0.05$).
   - Identify the optimal calibration parameter pair $(w^*, \sigma^*)$ per functional form.

3. **Runner 2: `r02_portfolio_direction_audit.jl` (13-Market Direction & Trust Vector Audit)**:
   - Feed the uncalibrated vs calibrated predictions into `src/Portfolio` simulation (`simulate_portfolio` / `run_portfolio_simulation`).
   - Audit all 13 betting directions independently:
     - 1X2: Home, Draw, Away
     - Totals: Over 0.5, Under 0.5, Over 1.5, Under 1.5, Over 2.5, Under 2.5, Over 3.5, Under 3.5
     - BTTS: Yes, No
   - Key test questions:
     - Does rate calibration rescue **Over 2.5** (historically negative ROI due to Jensen's inequality and Poisson right-tail over-dispersion)?
     - Does rate calibration eliminate the need for `CanonicalScottishLowerTrust` (which sets Over 2.5 to 0.0 and weights Away/Draw lower than Home)?
     - Compare:
       1. Raw `m12` + `CanonicalScottishLowerTrust` (Current champion: +136.6%, Sharpe 1.416, MaxDD -20.2%)
       2. Calibrated `m12` + Uniform Trust (`FlatTrust(1.0)` across all markets)
       3. Calibrated `m12` + `CanonicalScottishLowerTrust`
       4. Calibrated `m12` + newly re-optimized trust vector.

---

## 4. Acceptance Gates & Verification Ladder

Before any code is ported to `src/Calibration/`, the following gates must be met:

### Gate 1: Proper Scoring & Statistical Gating
* **LogLoss:** Calibrated `m12` must match or improve uncalibrated LogLoss ($\le 0.64337$).
* **ECE (Expected Calibration Error):** Must beat the Betfair closing line benchmark ($0.0139$) and match/improve uncalibrated `m12` ($0.0100$).
* **Stratified Edge LPD:** Must verify whether large edges ($>5\%$) benefit from model conviction (Inverse) or shrinkage (Standard).

### Gate 2: Portfolio Performance & Drawdown Preservation
* **Bankroll Growth:** Must achieve $> +130\%$ bankroll growth on the 40-fold Scottish Lower test slate.
* **Risk-Adjusted Return:** Annual Sharpe ratio must match or beat uncalibrated benchmark ($> 1.416$).
* **Max Drawdown:** Must NOT degrade maximum drawdown beyond $-20.5\%$ (current uncalibrated: $-20.2\%$).

---

## 5. Graduation Architecture (`src/Calibration/` Redesign)

Upon passing the acceptance gates:
1. Deprecate selection-level `BasicLogitShift` as an anti-pattern.
2. Introduce `GenerativeRateCalibrator <: AbstractLayerTwoModel` into `src/Calibration/types.jl`:
   ```julia
   struct GenerativeRateCalibrator <: AbstractLayerTwoModel
       method::Symbol # :inverse_gaussian, :standard_gaussian, :static_geometric
       w_base::Float64
       sigma::Float64
       w_max::Float64
   end
   ```
3. Implement `apply_calibration` to operate on `CountLatents` or MCMC draw tensors, outputting coherent calibrated prediction structures.
4. Export updated interfaces from `src/Calibration/calibration-module.jl`.
5. Add rigorous unit tests in `test/test_calibration.jl` asserting derivative market coherence ($\sum P(1X2) = 1.0$, $P(\text{Over}) + P(\text{Under}) = 1.0$).

---

## 6. Execution Rules & Database Boundaries

* **Hardware & Runtime:** Run on `mcmc-beast` via persistent Julia REPL or scripts with `pinthreads(:cores)` and `BLAS.set_num_threads(1)`.
* **Database Isolation:**
  - Read posterior draws and runs from `BF_EXPERIMENTS_DB_URL` (`mcmc_experiments`).
  - Read match odds and results from `BF_DB_URL` (`betdb`).
  - **STRICT PROHIBITION:** Do NOT connect to or write to `betdb.paper_runbook`. Live MatchDay console on port 8085 must not be touched or disturbed.
* **Documentation Integrity:** All findings, parameter sweeps, and portfolio returns must be recorded with exact numbers, tables, and SQL run IDs in `current_development/calibration_generative_eda/README.md`.
