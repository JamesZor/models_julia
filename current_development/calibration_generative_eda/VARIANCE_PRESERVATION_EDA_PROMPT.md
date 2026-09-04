# Work Package: Posterior Variance Preservation & Dispersion Transforms in Generative Rate Calibration

### 1. Motivation & Context
In Phase 3 (`current_development/calibration_generative_eda/`), generative rate calibration was verified at tradeable T−25 pre-match prices on Scottish Lower (40 walk-forward folds, 710 matches, 2024/25–2025/26). The standard calibrator (`std_w0.40_s0.15` in `l02_point_in_time_book.jl` and `r03_t25_book_and_calibration.jl`) achieved:
- **ECE halving**: 0.0093 vs market 0.0183 (and raw model 0.0151).
- **Strong Positive CLV**: +1.02% stake-weighted CLV, 52.0% positive hit rate.
- **Over 2.5 Rescued**: +21.46u PnL, +13.39% Kelly ROI (50% win rate at 2.10 avg odds), fixing the historical leak.
- **Drawdown Compressed**: Max Drawdown compressed from -16.15% (raw) down to -11.36% (and -7.76% under inverse).

However, at default risk scaling (`SlateDrawdown(23.0)`), the calibrated model compound return appeared lower (+72.85% vs +151.52% raw) because log-linear pooling $\log \mu_{\text{cal}} = w \log \mu + (1-w) \log \lambda_{\text{mkt}}$ contracts posterior log-variance by $w^2$ (~70.6% variance retained for standard, ~8.5% for inverse), causing Fractional Kelly to stake smaller amounts.

### 2. The Core Debate & Freedom to Experiment
You proposed building a **"variance-preserving pool"** so that posterior location shifts toward the market without the posterior variance contracting by $w^2$, preventing Kelly stake shrinking.

However, existing research in this repository (`eda/README.md`) raises important counter-evidence:
1. **The Jensen's Inequality Tail Distortion** (`eda/README.md` Discovery 2): Under Poisson intensity with posterior uncertainty, the predictive zero-goal mass is a mixture $\mathbb{E}[e^{-\Lambda}] \ge e^{-\mathbb{E}[\Lambda]}$. High posterior variance in $\Lambda$ mechanically inflates goalless draws and deep totals (Under 0.5, correct scores), manufacturing artificial Kelly edges on longshot noise.
2. **The Scale-Invariance Law** (`eda/README.md` Discovery 4): Under `SlateDrawdown(23.0)`, the drawdown constraint absorbs uniform stake reductions. Shorter stakes can simply be scaled up via the risk budget ($\lambda_{\text{risk}}$ or Fractional Kelly multiplier) to exploit the huge drawdown headroom (-11.4% vs -20.5% budget), which already proved to yield +134% to +202% returns without distorting posterior distributions.

**You have full freedom to experiment with your own ideas regarding this problem.** Whether you want to:
- Test a mean-shifted variance-preserved pool.
- Model the market as a noisy likelihood observation with explicit precision $\tau_{\text{mkt}}$.
- Develop an asymmetric or heteroskedastic dispersion transform (e.g. preserving variance on supremacy while shrinking totals variance, or vice versa).
- Or empirically demonstrate that pure risk-budget scaling ($\lambda_{\text{risk}} \in [10.0, 23.0]$, Fractional Kelly $0.30 \to 0.45$) strictly dominates any variance tampering on Sharpe, Calmar, and compounding efficiency.

### 3. Experimental Hypotheses
* **H1 (Variance Preservation)**: Shifting location while preserving raw posterior variance recovers Kelly sizing and compounding without degrading proper scores (LogLoss, ECE, Brier).
* **H2 (Jensen Tail Inflation Risk)**: Artificially maintaining raw variance when shifting location toward the market causes Jensen tail distortion ($\mathbb{E}[e^{-\Lambda}] \ge e^{-\mathbb{E}[\Lambda]}$), worsening O/U proper scores and baiting Kelly into over-betting longshot noise.
* **H3 (Scale-Invariance Domination)**: Leaving the natural $w^2$ contraction intact and instead scaling the portfolio risk budget ($\lambda_{\text{risk}}$ or Fractional Kelly multiplier) strictly dominates variance preservation in Sharpe, Calmar, and real compounding.
* **H4 (Agent Novel Scheme)**: Any innovative dispersion scheme you design (e.g. asymmetric dispersion, supremacy-only preservation, or Bayesian likelihood updating).

### 4. Implementation & Execution Protocol
All compute must run on remote node `mcmc-beast` via the persistent warm Julia REPL in tmux session `r68:0.0` (16 threads).
- **Zero writes to `betdb.paper_runbook`**.
- Do not touch MatchDay live console (8085) or replay console (8086).
- Prototype in `current_development/calibration_generative_eda/`:
  - `l03_variance_schemes.jl`: Implement the candidate dispersion transforms.
  - `r05_variance_experiments.jl`: Execute the diagnostic and portfolio sweeps.

### 5. Candidate Dispersion Schemes to Benchmark
1. **Scheme A (Baseline Coherent)**: Standard log-linear pool ($\operatorname{Var} = w^2 \sigma^2$).
2. **Scheme B (Full Mean-Shift Variance Preservation)**:
   $$\log \tilde{\mu}^{(s)} = \overline{\log \mu_{\text{pool}}} + \left(\log \mu^{(s)} - \overline{\log \mu}\right)$$
   (Preserving 100% of raw model log-variance around the calibrated mean rate).
3. **Scheme C (Intermediate Precision Shrinkage)**:
   Scale centered residuals by $\sqrt{w}$ such that $\operatorname{Var} = w \sigma^2$ (as if the market was an independent observation of equal precision).
4. **Scheme D (Your Own Novel Idea)**:
   Your proposed formulation (e.g. heteroskedastic dispersion, supremacy vs totals split, or likelihood de-biasing).
5. **Scheme E (Scale-Invariance / Risk Headroom Control)**:
   Scheme A with scaled drawdown budget ($\lambda_{\text{risk}} \in [10.0, 23.0]$) and Fractional Kelly $0.40$ to exploit the compressed drawdown headroom.

### 6. Verification Gates & Required Artefacts
- **Gate 1 (Proper Scores & Tail Diagnostics)**:
  - Compute LogLoss, ECE, Brier score across `m12_joint_hybrid_synergy` and `m05_joint_production_wealth` against T−25 tradeable prices.
  - Audit tail probabilities: Under 0.5, Under 1.5, and Over 3.5 to explicitly check for Jensen inflation.
- **Gate 2 (Portfolio & Directional Audit)**:
  - Run 11-direction portfolio simulation at T−25 tradeable prices using Canonical Scottish Lower Trust (`P1_conservative_tilt`).
  - Audit Over 2.5 performance across all schemes.
  - Compute Closing Line Value (CLV, flat and stake-weighted).
- **Persistence & Documentation**:
  - Save CSVs to `current_development/calibration_generative_eda/results/`:
    - `r05_variance_scores.csv`
    - `r05_variance_portfolio_summary.csv`
    - `r05_variance_direction_ledger.csv`
    - `r05_variance_clv.csv`
  - Record findings, tables, and conclusions in `current_development/calibration_generative_eda/README.md` under Section 8. State clearly whether H1, H2, H3, or H4 is supported.
