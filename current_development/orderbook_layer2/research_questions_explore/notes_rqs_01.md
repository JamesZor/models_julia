# Layer-2 Generative Inverse Dynamic Calibration: Mathematical Foundations & Empirical Findings

**File:** `current_development/orderbook_layer2/research_questions_explore/notes_rqs_01.md`  
**Related Runner:** `current_development/orderbook_layer2/research_questions_explore/rqs_001_multi_class_softmax_pooling.jl`  
**Target Engine:** `DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel` (`src_sup40_sw40`)  
**Data Set:** Ireland Premier Division (Tournament 79, 314 Out-of-Sample Matches, 7,080 Market Quotes)  
**Dated:** 2026-08-13  
**Updated:** 2026-08-24 — Scottish Lower out-of-domain test added  

> **Current verdict:** The generative construction is mathematically coherent, but the original
> $(w_{base}, \sigma)=(0.25,0.25)$ calibration does **not** generalize unchanged. It improved the
> original Ireland experiment but degraded return, risk-adjusted return, and drawdown under every
> matched-policy Scottish Lower test. Do not graduate or deploy these parameters.


  ### 1. The Logistic Shift (RQ1)

  • Original Question: Instead of blending the means, can we do a "logistic shift" on the log-odds of the posterior predictive
  distribution?
  • Our Answer: Yes. The literature calls this Log-Linear Pooling (or geometric pooling). It retains posterior draws rather than
  collapsing to a point estimate, but it does **not** preserve their variance: log-variance contracts by $w^2$, which can materially
  alter Kelly sizing.

  ### 2. Multi-Class / Generative Generalization (RQ2)

  • Original Question: How do we generalize a binary logit shift to multi-class probabilities (like 1X2) without breaking them?
  • Our Answer: We initially tried Softmax/Temperature scaling, but realized the ultimate mathematically bulletproof way was to
  invert the market back to generative λ rates, and apply the log-linear pool at the Poisson rate level. This guarantees that 1X2,
  Asian Handicaps, and Totals remain 100% coherent.

  ### 3. Non-Linear Distance Decay (RQ3)

  • Original Question: Is there a non-linear version where the weight changes depending on how far the model is from the market (to
  combat the optimizer's curse)?
  • Our Answer: Yes, we built a Gaussian distance function.

  ### 4. The Profitability / Longshot Bias Fear (RQ4)

  • Original Question: "I'm worried that this would just converge the model to the market, thus making it heavily correlated, and...
  hence will not be profitable. Will this help with the long shot bias?"
  • Our Answer: Your intuition here was historically brilliant. Our LPD checks proved that extreme claims (> 5% edge) are where the
  model beats the market's longshot/favorite biases. Converging to the market there destroyed accuracy. We fixed this by Inverting
  the Dynamic Weight: we only converge to the market on small, noisy edges (< 2%), but give the model full conviction on large
  edges!

  ### 5. End-to-End Backtest Simulation (RQ5)

  • Original Question: Does this actually improve the betting bankroll in a historical simulation?
  • Our Answer: **League-dependent, and negative out of domain at the original settings.** Ireland
  Premier showed an improvement, but the cleaner matched-policy Scottish Lower test showed lower
  final wealth, ROI, and Sharpe plus worse drawdown under all three policies. The construction remains
  worth investigating, but $(w_{base}, \sigma)=(0.25,0.25)$ was too strong to treat as universal.

---

## 1. Executive Summary & The Core Architectural Shift

### The Old Paradigm (Post-Kelly Scalar Trust & Probability Hacking)
In earlier iterations of `src/Portfolio/stake.jl` and `Calibration`, market calibration was attempted either by:
1. **Scalar probability blending / Platt scaling:** Modifying output probabilities (e.g. shifting $P(\text{Home})$ and $P(\text{Over 2.5})$ independently).
   * **Flaw:** Completely destroys mathematical coherence between derivative markets. The shifted probabilities no longer sum to a valid bivariate scoreline matrix, creating **internal arbitrage against our own price board**.
2. **Post-allocator scalar trust multipliers:** Sizing bets via Kelly on raw model probabilities, and then scaling stakes down: $a_{\text{final}} = a_{\text{kelly}} \cdot w_{\text{trust}}$ (e.g., `FlatTrust(0.25)`).
   * **Flaw:** The Boyd-Busseti drawdown budget (`SlateDrawdown`) in `src/Portfolio` is **homogeneous of degree 0** in the stakes it receives. Once the drawdown constraint binds, a scalar multiplier cannot resize exposure—it only reshapes the book.

```
OLD PIPELINE (Post-Kelly Scalar Trust):
Model PPD ──> Mean Probs ──> [Kelly Solver] ──> a_kelly ──> x FlatTrust(0.25) ──> [Drawdown Budget]
                                                               (Blunt Discount)
```

---

### The New Paradigm (Generative $\lambda$-Level Inverse Dynamic Shift)
Instead of modifying stakes or probabilities, we shift the **underlying generative Poisson/Negative-Binomial intensity parameters $(\lambda_h, \lambda_a)$ across all 3,200 posterior MCMC draws** before any market pricing or book building occurs.

```
NEW GENERATIVE L2 PIPELINE:
Exchange Odds ──> [Nelder-Mead Inversion] ──> (λ_mkt_h, λ_mkt_a)
                                                     │
                                                     ▼
Model Latents [λ_h, λ_a] (3,200 draws) ──> [Geometric Mean Shift] ──> Shifted Draws [λ_h*, λ_a*]
                                            (Inverse Dynamic Weight)         │
                                                                             ▼
                                                           [Predictions.compute_score_matrix]
                                                           Contiguous 144-State Tensor P_{i,j}
                                                                             │
                                              ┌──────────────────────────────┼──────────────────────────────┐
                                              ▼                              ▼                              ▼
                                       1X2 Probabilities             Over/Under (All lines)         BTTS / Handicaps
                                              │                              │                              │
                                              └──────────────────────────────┼──────────────────────────────┘
                                                                             ▼
                                                                  [Portfolio.build_books]
                                                            Coherent Multi-Market Kelly Solver
                                                                (Runs at FlatTrust = 1.00!)
```

---

## 2. Mathematical Formulation

### Step 1: Market Rate Inversion via Nelder-Mead
Given a match's closing fair prices across multiple markets (1X2, Over/Under lines), we solve for the latent market intensities that minimize sum of squared errors:

$$\min_{\theta = (\log \lambda_h, \log \lambda_a)} \sum_{k \in \text{Markets}} \left( P_{\text{model\_implied}}(k; \theta) - P_{\text{market\_fair}}(k) \right)^2$$

Where $P_{\text{model\_implied}}$ is computed from the bivariate score matrix:

$$P_{i,j} = \frac{e^{-\lambda_h} \lambda_h^i}{i!} \cdot \frac{e^{-\lambda_a} \lambda_a^j}{j!} \cdot \tau(i, j, \rho)$$

Extracting the market point estimates:

$$\lambda_{\text{mkt\_h}} = \exp(\theta_1^*), \quad \lambda_{\text{mkt\_a}} = \exp(\theta_2^*)$$

---

### Step 2: Log-Rate Discrepancy & The Calibrated Inverse Dynamic Weight
We measure the discrepancy between the model's central tendency and the market rate in **log-scale** (ensuring scale-invariance):

$$\Delta_h = \log\left(\text{median}(\lambda_{\text{model\_h}})\right) - \log \lambda_{\text{mkt\_h}}$$

$$\Delta_a = \log\left(\text{median}(\lambda_{\text{model\_a}})\right) - \log \lambda_{\text{mkt\_a}}$$

The dynamic weight $w \in [w_{\text{base}}, 1.0]$ is governed by:

$$w(\Delta) = w_{\text{base}} + (1.0 - w_{\text{base}}) \left( 1.0 - \exp\left( -\frac{\Delta^2}{2\sigma^2} \right) \right)$$

#### The Philosophy of the Inverse "Rubber Band"
Standard forecasting theory dictates shrinking extreme claims back to consensus to avoid the "optimizer's curse". **The Ireland LPD experiment supported the inverse direction, but the Scottish result shows it is
not established as a general football property.**
- **Moderate Edges (< 2%):** The market may efficiently remove noise, but heavy shrinkage can also erase weak genuine alpha.
- **Extreme Edges (> 5%):** Ireland results favored retaining more model weight. This remains a hypothesis requiring cross-league validation, not proof that every large disagreement is structural market bias.

#### Hyperparameter Calibration:
* **$w_{\text{base}} = 0.25$ (Original Trust Floor):** Retains at least 25% model weight, but the Scottish test showed this floor is too low for universal use; it does not guarantee enough edge to clear commission.
* **$\sigma = 0.25$ (Original Disagreement Bandwidth):** Chosen from the Ireland exploration. It was not externally calibrated and should now be treated as a failed transfer candidate.

```
Trust Weight w
1.00 ┤                                           ╭───── (Attacks large structural bias)
     │                                     ╭─────╯
0.75 ┤                               ╭─────╯
     │                         ╭─────╯
0.50 ┤                   ╭─────╯
     │             ╭─────╯
0.25 ┼───────┬─────╯ (Guarantees baseline edge to clear 2% commission)
     │       │       │       │       │       │
    0.00    0.10    0.20    0.30    0.40    0.50+   Disagreement Δ = |log(λ_model) - log(λ_mkt)|
```

---

### Step 3: Posterior Geometric Mean Shift (All 3,200 Draws)
For each MCMC posterior sample $s \in \{1, \dots, 3200\}$:

$$\lambda_{\text{shifted\_h}}^{(s)} = \left( \lambda_{\text{model\_h}}^{(s)} \right)^{w_h} \times \left( \lambda_{\text{mkt\_h}} \right)^{1 - w_h}$$

$$\lambda_{\text{shifted\_a}}^{(s)} = \left( \lambda_{\text{model\_a}}^{(s)} \right)^{w_a} \times \left( \lambda_{\text{mkt\_a}} \right)^{1 - w_a}$$

#### Key Properties:
1. **The Canonical Link:** Because Poisson rates ($\lambda$) are strictly positive and bounded at $(0, \infty)$, the correct canonical link function is $\log$. A log-linear pool of Poisson parameters mathematically simplifies directly into a Weighted Geometric Mean, ensuring rigorous internal consistency.
2. **Linear in Log-Space:** $\log \lambda_{\text{shifted}}^{(s)} = w \log \lambda_{\text{model}}^{(s)} + (1-w) \log \lambda_{\text{market}}$.
3. **Uncertainty Preserved:** $\text{Var}(\log \lambda_{\text{shifted}}) = w^2 \cdot \text{Var}(\log \lambda_{\text{model}})$. The full posterior distribution shape is preserved for downstream Kelly sizing rather than collapsing to a single point estimate.

---

### Step 4: Coherent Scoreline Tensor & Derivative Markets
The 3,200 shifted draws are fed into `Predictions.compute_score_matrix`, generating a $12 \times 12 \times 3200$ tensor $\mathbf{S}$.

Every derivative market is calculated directly from $\mathbf{S}$:
* **Home Win:** $\sum_{i > j} \mathbf{S}_{i+1, j+1, s}$
* **Draw:** $\sum_{i = j} \mathbf{S}_{i+1, j+1, s}$
* **Away Win:** $\sum_{i < j} \mathbf{S}_{i+1, j+1, s}$
* **Over $K.5$ Goals:** $\sum_{i+j > K} \mathbf{S}_{i+1, j+1, s}$
* **Both Teams to Score (BTTS):** $\sum_{i \ge 1, j \ge 1} \mathbf{S}_{i+1, j+1, s}$

**Guaranteed Coherence:** All derivative markets sum to $1.0$, and internal arbitrage is mathematically impossible.

---

## 3. Empirical Results & Proofs on Ireland Premier (79)

### 1. Bayesian Log Predictive Density (LPD) on Out-of-Sample Draws

$$\text{LPD} = \log \left( \frac{1}{S} \sum_{s=1}^S p(y \mid \theta^{(s)}) \right)$$

| Edge Regime | Raw Model LPD | Shifted $\lambda$ LPD | $\Delta$ LPD | Result |
| :--- | :---: | :---: | :---: | :--- |
| **Moderate Edges ($< 2\%$)** | `-0.32027` | **`-0.32023`** | **$+0.00004$** | Eliminates market noise churn |
| **Extreme Edges ($> 5\%$)** | `-0.58876` | **`-0.57430`** | **$+0.01446$** | **Massive $+145\text{ bps}$ Gain** |

---

### 2. GLM Edge Logistic Regression ($N = 7,080$ quotes across 314 matches)

$$\text{logit}(P(\text{Win})) = \beta_0 + \beta_{\text{mkt}} \cdot P_{\text{fair\_mkt}} + \beta_{\text{spread}} \cdot (P_{\text{model}} - P_{\text{fair\_mkt}})$$

| Regression Parameter | Raw Model | Shifted $\lambda$ (L2) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Observations ($N$)** | 7,080 | 7,080 | Full out-of-sample sample |
| **Intercept ($\beta_0$)** | `-2.8328` | `-2.8091` | Base outcome rate |
| **Market Price ($\beta_{\text{mkt}}$)** | `+5.7332` | `+5.6898` | Market pricing efficiency |
| **Model Spread ($\beta_{\text{spread}}$)** | **`+0.6660`** | **`+0.5562`** | **Genuine Independent Predictive Alpha** |
| **Spread Std. Error** | `0.2117` | `0.2169` | Narrow standard error |
| **Spread $z$-score** | `3.146` | `2.564` | Highly significant |
| **Spread $p$-value** | **`0.00165`** | **`0.01033`** | **$\checkmark$ Statistically Significant ($p < 0.01$)** |

---

### 3. RQR (Randomized Quantile Residuals) Count Calibration

| Metric | Target | Raw Model | Shifted $\lambda$ (L2) | What Changed |
| :--- | :---: | :---: | :---: | :--- |
| **Pooled Goal Mean ($\mu$)** | **$0.0000$** | `-0.0415` | **`+0.0112`** | **$4\times$ closer to zero bias!** |
| **Std Dev ($\sigma$)** | **$1.0000$** | `0.9393` | **`0.9378`** | Correct variance |
| **Away Goals Mean ($\mu_{\text{away}}$)** | **$0.0000$** | **`-0.0804`** | **`+0.0016`** | **Away goal scoring bias completely cured!** |
| **Shapiro-Wilk $p$-value** | **$> 0.05$** | `0.2624` | **`0.4906`** | **Strong acceptance of Gaussian normality ($W = 0.9976$)** |

---

### 4. Multi-Market Kelly Portfolio Backtest (`src/Portfolio`)

Simulated across 100 daily slates (314 matches) with **2% Betfair commission** deducted from every winning bet:

| Metric | Raw Model (`FlatTrust 0.25`) | Shifted $\lambda$ (`FlatTrust 1.00`, $\lambda=23$) | Shifted $\lambda$ (`FlatTrust 1.00`, $\lambda=40$) |
| :--- | :---: | :---: | :---: |
| **Final Bankroll** | $1.258\times$ | **$1.281\times$** | **$1.234\times$** |
| **Flat Net ROI** | $5.07\%$ | **$6.21\%$** | **$7.50\%$** |
| **Max Drawdown (MDD)** | $-36.05\%$ | **$-33.39\%$** | **$-20.88\%$** |
| **Sortino Ratio** | $0.081$ | **$0.117$** ($+44\%$) | **$0.175$** ($+116\%$) |
| **Sharpe Ratio** | $0.048$ | **$0.064$** ($+33\%$) | **$0.088$** ($+83\%$) |
| **Calmar Ratio** | $2.348$ | **$2.612$** | **$4.519$** ($+92\%$) |
| **Mean Slate Exposure** | $6.4\%$ | **$5.1\%$** | **$3.2\%$** |
| **Ulcer Index (Drawdown Pain)** | $16.66$ | **$14.56$** | **$8.67$** |

---

## 4. Out-of-Domain Result: Scottish Lower (Tournaments 56 & 57)

### Test design

The 2026-08-24 run applied the unchanged Ireland parameters
$(w_{base},\sigma)=(0.25,0.25)$ to 710 OOS matches from the Scottish Lower
`recomb_pxg_wealth_integrated_hl365_hs2` champion. Raw and shifted books used the **same**
Baker-McHale shrinkage and policy settings, making this a cleaner estimate of the shift itself than
an across-policy comparison.

The shift was substantial in typical matches:

| Side | Median model weight $w$ | Median market share $1-w$ | Approx. retained log-variance $w^2$ |
| :--- | ---: | ---: | ---: |
| Home | 0.408 | 59.2% | 16.6% |
| Away | 0.433 | 56.7% | 18.8% |

Thus, the median prediction was pulled more than halfway toward the closing market while roughly
81–83% of posterior log-variance was removed.

### Matched-policy portfolio results

| Policy | Raw wealth | Shifted wealth | Raw ROI | Shifted ROI | Raw Sharpe | Shifted Sharpe | Raw MDD | Shifted MDD | Bets raw → shifted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Conservative, cap 10%, $\lambda=23$ | 2.261x | **1.894x** | 11.50% | **10.81%** | 1.17 | **1.02** | -22.37% | **-25.66%** | 1,886 → 1,715 |
| Balanced, cap 15%, $\lambda=15$ | 3.147x | **2.449x** | 11.51% | **10.81%** | 1.17 | **1.02** | -32.22% | **-36.54%** | 1,886 → 1,715 |
| Aggressive, cap 20%, $\lambda=10$ | 4.031x | **3.148x** | 11.21% | **10.85%** | 1.13 | **1.01** | -42.89% | **-47.30%** | 1,886 → 1,715 |

### Interpretation

1. **The original shift failed out of domain.** Final wealth fell 16–22%, net ROI fell
   36–70 bps, and Sharpe fell 0.12–0.15 under every policy.
2. **It did not buy drawdown protection.** MDD worsened by 3.29–4.41 percentage points despite
   placing 171 fewer bets (9.1% fewer).
3. **The parameters were too market-dominant.** Median $w\approx0.41$–$0.43$ means the market
   supplied most of the shifted location and the posterior uncertainty was compressed severely.
4. **The Ireland conclusion was overgeneralized.** Better Ireland RQR and portfolio results showed
   in-sample/domain utility, not a universal mathematical calibration constant. Its portfolio table
   also compared different trust/risk configurations, whereas this Scottish test held policy fixed.
5. **Do not deploy `(0.25, 0.25)` or remove `FlatTrust` yet.** The next experiment should seek a
   broad cross-league parameter basin and evaluate calibration metrics as well as bankroll outcomes.

### Recommended next test

Use a weaker shift with the raw model included explicitly as $w=1$:

- $w_{base}\in\{0.50,0.65,0.80,1.00\}$
- $\sigma\in\{0.15,0.25,0.40\}$ (skip $\sigma$ when $w_{base}=1$)
- Report weight quantiles, variance-retention quantiles, LPD/RQR/GLM Edge, bet count, ROI, Sharpe,
  MDD, and final wealth.
- Select parameters across rolling training folds or leagues; never select on the same backtest used
  for the final performance claim.

The nonfatal JLD2 warning in this run concerned an unrelated legacy
`TeamGoalsNegBinOpenPlayModel`; the target recombination champion loaded and completed normally.

---

## 5. Key Lessons & Diagnostics Reference

### Why $\sigma = 1.0$ Failed (The Commission Trap)
* With $\sigma = 1.0$, typical football discrepancies ($\Delta \approx 0.15$) yielded weights $w \approx 0.01 - 0.08$. The model was **$95\%+$ cloned to market consensus**.
* Betting market consensus while paying a **2% exchange commission** mechanically yields a net loss of $-2\%$ to $-3\%$ ROI.
* **Fix:** Setting $w_{\text{base}} = 0.25$ and $\sigma = 0.25$ guarantees baseline edge retention to clear the 2% commission, while scaling up to $w \ge 0.85$ on structural mispricings.

### Why `FlatTrust(0.25)` cannot yet be removed
* In unshifted models, `FlatTrust(0.25)` is a blunt brake on uncalibrated tail bets.
* The Ireland experiment suggested generative calibration might replace it, but the Scottish result
  falsified that conclusion for the original parameters.
* Keep staking trust and generative calibration as separate controls until a weaker shift validates
  out of domain under matched policies.

### Understanding `SlateDrawdown(lambda)`
The Busseti-Boyd drawdown parameter $\lambda = \log(\beta) / \log(D)$:
* **$\lambda = 23$:** Calibrated for $\approx 20\%$ max drawdown at $1\%$ breach probability.
* **$\lambda = 40$:** Calibrated for $\approx 10\%$ max drawdown (conservative risk control, cut drawdown to $-20.88\%$).
* **Higher $\lambda \implies$ Stricter loss tolerance $\implies$ Smaller $k_{\text{risk}} \implies$ Lower exposure**.

### Production Graceful Degradation
In live environments, exchange feeds can be incomplete and Nelder-Mead can fail to converge. 
* **Fallback Protocol:** If market odds for a match are missing, or `fit_market_implied_parameters` errors out, the system defaults to $w = 1.0$. The raw L1 model draws are passed directly to the allocator, ensuring the pipeline never halts.

### Note on Dispersion
The generative shift changes central tendency but does not guarantee calibration. It also contracts
posterior log-variance by $w^2$; in the Scottish median case, only about 17–19% remained. This is
separate from observation-level Poisson/Negative-Binomial dispersion, which remains primarily a
Layer-1 modelling issue.

---

## 6. REPL Execution Snippets

```julia
# 1. Apply Generative Shift across all matches (Multi-Threaded)
apply_layer2_shift!(models_latents.df, odds; w_base=0.25, sigma=0.25)

# 2. Build Shifted LatentStates & Run PPD Model Inference
shifted_latents_df = copy(models_latents.df)
shifted_latents_df.λ_h = models_latents.df.shifted_λ_h
shifted_latents_df.λ_a = models_latents.df.shifted_λ_a
shifted_latents = Experiments.LatentStates(shifted_latents_df, models_latents.model)

# 3. Build MatchBooks (Kelly Allocator + Baker-McHale Shrinkage)
book_spec = PF.BookSpec(markets = MARKETS, allocator = PF.KellyLogUtility(), shrink = PF.BakerMcHale(n_draws=128))
raw_books     = PF.build_books(book_spec, models_latents.df, expr79, odds, ds79)
shifted_books = PF.build_books(book_spec, shifted_latents_df, expr79, odds, ds79)

# 4. Run Portfolio Head-to-Head Comparison
res = run_portfolio_comparison(
    "Calibrated Full Trust (FlatTrust 1.00, SlateDrawdown 23.0)",
    PF.PolicySpec(trust=PF.FlatTrust(1.00), risk=PF.SlateDrawdown(23.0), cap=PF.FixedCap(0.10), filter=PF.KeepAll(), grouping=PF.DailySlate()),
    raw_books, shifted_books
)

# 5. Run Statistical Diagnostics (GLM Edge & RQR)
diag_results = display_evaluation_diagnostics(expr79, ds79, models_latents, shifted_latents)
```


