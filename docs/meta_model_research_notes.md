# Meta Model (Layer 2) — Research Handover Notes
**Date:** 2026-05-26  
**Project:** BayesianFootball — Scottish Lower Leagues  
**Author:** Research session notes for white paper revision  

---

## 1. Overview and Motivation

The system has two layers:

- **Layer 1 (L1):** A Bayesian hierarchical model (Turing.jl, NUTS) that learns team strengths, dynamics, and goal-scoring distributions from historical data (goals, xG). Outputs a **Posterior Predictive Distribution (PPD)** over match outcomes.
- **Layer 2 / Meta Model:** Learns *how much to trust L1* on a given market at a given point in time, relative to the bookmaker's market price.

The core motivation is that **L1 is not uniformly well-calibrated across all markets and time periods**. The Meta Model should detect this structure and suppress bets during "bad regimes" where L1 is less reliable than the market.

---

## 2. Mathematical Formulation (Current Implementation)

### 2.1 Mixture Probability

For each match $i$ in week $w$, define the **Meta probability**:

$$Q_i = \theta_{t_w} \cdot p^{L1}_i + (1 - \theta_{t_w}) \cdot m_i$$

Where:
- $p^{L1}_i$ = L1 posterior predictive mean for match $i$ (scalar extracted from PPD)
- $m_i$ = vig-removed fair market probability (e.g. `prob_fair_close`)
- $\theta_{t_w} \in (0,1)$ = **time-varying trust in L1** for week $w$

### 2.2 Trust Dynamics (GRW)

$\theta_{t_w}$ evolves as a Gaussian Random Walk in logit-space (non-centred parameterisation for NUTS efficiency):

$$\text{logit}(\theta_w) = \alpha + \sum_{k=1}^{w} z_k \cdot \sigma_{\text{GRW}}$$

$$\alpha \sim \mathcal{N}(0, 1)$$
$$\sigma_{\text{GRW}} \sim \text{HalfNormal}(0.1)$$
$$z_k \sim \mathcal{N}(0, 1) \quad \text{(non-centred)}$$

The GRW drift is **mean-centred** before adding $\alpha$ so that $\alpha$ retains its interpretation as the global average trust level.

### 2.3 Likelihood

$$Y_i \sim \text{Bernoulli}(Q_i)$$

Where $Y_i = 1$ if the selection won (e.g. under 2.5 goals), 0 otherwise.

### 2.4 Team Bias Component (Hierarchical Extension)

An optional team-level correction (tested but found to be weak at current data volumes):

$$Q_i^{\text{corrected}} = \text{logistic}\left(\text{logit}(Q_i) + \delta_{\text{home}[i]} + \delta_{\text{away}[i]}\right)$$

$$\delta_k = \sigma_\delta \cdot z_k^{\delta}, \quad z_k^{\delta} \sim \mathcal{N}(0,1), \quad \sigma_\delta \sim \text{HalfNormal}(0.1)$$

### 2.5 Staking (McHale / Baker 2013)

For staking, we propagate **full posterior uncertainty** through Q. For each match, we construct a posterior distribution of $Q_i$ by sampling across all MCMC draws:

$$Q_i^{(s)} = \theta_{t_w}^{(s)} \cdot p^{L1}_i + (1 - \theta_{t_w}^{(s)}) \cdot m_i, \quad s = 1,\ldots,S$$

This vector $\{Q_i^{(s)}\}$ is passed to the BayesianKelly (`ExactBayesianKelly` / `BayesianKelly`) staking functions, which maximise expected log-growth accounting for the full distributional uncertainty — not just the mean.

---

## 3. Training Architecture

### 3.1 Data Pipeline

```
L1 ExperimentResults
    └── extract_oos_predictions(ds, exp_results)
            → latent_states (OOS chains, one per L1 fold)
    └── Predictions.model_inference(latent_states)
            → PPD (DataFrame: match_id, selection, distribution, p_L1_mean)
    └── innerjoin with ds.odds (prob_fair_close, odds_close, is_winner)
    └── innerjoin with ds.matches (match_date, home_team, away_team, season)
    └── Compute W (global week index from match_date)
    └── MetaModelData → Turing model → NUTS
```

### 3.2 Fold Structure (Expanding Window)

The Meta Model uses an **expanding window** of L1 OOS predictions:

| Meta Fold | Training Data | Evaluation Target |
|---|---|---|
| Fold 1 | 22/23 OOS (350 matches) | Diagnostic only |
| Fold 2 | 22/23 + 23/24 OOS (698 matches) | — |
| Fold 3 | 22/23 + 23/24 + 24/25 OOS (1057 matches) | → 25/26 forecast |
| Fold 4 | All seasons (1405 matches) | In-sample diagnostic |

### 3.3 Queued NUTS Execution

All `n_folds × n_chains` tasks are flattened into a single task queue and run concurrently via `Base.Semaphore(Threads.nthreads())`, mirroring the L1 `QueuedNUTSConfig` / `_train_queued` approach. Chains within a fold are combined via `chainscat` when all complete.

---

## 4. Experimental Results — Scottish Lower Leagues

### 4.1 Dataset

- **L1 model:** `DynamicGoalsModel` (goals only, no xG)
- **Seasons:** 22/23 → 25/26 (4 seasons, ~80 L1 weekly splits)
- **Markets tested:** `:over_15`, `:under_25`
- **Sampler:** `QueuedNUTSConfig(n_samples=500, n_chains=4, n_warmup=200)`

### 4.2 MCMC Diagnostics (over_15, single unified run)

| Metric | Value | Assessment |
|---|---|---|
| R-hat failures (>1.05) | 0 / 225 | ✅ Excellent |
| ESS (minimum) | 346 | ✅ Well above 100 |
| ESS (mean) | 648 | ✅ |
| Acceptance rate | 0.80 | ✅ On target |

### 4.3 Global Trust: over_15 Market

$$\hat{\alpha} = -1.01 \quad (\text{std}=0.83, \; 95\% \text{ CI: } [-2.55,\; +0.66])$$
$$\hat{\theta}_{\text{global}} = \text{logistic}(-1.01) \approx 0.27$$

**Interpretation:** The Meta Model globally trusts L1 at only **26.7%** on the over 1.5 goals market, weighting the market at 73.3%. The 95% CI spans from near-zero trust to roughly equal trust — L1's advantage is uncertain.

### 4.4 Regime Dynamics: over_15

$$\hat{\sigma}_{\text{GRW}} = 0.097 \quad (\text{std}=0.105)$$

- **Weekly $\theta_t$ range:** [0.240, 0.293] — very narrow
- **Std across weeks:** 0.017 — minimal regime variation
- **Quarterly trend:** Monotonically **decreasing** Q1→Q4:

| Quarter | Avg $\theta$ | Direction |
|---|---|---|
| Q1 (wk 1–49) | 0.289 | ↓ |
| Q2 (wk 50–98) | 0.275 | ↓ |
| Q3 (wk 99–147) | 0.259 | ↓ |
| Q4 (wk 148–196) | 0.246 | ↓ |

**Interpretation:** L1's reliability on over_15 is declining over time. The bookmaker's over_15 market is becoming more efficient relative to L1's signal.

### 4.5 Predictive Fit: over_15

| Model | Brier Score | Log-Loss | vs Market (Brier, bp) |
|---|---|---|---|
| Market (Baseline) | **0.16935** | **0.52104** | — |
| Meta Mixture | 0.16965 | 0.52199 | -3 bp *(worse)* |
| L1 Raw | 0.17129 | 0.52642 | -19 bp *(worse)* |

The Meta improves over L1 by **+16 bp** but is **-3 bp worse than the market**. The market wins on over_15.

### 4.6 Team Biases: over_15

- $\hat{\sigma}_\delta = 0.102$ (small)
- All team 95% CIs cross zero — **no statistically significant team-level biases** at current data volumes (~50 matches per team)
- Largest bias: cove-rangers $\delta = -0.013$ (std=0.14) — essentially noise

### 4.7 Staking Results: over_15

The Q mixture never produces positive edge vs bookmaker implied probability because:

$$Q_{\text{edge}} = \theta \cdot \underbrace{(p^{L1} - p_{\text{implied}})}_{\text{L1 edge}} + (1-\theta) \cdot \underbrace{(m_i - p_{\text{implied}})}_{\text{always negative (vig)}}$$

With $\theta = 0.27$ and market vig ≈ 9%:

$$Q_{\text{edge, max}} \approx 0.27 \times 0.09 - 0.73 \times 0.045 \approx -0.009$$

Even at L1's best individual match, Q never beats the implied price. **Zero bets placed.** This is the correct, intended behaviour — the Meta Model auto-kills markets where L1 has no systematic edge.

---

### 4.8 Results: under_25 Market (4-Fold Queued Run)

**Global trust across folds:**

| Fold | Matches | $\hat{\theta}_{\text{global}}$ |
|---|---|---|
| 1 | 350 | ~0.42 |
| 2 | 698 | ~0.41 |
| 3 | 1057 | ~0.40 |
| 4 | 1405 | ~0.42 |

$\theta \approx 0.40$ on under_25 — **meaningfully higher** than over_15 (0.27). L1 has more genuine signal here.

**$\theta_t$ dynamics (fold 4, all 196 weeks):**
- Range: [0.395, 0.441]
- Std: 0.0148 — moderate within-season variation

**Regime Filter Results (fold-4 chain, 25/26 holdout, 348 matches):**

| Regime | Bets | PnL | ROI |
|---|---|---|---|
| Good ($\theta \geq$ median) | 45 | **+0.200** | **+12.17%** |
| Bad ($\theta <$ median) | 39 | -0.130 | -6.65% |
| All bets (no filter) | 144 | +0.073 | +1.96% |

**Regime ROI spread: +18.8 percentage points.** Strong evidence that weekly regime structure is real and material on the under_25 market.

> **Caveat:** This is in-sample — fold-4 was trained on 25/26 data. The GRW had access to the full season outcomes when learning which weeks were "good". See §6 for the predictive gap.

---

## 5. Key Architectural Findings

### 5.1 The Vig Drag Problem

The mixture $Q = \theta \cdot p^{L1} + (1-\theta) \cdot m_i$ uses **vig-removed fair price** $m_i$ but the bet is placed at raw **bookmaker odds** (which include vig). This creates a structural negative drag in the edge calculation:

$$\text{Edge vs offered odds} = Q - \frac{1}{\text{odds}} = \theta(p^{L1} - \frac{1}{\text{odds}}) + (1-\theta)(m_i - \frac{1}{\text{odds}})$$

Since $m_i < \frac{1}{\text{odds}}$ always (vig makes implied > fair), the second term is **always negative**. For low $\theta$ markets (like over_15), this suppresses all Q edges.

**White paper implication:** The formulation should be explicit about whether $m_i$ is the fair price or the implied price, and how the mixture interacts with the vig. A possible reformulation: replace $m_i$ with $\frac{1}{\text{odds}}$ (the implied price), so the mixture is between L1 and the bookmaker's implied probability directly.

### 5.2 The GRW Extrapolation Problem

The GRW θ_t is an **in-sample latent process** — it has no ability to forecast into unseen time periods. For OOS evaluation:

- Fold-3 chain (trained on weeks 1..145) applied to fold-4 OOS (weeks 149..196): all OOS weeks get `θ_t[145]` (the terminal value) — no discrimination between weeks.
- **The GRW is retrospective, not predictive.**

**White paper implication:** The regime signal is currently a diagnostic tool, not a betting signal. For genuine predictive use, the architecture needs either:
1. **Weekly rolling retraining** of the Meta Model (computationally expensive but correct)
2. A **one-step-ahead predictive distribution** for $\theta_{T+1}$ given $\theta_T$ and the GRW posterior — this is mathematically tractable as $\text{logit}(\theta_{T+1}) \sim \mathcal{N}(\text{logit}(\hat{\theta}_T),\, \sigma^2_{\text{GRW}})$

### 5.3 Mean vs Distribution in the Mixture

**Current implementation:** $Q_i$ uses a **scalar** $p^{L1}_i = \mathbb{E}[\text{PPD}]$, discarding L1's distributional uncertainty.

**Richer formulation:** Propagate the full PPD into the mixture. For each MCMC draw $s$:

$$Q_i^{(s)} = \theta_{t_w}^{(s)} \cdot p^{L1,(s)}_i + (1-\theta_{t_w}^{(s)}) \cdot m_i$$

Where $p^{L1,(s)}_i$ is a sample from the L1 PPD for match $i$. This is implemented in `staking.jl` for the Kelly calculation but **not yet in the Turing likelihood** — the likelihood still uses the scalar L1 mean.

**White paper implication:** The full Bayesian treatment should integrate over L1's uncertainty in the likelihood:

$$\mathcal{L}(Y_i | \theta_{t_w}, m_i) = \int p^{L1} \cdot \left[\theta_{t_w} p^{L1} + (1-\theta_{t_w}) m_i\right]^{Y_i} \cdot \left[1 - \theta_{t_w} p^{L1} - (1-\theta_{t_w}) m_i\right]^{1-Y_i} d\mathcal{F}^{L1}$$

This is a **mixture of Bernoullis marginalised over the PPD**, which may be approximated by passing L1 posterior samples as data or via importance weighting.

### 5.4 The Regime Filter as a Bet Gate vs Probability Adjuster

**Two distinct uses of θ_t:**

1. **Probability Adjuster** (current): $Q = \theta p^{L1} + (1-\theta) m$ → replaces the betting probability. Loses most of L1's edge when θ is small.

2. **Bet Gate** (more powerful in practice): Use θ as a weekly on/off switch. In "good regime" weeks (θ > threshold), bet using L1's **raw PPD** with BayesianKelly. In "bad regime" weeks, don't bet at all.

The empirical results strongly favour the **gate approach**: 45 bets at +12.17% ROI (good) vs 39 bets at -6.65% (bad), a +18.8pp spread.

---

## 6. Open Questions for the White Paper

### Mathematical Questions

1. **Vig-corrected mixture:** Should the mixture target be $\frac{1}{\text{odds}}$ (bookmaker implied) or $m_i$ (vig-free fair price)? What is the correct economic interpretation of each?

2. **Full marginalisation:** What is the computational cost and accuracy improvement of marginalising the Bernoulli likelihood over the full L1 PPD vs using the scalar mean?

3. **One-step-ahead GRW forecast:** Can we derive a closed-form predictive posterior for $\theta_{T+1}$ given the in-sample GRW trajectory? How does predictive uncertainty grow as a function of forecast horizon and $\sigma_{\text{GRW}}$?

4. **Optimal regime threshold:** What is the theoretically optimal threshold for the bet gate? Is it the median of θ_t, or should it be derived from the Kelly criterion (i.e. only bet when expected log-growth is positive under Q)?

### Architecture Questions

5. **Fold granularity:** The current fold structure (seasonal expanding window) gives too few folds for rigorous OOS validation. Weekly rolling estimation is theoretically correct but expensive. What is the minimum data volume per fold for the Meta Model to converge reliably?

6. **Team bias identification:** With ~50 matches per team in a single fold, team-specific δ estimates are all noise. How many seasons of data are needed for team-level biases to become identifiable? Is a pooled "opponent class" hierarchy (e.g. by league tier) more appropriate?

7. **Multi-market Meta Model:** Can a single Meta Model learn joint trust across multiple markets simultaneously (over_15, under_25, home, away), sharing a common GRW and team hierarchy but with market-specific intercepts?

8. **L1 model heterogeneity:** If L1 is updated (e.g. DynamicGoalsModel → DynamicXGModel), the historical OOS predictions change. How should the Meta Model handle structural breaks in the L1 signal?

### Practical / Betting Questions

9. **Rolling retraining cadence:** How frequently should the Meta Model be retrained in production? Every week (expensive), every matchday, or every season?

10. **Correlation between markets:** The over_15 and under_25 regimes may be anti-correlated. Should staking be capped across correlated market bets from the same regime?

---

## 7. File Map (Current Development)

```
current_development/MetaModels/
├── src/
│   ├── MetaModels.jl               ← Module entry point
│   ├── types.jl                    ← AbstractMetaModel hierarchy
│   ├── staking.jl                  ← Full Q posterior + McHale Kelly
│   ├── components/
│   │   ├── dynamics.jl             ← GRW dynamics config + Turing builder
│   │   └── teams.jl                ← Hierarchical team bias config + builder
│   ├── engines/
│   │   └── mixture_engine.jl       ← ConvexMixtureMetaModel + Turing assembly
│   └── training/
│       └── workflow.jl             ← Queued fold run_meta_experiment
├── r03_meta_framework_runner.jl    ← Single unified run (MVP, now deprecated)
├── r04_queued_fold_runner.jl       ← Full queued fold runner (current)
└── analysis_headless.jl            ← Complete text-based diagnostics (no plots)
```

---

## 8. Summary of What Works, What Doesn't

| Component | Status | Notes |
|---|---|---|
| MCMC convergence | ✅ Works well | R-hat 0 failures, ESS >300 on all tested runs |
| Global α (market-level trust) | ✅ Highly informative | Clear differentiation: over_15 (θ=0.27) vs under_25 (θ=0.40) |
| GRW dynamics (in-sample) | ✅ Finds real regime structure | +18.8pp ROI spread on under_25 |
| GRW dynamics (OOS forecast) | ❌ Doesn't extrapolate | Terminal θ used for all future weeks — no predictive signal |
| Team biases | ❌ Too weak at current data volume | All CIs cross zero, σ_δ≈0.10 but no identifiable signal |
| Q-mixture for staking | ⚠️ Partially works | Vig drag suppresses all bets on low-θ markets (over_15). Works on higher-θ markets |
| Bet gate (θ as filter) | ✅ Most promising result | +12.17% vs -6.65% ROI split in-sample on under_25 |
| Queued multi-fold training | ✅ Architecture correct | 4 folds × 4 chains = 16 tasks, concurrent execution working |

---

*End of handover notes.*
