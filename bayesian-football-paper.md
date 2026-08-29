# A Bayesian Framework for Football Match Outcome Prediction Using MCMC Methods

## Abstract

A comprehensive Bayesian framework for predicting football match outcomes using Markov Chain Monte Carlo (MCMC) methods. We extend the classical Maher model to incorporate team-specific attack and defense parameters with a home advantage factor, employing a Poisson likelihood for goal scoring. Our key contribution lies in the probabilistic treatment of betting decisions through posterior predictive distributions, where we develop an adaptive confidence threshold framework based on Neyman-Pearson decision theory. We propose a context-dependent threshold function that accounts for match characteristics and team-specific historical performance, optimized through empirical risk minimization on historical betting data.

## 1. Introduction

The prediction of football match outcomes presents a challenging problem at the intersection of sports analytics and statistical modeling. Traditional approaches often produce point estimates of match probabilities, failing to capture the inherent uncertainty in predictions. We address this limitation by developing a fully Bayesian framework that propagates parameter uncertainty through to betting decisions.

Our approach consists of three main components:
1. A hierarchical Bayesian model for team strengths based on the Maher framework
2. Posterior predictive distributions for match outcome probabilities
3. An adaptive, context-dependent decision framework for value betting

## 2. The Hierarchical Bayesian Model

### 2.1 Model Specification

We model the number of goals scored by each team in a match as independent Poisson random variables. For a match between home team $i$ and away team $j$, let:
- $Y_{ij}^{(h)}$ = goals scored by home team $i$
- $Y_{ij}^{(a)}$ = goals scored by away team $j$

The likelihood is specified as:
$$Y_{ij}^{(h)} \sim \text{Poisson}(\lambda_{ij})$$
$$Y_{ij}^{(a)} \sim \text{Poisson}(\mu_{ij})$$

where the expected goals are:
$$\lambda_{ij} = \alpha_i \cdot \beta_j \cdot \gamma$$
$$\mu_{ij} = \alpha_j \cdot \beta_i$$

Here:
- $\alpha_i > 0$ represents the attacking strength of team $i$
- $\beta_i > 0$ represents the defensive weakness of team $i$
- $\gamma > 0$ represents the home advantage factor

### 2.2 Prior Specification

To ensure positivity constraints and computational stability, we work in log-space:

$$\log(\alpha_i) \sim \mathcal{N}(0, \sigma_{\alpha}^2), \quad i = 1, \ldots, n$$
$$\log(\beta_i) \sim \mathcal{N}(0, \sigma_{\beta}^2), \quad i = 1, \ldots, n$$
$$\log(\gamma) \sim \mathcal{N}(\log(1.3), 0.2^2)$$

The hyperpriors for the variance parameters are:
$$\sigma_{\alpha} \sim \text{TruncatedNormal}(0, 1, 0, \infty)$$
$$\sigma_{\beta} \sim \text{TruncatedNormal}(0, 1, 0, \infty)$$

### 2.3 Identifiability Constraints

To ensure model identifiability, we impose the constraint $\prod_{i=1}^n \alpha_i = 1$, implemented through centering in log-space:

$$\log(\alpha_i^*) = \log(\alpha_i) - \frac{1}{n}\sum_{k=1}^n \log(\alpha_k)$$
$$\log(\beta_i^*) = \log(\beta_i) - \frac{1}{n}\sum_{k=1}^n \log(\beta_k)$$

### 2.4 Posterior Inference

Given observed match data $\mathcal{D} = \{(i_k, j_k, y_k^{(h)}, y_k^{(a)})\}_{k=1}^{m}$, the posterior distribution is:

$$p(\boldsymbol{\alpha}, \boldsymbol{\beta}, \gamma, \sigma_{\alpha}, \sigma_{\beta} | \mathcal{D}) \propto p(\mathcal{D} | \boldsymbol{\alpha}, \boldsymbol{\beta}, \gamma) \cdot p(\boldsymbol{\alpha} | \sigma_{\alpha}) \cdot p(\boldsymbol{\beta} | \sigma_{\beta}) \cdot p(\gamma) \cdot p(\sigma_{\alpha}) \cdot p(\sigma_{\beta})$$

We sample from this posterior using Hamiltonian Monte Carlo (HMC), generating $S$ samples: $\{\boldsymbol{\theta}^{(s)}\}_{s=1}^S$ where $\boldsymbol{\theta}^{(s)} = (\boldsymbol{\alpha}^{(s)}, \boldsymbol{\beta}^{(s)}, \gamma^{(s)})$.

## 3. Posterior Predictive Distributions for Match Outcomes

### 3.1 From Parameters to Match Probabilities

For each MCMC sample $s$, we compute the expected goals:
$$\lambda_{ij}^{(s)} = \alpha_i^{(s)} \cdot \beta_j^{(s)} \cdot \gamma^{(s)}$$
$$\mu_{ij}^{(s)} = \alpha_j^{(s)} \cdot \beta_i^{(s)}$$

The probability of each match outcome is computed using the bivariate Poisson distribution. For computational efficiency, we use a truncated sum:

$$P_{\text{HW}}^{(s)} = \sum_{h=1}^{K} \sum_{a=0}^{h-1} P(Y^{(h)} = h | \lambda_{ij}^{(s)}) \cdot P(Y^{(a)} = a | \mu_{ij}^{(s)})$$

$$P_{\text{D}}^{(s)} = \sum_{g=0}^{K} P(Y^{(h)} = g | \lambda_{ij}^{(s)}) \cdot P(Y^{(a)} = g | \mu_{ij}^{(s)})$$

$$P_{\text{AW}}^{(s)} = \sum_{a=1}^{K} \sum_{h=0}^{a-1} P(Y^{(h)} = h | \lambda_{ij}^{(s)}) \cdot P(Y^{(a)} = a | \mu_{ij}^{(s)})$$

where $K$ is chosen such that $P(Y > K | \lambda) < \epsilon$ for numerical tolerance $\epsilon$.

This yields posterior predictive distributions:
$$\mathcal{P}_{\text{HW}} = \{P_{\text{HW}}^{(1)}, \ldots, P_{\text{HW}}^{(S)}\}$$
$$\mathcal{P}_{\text{D}} = \{P_{\text{D}}^{(1)}, \ldots, P_{\text{D}}^{(S)}\}$$
$$\mathcal{P}_{\text{AW}} = \{P_{\text{AW}}^{(1)}, \ldots, P_{\text{AW}}^{(S)}\}$$

### 3.2 Value Assessment

For market odds $O_{\text{outcome}}$, the implied probability is $p_{\text{market}} = 1/O_{\text{outcome}}$. A bet has positive expected value when:

$$P_{\text{model}} \times O_{\text{outcome}} > 1$$

The probability of value for each outcome is:
$$\mathbb{P}(\text{Value} | \text{outcome}) = \frac{1}{S}\sum_{s=1}^S \mathbb{I}(P_{\text{outcome}}^{(s)} \times O_{\text{outcome}} > 1)$$

## 4. Time-Adaptive Decision Framework

### 4.1 Dynamic Context-Dependent Confidence Thresholds

The optimal confidence threshold evolves over time due to changing market efficiency and model performance. We propose a time-adaptive threshold function:

$$c_t(\text{outcome}, i, j, \mathbf{x}) = c_{\text{base}, t}^{(\text{outcome})} \cdot \psi_{i,t} \cdot \psi_{j,t} \cdot f(\mathbf{x})$$

where the subscript $t$ denotes time-dependence, reflecting that:
- Market efficiency increases over time as bookmakers refine their models
- Model performance varies as it adapts to new data and league dynamics
- Team-specific factors evolve with squad changes and tactical shifts

The base thresholds $c_{\text{base}, t}^{(\text{outcome})}$ are no longer static but are re-optimized periodically to reflect current market conditions.

### 4.2 Rolling Window Optimization

We implement a rolling window approach for threshold optimization. At time $t$, we optimize thresholds using only recent historical data:

$$\mathcal{D}_t^{\text{window}} = \{(m, d_m, r_m) : t - W \leq t_m < t\}$$

where $W$ is the window size (e.g., 3 seasons) and $(m, d_m, r_m)$ represents match $m$ with decision $d_m$ and realized return $r_m$.

The optimization problem becomes:
$$\{\hat{c}_{\text{base}, t}^{(\cdot)}, \hat{\eta}_t\} = \arg\min_{c_{\text{base}}^{(\cdot)}, \eta} \sum_{m \in \mathcal{D}_t^{\text{window}}} L(d_m, r_m)$$

### 4.3 Exponentially Weighted Optimization

A more sophisticated approach uses exponentially decaying weights to emphasize recent performance while retaining information from older matches:

$$w_m(t) = \exp\left(-\lambda \cdot (t - t_m)\right)$$

where $\lambda$ is the decay rate and $t_m$ is the time of match $m$.

The weighted optimization becomes:
$$\{\hat{c}_{\text{base}, t}^{(\cdot)}, \hat{\eta}_t\} = \arg\min_{c_{\text{base}}^{(\cdot)}, \eta} \sum_{m \in \mathcal{D}_t} w_m(t) \cdot L(d_m, r_m)$$

The weighted ROI for threshold selection is:
$$\text{ROI}_{\text{weighted}}(c) = \frac{\sum_{m \in \mathcal{D}_t} w_m(t) \cdot \text{Profit}_m(c)}{\sum_{m \in \mathcal{D}_t} w_m(t) \cdot \text{Stake}_m(c)}$$

### 4.4 Adaptive Team-Specific Factors

Team-specific adjustment factors also evolve over time:

$$\psi_{i,t} = \exp\left(\eta_t \cdot \left(\frac{W_{i,t} - L_{i,t}}{W_{i,t} + L_{i,t} + \epsilon} - 0.5\right)\right)$$

where $W_{i,t}$ and $L_{i,t}$ are calculated using the same exponential weighting:

$$W_{i,t} = \sum_{m \in \mathcal{M}_i} w_m(t) \cdot \mathbb{I}(\text{profit}_m > 0)$$
$$L_{i,t} = \sum_{m \in \mathcal{M}_i} w_m(t) \cdot \mathbb{I}(\text{profit}_m \leq 0)$$

where $\mathcal{M}_i$ represents matches involving team $i$.

### 4.5 Market Efficiency Adjustment

To account for increasing market efficiency, we introduce a time-dependent scaling factor:

$$\kappa(t) = 1 + \rho \cdot \log\left(1 + \frac{t - t_0}{T_{\text{scale}}}\right)$$

where:
- $\rho$ controls the rate of efficiency adjustment
- $t_0$ is the reference time (start of data)
- $T_{\text{scale}}$ is a time scale parameter

The final threshold becomes:
$$c_t^{\text{final}}(\text{outcome}, i, j, \mathbf{x}) = \kappa(t) \cdot c_t(\text{outcome}, i, j, \mathbf{x})$$

### 4.6 Online Learning Framework

For real-time adaptation, we implement an online learning approach using stochastic gradient descent:

$$c_{\text{base}, t+1}^{(\text{outcome})} = c_{\text{base}, t}^{(\text{outcome})} - \alpha_{\text{lr}} \cdot \nabla_c L(d_t, r_t)$$

where $\alpha_{\text{lr}}$ is the learning rate, potentially scheduled as:
$$\alpha_{\text{lr}}(t) = \frac{\alpha_0}{1 + \beta \cdot t}$$

### 4.7 Performance Monitoring and Regime Detection

We monitor the model's performance using a moving average of recent returns:

$$\bar{R}_t = \frac{1}{N_{\text{recent}}} \sum_{m=t-N_{\text{recent}}}^{t} r_m$$

When performance degrades significantly (e.g., $\bar{R}_t < \tau_{\text{min}}$), we trigger:
1. Immediate threshold recalibration
2. Potential model retraining with updated data
3. Temporary increase in confidence requirements

### 4.8 Multi-Scale Temporal Adaptation

We propose a multi-scale approach combining different time horizons:

$$c_t^{\text{multi}}(\text{outcome}) = \omega_{\text{short}} \cdot c_t^{\text{short}} + \omega_{\text{med}} \cdot c_t^{\text{medium}} + \omega_{\text{long}} \cdot c_t^{\text{long}}$$

where:
- $c_t^{\text{short}}$ uses the last month of data (tactical changes, injuries)
- $c_t^{\text{medium}}$ uses the current season (team form)
- $c_t^{\text{long}}$ uses 3 seasons (stable patterns)

The weights $\omega$ can be optimized or set based on domain knowledge.

### 4.9 Neyman-Pearson Framework

We formulate the betting decision as a hypothesis testing problem:
- $H_0$: The bet has negative expected value
- $H_1$: The bet has positive expected value

Under the Neyman-Pearson lemma, the optimal test for a fixed Type I error rate $\alpha$ is:

$$\text{Bet if } \frac{\mathbb{P}(\mathcal{P}_{\text{outcome}} | H_1)}{\mathbb{P}(\mathcal{P}_{\text{outcome}} | H_0)} > k_{\alpha}$$

In practice, we approximate this by:

$$\text{Bet if } \mathbb{P}(\text{Value} | \text{outcome}) > c_t(\text{outcome}, i, j, \mathbf{x})$$

### 4.10 ROC-Based Calibration

For each outcome type, we can construct ROC curves by varying the threshold:

$$\text{TPR}(c) = \frac{\sum_{m \in \mathcal{M}_+} \mathbb{I}(\mathbb{P}(\text{Value}_m) > c)}{|\mathcal{M}_+|}$$

$$\text{FPR}(c) = \frac{\sum_{m \in \mathcal{M}_-} \mathbb{I}(\mathbb{P}(\text{Value}_m) > c)}{|\mathcal{M}_-|}$$

where $\mathcal{M}_+$ and $\mathcal{M}_-$ are sets of matches with positive and negative realized returns.

The optimal threshold can be selected to maximize:
$$J(c) = \text{TPR}(c) - \text{FPR}(c)$$

or using a utility-weighted criterion:
$$U(c) = u_+ \cdot \text{TPR}(c) - u_- \cdot \text{FPR}(c)$$

where $u_+$ and $u_-$ represent the utilities of correct and incorrect bets.

## 5. Implementation Algorithm

```
Algorithm 1: Adaptive Bayesian Betting Strategy
---------------------------------------------------
Input: Match (i,j), Market odds O, Historical data D
Output: Betting decisions for HW, D, AW

1. Sample from posterior: {θ^(s)}_{s=1}^S ~ p(θ|D)
2. For each sample s:
   a. Compute λ^(s), μ^(s) from θ^(s)
   b. Calculate P_HW^(s), P_D^(s), P_AW^(s)
3. For each outcome ∈ {HW, D, AW}:
   a. Compute P(Value|outcome)
   b. Calculate ψ_i, ψ_j from historical performance
   c. Determine c(outcome, i, j, x)
   d. Decision: Bet if P(Value|outcome) > c(outcome, i, j, x)
```

```
Algorithm 2: Time-Adaptive Bayesian Betting Strategy
---------------------------------------------------
Input: Match (i,j) at time t, Market odds O, Historical data D_t
Output: Betting decisions for HW, D, AW

1. Check if recalibration needed:
   If t mod T_recal = 0 or performance trigger:
      a. Compute weights w_m(t) for all m in D_t
      b. Optimize c_base,t using weighted backtesting
      c. Update team factors ψ_i,t for all teams
      
2. Sample from posterior: {θ^(s)}_{s=1}^S ~ p(θ|D_t)

3. For each sample s:
   a. Compute λ^(s), μ^(s) from θ^(s)
   b. Calculate P_HW^(s), P_D^(s), P_AW^(s)
   
4. For each outcome ∈ {HW, D, AW}:
   a. Compute P(Value|outcome)
   b. Calculate time-adjusted threshold:
      c_t^final = κ(t) · c_base,t^(outcome) · ψ_i,t · ψ_j,t · f(x)
   c. Decision: Bet if P(Value|outcome) > c_t^final
   
5. After match resolution:
   a. Update performance metrics
   b. Store (m, d_m, r_m) for future recalibration
```

## 6. Practical Considerations

### 6.1 Computational Efficiency

For real-time betting decisions, we recommend:
1. Pre-compute team-specific factors $\psi_i$ periodically
2. Use parallel processing for MCMC chains
3. Cache posterior samples for recent model fits
4. Consider variational inference for faster approximate posteriors

### 6.2 Model Validation

The framework should be validated through:
1. **Posterior predictive checks**: Compare predicted vs actual score distributions
2. **Cross-validation**: Time-series split to avoid look-ahead bias
3. **Calibration plots**: Reliability diagrams for predicted probabilities
4. **Betting performance metrics**: ROI, Sharpe ratio, maximum drawdown

### 6.3 Backtesting with Time-Awareness

We implement walk-forward analysis to avoid look-ahead bias:

1. **Training Period**: Seasons 1-3 to fit model and optimize initial thresholds
2. **Validation Period**: Season 4 to tune hyperparameters ($\lambda$, window size)
3. **Test Period**: Season 5+ for out-of-sample evaluation

### 6.4 Performance Metrics Over Time

Track time-varying metrics:
- Rolling Sharpe ratio: $\text{SR}_t = \frac{\bar{R}_t}{\sigma_{R,t}}$
- Time-weighted ROI using exponential weighting
- Regime-specific performance during different market conditions

## 7. Extensions and Future Work

1. **Time-varying parameters**: Allow team strengths to evolve over time using state-space models
2. **Correlation structure**: Model correlation between home and away goals for more accurate draw predictions
3. **Feature enrichment**: Incorporate additional covariates (injuries, weather, importance)
4. **Multi-market integration**: Extend to over/under and Asian handicap markets
5. **Ensemble approaches**: Combine multiple models with different temporal scales
6. **Deep learning integration**: Use neural networks for feature extraction while maintaining Bayesian framework

## 8. Conclusion

This enhanced framework addresses the dynamic nature of betting markets through time-adaptive confidence thresholds. By incorporating rolling window optimization, exponential weighting, and market efficiency adjustments, the model maintains its edge as both the market and the underlying sport evolve. The multi-scale temporal adaptation ensures robustness across different time horizons while the online learning component enables real-time responsiveness to performance changes. The key innovation lies in treating the confidence threshold as a context-dependent and time-evolving function optimized through historical performance, addressing the practical observation that certain teams (like Celtic) and certain market conditions require special consideration in betting strategies.

## References

[Add relevant references here]

## Appendix A: Julia Implementation Details

### A.1 Basic Maher Model

```julia
@model function basic_maher_model_raw(home_team_ids, away_team_ids, 
                                      home_goals, away_goals, n_teams)
    # PRIORS (in log space for positivity constraints)
    σ_attack ~ truncated(Normal(0, 1), 0, Inf)
    log_α_raw ~ MvNormal(zeros(n_teams), σ_attack * I)
    
    σ_defense ~ truncated(Normal(0, 1), 0, Inf)
    log_β_raw ~ MvNormal(zeros(n_teams), σ_defense * I)
    
    log_γ ~ Normal(log(1.3), 0.2)
    
    # IDENTIFIABILITY CONSTRAINT
    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)
    
    # Transform to original scale
    α = exp.(log_α)
    β = exp.(log_β)
    γ = exp(log_γ)
    
    # LIKELIHOOD
    n_matches = length(home_goals)
    for k in 1:n_matches
        i = home_team_ids[k]
        j = away_team_ids[k]
        
        λ = α[i] * β[j] * γ
        μ = α[j] * β[i]
        
        home_goals[k] ~ Poisson(λ)
        away_goals[k] ~ Poisson(μ)
    end
    
    return (α = α, β = β, γ = γ)
end
```

### A.2 Market Odds Processing

```julia
struct Odds1x2
    home::Float64
    draw::Float64
    away::Float64
end

function estimate_missing_odds(home, draw, away)
    non_missing = sum([!ismissing(home), !ismissing(draw), !ismissing(away)])
    
    if non_missing < 2
        return nothing
    end
    
    if non_missing == 3
        return Odds1x2(home, draw, away)
    end
    
    # Estimate missing value using implied probabilities
    if ismissing(home)
        prob_draw = 1 / draw
        prob_away = 1 / away
        prob_home = 1 - prob_draw - prob_away
        home_est = prob_home > 0 ? 1 / prob_home : 999.0
        return Odds1x2(home_est, draw, away)
    elseif ismissing(draw)
        prob_home = 1 / home
        prob_away = 1 / away
        prob_draw = 1 - prob_home - prob_away
        draw_est = prob_draw > 0 ? 1 / prob_draw : 999.0
        return Odds1x2(home, draw_est, away)
    else  # ismissing(away)
        prob_home = 1 / home
        prob_draw = 1 / draw
        prob_away = 1 - prob_home - prob_draw
        away_est = prob_away > 0 ? 1 / prob_away : 999.0
        return Odds1x2(home, draw, away_est)
    end
end
```

## Appendix B: Mathematical Notation Summary

| Symbol | Description |
|--------|-------------|
| $\alpha_i$ | Attack strength of team $i$ |
| $\beta_i$ | Defensive weakness of team $i$ |
| $\gamma$ | Home advantage factor |
| $\lambda_{ij}$ | Expected goals for home team $i$ vs away team $j$ |
| $\mu_{ij}$ | Expected goals for away team $j$ vs home team $i$ |
| $c_t(\cdot)$ | Time-adaptive confidence threshold function |
| $\psi_{i,t}$ | Team-specific adjustment factor at time $t$ |
| $w_m(t)$ | Exponential weight for match $m$ at time $t$ |
| $\kappa(t)$ | Market efficiency adjustment factor |
| $\mathcal{P}_{\text{outcome}}$ | Posterior predictive distribution for outcome |
| $S$ | Number of MCMC samples |
| $W$ | Window size for rolling optimization |
| $\lambda$ | Decay rate for exponential weighting |
