# Bayesian Scoring of Posterior Predictive Distributions

**Research question:** For a Bayesian model outputting a full PPD (a vector of samples representing p(w)), is there a proper Bayesian scoring rule that uses the whole distribution rather than collapsing to the mean first? Is the "histogram → area → log loss" idea valid?

---

## Executive Summary

- **Yes, your histogram intuition is exactly right.** The proper Bayesian log score computes `log(area under PPD at the observed outcome)`, which for discrete outcomes simplifies to `log(mean probability over posterior samples)`.
- The correct formula from MCMC samples is **LPD = log(1/S · Σ p(y | θ^s))** — log of the *average* likelihood, not the average of the log likelihoods.
- This is called the **Log Predictive Density (LPD)** and its negation is the Bayesian log loss. Classical log loss (applied to a point estimate) is a biased approximation of it.
- Jensen's inequality proves that `log(E[p]) > E[log p]` — so averaging log-likelihoods is always pessimistic relative to the true LPD.
- For discrete outcomes (H/D/A football results), no histogram or KDE is needed — averaging the class probability over chains is exact.
- For continuous outcomes, **CRPS** is often the more robust alternative to log score because it doesn't require density estimation.

---

## Findings

### 1. The Proper Bayesian Log Score (LPD)

The full Bayesian predictive distribution marginalises out parameter uncertainty:

```
p(y_new | y_observed) = ∫ p(y_new | θ) p(θ | y_observed) dθ
```

In MCMC, this integral is approximated by averaging over S posterior draws `{θ^(1), ..., θ^(S)}`:

```
p(y_new | y_observed) ≈ (1/S) Σ_s p(y_new | θ^(s))
```

The **Log Predictive Density** is then:

```
LPD(y_new) = log[ (1/S) Σ_s p(y_new | θ^(s)) ]
```

And the Bayesian log loss is simply `-LPD`.

This is the proper Bayesian scoring rule: it evaluates the *entire mixture* of predictions (one per posterior draw), not any single draw. Source: [Vehtari et al. 2017 (arXiv:1507.04544)](https://arxiv.org/abs/1507.04544).

### 2. Your Histogram Intuition Decoded

You described: "make a histogram of the PPD samples, compute the area in the bin containing the observed outcome, then log-loss the area."

This is correct in principle and maps precisely to the LPD:

| Your language | Technical term |
|---|---|
| PPD samples (vector of numbers) | Monte Carlo draws from the posterior predictive |
| Histogram / bin the numbers | Density estimation over PPD |
| Compute the area in the observed bin | Estimate p(y_observed) from the predictive distribution |
| Log loss the area | `-log(p(y_observed))` = Bayesian log loss |

**For discrete outcomes** (H/D/A football results), the histogram step is unnecessary. Each posterior draw θ^(s) gives you a closed-form categorical probability `p(class | θ^(s))`. Averaging those probabilities *is* the area computation — no binning needed:

```
p(H | y_observed) ≈ (1/S) Σ_s p(H | θ^(s))
LPD = log( p(H | y_observed) )
```

**For continuous outcomes** (e.g. goal totals), when the PPD is a vector of draws `ỹ^(s)` rather than a vector of closed-form densities, you DO need a density estimation step (histogram or KDE) to evaluate the density at the actual observed value.

### 3. Three Candidate Formulas and Which Is Correct

For discrete outcomes (the football 3-class case), three approaches are often confused:

| Method | Formula | Proper? |
|---|---|---|
| **(a)** Average probabilities, then log-loss | `-log( (1/S) Σ p(class\|θ^s) )` | **YES — this is the LPD** |
| **(b)** Log of average probability | `log( (1/S) Σ p(class\|θ^s) )` | **YES — same thing, opposite sign** |
| **(c)** Average of log-probabilities | `(1/S) Σ log p(class\|θ^s)` | **NO — underestimates LPD** |

**(a) and (b) are mathematically identical** — one is a "score" (higher is better), the other is a "loss" (lower is better). Both correctly score the full predictive mixture.

**(c) is wrong** because of **Jensen's inequality**: since log is concave, `log(E[X]) ≥ E[log(X)]`. Method (c) always gives a more pessimistic score than the true LPD. It penalises individual posterior draws that are confidently wrong, rather than rewarding the model for spreading its uncertainty correctly.

Intuitively: the model is saying "I'm not sure, but here are 1000 plausible parameter values". The correct score asks "what probability does this *committee* assign to the outcome?" (average then log). The wrong score asks "on average, how confident was each *individual committee member*?" (average of logs).

### 4. Numerical Computation (log-sum-exp trick)

When computing LPD in code, raw likelihoods underflow to zero on log scales. The stable formula using the **log-sum-exp trick**:

```julia
# For each observation y, given S posterior samples of log-likelihoods:
log_liks = [logpdf(model(θ_s), y) for θ_s in posterior_samples]   # S values

# Numerically stable log-mean-exp:
lmax = maximum(log_liks)
lpd = lmax + log(mean(exp.(log_liks .- lmax)))
```

This is equivalent to `log((1/S) Σ exp(log_lik_s))` but avoids float underflow. Source: [LogSumExp (Wikipedia)](https://en.wikipedia.org/wiki/LogSumExp).

For histogram/KDE approaches on continuous PPD samples, the key pitfalls are:
- **Too-narrow bins** → empty bins → `-Inf` log scores
- **Too-wide bins** → over-smoothing, biased density estimate
- **KDE** is more robust but bandwidth selection matters; still struggles in high dimensions

### 5. Proper Scoring Rules — Full Landscape

| Score | Use case | Needs density estimation? | Distance-sensitive? |
|---|---|---|---|
| **Log Score (LPD)** | Any outcome; model comparison | For continuous PPD samples, yes | No (local: evaluates only at observed point) |
| **CRPS** | Continuous univariate outcomes | **No** (computed from samples directly) | **Yes** (penalises near-misses less) |
| **RPS (Ranked Probability Score)** | Ordered categorical (e.g. H/D/A) | No | Yes |
| **Brier Score** | Binary or multi-class | No | No |
| **Energy Score** | Multivariate continuous | No | Yes |

**For your football model specifically:**
- H/D/A result: use **Log Score** (formula (a)/(b) above) or **RPS** (accounts for ordinal distance H→D→A)
- Goal totals (continuous): use **CRPS** — it works directly on MCMC samples without KDE and is more robust
- Both are proper scoring rules

### 6. WAIC and LOO-CV (Standard Bayesian Model Comparison)

The standard Bayesian workflow aggregates the LPD across all N observations:

```
ELPD = Σ_i log[ (1/S) Σ_s p(y_i | θ^(s)) ]
```

This is the **Expected Log Pointwise Predictive Density**, estimated either by:

- **WAIC**: Computes LPD from full-data posterior, subtracts a variance penalty `Σ_i Var_s[log p(y_i|θ^(s))]` to estimate out-of-sample performance. Can fail silently.
- **PSIS-LOO**: Approximates leave-one-out CV via importance sampling without refitting. Provides **Pareto-k diagnostics**: `k > 0.7` means the approximation is unreliable for that observation (usually an outlier or model misspecification). Preferred over WAIC.

In `Turing.jl`, `ParetoSmooth.jl` implements PSIS-LOO. In Python, the `arviz` package does the same.

---

## Open Questions / Disagreements

- Workers agree that RPS is appropriate for ordered H/D/A outcomes, but no source explicitly validates it for a posterior-averaged setting. The formula `RPS = (1/S) Σ_s RPS(F_s, y)` should work by linearity, but double-check.
- For your bivariate Poisson / copula goals models (continuous latent → discrete score), the cleanest approach depends on whether you score the *marginal class probabilities* (use log score on averaged H/D/A probs) or the *joint score distribution* (use log score on averaged joint PMF). Both are proper; they measure different things.

## Unverified

- The claim that RPS (discrete CRPS for ordered categories) is strictly proper for Bayesian averaged predictions was not independently corroborated across two sources. Treat as very likely true but check before publishing.

---

## Bottom Line for Your Model

Given your PPD is a vector of MCMC samples `θ^(1..S)`, each giving categorical probs `p(H|θ^s), p(D|θ^s), p(A|θ^s)`:

```julia
# Correct Bayesian log score for match i with actual outcome k ∈ {H,D,A}
p_bar_k = mean(p_k_given_theta[s] for s in 1:S)   # average over chains
lpd_i = log(p_bar_k)                               # LPD for this match
log_loss_i = -lpd_i                                 # loss formulation
```

This is strictly proper, uses the full PPD, requires no histogram, and is exactly what your histogram intuition was aiming at.

---

*Research conducted 2026-06-28 via Antigravity parallel web research. Primary source: Vehtari, Gelman, Gabry (2017) "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC" — arXiv:1507.04544.*
