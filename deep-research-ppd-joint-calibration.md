# Comparing PPD Predictions to Realized Binary Outcomes Across a Backtest

**Research question:** For each match in a walk-forward backtest, the model produces a PPD (vector of samples giving p(under_2.5)). We observe a bool for each match. Is there a way to construct a "joint PPD" over all matches and compare to the realized count of successes? How do you properly validate a sequence of PPDs against realized binary outcomes?

---

## Executive Summary

- **Yes, you can build a joint PPD for the total count of successes** — it follows a Poisson binomial distribution, approximated via simulation. Compare your actual count of under_2.5 matches against this distribution as a "season-level posterior predictive check."
- **Your PPD vector already tells you the predicted p per match** — the mean of that vector is the Bayesian probability estimate. This collapses the PPD to a scalar correctly (as established in the previous research: `mean(ppd_samples)`).
- **Four complementary tests** should be run together: reliability diagram, season-level PPC (joint count), LPD/Brier per-line scoring, and a temporal runs test for autocorrelation.
- **Standard PIT / rank histograms don't work well for binary outcomes** — use randomized PIT (jittering) or just use reliability diagrams instead.

---

## Findings

### 1. The Joint PPD for Total Successes — Your Core Question

You asked: "can we create a joint PPD across all under_2.5 events and compare to the realized count?"

**Yes, and here is exactly how:**

For N matches in your backtest, each match i has S posterior samples giving probabilities `p_i^(1), ..., p_i^(S)`. To build the joint predictive distribution for "total successes across the season":

```julia
# For each MCMC draw s:
#   For each match i: simulate one Bernoulli outcome using p_i^(s)
#   Sum the N outcomes → one simulated season total

function season_joint_ppd(ppd_samples::Matrix{Float64})
    # ppd_samples: N_matches × S_posterior_draws
    N, S = size(ppd_samples)
    simulated_totals = zeros(Int, S)
    for s in 1:S
        simulated_totals[s] = sum(rand() < ppd_samples[i, s] for i in 1:N)
    end
    return simulated_totals
end

# Then compare to actual count:
actual_total = sum(realized_outcomes)   # count of under_2.5 = true
predicted_dist = season_joint_ppd(ppd_samples)

# Posterior predictive p-value:
ppp = mean(predicted_dist .>= actual_total)

# Credible interval check:
ci_90 = quantile(predicted_dist, [0.05, 0.95])
```

This is called a **Season-Level Posterior Predictive Check (PPC)**. The resulting `predicted_dist` is the joint predictive distribution the user was asking about — a histogram of "how many under_2.5 matches would we expect, given our model's uncertainty across all N matches."

**What the distribution is:** The sum of N independent Bernoullis with different p_i is a **Poisson binomial distribution**. There's no closed-form formula for heterogeneous p_i, so simulation is the right approach. Source: [Wikipedia: Poisson binomial distribution](https://en.wikipedia.org/wiki/Poisson_binomial_distribution).

**What to look for:**
- Actual count inside the 90% credible interval → model is well-specified at the aggregate level
- Actual count in the lower tail → model is systematically over-predicting under_2.5
- Actual count in the upper tail → model is systematically under-predicting under_2.5
- Actual count always outside CI across multiple seasons → structural bias

---

### 2. Per-Line Calibration: The Reliability Diagram

The season-level PPC tests aggregate bias. The reliability diagram tests whether the model is calibrated at each probability level.

**How it works:**
1. For each match i, compute `p̂_i = mean(ppd_samples[i, :])` — the Bayesian predicted probability
2. Bin all matches by their `p̂_i` (e.g., 10 bins: [0,0.1), [0.1,0.2), ..., [0.9,1.0])
3. For each bin: plot mean `p̂_i` in bin (x-axis) vs. empirical hit rate (y-axis)
4. A perfectly calibrated model lies on the 45° diagonal

```julia
function reliability_diagram(p_hat::Vector{Float64}, outcomes::Vector{Bool}; nbins=10)
    bins = range(0, 1, length=nbins+1)
    x_vals, y_vals, n_per_bin = Float64[], Float64[], Int[]
    for b in 1:nbins
        lo, hi = bins[b], bins[b+1]
        mask = (p_hat .>= lo) .& (p_hat .< hi)
        if sum(mask) > 0
            push!(x_vals, mean(p_hat[mask]))
            push!(y_vals, mean(outcomes[mask]))
            push!(n_per_bin, sum(mask))
        end
    end
    return x_vals, y_vals, n_per_bin
end

p_hat = [mean(ppd_samples[i, :]) for i in 1:N]
x, y, n = reliability_diagram(p_hat, realized_outcomes)
```

**Key point:** For PPD-backed predictions, the predicted probability to use on the x-axis is `mean(ppd_samples[i, :])` — the expectation over the posterior. This is mathematically equivalent to `E_θ[p(under_2.5 | θ)]`, the marginalized Bayesian probability. Using the posterior mean here is correct (as distinct from applying log loss, where you want `log(mean(p))` not `mean(log(p))`).

---

### 3. Per-Line Scoring: LPD and Brier

For match-level scoring (usable for model comparison and A/B testing):

**Brier Score:**
```julia
brier_i = (mean(ppd_samples[i, :]) - realized_outcomes[i])^2
mean_brier = mean(brier_i for i in 1:N)
```

**Log Predictive Density (LPD):**
```julia
# For binary outcome y_i ∈ {0, 1}:
# p_bar = mean probability the event occurred (under_2.5 = true)
function lpd_binary(ppd_samples, outcomes)
    N, S = size(ppd_samples)
    lpd = zeros(N)
    for i in 1:N
        p_bar = mean(ppd_samples[i, :])   # E[p(under_2.5=true | θ)]
        p_obs = outcomes[i] ? p_bar : (1 - p_bar)
        lpd[i] = log(p_obs)
    end
    return lpd
end

elpd = sum(lpd_binary(ppd_samples, realized_outcomes))
```

**Brier score decomposition** (Brier = Reliability + Resolution - Uncertainty):
- **Reliability** (calibration error): how far the reliability curve deviates from the diagonal
- **Resolution** (sharpness): how much the forecasts deviate from the base rate — higher is better
- **Uncertainty**: the base rate variance — fixed by the data, not the model

A model can score better than another on Brier purely by being sharper (more extreme probabilities) even if miscalibrated. Always plot the reliability diagram alongside the score.

---

### 4. What Doesn't Work — PIT and Rank Histograms for Binary

The **Probability Integral Transform (PIT)** normally works by evaluating `F(y)` at the observed outcome and checking for uniformity. For binary outcomes, `F(y)` is a step function with only two values (F(0) and F(1)), so the PIT histogram is degenerate — it can't detect miscalibration the way it does for continuous forecasts.

The **Randomized PIT** (jittering) fixes this: instead of evaluating exactly at F(y), draw a uniform random variable from `[F(y−1), F(y)]`. This recovers approximate uniformity under the null and can be tested with a KS test. But in practice, for binary outcomes, the reliability diagram is simpler and more interpretable.

**Rank histograms** are essentially inapplicable for binary outcomes — ranking a 0 or 1 among an ensemble of 0s and 1s produces severe ties that destroy the rank structure.

---

### 5. Testing for Temporal Autocorrelation (Runs Test)

One thing the above tools miss: whether the model's errors are temporally clustered (e.g., consistently wrong in December, right in August). This matters because walk-forward backtests are temporal sequences.

A **Runs Test** on the binary residuals (was the model wrong on each match?) tests whether errors are more or less clustered than a random sequence:

```julia
# residuals[i] = 1 if model was "surprised" (outcome in tail of PPD), 0 otherwise
# A significant runs test → temporal clustering in errors → model missing some time-varying structure
```

In practice: compute `sign(outcomes[i] - p_hat[i])` for each match, count consecutive runs of the same sign, and compare to the expected number under independence.

---

### 6. Putting It All Together: Recommended Workflow

For a walk-forward backtest with N matches and PPD samples per match:

| Test | What it tells you | Computation |
|---|---|---|
| **Reliability diagram** | Calibration per probability level | `mean(ppd[i,:])` → bin → hit rate |
| **Season-level PPC** | Aggregate count bias | Simulate joint count distribution |
| **LPD** | Full per-match scoring (proper) | `log(mean(ppd[i,:]))` for each match |
| **Brier score** | Decomposable into calibration + sharpness | `(mean(ppd[i,:]) - y_i)^2` |
| **Runs test** | Temporal autocorrelation in errors | Count sign-change runs |

The season-level PPC (your original question) and the reliability diagram are complementary:
- **Reliability diagram** tells you *where* in probability space the model is biased (e.g., over-confident at 70%)
- **Season PPC** tells you if the aggregate count is right even if individual probabilities are right

A model can be perfectly calibrated on the reliability diagram but still put the actual season count in the tail of its joint PPD — this would indicate that the model is treating events as more independent than they actually are (intra-season correlation, structural under-dispersion).

---

## Open Questions

- Whether the Poisson binomial simulation correctly handles cases where matches in the same gameweek are correlated (shared schedule effects) — if they are, the joint count CI will be too narrow and the PPC will have too many "significant" p-values. This is a known limitation of treating N events as independent.
- The specific structure for the user's bivariate Poisson model means the home/away goal counts are correlated per match; but across matches, the independence assumption should hold if teams don't play each other again soon.

## Unverified

- The claim that the runs test is commonly applied to binary prediction sequences in sports forecasting specifically — while well-established in sequential testing generally, no sports-specific citation was confirmed.

---

*Research conducted 2026-06-28 via Antigravity parallel web research.*
