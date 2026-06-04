# Bernoulli-Gamma Hurdle Columns — Interpretation Guide

## The Model

Every row in your tearsheet represents a specific **(model × market)** slice — e.g. `DixonColes_Market_Hierarchical` on `over_25`. For each slice, the hurdle fits:

$$R_i \sim \begin{cases} \text{Gamma}(\alpha, \beta) & \text{with probability } p \quad \text{(win)} \\ -1 & \text{with probability } 1-p \quad \text{(lose entire stake)} \end{cases}$$

where $R_i = \text{pnl}_i / \text{stake}_i$ is the per-bet ROI.

---

## Column Reference

### `hurdle_p` — Win Probability
**What it is:** The fraction of active bets that were winners (ROI > 0).

**How to use it:**
- Compare `hurdle_p` across models for the same market. If `DixonColes_Market_Hierarchical` has `p=0.48` on `over_25` but `DoublePoisson_NoMarket` has `p=0.47`, the Dixon-Coles model is picking winners slightly more often.
- Compare `hurdle_p` against the implied break-even probability from average odds. If avg odds are 2.0, break-even is 50%. A `hurdle_p` of 0.48 looks bad — but the Gamma tells you *how much* you win when you do win.

> [!TIP]
> `hurdle_p` alone is misleading for profitability. A model with `p=0.30` can be far more profitable than one with `p=0.55` if it's backing longshots with huge positive ROI when correct.

---

### `hurdle_shape` (α) and `hurdle_scale` (β) — Gamma Parameters
**What they are:** The shape and scale of the fitted Gamma distribution over *winning* ROIs only.

- **Mean of winning ROI** = α × β = `hurdle_shape * hurdle_scale`
- **Variance of winning ROI** = α × β²

**How to use them:**

| Pattern | α (shape) | β (scale) | Interpretation |
|---------|-----------|-----------|----------------|
| Consistent small wins | High (>5) | Low (<0.5) | Tight distribution — model bets on near-certainties at short odds. Low variance, predictable. |
| Occasional big wins | Low (<2) | High (>1) | Heavy right tail — model finds rare large-edge bets. High variance, high upside. |
| Balanced | ~2–4 | ~0.3–0.8 | Healthy mix of moderate wins. |

**EDA idea:** Plot the fitted `Gamma(α, β)` density for your top 2–3 models on the same market. If `DixonColes_Market_Hierarchical` has a fatter right tail than `DoublePoisson_Market`, it means the hierarchical model is finding *larger* edges when it does find them.

```julia
using Plots, Distributions
# Compare winning ROI distributions for two models on over_25
d1 = Gamma(shape_model_1, scale_model_1)
d2 = Gamma(shape_model_2, scale_model_2)
plot(x -> pdf(d1, x), 0, 5, label="DC Hierarchical", lw=2)
plot!(x -> pdf(d2, x), 0, 5, label="DP Market", lw=2)
```

---

### `hurdle_E_R` — Parametric Expected ROI
**What it is:** The expected ROI per bet, derived from the fitted distribution:

$$E[R] = p \cdot \mu_{\text{pos}} - (1 - p)$$

where $\mu_{\text{pos}} = \alpha \cdot \beta$ is the mean winning ROI.

**How to use it:**
- This is the *smoothed* version of your empirical `roi_pct / 100`. Because it comes from a fitted distribution rather than raw data, it's more stable for small sample sizes.
- **Positive = profitable model on this market.** Negative = the model is losing money here.
- Unlike raw ROI, this is robust to a single lucky/unlucky outlier bet distorting the average.

> [!IMPORTANT]
> `hurdle_E_R` is the single best column for ranking model-market combinations by expected profitability when you have limited data (e.g., < 50 bets per slice).

---

### `hurdle_sharpe` — Parametric Sharpe Ratio
**What it is:** The signal-to-noise ratio of the model's edge:

$$\text{Sharpe} = \frac{E[R]}{\sigma_R}$$

where $\sigma_R$ is derived from the full hurdle distribution (not just the equity curve).

**How to use it:**
- **Sharpe > 0.3** → Strong, reliable edge. The model consistently finds value.
- **Sharpe 0.1–0.3** → Edge exists but is noisy. Needs volume to realise profits.
- **Sharpe < 0.1** → Edge is indistinguishable from noise. Avoid this model-market combination.
- **Negative** → Model is losing money.

**Key difference from the existing `SharpeRatio` column:** The existing Sharpe in your tearsheet is computed from the equity curve (cumulative PnL diffs). The hurdle Sharpe is computed from the *fitted distributional model* of per-bet ROI. The hurdle version is:
1. More robust to bet ordering (equity curve Sharpe is path-dependent).
2. More interpretable for comparing across markets with different bet frequencies.

---

### `hurdle_G` — Parametric Geometric Growth Rate ⭐
**What it is:** The expected *compounded* growth rate per bet, estimated via Monte Carlo:

$$G = \exp\!\Big((1-p)\cdot\log(1 - \bar{f}) + p \cdot \mathbb{E}\big[\log(1 + \bar{f} \cdot Y)\big]\Big) - 1$$

where $Y \sim \text{Gamma}(\alpha, \beta)$ and $\bar{f}$ is the average Kelly stake fraction.

**Why this is your most important column:**
- This is the **Kelly criterion's native metric**. A positive `hurdle_G` means your Kelly staking is growing your bankroll at that rate per bet. A negative `hurdle_G` means you're destroying wealth.
- It correctly accounts for the *asymmetry of compounding*: losing 50% requires gaining 100% to recover. Simple ROI misses this entirely.
- It's the only metric that properly answers: **"If I keep betting this model on this market forever, will I go broke or get rich?"**

**How to use it:**
- `hurdle_G > 0` → This model-market combination grows your bankroll. Deploy it.
- `hurdle_G ≈ 0` → Break-even after compounding. Not worth the risk.
- `hurdle_G < 0` → Wealth destruction. Kill this market for this model.

> [!CAUTION]
> A model can have positive `hurdle_E_R` (positive average ROI) but **negative `hurdle_G`** (negative growth). This happens when the variance is too high relative to the edge — the compounding drag from large losses outweighs the arithmetic average of wins. This is the classic "overbetting" signal.

**EDA idea — The Money Table:**
```julia
# Filter to only profitable growth markets
profitable = subset(tearsheet, :hurdle_G => ByRow(>(0.0)))
sort!(profitable, :hurdle_G, rev=true)
show(profitable[!, [:model_name, :selection, :hurdle_G, :hurdle_sharpe, :hurdle_p, :hurdle_n_bets]])
```

---

### `hurdle_G_emp` — Empirical Geometric Growth Rate
**What it is:** The same growth rate concept, but computed directly from the raw data:

$$G_{\text{emp}} = \exp\!\Big(\frac{1}{n}\sum_{i=1}^{n} \log(1 + s_i \cdot r_i)\Big) - 1$$

where $s_i$ is the actual stake and $r_i$ is the actual ROI for bet $i$.

**How to use it:**
- Compare `hurdle_G` (parametric) against `hurdle_G_emp` (empirical) as a **goodness-of-fit diagnostic**.
- If they agree closely → The Bernoulli-Gamma model is a good fit for this market's ROI distribution.
- If they diverge significantly → The distribution shape may not be well-captured by a Gamma (e.g., bimodal winning ROIs from mixing short-odds and long-odds bets).

---

### `hurdle_n_bets` — Active Bet Count
**What it is:** Number of bets with stake > 0 in this group.

**How to use it:**
- **Statistical significance filter.** Treat any slice with `hurdle_n_bets < 30` with extreme caution — the fitted parameters will be noisy.
- Cross-reference with `hurdle_sharpe`: a Sharpe of 0.5 on 200 bets is gold. A Sharpe of 0.5 on 8 bets is noise.

---

### `hurdle_avg_stake` — Mean Stake Fraction
**What it is:** The average Kelly stake size across active bets in this group.

**How to use it:**
- Reveals how *confident* your model is in this market. A high `hurdle_avg_stake` means the model is finding large edges and staking aggressively.
- If `hurdle_avg_stake` is high but `hurdle_G` is low/negative, the model is **overbetting** — it thinks it has a big edge but doesn't. This is a critical risk signal.
- Compare across models: if `DixonColes_Market_Hierarchical` achieves similar `hurdle_G` to `DoublePoisson_Market` but with *lower* `hurdle_avg_stake`, the Dixon-Coles model is more capital-efficient.

---

## Recommended EDA Workflow

### Step 1: The Growth Filter
```julia
# Which model-market combos actually grow wealth?
growth_table = subset(tearsheet, :hurdle_G => ByRow(>(0.0)), :hurdle_n_bets => ByRow(>=(30)))
sort!(growth_table, :hurdle_G, rev=true)
cols = [:model_name, :selection, :hurdle_G, :hurdle_G_emp, :hurdle_sharpe, :hurdle_E_R, :hurdle_p, :hurdle_n_bets]
show(growth_table[!, cols], allrows=true)
```

### Step 2: The Overbetting Diagnostic
```julia
# Positive expected ROI but negative growth = overbetting
overbets = subset(tearsheet, :hurdle_E_R => ByRow(>(0.0)), :hurdle_G => ByRow(<(0.0)))
show(overbets[!, [:model_name, :selection, :hurdle_E_R, :hurdle_G, :hurdle_avg_stake]])
```

### Step 3: Model Comparison on Core Markets
```julia
core_markets = [:over_25, :under_25, :home, :draw, :away, :btts_yes, :btts_no]
core = subset(tearsheet, :selection => ByRow(in(core_markets)))
for mkt in core_markets
    println("\n=== $mkt ===")
    sub = subset(core, :selection => ByRow(==(mkt)))
    sort!(sub, :hurdle_G, rev=true)
    show(sub[!, [:model_name, :hurdle_G, :hurdle_sharpe, :hurdle_p, :hurdle_avg_stake]], allrows=true)
end
```

### Step 4: Distribution Shape Analysis
```julia
# Which markets have the "best" winning ROI distribution?
# High shape + moderate scale = consistent, bankable wins
show(sort(tearsheet[!, [:model_name, :selection, :hurdle_shape, :hurdle_scale, :hurdle_p, :hurdle_G]], :hurdle_shape, rev=true), allrows=true)
```
