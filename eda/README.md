# Stochastic Optimal Control & Portfolio Trust Architecture: Research Findings

This directory contains the complete three-part research and empirical audit investigating portfolio-level risk capacity, market selection, directional alpha, and multi-tier conviction staking for the Scottish Lower market universe.

---

## 1. Quick Reference & Core Deliverables

| Part | Topic | Script | Report | Key Takeaway |
|---|---|---|---|---|
| **Part 1** | **Market Capacity Cannibalization & The Shadow Price Law** | `eda/eda_capacity_cannibalization.jl` | [`STOCHASTIC_CONTROL_CAPACITY_REPORT.md`](STOCHASTIC_CONTROL_CAPACITY_REPORT.md) | Fringe lines eat 29% of budget on constrained slates returning -13.2%; removing them yields a +13.1% shadow value. Deep totals suffer from Jensen's inequality tail inflation. |
| **Part 2** | **Asymmetric / Directional Selection Alpha** | `eda/eda_asymmetric_selection_trust.jl` | [`ASYMMETRIC_SELECTION_TRUST_REPORT.md`](ASYMMETRIC_SELECTION_TRUST_REPORT.md) | O/U 2.5 is not symmetric: Under 2.5 is super-alpha (+18.7% ROI), while Over 2.5 is toxic (-11.4% ROI). Driven by the behavioral "public over bias". |
| **Part 3** | **Multi-Tier Conviction & The Scale-Invariance Law** | `eda/eda_multitier_trust.jl` | [`MULTITIER_TRUST_REPORT.md`](MULTITIER_TRUST_REPORT.md) | `SlateDrawdown(23.0)` makes portfolio allocation scale-invariant: absolute trust values are irrelevant; only the Tier 1 : Tier 2 ratio matters. `P1_conservative_tilt` (ratio 1.4) achieves +163.1% return and 1.667 Sharpe. |

---

## 2. Executive Synthesis of Discoveries

### Discovery 1: The Knapsack Capacity Shadow Price ($\lambda_t > 0$)
* In football portfolio construction, simultaneous matches settle together, creating a **joint knapsack problem** with finite risk capacity ($\sum f_i \le C_{\text{slate}}$).
* When slates are capacity-constrained, fringe markets (`O/U 0.5`, `1.5`, `3.5`, and `BTTS`) consume **29.17% of all staked capital**, delivering **-13.19% ROI**, while core 1X2 and O/U 2.5 capital delivers **+12.62% ROI**.
* True opportunity cost is not simply subtracting fringe P&L; re-solving the Kelly problem without fringe lines on constrained dates unlocks **+26.46 percentage points of incremental return**, establishing an empirical shadow price of **+13.1% P&L per unit of fringe stake eliminated**.

### Discovery 2: The Jensen's Inequality Tail Distortion in Deep Totals
* Why did `Under 0.5` lose **-30.70% ROI**?
* Under Poisson intensity with posterior uncertainty, the predictive zero-goal mass is a mixture:
  $$\mathbb{E}[e^{-\Lambda}] \ge e^{-\mathbb{E}[\Lambda]} \quad (\text{via Jensen's inequality})$$
* Posterior variance in $\Lambda$ mechanically inflates predicted goalless draws (model predicted 6.97% vs. 3.36% realized). Because longshot decimal odds are high ($o \approx 15\text{--}25$), small probability errors manufacture fake Kelly edges ($o \cdot p - 1 > 0$), baiting the allocator into taking toxic positions.

### Discovery 3: Asymmetric Alpha in O/U 2.5 (The "Public Over" Bias)
* All 13 directions across 6 markets were scored across 40 walk-forward folds (710 matches, 100 slates).
* **Only four directions possess positive alpha**:
  1. `1X2 Home` (+11.67% ROI, IR 1.15)
  2. `1X2 Draw` (+10.28% ROI, IR 0.36)
  3. `1X2 Away` (+5.63% ROI, IR 0.65)
  4. `O/U 2.5 Under` (**+18.67% ROI**, IR 1.24)
* `Over 2.5` lost **-11.40% ROI**. This is the classic behavioral **"Over Bias"**: retail punters bet goals for entertainment, causing bookmakers to shade Over prices shorter and leaving persistent, harvestable value on Under 2.5.
* "Under" is not a universal factor: expanding to `Under 1.5` and `Under 3.5` degraded performance, confirming the anomaly is isolated to the 2.5 strike where institutional liquidity meets retail sentiment.

### Discovery 4: The Scale-Invariance Law Under `SlateDrawdown`
* The 20% `FixedCap` never binds in any tested policy (`n_capped = 0`).
* Instead, `SlateDrawdown(23.0)` is the sole active constraint, solving for scalar $k_{\text{risk}}$ to normalize portfolio tail loss to a fixed budget.
* Consequence: **Absolute trust levels do not matter; only the ratio between conviction tiers governs the portfolio.**
  - $\tau = (0.40, 0.20)$ and $\tau = (0.50, 0.25)$ both have ratio $2.0$ and produce **bit-for-bit identical portfolios** down to 10 decimal places.
  - Doubling trust simply cuts $k_{\text{risk}}$ in half, leaving realized stakes unchanged.

---

## 3. Evolution of Headline Performance (40 Folds, Model `m12`)

| Stage | Policy Specification | Terminal Return | Annual Sharpe | Sortino | Calmar | Max Drawdown | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| **Status Quo** | 6 markets, flat $\tau = 0.30$ | +123.94% | 1.333 | 1.637 | 5.91 | -20.97% | Baseline before EDA |
| **Symmetric Pruning** | 1X2 + O/U 2.5 (Over & Under) @ $\tau = 0.30$ | +141.53% | 1.485 | 1.839 | 7.15 | -19.80% | Eliminates fringe cannibalization |
| **Asymmetric Core** | 1X2 + Under 2.5 only @ $\tau = 0.30$ | +143.91% | 1.516 | 1.886 | 7.83 | -18.39% | Prunes losing Over 2.5 |
| **Conservative Tilt (P1)** | Tier 1 @ 0.35, Tier 2 @ 0.25 (Ratio 1.4) | **+160.76%** | **1.645** | **2.088** | **8.06** | **-19.95%** | **Production Champion** |
| **Conviction Tilt (P2)** | Tier 1 @ 0.40, Tier 2 @ 0.20 (Ratio 2.0) | +168.98% | 1.742 | 2.183 | 7.93 | -21.31% | Maximum return / higher DD |

*(On sensitivity model `m13_joint_composite`, `P1_conservative_tilt` reached **+165.34% return** and a **1.689 Sharpe**).*

---

## 4. Production Rule & Implementation Specification

For MatchDay operational execution (`src/MatchDay/`) and portfolio simulation (`src/Portfolio/`), the canonical trust model is **`P1_conservative_tilt`**:

```julia
# Canonical Production Trust Specification (Scottish Lower)
trust_spec = SelectionTrust(
    # Tier 1: Super Alpha (Trust = 0.35)
    ("1x2", 0.0, :home)       => 0.35,
    ("over_under", 2.5, :under) => 0.35,

    # Tier 2: Moderate Diversifiers (Trust = 0.25)
    ("1x2", 0.0, :draw)       => 0.25,
    ("1x2", 0.0, :away)       => 0.25,

    # Tier 3: All Other Directions Strictly Gated / Pruned (Trust = 0.00)
    # (Over 2.5, O/U 0.5, O/U 1.5, O/U 3.5, BTTS)
    default = 0.00
)
```

### Operational Rules for Future Work:
1. **Do not re-enable Over 2.5, O/U 0.5, 1.5, 3.5, or BTTS** without a dedicated, verified likelihood re-calibration addressing the Jensen tail inflation.
2. **Do not deploy exploratory probe weights** ($\tau = 0.05$): they generated -$3.90 PnL and 200 unnecessary bets, creating dead commission friction.
3. **Always reason about trust as a ratio between tiers** rather than raw floats, because `SlateDrawdown` absorbs scalar multiples.
