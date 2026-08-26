# 3. Starting-XI Squad Wealth Submodel

**Module:** `BayesianFootball.Models.PreGame`  
**Related Structs:** `LinearSquadWealthConfig`, `NoSquadWealthConfig`, `SquadWealthFeature`

---

## 🎯 Motivation

In domestic cup competitions, promotion/relegation playoffs, and transfer windows, team rosters experience significant talent shifts that static team indices cannot immediately track. 

By calculating the total Transfermarkt valuation of the **11 confirmed starting players** for each team, we obtain an instantaneous proxy for lineup quality and squad depth differential.

---

## 📐 Mathematical Formulation

### 1. Match Wealth Differential ($\Delta W_m$)
For match $m$, let $L_{h,m}$ and $L_{a,m}$ be the mean log values of the confirmed home and away starters after fixed-value player imputation:
$$D_m = L_{h,m} - L_{a,m}$$
$$\Delta W_m = \frac{D_m}{s_W}$$

Here $s_W$ is the fixed `SquadWealthFeature.log_scale` configuration (default 1.0). It is deliberately **not** estimated from the fixture sample: adding or deleting future matches must not change historical features.

### 2. Linear Latent Rate Shift
Squad wealth acts as an additive linear shift inside the log-rate of open-play intensity:
$$\log \mu_{\text{open}, h} = \dots + w_{\text{wealth}} \cdot \Delta W_m$$
$$\log \mu_{\text{open}, a} = \dots - w_{\text{wealth}} \cdot \Delta W_m$$

### 3. Prior Specification
Because greater squad wealth should unambiguously increase chance creation and reduce conceded threat, we use a positive truncated normal prior:
$$w_{\text{wealth}} \sim \text{TruncatedNormal}(0.10, 0.05, a=0.0)$$

---

## 🔍 Feature Extraction Pipeline (`SquadWealthFeature`)

1. **Extractor:** `src/features/extractors/open_play_extractors.jl`.
2. **Data Source:** Only match-scoped rows in `ds.lineups` for the requested fold IDs, filtered by `is_substitute == false`.
3. **Point-in-time contract:** By default each observed value requires a valuation timestamp strictly preceding kickoff. Values from global/latest-player catalogs are not backfilled into historical matches.
4. **Player imputation:** Missing or invalid players use the fixed `fallback_default` value.
5. **Match fallback:** Both clubs must have at least one timestamp-safe observed value. Otherwise $\Delta W_m = 0.0$ and `flat_wealth_fallback = 1`.
6. **Perturbation safety:** No normalization moment or lookup map is fitted over future fixture rows.

---

## 📊 Empirical Impact

Including Starting-XI $\Delta W$ in the Scottish Lower division backtest yielded:
- **Wealth Growth:** Increased final bankroll from **3.004x $\to$ 3.147x** on Betfair Exchange.
- **Sharpe Ratio:** Increased Kelly Sharpe from **1.08 $\to$ 1.17** (Rank #1).
- **Max Drawdown:** Reduced peak-to-trough drawdown from **-37.75% $\to$ -32.22%**.
