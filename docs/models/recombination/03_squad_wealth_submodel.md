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
For match $m$ with home Starting-XI market value $V_{\text{home}, m}$ and away Starting-XI market value $V_{\text{away}, m}$:
$$\text{Raw Differential: } D_m = \log(V_{\text{home}, m} + \epsilon) - \log(V_{\text{away}, m} + \epsilon)$$
$$\Delta W_m = \frac{D_m - \bar{D}}{\sigma_D} \quad (\text{Standardized to Mean 0, Std 1})$$

### 2. Linear Latent Rate Shift
Squad wealth acts as an additive linear shift inside the log-rate of open-play intensity:
$$\log \mu_{\text{open}, h} = \dots + w_{\text{wealth}} \cdot \Delta W_m$$
$$\log \mu_{\text{open}, a} = \dots - w_{\text{wealth}} \cdot \Delta W_m$$

### 3. Prior Specification
Because greater squad wealth should unambiguously increase chance creation and reduce conceded threat, we use a positive truncated normal prior:
$$w_{\text{wealth}} \sim \text{TruncatedNormal}(0.10, 0.05, a=0.0)$$

---

## 🔍 Feature Extraction Pipeline (`SquadWealthFeature`)

1. **Extractor:** `src/features/extractors/open_play_extractors.jl`
2. **Data Source:** `ds.lineups` filtered by `is_substitute == false`.
3. **Imputation:** If a player has a missing or zero valuation, the tier median is imputed.
4. **Fallback:** If a match has no lineup data, $\Delta W_m = 0.0$ (neutral match).

---

## 📊 Empirical Impact

Including Starting-XI $\Delta W$ in the Scottish Lower division backtest yielded:
- **Wealth Growth:** Increased final bankroll from **3.004x $\to$ 3.147x** on Betfair Exchange.
- **Sharpe Ratio:** Increased Kelly Sharpe from **1.08 $\to$ 1.17** (Rank #1).
- **Max Drawdown:** Reduced peak-to-trough drawdown from **-37.75% $\to$ -32.22%**.
