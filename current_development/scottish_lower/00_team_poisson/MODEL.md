# Model 00 — Pure Team-level Poisson Baseline

**Engine:** `DynamicPoissonGoalsTimeDecayModel` (defined in `l01_model.jl`).
**Role:** The minimal pregame goals model. Equidispersed baseline ($\text{Var}(Y) = \text{E}[Y] = \lambda$).
Judged against `01_team_poisson` (Negative Binomial, $r \approx 22$) on identical fixtures and markets.

---

## What it is

A hierarchical attack/defence rating model on full-time goals with exponential time decay on the likelihood weights. Goals are modelled as **pure Poisson** rather than Negative Binomial:

$$\text{Var}(Y) = \text{E}[Y] = \lambda$$

The likelihood is evaluated directly in **log-intensity space** ($\eta = \log \lambda$), eliminating redundant $\exp \to \log$ round-trips and minimizing the ReverseDiff computation tape instructions.

---

## What it is NOT

- **Not Negative Binomial**: There is no dispersion parameter $r$ or `disp.log_r`.
- **Not market-anchored**: Sees only historical match scores and dates.
- **Not decomposed**: Gross goals (including penalties and own goals) are modelled directly.
- **Not player-aware**: Team-level only.
- **Not correlated**: Home and away goals are conditionally independent given intensities $\lambda_h, \lambda_a$ (no Dixon-Coles or Copula correlation).

---

## Equations

For match $i$ between home team $h(i)$ and away team $a(i)$ in season $s(i)$ and calendar month $m(i)$:

### 1. Log-intensities (Linear Predictors)
```
η_h(i) = μ_base[s(i)] + δ_month[m(i)] + γ_h(i) + α_h(i) + β_a(i)
η_a(i) = μ_base[s(i)] + δ_month[m(i)]           + α_a(i) + β_h(i)
```

Intensities in natural space:
```
λ_h(i) = exp(η_h(i))
λ_a(i) = exp(η_a(i))
```

### 2. Team Ratings
Team attack ($\alpha$) and defence leak ($\beta$) are non-centred and zero-sum:

```
α_scaled = raw_a .* σ_a        α = α_scaled .- mean(α_scaled)
β_scaled = raw_d .* σ_d        β = β_scaled .- mean(β_scaled)

raw_a, raw_d ~ Normal(0, 1)^n_teams
σ_a ~ Gamma(2.0, 0.15)         σ_d ~ Gamma(2.0, 0.15)
```

`α` is attacking strength, `β` is defensive leak (enters the opponent's intensity).

### 3. Log-Poisson Likelihood (with exponential time decay)
With half-life $H$ days and $\Delta_i$ days before the cutoff boundary:
```
w_i = 0.5 ^ (Δ_i / H)

log p(y_h(i) | η_h(i)) = y_h(i) · η_h(i) − exp(η_h(i)) − log(y_h(i)!)
log p(y_a(i) | η_a(i)) = y_a(i) · η_a(i) − exp(η_a(i)) − log(y_a(i)!)

logprob += Σ_i w_i · log p(y_h(i) | η_h(i)) + Σ_i w_i · log p(y_a(i) | η_a(i))
```

---

## Component Menu

| Slot | Default | Alternatives |
|---|---|---|
| Interception | `GlobalInterception(μ = Normal(0.2, 0.1))` | `SeasonalInterception`, `HierarchicalMonthlyInterception` |
| Dispersion | **None** (Pure Poisson) | `GlobalDispersion` (see Model 01) |
| Home advantage | `GlobalHomeAdvantage(γ_global = Normal(0.2, 0.2))` | `HierarchicalTeamHomeAdvantage`, `HierarchicalLeagueHomeAdvantage` |
| Dynamics | `TimeDecayDynamics(days_half_life = 180)` | any half-life; `σ_att`/`σ_def` priors |

---

## Required Features

`Features.required_features` returns:
- `TeamIDsFeature`
- `GoalsFeature`
- `DatesFeature`
- `MonthFeature`
- `TimeIndicesFeature`

---

## Sampled Sites Manifest

| Component | Sampled sites | Count |
|---|---|---|
| Interception | `inter.μ` | 1 |
| Home advantage | `ha.γ_global` | 1 |
| Dynamics | `dyn.σ_a`, `dyn.σ_d` | 2 |
| Team ratings | `dyn.raw_a[1:N]`, `dyn.raw_d[1:N]` | $2 \times N_{\text{teams}}$ |
| **Total** | | $4 + 2 \times N_{\text{teams}}$ (50 for 23 teams) |

---

## Differences vs Model 01 (`01_team_poisson`)

1. **Parameters**: 50 parameters vs 51 in Model 01 (no `disp.log_r`).
2. **Likelihood Computation**: Direct log-Poisson evaluation avoids constructing Negative Binomial distributions and $\exp \to \log$ round-trips.
3. **Score Matrix Dispatch**: Dispatches to `src/predictions/score_computation/poisson.jl` producing exact Poisson joint score matrices.
