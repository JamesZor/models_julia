# Architectural Design: Player-Level Lineup Dynamics & Bench Depth in `BayesianFootball.jl`

> **Document Status:** Complete Design Specification & Research Plan
> **Target Branch:** `feat/player-lineup-dynamics`
> **Directory:** `current_development/player_lineup_dynamics/`
> **Reference Paper:** Kharrat, López Peña & McHale (2019), *Plus-Minus Player Ratings For Soccer*; Hvattum et al. (2020)

---

## 1. Executive Summary & Objective

In standard Bayesian football models (e.g. Maher 1982, Dixon-Coles 1997), team offensive and defensive capabilities are estimated through latent team parameters $\alpha_i$ (attack) and $\beta_i$ (defense). While effective, team-level models cannot react to:
1. **Starting Lineup Shocks:** Missing key goalscorers, tactical rotations, or suspended defenders.
2. **Bench Depth Quality:** Teams with high-quality bench substitutes who gain significant second-half advantages when opponents fatigue.
3. **Mid-Season Roster Turnover:** Loans, January transfers, and returning injured players.

**Objective:** Design and implement a **Pure Player-Level Lineup Dynamics Engine** for `CountModelBuilder` that replaces team-level attack/defense parameters with starting lineup and bench-weighted ratings derived from **Regularized Adjusted Plus-Minus (RAPM)** (e.g. pxG RAPM, Shots RAPM, or SofaScore ratings).

---

## 2. Mathematical Formulations

In all formulations below, team offensive and defensive capabilities are derived directly from the lineup vectors $R_{\text{home}}, R_{\text{away}}$:

$$\begin{aligned}
\log \lambda_{\text{home}} &= \mu_{\text{base}} + \gamma_{\text{ha}} + \text{att}_{\text{home}} + \text{def}_{\text{away}} + \sum_{k} w_k x_{k,\text{home}} \\
\log \lambda_{\text{away}} &= \mu_{\text{base}} + \text{att}_{\text{away}} + \text{def}_{\text{home}} + \sum_{k} w_k x_{k,\text{away}}
\end{aligned}$$

Where:
* $\mu_{\text{base}}$: Base league goal expectancy ($\sim \mathcal{N}(0.2, 0.1)$).
* $\gamma_{\text{ha}}$: Home ground advantage ($\sim \mathcal{N}(0.2, 0.2)$).
* $\text{att}_h$: Offensive rating produced by Home's squad.
* $\text{def}_a$: Defensive concession vulnerability of Away's squad.

---

### Formulation 1: Outfield Starters Sum (The Base Engine)
Sums all 10 outfield starting players ($D + M + F$), explicitly omitting the goalkeeper (as GK RAPM is noise-dominated across backup stints):

$$R_h = \sum_{p \in \text{Starters}_h \setminus \{\text{GK}\}} r_p, \quad R_a = \sum_{p \in \text{Starters}_a \setminus \{\text{GK}\}} r_p$$

$$\begin{aligned}
\text{att}_h &= w_{\text{att}} \cdot R_h, \quad \text{def}_a = - w_{\text{def}} \cdot R_a \\
\text{att}_a &= w_{\text{att}} \cdot R_a, \quad \text{def}_h = - w_{\text{def}} \cdot R_h
\end{aligned}$$

*Priors:* $w_{\text{att}} \sim \mathcal{N}(0.0, 0.3)$, $w_{\text{def}} \sim \mathcal{N}(0.0, 0.3)$ (unconstrained, allowing the model to shrink to zero if ratings carry no signal).

---

### Formulation 2: Starters + Bench-Depth Weighted Sum (The Squad Advantage)
A match is contested by 11 starters plus up to 5 second-half substitutes. A squad with a strong bench maintains high pressing intensity and scoring rate in the final 30 minutes.

$$\tilde{R}_h = \sum_{p \in \text{Starters}_h \setminus \{\text{GK}\}} r_p + \sum_{b \in \text{Bench}_h \setminus \{\text{GK}\}} w_{\text{bench}} \cdot r_b$$

$$\tilde{R}_a = \sum_{p \in \text{Starters}_a \setminus \{\text{GK}\}} r_p + \sum_{b \in \text{Bench}_a \setminus \{\text{GK}\}} w_{\text{bench}} \cdot r_b$$

Where:
* $w_{\text{bench}} \in [0.10, 0.35]$ (e.g. Fixed $0.25$, or learned via prior $w_{\text{bench}} \sim \text{truncated}(\mathcal{N}(0.25, 0.10), 0.0, 1.0)$).
* Reflects that substitutes play an average of $\sim 20\text{--}30\text{ minutes}$ ($22\%\text{--}33\%$ of full match duration).

$$\begin{aligned}
\text{att}_h &= w_{\text{att}} \cdot \tilde{R}_h, \quad \text{def}_a = - w_{\text{def}} \cdot \tilde{R}_a \\
\text{att}_a &= w_{\text{att}} \cdot \tilde{R}_a, \quad \text{def}_h = - w_{\text{def}} \cdot \tilde{R}_h
\end{aligned}$$

---

### Formulation 3: Positional Attack/Defense Lineup Vectors
Splits the outfield starters and bench into positional groups: Defenders ($D$), Midfielders ($M$), and Forwards ($F$):

$$\begin{aligned}
R_{h,F} &= \sum_{p \in \text{Forwards}_h} r_p + w_{\text{bench}} \sum_{b \in \text{Bench\_F}_h} r_b \\
R_{h,M} &= \sum_{p \in \text{Midfielders}_h} r_p + w_{\text{bench}} \sum_{b \in \text{Bench\_M}_h} r_b \\
R_{h,D} &= \sum_{p \in \text{Defenders}_h} r_p + w_{\text{bench}} \sum_{b \in \text{Bench\_D}_h} r_b
\end{aligned}$$

Attack and defense are asymmetric linear combinations:
$$\begin{aligned}
\text{att}_h &= w_{F,\text{att}} R_{h,F} + w_{M,\text{att}} R_{h,M} \\
\text{def}_h &= - w_{D,\text{def}} R_{h,D} - w_{M,\text{def}} R_{h,M}
\end{aligned}$$

*Priors:* $w_{F,\text{att}}, w_{M,\text{att}}, w_{D,\text{def}}, w_{M,\text{def}} \sim \mathcal{N}(0.0, 0.3)$.

---

### Formulation 4: Expected Minute-Weighted Lineup Rating
Weights each player by their expected minutes in the match ($\mathbb{E}[m_p] / 90.0$):

$$R_h = \sum_{p \in \text{Match Squad}_h \setminus \{\text{GK}\}} \left( \frac{\bar{m}_p}{90.0} \right) r_p$$

Where $\bar{m}_p$ is the historical 5-match rolling average of minutes played by player $p$.

---

## 3. Two-Arm Joint Integration (Gamma Proxy xG + Poisson Goals)

When paired with the Two-Arm Joint observation likelihood:
* **Arm 1 (Proxy xG):** $\text{pxG} \sim \text{Gamma}(\nu, \mu / \nu)$ where $\nu \sim \text{truncated}(\mathcal{N}(4.0, 1.5), 0.5, \infty)$.
* **Arm 2 (Goals):** $\text{Goals} \sim \text{Poisson}(\kappa \cdot \mu)$ where $\log \kappa \sim \mathcal{N}(0, 0.2)$.
* **Latents:** $\lambda = \kappa \cdot \mu$.

The player lineup ratings govern $\mu$ directly, propagating player quality into both shot-generation intensity and goal conversion!

---

## 4. Multi-Tier EDA & Empirical Bake-Off Protocol

To evaluate which player lineup formulation provides the strongest predictive power, the research will execute an empirical bake-off across 3 league tiers:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MULTI-TIER EVALUATION GAUNTLET                                                                   │
├──────────────────────────────────────┬───────────────────────────────────┬───────────────────────┤
│ Tier Scope A: England + Scotland     │ Tier Scope B: Scotland All Tiers  │ Tier Scope C: Target  │
│ (Tiers 1, 2, 3, 84, 54, 55, 56, 57)  │ (Tiers 54, 55, 56, 57)            │ Scottish Lower (56/57)│
│ - 9,009 Matches                      │ - 2,009 Matches                   │ - 710 Target Matches  │
│ - Official SofaScore xG & Ratings    │ - Cross-Tier Movers (18% shared)  │ - Pure RAPM Lineups   │
│ - Ground Truth Benchmark             │ - High & Low League Transfer      │ - Primary Production  │
└──────────────────────────────────────┴───────────────────────────────────┴───────────────────────┘
```

### Metrics to Report:
1. **Correlation ($r$) & Rank Correlation ($\rho$)** vs. Out-of-Sample Match Goal Differential and Supremacy.
2. **Held-Out Proper Scoring:** Out-of-Sample LogLoss, CRPS, and Brier Score against Betfair closing market odds.
3. **Information Gain ($R^2$ increment):** Marginal predictive gain over baseline squad wealth and team-level Poisson.
4. **MCMC Latency & Memory:** Sampling speed (ms per iteration) and ESS efficiency under Turing/ReverseDiff.

---

## 5. Julia Component Architecture & Composable Builder Syntax

### New Types in `src/models/pregame/builder/`:

```julia
# 1. Lineup Aggregation Abstract Type & Concrete Strategies
abstract type AbstractPlayerAggregation end

struct OutfieldPlayerAggregation <: AbstractPlayerAggregation end

Base.@kwdef struct BenchWeightedPlayerAggregation <: AbstractPlayerAggregation
    w_bench::Float64 = 0.25
end

struct PositionalPlayerAggregation <: AbstractPlayerAggregation end
struct MinuteWeightedPlayerAggregation <: AbstractPlayerAggregation end

# 2. PlayerLineupDynamics Component
Base.@kwdef struct PlayerLineupDynamics{F<:Features.AbstractFeatureConfig, A<:AbstractPlayerAggregation} <: AbstractDynamicsConfig
    feature::F                  = Features.XGPlusMinusFeature() # pxG RAPM
    aggregation::A              = OutfieldPlayerAggregation()
    w_att_prior::Distribution   = Normal(0.0, 0.3)
    w_def_prior::Distribution   = Normal(0.0, 0.3)
    w_bench_prior::Union{Nothing, Distribution} = nothing
end
```

### Clean Composable Builder Construction:

```julia
using BayesianFootball

# Construct Model Modularly:
model = CountModelBuilder(:player_pxg_rapm_bench_model) |>
    add(GlobalInterception()) |>
    add(GlobalHomeAdvantage()) |>
    add(PlayerLineupDynamics(
        feature     = Features.XGPlusMinusFeature(),
        aggregation = BenchWeightedPlayerAggregation(w_bench = 0.25)
    )) |>
    add(ProductionWealthCovariate()) |>
    add(JointGammaPoissonObservation()) |>
    build
```

---

## 6. Strict Julia Coding Rules & Best Practices (For AI Agents)

When implementing the Julia code, adhere strictly to `docs/guides/julia_coding_context_for_agents.md`:

1. **Zero Runtime Allocations in Likelihoods:**
   * Never allocate dynamic arrays or dictionaries inside Turing model loops.
   * Pre-extract contiguous `Vector{Float64}` rating arrays in `FeatureSet` (`flat_home_outfield_rating`, `flat_away_outfield_rating`, etc.).
2. **ReverseDiff AD Safety:**
   * Avoid mutating global buffers during autodiff tape execution.
   * Use broadcast masking `ifelse.(mask, val, 0.0)` instead of control flow branches on dual numbers.
3. **Type Stability:**
   * Fully parameterize all structs (`struct PlayerLineupDynamics{F, A}`).
4. **Testing Protocol:**
   * All feature extraction and builder assembly must pass 100% unit tests locally before pushing.
