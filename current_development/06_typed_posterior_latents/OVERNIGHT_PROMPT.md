# BRIEFING: TYPED POSTERIOR LATENTS PROTOTYPE (`06_typed_posterior_latents`)

> **Objective:** Build, test, and verify the `AbstractPosteriorLatents` architecture in `current_development/06_typed_posterior_latents/` to replace the slow, un-typed `latents.df` DataFrame storage with zero-allocation, typed dense matrix containers across all model families.

---

## 1. Problem Statement & Motivation

Currently, `BayesianFootball` extracts posterior parameter samples for out-of-sample (OOS) fixtures into a `DataFrame` (`latents.df`) where high-dimensional MCMC sample vectors (e.g., 3,200 posterior draws per fixture) are stored inside individual DataFrame cells.

### Why `latents.df` Must Be Replaced:
1. **Type Instability & Boxing**: Storing arrays inside DataFrame cells degrades column types to `Vector{Any}` or nested arrays, causing massive heap allocations.
2. **I/O & Memory Churn**: Serializing large DataFrames to disk and unpacking them row-by-row during score matrix calculation is slow.
3. **Inflexible Ad-Hoc Columns**: Different models invent different column names (`:lambda_home`, `:true_xg_h`, `:phi`, `:w_wealth`).

---

## 2. Target Architecture

Build the following modular files in `current_development/06_typed_posterior_latents/`:

```
current_development/06_typed_posterior_latents/
├── l01_latents.jl       # AbstractPosteriorLatents type hierarchy
├── l02_extract.jl       # extract_latents(model, chain, oos_df, ds)
├── l03_score_grids.jl   # compute_score_grid(latents, match_idx) & price_market
├── l04_parity.jl        # Mathematical parity harness vs legacy latents.df
└── r01_demo.jl          # Deterministic verification runner & allocation benchmarks
```

---

## 3. Component Details & Multiple Dispatch Contracts

### 3.1 Type Hierarchy (`l01_latents.jl`)
```julia
abstract type AbstractPosteriorLatents end

# 1. Standard Count Models (Poisson, NegBin, Wealth, Distance from 05_composable_count_builder)
struct CountLatents{T<:Real, Obs} <: AbstractPosteriorLatents
    match_ids::Vector{Int}
    λ_home::Matrix{T}                 # (n_matches, n_draws)
    λ_away::Matrix{T}                 # (n_matches, n_draws)
    observation_params::Obs           # `nothing` for Poisson, or (; r_h, r_a) for NegBin
end

# 2. Recombination Models (DynamicPxGRecombModel: Open Play + Penalty + Own Goals + pxG)
struct RecombLatents{T<:Real} <: AbstractPosteriorLatents
    match_ids::Vector{Int}
    λ_open_h::Matrix{T}               # (n_matches, n_draws)
    λ_open_a::Matrix{T}
    λ_pen_h::Matrix{T}
    λ_pen_a::Matrix{T}
    λ_og_h::Matrix{T}
    λ_og_a::Matrix{T}
    pxg_h::Matrix{T}
    pxg_a::Matrix{T}
end

# 3. Market Smile Models (from current_development/smile_negbin/)
struct SmileLatents{T<:Real, Obs} <: AbstractPosteriorLatents
    match_ids::Vector{Int}
    λ_home::Matrix{T}                 # (n_matches, n_draws)
    λ_away::Matrix{T}                 # (n_matches, n_draws)
    observation_params::Obs           # `nothing` for Poisson, or (r_h, r_a) for NegBin
    λ_tot::Matrix{T}                  # (n_matches, n_draws) - market total intensity
    φ::Array{T, 3}                    # (n_matches, n_strikes, n_draws) - smile shape curve
    strikes::Vector{Float64}          # [0.5, 1.5, 2.5, 3.5, 4.5]
end
```

### 3.2 Extraction Layer (`l02_extract.jl`)
* Implement `extract_latents(model, chain, oos_fixtures, ds)` for:
  1. `CompiledCountModel` (from `05_composable_count_builder`) $\to$ returns `CountLatents`.
  2. `DynamicPxGRecombModel` $\to$ returns `RecombLatents`.
  3. `DynamicSmileDoublePoisson...` / `DynamicSmileDoubleNegBin...` $\to$ returns `SmileLatents`.

### 3.3 Score Grids & Market Pricing (`l03_score_grids.jl`)
* Implement `compute_score_grid(latents, match_idx; max_goals=12)`:
  * `CountLatents`: Outer product of Poisson or Negative Binomial marginals.
  * `RecombLatents`: Discrete Poisson convolution of open play, penalties, and own goals.
  * `SmileLatents`: Computes $(12 \times 12)$ discrete grid and returns `SmileScoreMatrix(grid, λ_tot, φ, strikes)`.
* Implement `price_market(grid_or_container, market)`:
  * `Market1X2`: Home Win, Draw, Away Win probabilities.
  * `MarketBTTS`: BTTS Yes/No probabilities.
  * `MarketOverUnder`: Over/Under probabilities (using smile curve $\Lambda(K) = \lambda_{\text{tot}} \cdot \phi(K)$ for `SmileScoreMatrix`).

### 3.4 Verification & Parity Harness (`l04_parity.jl` & `r01_demo.jl`)
* Verify **bit-identical mathematical parity** ($| \Delta | < 10^{-12}$) between `AbstractPosteriorLatents` and legacy `latents.df`.
* Verify **0 heap allocations** during score grid calculation on contiguous slices.
* Output a clean summary table of timing, memory, and parity.

---

## 4. Execution Rules
* **Deterministic Only**: Do NOT run heavy MCMC grids on `archpc`—use synthetic parameters or cached chains.
* **Keep Code Clean**: Follow the Loader/Runner pattern.
