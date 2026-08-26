# TP04 verification walkthrough — deliberately stops after Gate 5.
# Question: does this optional pregame covariate preserve the pure-Poisson
# contract from feature construction through posterior score matrices?
# This is not a full-fold or betting experiment.

# %% ===========================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using DataFrames
using Dates
include(joinpath(@__DIR__, "..", "_protocol", "ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol
include(joinpath(@__DIR__, "l01_model.jl"))
include(joinpath(@__DIR__, "l02_equations.jl"))
include(joinpath(@__DIR__, "l03_adapter.jl"))

# %% ===========================================================================
# 2. Configuration and data snapshot
# ==============================================================================
TP04_CONTRACT = sl_contract()
TP04_ADAPTER = TP04Adapter(half_life_days=180.0)
TP04_DS = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower(); max_age_hours = 100_000)
TP04_FOLDS = sl_build_folds(TP04_DS, TP04_CONTRACT)

# %% ===========================================================================
# 3. Gates 0--2 — contract, configuration, and leakage-safe features
# ==============================================================================
@assert sl_gate_table("0. Contract", sl_gate_contract(TP04_DS, TP04_FOLDS, TP04_CONTRACT))
@assert sl_gate_table("1. Config", sl_gate_config(TP04_ADAPTER, TP04_CONTRACT))
TP04_G2, TP04_FEATURES = sl_gate_features(TP04_DS, TP04_FOLDS, TP04_ADAPTER, TP04_CONTRACT)
@assert sl_gate_table("2. Features", TP04_G2)

# %% ===========================================================================
# 4. Gate 3 — independent density and AD checks (no MCMC sampling)
# ==============================================================================
@assert sl_gate_table("3a. Equation parity", sl_gate_equation_parity(TP04_ADAPTER, TP04_FEATURES[1]))
TP04_G3B, TP04_GRAD = sl_gate_gradients(TP04_ADAPTER, TP04_FEATURES[1])
@assert sl_gate_table("3b. Gradient health", TP04_G3B)

# %% ===========================================================================
# 5. Gate 4 — synthetic posterior extraction and fallback checks
# ==============================================================================
@assert sl_gate_table("4a. Synthetic extraction", sl_gate_extraction_synthetic(TP04_ADAPTER, TP04_FEATURES[1]))
@assert sl_gate_table("4c. Fallback extraction", sl_gate_extraction_fallbacks(TP04_ADAPTER, TP04_FEATURES[1]))

# %% ===========================================================================
# 6. Gate 5 — synthetic posterior score matrices and market identities
# ==============================================================================
TP04_DRAWS = sl_synthetic_draws(TP04_ADAPTER, Int(TP04_FEATURES[1].data[:n_teams]), 8)
TP04_CHAIN = sl_synthetic_chain(TP04_ADAPTER, TP04_DRAWS; n_chains=2)
TP04_FIXTURES = sl_synthetic_fixtures(TP04_ADAPTER, TP04_FEATURES[1].data[:team_map]; n=6)
TP04_PRICED = sl_extract_parameters(TP04_ADAPTER, TP04_FIXTURES, TP04_FEATURES[1], TP04_CHAIN)
TP04_LATENTS = DataFrame(match_id=Int[], λ_h=Vector{Float64}[], λ_a=Vector{Float64}[])
for row in eachrow(TP04_FIXTURES)
    p = TP04_PRICED[Int(row.match_id)]
    push!(TP04_LATENTS, (Int(row.match_id), p.λ_h, p.λ_a))
end
@assert sl_gate_table("5a. Score dispatch", sl_gate_score_dispatch(TP04_ADAPTER, first(eachrow(TP04_LATENTS)); max_goals=TP04_CONTRACT.max_goals))
@assert sl_gate_table("5b. Score grid", sl_gate_score_grid(TP04_ADAPTER, TP04_LATENTS, TP04_CONTRACT))
@assert sl_gate_table("5c. Market identities", sl_gate_market_identities(TP04_ADAPTER, TP04_LATENTS, TP04_CONTRACT))

# STOP. r01_train_all.jl is the explicit opt-in full-grid launcher after review.
