# TP03 verification walkthrough — deliberately stops after Gate 5.
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
TP03_CONTRACT = sl_contract()
TP03_ADAPTER = TP03Adapter(half_life_days=180.0)
TP03_DS = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower(); max_age_hours = 100_000)
TP03_FOLDS = sl_build_folds(TP03_DS, TP03_CONTRACT)

# %% ===========================================================================
# 3. Gates 0--2 — contract, configuration, and leakage-safe features
# ==============================================================================
@assert sl_gate_table("0. Contract", sl_gate_contract(TP03_DS, TP03_FOLDS, TP03_CONTRACT))
@assert sl_gate_table("1. Config", sl_gate_config(TP03_ADAPTER, TP03_CONTRACT))
TP03_G2, TP03_FEATURES = sl_gate_features(TP03_DS, TP03_FOLDS, TP03_ADAPTER, TP03_CONTRACT)
@assert sl_gate_table("2. Features", TP03_G2)

# %% ===========================================================================
# 4. Gate 3 — independent density and AD checks (no MCMC sampling)
# ==============================================================================
@assert sl_gate_table("3a. Equation parity", sl_gate_equation_parity(TP03_ADAPTER, TP03_FEATURES[1]))
TP03_G3B, TP03_GRAD = sl_gate_gradients(TP03_ADAPTER, TP03_FEATURES[1])
@assert sl_gate_table("3b. Gradient health", TP03_G3B)

# %% ===========================================================================
# 5. Gate 4 — synthetic posterior extraction and fallback checks
# ==============================================================================
@assert sl_gate_table("4a. Synthetic extraction", sl_gate_extraction_synthetic(TP03_ADAPTER, TP03_FEATURES[1]))
@assert sl_gate_table("4c. Fallback extraction", sl_gate_extraction_fallbacks(TP03_ADAPTER, TP03_FEATURES[1]))

# %% ===========================================================================
# 6. Gate 5 — synthetic posterior score matrices and market identities
# ==============================================================================
TP03_DRAWS = sl_synthetic_draws(TP03_ADAPTER, Int(TP03_FEATURES[1].data[:n_teams]), 8)
TP03_CHAIN = sl_synthetic_chain(TP03_ADAPTER, TP03_DRAWS; n_chains=2)
TP03_FIXTURES = sl_synthetic_fixtures(TP03_ADAPTER, TP03_FEATURES[1].data[:team_map]; n=6)
TP03_PRICED = sl_extract_parameters(TP03_ADAPTER, TP03_FIXTURES, TP03_FEATURES[1], TP03_CHAIN)
TP03_LATENTS = DataFrame(match_id=Int[], λ_h=Vector{Float64}[], λ_a=Vector{Float64}[])
for row in eachrow(TP03_FIXTURES)
    p = TP03_PRICED[Int(row.match_id)]
    push!(TP03_LATENTS, (Int(row.match_id), p.λ_h, p.λ_a))
end
@assert sl_gate_table("5a. Score dispatch", sl_gate_score_dispatch(TP03_ADAPTER, first(eachrow(TP03_LATENTS)); max_goals=TP03_CONTRACT.max_goals))
@assert sl_gate_table("5b. Score grid", sl_gate_score_grid(TP03_ADAPTER, TP03_LATENTS, TP03_CONTRACT))
@assert sl_gate_table("5c. Market identities", sl_gate_market_identities(TP03_ADAPTER, TP03_LATENTS, TP03_CONTRACT))

# STOP. r01_train_all.jl is the explicit opt-in full-grid launcher after review.
