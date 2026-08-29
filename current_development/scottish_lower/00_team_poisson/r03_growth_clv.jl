# ==============================================================================
# MODEL 00 — PURE POISSON BASELINE : GATE 7 GROWTH AND CLV
# ==============================================================================
# Builds Portfolio-Kelly books from an accepted persisted grid and Betfair closing
# odds. It does not fit a model. Set ENV["TP00_GRID_PATH"] to r01's artifact.

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using DataFrames
using Statistics

const TP00_ROOT = let d = dirname(@__DIR__)
    isfile(joinpath(d, "_protocol", "ScottishLowerProtocol.jl")) ? d :
        "current_development/scottish_lower"
end
include(joinpath(TP00_ROOT, "_protocol/ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol
include(joinpath(TP00_ROOT, "00_team_poisson/l01_model.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l02_equations.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l03_adapter.jl"))

# %%
# ==============================================================================
# 2. Configuration, persisted grid, and OOS posterior
# ==============================================================================
TP00_CONTRACT = sl_contract()
TP00_ADAPTER = TP00Adapter(half_life_days = 180.0)
TP00_GRID_PATH = get(ENV, "TP00_GRID_PATH", "")
@assert !isempty(TP00_GRID_PATH) "Set ENV[\"TP00_GRID_PATH\"] to an accepted r01 grid artifact."
TP00_DS = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
TP00_GRID_RESULTS = sl_load_experiment(TP00_GRID_PATH)
TP00_FOLDS = sl_build_folds(TP00_DS, TP00_CONTRACT)
TP00_GATE3 = sl_gate_convergence(TP00_GRID_RESULTS, TP00_ADAPTER, TP00_CONTRACT; expected_folds = length(TP00_FOLDS))
@assert sl_gate_table("3. Persisted grid convergence", TP00_GATE3)
TP00_GRID_LATENTS = BayesianFootball.Experiments.extract_oos_predictions(TP00_DS, TP00_GRID_RESULTS; force = true)
TP00_OOS_IDS = Set(Int.(TP00_GRID_LATENTS.df.match_id))

# %%
# ==============================================================================
# 3. GATE 7a — Betfair-close book construction
# ==============================================================================
TP00_BF_ODDS = sl_betfair_odds_df(TP00_DS, TP00_CONTRACT; ids = TP00_OOS_IDS)
TP00_BOOK_SPEC = sl_book_spec(TP00_CONTRACT)
TP00_BOOKS_BF = BayesianFootball.Portfolio.build_books(TP00_BOOK_SPEC, TP00_GRID_LATENTS.df, TP00_GRID_RESULTS, TP00_BF_ODDS, TP00_DS)
TP00_GATE7A = sl_gate_books(TP00_BOOKS_BF, TP00_GRID_LATENTS.df, TP00_BF_ODDS)
@assert sl_gate_table("7a. Book construction", TP00_GATE7A)

# %%
# ==============================================================================
# 4. GATE 7b — Portfolio simulation integrity
# ==============================================================================
TP00_POLICY = sl_growth_policies(TP00_CONTRACT)[1].policy
TP00_SLATES = BayesianFootball.Portfolio.group(TP00_POLICY.grouping, TP00_BOOKS_BF)
TP00_TRAJECTORY = BayesianFootball.Portfolio.simulate(TP00_POLICY, TP00_SLATES)
TP00_GATE7B = sl_gate_simulation(TP00_TRAJECTORY, TP00_SLATES, TP00_CONTRACT)
@assert sl_gate_table("7b. Simulation integrity", TP00_GATE7B)

# %%
# ==============================================================================
# 5. GATE 7c — Growth verdict and declared policy sweep
# ==============================================================================
TP00_GROWTH = sl_growth_table(TP00_BOOKS_BF, TP00_CONTRACT)
TP00_GATE7C = sl_gate_growth(TP00_GROWTH)
@assert sl_gate_table("7c. Growth verdict", TP00_GATE7C)
TP00_POLICY_SWEEP = ScottishLowerProtocol.sl_sweep_policy(TP00_BOOKS_BF, TP00_CONTRACT)
