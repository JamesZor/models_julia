# ==============================================================================
# MODEL 01 — TEAM-LEVEL BASELINE: GATE 7 GROWTH AND CLV
# ==============================================================================
# This consumes a persisted full-grid posterior and Betfair closing quotes. It
# reports a declared Portfolio-Kelly backtest, not evidence of a deployable edge.
# Usage: TP01_GRID_PATH=/path/to/artifact julia --project -e 'include("...")'
#

# %%
# ==============================================================================
# 1. Packages, implementation, and persisted input
# ==============================================================================

using BayesianFootball
using DataFrames

const TP01_ROOT = "current_development/scottish_lower"
const TP01_GRID_PATH = get(ENV, "TP01_GRID_PATH", "")
isempty(TP01_GRID_PATH) && error("set TP01_GRID_PATH to the r01 persisted grid artifact")

include(joinpath(TP01_ROOT, "_protocol/ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol
include(joinpath(TP01_ROOT, "01_team_poisson/l01_model.jl"))
include(joinpath(TP01_ROOT, "01_team_poisson/l02_equations.jl"))
include(joinpath(TP01_ROOT, "01_team_poisson/l03_adapter.jl"))


# %%
# ==============================================================================
# 2. Contract, data, and accepted grid artifact
# ==============================================================================

TP01_contract = sl_contract()
TP01_model = tp_model(half_life_days = 180.0)
TP01_adapter = TP01Adapter(TP01_model)
TP01_ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
TP01_grid_loaded = sl_load_experiment(TP01_GRID_PATH)
TP01_folds = sl_build_folds(TP01_ds, TP01_contract)

TP01_gate0 = sl_gate_contract(TP01_ds, TP01_folds, TP01_contract)
@assert sl_gate_table("0. Contract", TP01_gate0)
TP01_gate1 = sl_gate_config(TP01_adapter, TP01_contract)
@assert sl_gate_table("1. Config", TP01_gate1)
TP01_gate6_0 = sl_gate_convergence(
    TP01_grid_loaded, TP01_adapter, TP01_contract; expected_folds = length(TP01_folds))
@assert sl_gate_table("6.0 Grid convergence (loaded artifact)", TP01_gate6_0)

TP01_grid_latents = BayesianFootball.Experiments.extract_oos_predictions(
    TP01_ds, TP01_grid_loaded; force = true)
TP01_oos_ids = Set(Int.(TP01_grid_latents.df.match_id))


# %%
# ==============================================================================
# 3. GATE 7a — Closing-price book construction
# ==============================================================================
# DeArb is the execution policy. It prevents a recording artefact in close quotes
# from being interpreted as model edge.

TP01_betfair_odds = sl_betfair_odds_df(TP01_ds, TP01_contract; ids = TP01_oos_ids)
TP01_book_spec = sl_book_spec(TP01_contract)
TP01_books = BayesianFootball.Portfolio.build_books(
    TP01_book_spec,
    TP01_grid_latents.df,
    TP01_grid_loaded,
    TP01_betfair_odds,
    TP01_ds,
)
TP01_gate7a = sl_gate_books(TP01_books, TP01_grid_latents.df, TP01_betfair_odds)
@assert sl_gate_table("7a. Book construction", TP01_gate7a)


# %%
# ==============================================================================
# 4. GATE 7b — Simulation integrity
# ==============================================================================

TP01_policy0 = sl_growth_policies(TP01_contract)[1].policy
TP01_slates0 = BayesianFootball.Portfolio.group(TP01_policy0.grouping, TP01_books)
TP01_trajectory0 = BayesianFootball.Portfolio.simulate(TP01_policy0, TP01_slates0)
TP01_gate7b = sl_gate_simulation(TP01_trajectory0, TP01_slates0, TP01_contract)
@assert sl_gate_table("7b. Simulation integrity", TP01_gate7b)


# %%
# ==============================================================================
# 5. GATE 7c — Growth result and declared policy sweep
# ==============================================================================

TP01_growth = sl_growth_table(TP01_books, TP01_contract)
TP01_gate7c = sl_gate_growth(TP01_growth)
@assert sl_gate_table("7c. Growth verdict", TP01_gate7c)
TP01_growth

# This reuses built books; it is a policy sensitivity report, not selection.
TP01_policy_sweep = ScottishLowerProtocol.sl_sweep_policy(
    TP01_books,
    TP01_contract;
    trusts = [0.15, 0.3, 0.5, 1.0],
    lambdas = [15.0, 23.0, 35.0],
)
