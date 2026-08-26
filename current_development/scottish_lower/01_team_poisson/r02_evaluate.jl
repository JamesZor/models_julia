# ==============================================================================
# MODEL 01 — TEAM-LEVEL BASELINE: GATE 6 EVALUATION
# ==============================================================================
# This evaluates only a persisted, converged full-grid artifact. It reports proper
# scores and market-free shape diagnostics; it is not a staking or growth claim.
# Usage: TP01_GRID_PATH=/path/to/artifact julia --project -e 'include("...")'
#

# %%
# ==============================================================================
# 1. Packages, implementation, and persisted input
# ==============================================================================

using BayesianFootball
using DataFrames
using Statistics

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
# 2. Contract, model, data, and persisted grid load
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


# %%
# ==============================================================================
# 3. OOS extraction and Gate 5 pricing checks
# ==============================================================================

TP01_grid_latents = BayesianFootball.Experiments.extract_oos_predictions(
    TP01_ds, TP01_grid_loaded; force = true)
TP01_oos_ids = Set(Int.(TP01_grid_latents.df.match_id))
TP01_gate5a = sl_gate_score_dispatch(
    TP01_adapter, first(eachrow(TP01_grid_latents.df)); max_goals = TP01_contract.max_goals)
@assert sl_gate_table("5a. Score matrix dispatch", TP01_gate5a)
TP01_gate5b = sl_gate_score_grid(TP01_adapter, TP01_grid_latents.df, TP01_contract)
@assert sl_gate_table("5b. Score matrix grid", TP01_gate5b)
TP01_gate5c = sl_gate_market_identities(TP01_adapter, TP01_grid_latents.df, TP01_contract)
@assert sl_gate_table("5c. Market identities", TP01_gate5c)


# %%
# ==============================================================================
# 4. GATE 6a–6b — Baseline book integrity and alignment
# ==============================================================================

TP01_book_b365 = sl_market_book(TP01_ds.odds, TP01_contract; ids = TP01_oos_ids)
TP01_book_bf = ScottishLowerProtocol.sl_betfair_book(
    TP01_ds, TP01_contract, TP01_book_b365; ids = TP01_oos_ids)
TP01_book_bf, TP01_partial_bf = sl_drop_incomplete(TP01_book_bf)
@assert sl_gate_table("6a. Book integrity (Bet365 close)", sl_gate_book_integrity(TP01_book_b365, TP01_contract))
@assert sl_gate_table("6a. Book integrity (Betfair close)", sl_gate_book_integrity(TP01_book_bf, TP01_contract))

TP01_model_book, TP01_fixture_metrics = sl_model_book(
    TP01_adapter, TP01_grid_latents, TP01_ds, TP01_contract)
TP01_joined_books = sl_join_books(TP01_model_book, Dict(
    "bet365" => TP01_book_b365,
    "betfair" => TP01_book_bf,
))
TP01_gate6b = sl_gate_alignment(TP01_joined_books, TP01_model_book)
@assert sl_gate_table("6b. Alignment", TP01_gate6b)


# %%
# ==============================================================================
# 5. GATE 6c–6e — Shape, scores, and incremental information
# ==============================================================================

TP01_gate6c = sl_gate_shape(TP01_fixture_metrics)
@assert sl_gate_table("6c. Shape (RQR / LPD)", TP01_gate6c)
TP01_scores_b365 = sl_score_table(TP01_joined_books["bet365"])
TP01_edges_b365 = ScottishLowerProtocol.sl_edge_table(TP01_joined_books["bet365"])
TP01_gate6e = sl_gate_not_broken(TP01_scores_b365)
@assert sl_gate_table("6e. Not broken (vs Bet365 close)", TP01_gate6e)

TP01_summary = ScottishLowerProtocol.sl_summary(TP01_joined_books)
TP01_shape_summary = ScottishLowerProtocol.sl_summary_shape(TP01_fixture_metrics)
TP01_fold_weighting = ScottishLowerProtocol.sl_fold_weighting_check(TP01_joined_books["bet365"], TP01_folds)
TP01_summary
