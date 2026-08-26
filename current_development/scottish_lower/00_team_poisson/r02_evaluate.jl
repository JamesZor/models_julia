# ==============================================================================
# MODEL 00 — PURE POISSON BASELINE : GATE 6 OOS EVALUATION
# ==============================================================================
# Scores only an already persisted, accepted grid. It never samples or silently
# regenerates predictions. Before including, set ENV["TP00_GRID_PATH"] to the
# exact artifact path printed by r01_train.jl.

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using DataFrames
using Statistics

const TP00_ROOT = "current_development/scottish_lower"
include(joinpath(TP00_ROOT, "_protocol/ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol
include(joinpath(TP00_ROOT, "00_team_poisson/l01_model.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l02_equations.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l03_adapter.jl"))

# %%
# ==============================================================================
# 2. Configuration, persisted artifact, and data snapshot
# ==============================================================================
TP00_CONTRACT = sl_contract()
TP00_ADAPTER = TP00Adapter(half_life_days = 180.0)
TP00_GRID_PATH = get(ENV, "TP00_GRID_PATH", "")
@assert !isempty(TP00_GRID_PATH) "Set ENV[\"TP00_GRID_PATH\"] to an accepted r01 grid artifact."
TP00_DS = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
TP00_GRID_RESULTS = sl_load_experiment(TP00_GRID_PATH)
TP00_FOLDS = sl_build_folds(TP00_DS, TP00_CONTRACT)

# %%
# ==============================================================================
# 3. Gate 3 promotion check and persisted OOS extraction
# ==============================================================================
TP00_GATE3 = sl_gate_convergence(TP00_GRID_RESULTS, TP00_ADAPTER, TP00_CONTRACT; expected_folds = length(TP00_FOLDS))
@assert sl_gate_table("3. Persisted grid convergence", TP00_GATE3)
TP00_GRID_LATENTS = BayesianFootball.Experiments.extract_oos_predictions(TP00_DS, TP00_GRID_RESULTS; force = true)
TP00_OOS_IDS = Set(Int.(TP00_GRID_LATENTS.df.match_id))

# %%
# ==============================================================================
# 4. GATE 6a — Closing-book integrity
# ==============================================================================
TP00_BOOK_B365 = sl_market_book(TP00_DS.odds, TP00_CONTRACT; ids = TP00_OOS_IDS)
TP00_BOOK_BF, TP00_BF_PARTIAL = sl_drop_incomplete(ScottishLowerProtocol.sl_betfair_book(TP00_DS, TP00_CONTRACT, TP00_BOOK_B365; ids = TP00_OOS_IDS))
@assert sl_gate_table("6a. Bet365 book integrity", sl_gate_book_integrity(TP00_BOOK_B365, TP00_CONTRACT))
@assert sl_gate_table("6a. Betfair book integrity", sl_gate_book_integrity(TP00_BOOK_BF, TP00_CONTRACT))

# %%
# ==============================================================================
# 5. GATES 6b--6d — Model book, alignment, shape, and proper scores
# ==============================================================================
TP00_MODEL_BOOK, TP00_FIXTURES = sl_model_book(TP00_ADAPTER, TP00_GRID_LATENTS, TP00_DS, TP00_CONTRACT)
TP00_JOINED = sl_join_books(TP00_MODEL_BOOK, Dict("bet365" => TP00_BOOK_B365, "betfair" => TP00_BOOK_BF))
TP00_GATE6B = sl_gate_alignment(TP00_JOINED, TP00_MODEL_BOOK)
TP00_GATE6C = sl_gate_shape(TP00_FIXTURES)
TP00_SCORES_B365 = sl_score_table(TP00_JOINED["bet365"])
TP00_EDGES_B365 = ScottishLowerProtocol.sl_edge_table(TP00_JOINED["bet365"])
TP00_GATE6D = sl_gate_not_broken(TP00_SCORES_B365)
@assert sl_gate_table("6b. Alignment", TP00_GATE6B)
@assert sl_gate_table("6c. Shape (RQR / LPD / draw rate)", TP00_GATE6C)
@assert sl_gate_table("6d. Not broken (vs Bet365 close)", TP00_GATE6D)
TP00_EVALUATION = ScottishLowerProtocol.sl_summary(TP00_JOINED)
TP00_SHAPE_SUMMARY = ScottishLowerProtocol.sl_summary_shape(TP00_FIXTURES)
TP00_FOLD_WEIGHTING = ScottishLowerProtocol.sl_fold_weighting_check(TP00_JOINED["bet365"], TP00_FOLDS)
