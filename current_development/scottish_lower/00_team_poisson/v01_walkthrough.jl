# ==============================================================================
# MODEL 00 — PURE POISSON BASELINE : THE WALKTHROUGH
# ==============================================================================
#
# WHAT THIS IS
#   A correctness walkthrough of Model 00 (Pure Poisson with Log-Intensity
#   Formulation), covering Gates 0 through 7:
#     Gates 0-2: Contract, Config, Features (anti-leakage)
#     Gate 3:    Sampling (Parity against l02, Gradients, Smoke MCMC)
#     Gate 4:    Extraction (Synthetic parity, Real chain plumbing, Fallbacks)
#     Gate 5:    Score Matrix (Poisson dispatch, Grid parity, Market identities)
#     Gate 6:    Evaluation (Proper scores, Shape RQR/LPD, Calibration)
#     Gate 7:    Growth & Staking (Betfair close, Portfolio-Kelly, CLV)
#
# HOW TO RUN
#   Send one numbered block at a time from nvim (kitty-runner). Blocks are
#   independently sendable in order; each leaves its output in a named global so
#   you can inspect it afterwards.
#
# BEFORE THE SMOKE RUN (block 7), start Julia as:
#   julia --project -t 16
# and set up ThreadPinning + single-threaded BLAS.
#
# ==============================================================================


# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using DataFrames
using Distributions
using Statistics

const TP00_ROOT = "current_development/scottish_lower"

include(joinpath(TP00_ROOT, "_protocol/config.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l01_model.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l02_equations.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l03_gates.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l04_sampling_gates.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l05_extraction_gates.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l06_score_matrix_gates.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l07_evaluation_gates.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l08_growth_gates.jl"))


# %%
# ==============================================================================
# 2. GATE 0 — Contract
# ==============================================================================

tp00_contract = sl_contract()
sl_describe(tp00_contract)

tp00_ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())

tp00_folds = tp00_build_folds(tp00_ds, tp00_contract)
tp00_fold_table(tp00_ds, tp00_folds)

tp00_gate0 = tp00_gate_contract(tp00_ds, tp00_folds, tp00_contract)
@assert sl_gate_table("0. Contract", tp00_gate0)


# %%
# ==============================================================================
# 3. GATE 1 — Config
# ==============================================================================

tp00_menu()

tp00_engine = tp00_model(
    half_life_days = 180.0,
)

tp00_describe(tp00_engine)

tp00_gate1 = tp00_gate_config(tp00_engine, tp00_contract)
@assert sl_gate_table("1. Config", tp00_gate1)


# %%
# ==============================================================================
# 4. GATE 2 — Features
# ==============================================================================

tp00_gate2, tp00_features = tp00_gate_features(tp00_ds, tp00_folds, tp00_engine, tp00_contract)
@assert sl_gate_table("2. Features", tp00_gate2)


# %%
# ==============================================================================
# 5. GATE 3a — Equation parity
# ==============================================================================
#
# DynamicPPL scores the Turing model; l02_equations.jl scores independent log-Poisson.

tp00_gate3a = tp00_gate_equation_parity(tp00_engine, tp00_features[1])
@assert sl_gate_table("3a. Equation parity", tp00_gate3a)


# %%
# ==============================================================================
# 6. GATE 3b — Gradient health
# ==============================================================================

tp00_gate3b, tp00_grad = tp00_gate_gradients(tp00_engine, tp00_features[1])
@assert sl_gate_table("3b. Gradient health", tp00_gate3b)


# %%
# ==============================================================================
# 7. GATE 3c — Smoke run   ***THIS ONE SAMPLES***
# ==============================================================================

using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1)

tp00_smoke_results, tp00_smoke_path = tp00_run_smoke(tp00_ds, tp00_engine, tp00_contract)

tp00_gate3c = tp00_gate_convergence(tp00_smoke_results, tp00_contract; expected_folds = 1)
@assert sl_gate_table("3c. Smoke convergence", tp00_gate3c)

println("Saved smoke chain to: ", tp00_smoke_path)


# %%
# ==============================================================================
# 8. GATE 4 — Extraction
# ==============================================================================

tp00_gate4a = tp00_gate_extraction_synthetic(tp00_engine, tp00_features[1])
@assert sl_gate_table("4a. Extraction parity (synthetic chain)", tp00_gate4a)

tp00_gate4c = tp00_gate_extraction_fallbacks(tp00_engine, tp00_features[1])
@assert sl_gate_table("4c. Extraction fallbacks", tp00_gate4c)

tp00_smoke_loaded = tp00_load_smoke(tp00_smoke_path)
tp00_gate4b, tp00_latents = tp00_gate_extraction_real(tp00_ds, tp00_smoke_loaded, tp00_contract)
@assert sl_gate_table("4b. Extraction plumbing (real chain)", tp00_gate4b)


# %%
# ==============================================================================
# 9. GATE 5 — Score matrix
# ==============================================================================

tp00_gate5a = tp00_gate_score_dispatch(tp00_engine, first(eachrow(tp00_latents.df));
                                       max_goals = tp00_contract.max_goals)
@assert sl_gate_table("5a. Score matrix dispatch", tp00_gate5a)

tp00_gate5b = tp00_gate_score_grid(tp00_engine, tp00_latents.df, tp00_contract)
@assert sl_gate_table("5b. Score matrix grid", tp00_gate5b)

tp00_gate5c = tp00_gate_market_identities(tp00_engine, tp00_latents.df, tp00_contract)
@assert sl_gate_table("5c. Market identities", tp00_gate5c)

tp00_market_summary(tp00_engine, tp00_latents.df, tp00_contract; n_rows = 8)


# %%
# ==============================================================================
# 10. THE FULL GRID   ***THIS SAMPLES ALL 20 FOLDS***
# ==============================================================================

tp00_grid_results, tp00_grid_path = tp00_run_grid(tp00_ds, tp00_engine, tp00_contract)

tp00_gate10 = tp00_gate_convergence(tp00_grid_results, tp00_contract; expected_folds = length(tp00_folds))
@assert sl_gate_table("6.0 Grid convergence (all folds)", tp00_gate10)

tp00_grid_latents = Experiments.extract_oos_predictions(tp00_ds, tp00_grid_results; force = true)
nrow(tp00_grid_latents.df)


# %%
# ==============================================================================
# 11. GATE 6 — Evaluation
# ==============================================================================

tp00_oos_ids = Set(Int.(tp00_grid_latents.df.match_id))

tp00_mb_b365 = tp00_market_book(tp00_ds.odds, tp00_contract; ids = tp00_oos_ids)
tp00_mb_bf, tp00_n_partial = tp00_drop_incomplete(
    tp00_betfair_book(tp00_ds, tp00_contract, tp00_mb_b365; ids = tp00_oos_ids))

@assert sl_gate_table("6a. Book integrity (Bet365 close)", tp00_gate_book_integrity(tp00_mb_b365, tp00_contract))
@assert sl_gate_table("6a. Book integrity (Betfair close)", tp00_gate_book_integrity(tp00_mb_bf, tp00_contract))

tp00_model_bk, tp00_fx = tp00_model_book(tp00_engine, tp00_grid_latents, tp00_ds, tp00_contract)

tp00_books = Dict("bet365" => tp00_mb_b365, "betfair" => tp00_mb_bf)
tp00_j     = tp00_join_books(tp00_model_bk, tp00_books)
@assert sl_gate_table("6b. Alignment", tp00_gate_alignment(tp00_j, tp00_model_bk))

@assert sl_gate_table("6c. Shape (RQR / LPD)", tp00_gate_shape(tp00_fx))
@assert sl_gate_table("6d. Draw deficit", tp00_gate_draw_deficit(tp00_fx))

tp00_scores_b365 = tp00_score_table(tp00_j["bet365"])
tp00_edges_b365  = tp00_edge_table(tp00_j["bet365"])
@assert sl_gate_table("6e. Not broken (vs Bet365 close)", tp00_gate_not_broken(tp00_scores_b365, tp00_edges_b365))

tp00_summary(tp00_j)
tp00_summary_shape(tp00_fx)


# %%
# ==============================================================================
# 12. GATE 7 — Growth
# ==============================================================================

tp00_odds_bf   = tp00_betfair_odds_df(tp00_ds, tp00_contract; ids = tp00_oos_ids)
tp00_spec      = tp00_book_spec(tp00_contract)
tp00_books_bf  = Pf.build_books(tp00_spec, tp00_grid_latents.df, tp00_grid_results, tp00_odds_bf, tp00_ds)

@assert sl_gate_table("7a. Book construction", tp00_gate_books(tp00_books_bf, tp00_grid_latents.df, tp00_odds_bf))

tp00_pol0    = tp00_growth_policies(tp00_contract)[1].policy
tp00_slates  = Pf.group(tp00_pol0.grouping, tp00_books_bf)
tp00_traj0   = Pf.simulate(tp00_pol0, tp00_slates)
@assert sl_gate_table("7b. Simulation integrity", tp00_gate_simulation(tp00_traj0, tp00_slates, tp00_contract))

tp00_growth = tp00_growth_table(tp00_books_bf, tp00_contract)
@assert sl_gate_table("7c. Growth verdict", tp00_gate_growth(tp00_growth))
tp00_growth

tp00_sweep = tp00_sweep_policy(tp00_books_bf, tp00_contract)
