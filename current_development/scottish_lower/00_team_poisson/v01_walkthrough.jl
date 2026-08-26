# ==============================================================================
# MODEL 00 — PURE POISSON BASELINE : THE WALKTHROUGH
# ==============================================================================
#
# WHAT THIS IS
#   A correctness walkthrough of Model 00 (Pure Poisson with Log-Intensity
#   Formulation), one stage at a time. It answers:
#   is the model that gets FITTED the same model that gets DOCUMENTED and the
#   same model that gets PRICED?
#
# HOW TO RUN
#   Send one numbered block at a time from nvim (kitty-runner). Blocks are
#   independently sendable in order; each leaves its output in a named global so
#   you can inspect it afterwards.
#
# GATE COVERAGE IN THIS FILE
#   [x] 0. Contract        [x] 1. Config        [x] 2. Features
#   [x] 3. Sampling        [x] 4. Extraction    [x] 5. Score matrix
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
# The load-bearing gate. DynamicPPL scores the Turing model; l02_equations.jl
# scores an independent log-Poisson implementation.

tp00_gate3a = tp00_gate_equation_parity(tp00_engine, tp00_features[1])
@assert sl_gate_table("3a. Equation parity", tp00_gate3a)


# %%
# ==============================================================================
# 6. GATE 3b — Gradient health
# ==============================================================================
#
# Four independent routes: fresh ReverseDiff, compiled tape, ForwardDiff, FiniteDiff.

tp00_gate3b, tp00_grad = tp00_gate_gradients(tp00_engine, tp00_features[1])
@assert sl_gate_table("3b. Gradient health", tp00_gate3b)


# %%
# ==============================================================================
# 7. GATE 3c — Smoke run   ***THIS ONE SAMPLES***
# ==============================================================================
#
# One fold, 4 chains x 500/500, persisted through src/experiments.

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
#
# 4a: Synthetic chain parity against l02_equations.jl
# 4c: Fallbacks
# 4b: Real chain plumbing

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

# tp00_grid_results, tp00_grid_path = tp00_run_grid(tp00_ds, tp00_engine, tp00_contract)
# tp00_gate10 = tp00_gate_convergence(tp00_grid_results, tp00_contract; expected_folds = length(tp00_folds))
# @assert sl_gate_table("6.0 Grid convergence (all folds)", tp00_gate10)
