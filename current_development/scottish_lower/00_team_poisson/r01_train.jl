# ==============================================================================
# MODEL 00 — PURE POISSON BASELINE : FULL GRID TRAIN RUNNER
# ==============================================================================
#
# RUNNER. Executes full walk-forward grid across all 20 folds on 24/25.
#
#   20 folds x 4 chains = 80 queued MCMC tasks
#   800 warmup + 800 samples per chain
#   Queued concurrency on 16 CPU cores
#
# Usage (server REPL):
#   include("current_development/scottish_lower/00_team_poisson/r01_train.jl")
#
# ==============================================================================

using BayesianFootball
using ThreadPinning
using LinearAlgebra
using Dates

const TP00_ROOT = "current_development/scottish_lower"

include(joinpath(TP00_ROOT, "_protocol/config.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l01_model.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l02_equations.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l03_gates.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l04_sampling_gates.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l05_extraction_gates.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l06_score_matrix_gates.jl"))

# Thread setup
pinthreads(:cores)
BLAS.set_num_threads(1)

println("=" ^ 74)
println("STARTING MODEL 00 FULL GRID MCMC RUN ($(Threads.nthreads()) threads)")
println("=" ^ 74)

tp00_contract = sl_contract()
tp00_engine   = tp00_model()
tp00_ds       = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
tp00_folds    = tp00_build_folds(tp00_ds, tp00_contract)

tp00_grid_results, tp00_grid_path = tp00_run_grid(tp00_ds, tp00_engine, tp00_contract)

println("Full grid run completed. Saved to:")
println("  ", tp00_grid_path)

tp00_gate10 = tp00_gate_convergence(tp00_grid_results, tp00_contract; expected_folds = length(tp00_folds))
sl_gate_table("6.0 Grid convergence (all folds)", tp00_gate10)
