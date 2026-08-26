# ==============================================================================
# MODEL 00 — PURE POISSON BASELINE : FULL GRID TRAINING
# ==============================================================================
# Question: can the protocol-validated pure-Poisson baseline produce an accepted
# walk-forward posterior for every development fold? This runner trains only;
# r02 evaluates the persisted artifact and r03 stakes it at Betfair close.
# Run on beast, never locally: include(".../r01_train.jl").

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using ThreadPinning
using LinearAlgebra

const TP00_ROOT = "current_development/scottish_lower"
include(joinpath(TP00_ROOT, "_protocol/ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol
include(joinpath(TP00_ROOT, "00_team_poisson/l01_model.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l02_equations.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l03_adapter.jl"))

# %%
# ==============================================================================
# 2. Configuration, runtime, and output artifact
# ==============================================================================
TP00_CONTRACT = sl_contract()
TP00_ADAPTER = TP00Adapter(half_life_days = 180.0)
TP00_ARTIFACT_DIR = sl_artifact_dir(TP00_ADAPTER, TP00_CONTRACT)
pinthreads(:cores)
BLAS.set_num_threads(1)
println("Model 00 grid artifact directory: ", TP00_ARTIFACT_DIR)

# %%
# ==============================================================================
# 3. Data snapshot, folds, and preflight Gates 0--2
# ==============================================================================
TP00_DS = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
TP00_FOLDS = sl_build_folds(TP00_DS, TP00_CONTRACT)
TP00_GATE0 = sl_gate_contract(TP00_DS, TP00_FOLDS, TP00_CONTRACT)
@assert sl_gate_table("0. Contract", TP00_GATE0)
TP00_GATE1 = sl_gate_config(TP00_ADAPTER, TP00_CONTRACT)
@assert sl_gate_table("1. Config", TP00_GATE1)
TP00_GATE2, TP00_FEATURES = sl_gate_features(TP00_DS, TP00_FOLDS, TP00_ADAPTER, TP00_CONTRACT)
@assert sl_gate_table("2. Features", TP00_GATE2)

# %%
# ==============================================================================
# 4. Full queued MCMC grid and Gate 3 convergence   ***SAMPLES***
# ==============================================================================
TP00_GRID_RESULTS, TP00_GRID_PATH = sl_run_experiment(TP00_DS, TP00_ADAPTER, TP00_CONTRACT; smoke = false)
TP00_GATE3 = sl_gate_convergence(TP00_GRID_RESULTS, TP00_ADAPTER, TP00_CONTRACT; expected_folds = length(TP00_FOLDS))
@assert sl_gate_table("3. Grid convergence", TP00_GATE3)
println("Accepted grid artifact: ", TP00_GRID_PATH)
println("For r02/r03: ENV[\"TP00_GRID_PATH\"] = \"", TP00_GRID_PATH, "\"")
