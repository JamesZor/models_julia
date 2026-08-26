# ==============================================================================
# MODEL 01 — TEAM-LEVEL BASELINE: FULL GRID TRAINING
# ==============================================================================
# Question: can the documented DynamicGoalsTimeDecayModel produce converged,
# walk-forward posteriors on every development fold? This runner trains and
# persists only; r02 evaluates accepted artifacts and r03 applies staking.
#

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using ThreadPinning
using LinearAlgebra

const TP01_ROOT = "current_development/scottish_lower"
include(joinpath(TP01_ROOT, "_protocol/ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol
include(joinpath(TP01_ROOT, "01_team_poisson/l01_model.jl"))
include(joinpath(TP01_ROOT, "01_team_poisson/l02_equations.jl"))
include(joinpath(TP01_ROOT, "01_team_poisson/l03_adapter.jl"))


# %%
# ==============================================================================
# 2. Configuration, runtime, and artifact identity
# ==============================================================================

TP01_contract = sl_contract()
TP01_model = tp_model(half_life_days = 180.0)
TP01_adapter = TP01Adapter(TP01_model)
TP01_grid_dir = sl_artifact_dir(TP01_adapter, TP01_contract)
sl_describe(TP01_contract)
println("immutable grid artifact directory: ", TP01_grid_dir)

pinthreads(:cores)
BLAS.set_num_threads(1)
@assert Threads.nthreads() == TP01_contract.queue_tasks


# %%
# ==============================================================================
# 3. Data snapshot, folds, and preflight gates
# ==============================================================================

TP01_ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
TP01_folds = sl_build_folds(TP01_ds, TP01_contract)
TP01_gate0 = sl_gate_contract(TP01_ds, TP01_folds, TP01_contract)
@assert sl_gate_table("0. Contract", TP01_gate0)
TP01_gate1 = sl_gate_config(TP01_adapter, TP01_contract)
@assert sl_gate_table("1. Config", TP01_gate1)
TP01_gate2, TP01_features = sl_gate_features(TP01_ds, TP01_folds, TP01_adapter, TP01_contract)
@assert sl_gate_table("2. Features", TP01_gate2)


# %%
# ==============================================================================
# 4. Full-grid training and persistence (MCMC)
# ==============================================================================
# The native queue flattens folds × chains and fills the declared physical cores.

TP01_grid_results, TP01_grid_path = sl_run_experiment(
    TP01_ds, TP01_adapter, TP01_contract; smoke = false)
println("saved grid artifact: ", TP01_grid_path)


# %%
# ==============================================================================
# 5. Gate 6.0 — convergence promotion
# ==============================================================================

TP01_gate6_0 = sl_gate_convergence(
    TP01_grid_results, TP01_adapter, TP01_contract; expected_folds = length(TP01_folds))
@assert sl_gate_table("6.0 Grid convergence (all folds)", TP01_gate6_0)

# Persisted load, rather than this in-memory object, is the input to r02/r03.
TP01_grid_loaded = sl_load_experiment(TP01_grid_path)
println("promoted artifact reload succeeded: ", TP01_grid_path)
