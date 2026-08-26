# ==============================================================================
# Scottish Lower — unified Poisson feature-extension grid launcher
# ==============================================================================
# Hypothesis: point-in-time starting-XI wealth and static away-travel fatigue add
# walk-forward information beyond the clean team-Poisson model.  All arms use the
# same data and split contract; this script does not compare betting strategies.
#
# Persistence: each arm writes immutable artifacts under data/scottish_lower/
# <model-name>/<protocol-hash>/.  Set SL_RUN_GRIDS=true only after the matching
# v01_walkthrough.jl passed Gates 0--5 on this commit.

# %% ===========================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using ThreadPinning
using LinearAlgebra

const SL_ALL_ROOT = "current_development/scottish_lower"
include(joinpath(SL_ALL_ROOT, "_protocol", "ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol

include(joinpath(SL_ALL_ROOT, "02_poisson_wealth", "l01_model.jl"))
include(joinpath(SL_ALL_ROOT, "02_poisson_wealth", "l02_equations.jl"))
include(joinpath(SL_ALL_ROOT, "02_poisson_wealth", "l03_adapter.jl"))
include(joinpath(SL_ALL_ROOT, "03_poisson_distance", "l01_model.jl"))
include(joinpath(SL_ALL_ROOT, "03_poisson_distance", "l02_equations.jl"))
include(joinpath(SL_ALL_ROOT, "03_poisson_distance", "l03_adapter.jl"))
include(joinpath(SL_ALL_ROOT, "04_poisson_wealth_distance", "l01_model.jl"))
include(joinpath(SL_ALL_ROOT, "04_poisson_wealth_distance", "l02_equations.jl"))
include(joinpath(SL_ALL_ROOT, "04_poisson_wealth_distance", "l03_adapter.jl"))

# %% ===========================================================================
# 2. Configuration, runtime, and output directories
# ==============================================================================
const SL_ALL_CONTRACT = sl_contract()
const SL_ALL_ADAPTERS = (
    TP02Adapter(half_life_days=180.0),
    TP03Adapter(half_life_days=180.0),
    TP04Adapter(half_life_days=180.0),
)
const SL_ALL_RUN_GRIDS = lowercase(get(ENV, "SL_RUN_GRIDS", "false")) in ("1", "true", "yes")
pinthreads(:cores)
BLAS.set_num_threads(1)

# %% ===========================================================================
# 3. Data snapshot and common temporal splits
# ==============================================================================
SL_ALL_DS = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower(); max_age_hours = 100_000)
SL_ALL_FOLDS = sl_build_folds(SL_ALL_DS, SL_ALL_CONTRACT)
@assert sl_gate_table("0. Shared contract", sl_gate_contract(SL_ALL_DS, SL_ALL_FOLDS, SL_ALL_CONTRACT))

# %% ===========================================================================
# 4. Per-arm feature preflight gates (no sampling)
# ==============================================================================
for adapter in SL_ALL_ADAPTERS
    println("\nPreflight: ", sl_model_name(adapter), " → ", sl_artifact_dir(adapter, SL_ALL_CONTRACT))
    @assert sl_gate_table("1. $(sl_model_name(adapter)) config", sl_gate_config(adapter, SL_ALL_CONTRACT))
    gates, _ = sl_gate_features(SL_ALL_DS, SL_ALL_FOLDS, adapter, SL_ALL_CONTRACT)
    @assert sl_gate_table("2. $(sl_model_name(adapter)) features", gates)
end

# %% ===========================================================================
# 5. Full queued MCMC grids — explicit overnight opt-in
# ==============================================================================
if !SL_ALL_RUN_GRIDS
    println("Preflight complete; no chains sampled. Set SL_RUN_GRIDS=true after each arm passes v01 Gates 0--5.")
else
    # QueuedNUTS flattens fold × chain work; do not add another scheduler.
    for adapter in SL_ALL_ADAPTERS
        results, path = sl_run_experiment(SL_ALL_DS, adapter, SL_ALL_CONTRACT; smoke=false)
        gates = sl_gate_convergence(results, adapter, SL_ALL_CONTRACT; expected_folds=length(SL_ALL_FOLDS))
        @assert sl_gate_table("3. $(sl_model_name(adapter)) grid convergence", gates)
        println("Accepted artifact: ", path)
    end
end
