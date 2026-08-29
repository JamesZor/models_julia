# ==============================================================================
# Scottish Lower — unified Poisson feature-extension grid launcher (24/25 & 25/26)
# ==============================================================================
# Covers 40 walk-forward folds across both seasons: "24/25" and "25/26".
# All 4 arms (00_team_poisson, 02_poisson_wealth, 03_poisson_distance, 04_poisson_wealth_distance)
# sample using QueuedNUTS across 16 physical cores on mcmc-beast.

# %% ===========================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using ThreadPinning
using LinearAlgebra
using Printf

const SL_ALL_ROOT = let d = @__DIR__
    isfile(joinpath(d, "_protocol", "ScottishLowerProtocol.jl")) ? d :
        "current_development/scottish_lower"
end
include(joinpath(SL_ALL_ROOT, "_protocol", "ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol

include(joinpath(SL_ALL_ROOT, "00_team_poisson", "l01_model.jl"))
include(joinpath(SL_ALL_ROOT, "00_team_poisson", "l02_equations.jl"))
include(joinpath(SL_ALL_ROOT, "00_team_poisson", "l03_adapter.jl"))
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
# 2. Configuration (2-Season Contract: 24/25 & 25/26)
# ==============================================================================
const SL_ALL_CONTRACT = sl_contract(
    dev_seasons = ["24/25", "25/26"],
    sealed_seasons = ["26/27"]
)

const SL_ALL_REQUESTED = filter(!isempty, strip.(split(get(ENV, "SL_ARMS", ""), ",")))
const SL_ALL_ADAPTERS = let all_arms = (
        TP00Adapter(half_life_days=180.0),
        TP02Adapter(half_life_days=180.0),
        TP03Adapter(half_life_days=180.0),
        TP04Adapter(half_life_days=180.0),
    )
    selected = isempty(SL_ALL_REQUESTED) ? collect(all_arms) :
        filter(a -> sl_model_name(a) in SL_ALL_REQUESTED, collect(all_arms))
    isempty(selected) && error("SL_ARMS matched no arm: $(SL_ALL_REQUESTED). " *
        "Known arms: $(join(sl_model_name.(all_arms), ", "))")
    tuple(selected...)
end
const SL_ALL_RUN_GRIDS = lowercase(get(ENV, "SL_RUN_GRIDS", "false")) in ("1", "true", "yes")

pinthreads(:cores)
BLAS.set_num_threads(1)

# %% ===========================================================================
# 3. Data snapshot and common temporal splits
# ==============================================================================
SL_ALL_DS = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower(); max_age_hours = 100_000)
SL_ALL_FOLDS = sl_build_folds(SL_ALL_DS, SL_ALL_CONTRACT)
println("=" ^ 74)
println("SCOTTISH LOWER 2-SEASON GRID (24/25 & 25/26)")
println("=" ^ 74)
println("Total Walk-Forward Folds: ", length(SL_ALL_FOLDS))
@assert sl_gate_table("0. Shared contract", sl_gate_contract(SL_ALL_DS, SL_ALL_FOLDS, SL_ALL_CONTRACT))

# %% ===========================================================================
# 4. Per-arm feature preflight gates (no sampling)
# ==============================================================================
SL_ALL_PREFLIGHT_OK = Dict{String,Bool}()
for adapter in SL_ALL_ADAPTERS
    name = sl_model_name(adapter)
    println("\nPreflight: ", name, " → ", sl_artifact_dir(adapter, SL_ALL_CONTRACT))
    SL_ALL_PREFLIGHT_OK[name] = try
        config_pass = sl_gate_table("1. $name config", sl_gate_config(adapter, SL_ALL_CONTRACT))
        gates, _ = sl_gate_features(SL_ALL_DS, SL_ALL_FOLDS, adapter, SL_ALL_CONTRACT)
        features_pass = sl_gate_table("2. $name features", gates)
        config_pass && features_pass
    catch err
        println("!! ", name, " preflight ERRORED — arm will not be sampled:")
        showerror(stdout, err, catch_backtrace())
        println()
        false
    end
    SL_ALL_PREFLIGHT_OK[name] || println("!! ", name, " FAILED preflight — SKIPPED for sampling.")
end

# %% ===========================================================================
# 5. Full queued MCMC grids (40 Folds x 4 Chains)
# ==============================================================================
SL_ALL_ARM_STATUS = Dict{String,String}()
SL_ALL_ARM_PATH = Dict{String,String}()
SL_ALL_ARM_FAILED_GATES = Dict{String,Vector{String}}()

if !SL_ALL_RUN_GRIDS
    println("\nPreflight complete; no chains sampled. Set SL_RUN_GRIDS=true to launch 40-fold sampling.")
else
    for adapter in SL_ALL_ADAPTERS
        name = sl_model_name(adapter)
        if !SL_ALL_PREFLIGHT_OK[name]
            println("\nSkipping ", name, " — failed preflight in section 4, not sampled.")
            SL_ALL_ARM_STATUS[name] = "skipped (preflight failed)"
            continue
        end
        println("\n>>> LAUNCHING SAMPLING: ", name, " (40 Folds x 4 Chains) ...")
        t_start = time()
        SL_ALL_ARM_STATUS[name] = try
            results, path = sl_run_experiment(SL_ALL_DS, adapter, SL_ALL_CONTRACT; smoke=false)
            SL_ALL_ARM_PATH[name] = path
            elapsed = time() - t_start
            @printf("    Completed sampling in %.1f minutes. Evaluating convergence...\n", elapsed / 60.0)
            
            gates = sl_gate_convergence(results, adapter, SL_ALL_CONTRACT; expected_folds=length(SL_ALL_FOLDS))
            if sl_gate_table("3. $name grid convergence", gates)
                println("Accepted artifact: ", path)
                "accepted"
            else
                SL_ALL_ARM_FAILED_GATES[name] = [String(r.name) for r in gates if !r.pass]
                println("NOT ACCEPTED — ", name, " failed convergence gates. Chains are still on disk at: ", path)
                "failed convergence gates"
            end
        catch err
            println("!! ", name, " ERRORED during sampling or gating:")
            showerror(stdout, err, catch_backtrace())
            println()
            "errored"
        end
    end
end

# %% ===========================================================================
# 6. End-of-run summary
# ==============================================================================
SL_ALL_SUMMARY = map(SL_ALL_ADAPTERS) do adapter
    name = sl_model_name(adapter)
    if !SL_ALL_PREFLIGHT_OK[name]
        (; name, sampled = false, accepted = false, label = "FAILED PREFLIGHT — never sampled")
    elseif !SL_ALL_RUN_GRIDS
        (; name, sampled = false, accepted = false, label = "preflight passed — grids not run (SL_RUN_GRIDS unset)")
    elseif get(SL_ALL_ARM_STATUS, name, "") == "accepted"
        (; name, sampled = true, accepted = true, label = "ACCEPTED — $(get(SL_ALL_ARM_PATH, name, "?"))")
    else
        failed = join(get(SL_ALL_ARM_FAILED_GATES, name, String[]), ", ")
        status = get(SL_ALL_ARM_STATUS, name, "unknown")
        path = get(SL_ALL_ARM_PATH, name, "")
        (; name, sampled = !isempty(path), accepted = false,
           label = "NOT ACCEPTED — " * (isempty(failed) ? status : "$status: $failed") *
                   (isempty(path) ? "" : "\n" * " "^30 * "chains: " * path))
    end
end

println()
println("=" ^ 74)
println("SCOTTISH LOWER 2-SEASON GRID LAUNCHER — ARM SUMMARY")
println("=" ^ 74)
for s in SL_ALL_SUMMARY
    println("  ", rpad(s.name, 30), s.label)
end
println("=" ^ 74)
println("  ", count(s -> s.accepted, SL_ALL_SUMMARY), " accepted / ",
        count(s -> s.sampled, SL_ALL_SUMMARY), " sampled / ", length(SL_ALL_SUMMARY), " arms")
println("=" ^ 74)
