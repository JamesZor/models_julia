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

# Works both when this file is `include`d (nested includes resolve relative to
# THIS file's directory, so `@__DIR__` is the only correct root) and when the
# `# %%` cells are pasted into a REPL from the repository root.
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
# 2. Configuration, runtime, and output directories
# ==============================================================================
const SL_ALL_CONTRACT = sl_contract()
# SL_ARMS optionally restricts the run to a subset, e.g.
#   SL_ARMS="02_poisson_wealth,04_poisson_wealth_distance"
# Unset means every arm.  Used to re-run a single arm without re-sampling the
# others, which would otherwise leave two artifacts for the same model.
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
@assert sl_gate_table("0. Shared contract", sl_gate_contract(SL_ALL_DS, SL_ALL_FOLDS, SL_ALL_CONTRACT))


# %% ===========================================================================
# 4. Per-arm feature preflight gates (no sampling)
# ==============================================================================
# Preflight covers the config and anti-leakage gates.  An arm that fails here is
# skipped for sampling entirely: sampling on an untrusted feature set would waste
# ~16 minutes of MCMC on a result nobody can use.  Neither a failing gate nor a
# raised exception may stop the remaining arms from being preflighted, so the
# verdict is captured rather than asserted.
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
        # Never swallowed: the full error and backtrace are printed, the arm is
        # simply not promoted to sampling.
        println("!! ", name, " preflight ERRORED — arm will not be sampled:")
        showerror(stdout, err, catch_backtrace())
        println()
        false
    end
    SL_ALL_PREFLIGHT_OK[name] || println("!! ", name, " FAILED preflight — SKIPPED for sampling.")
end

# %% ===========================================================================
# 5. Full queued MCMC grids — explicit overnight opt-in
# ==============================================================================
# A convergence-gate failure on one arm must not destroy the queue: that arm's
# sampling already succeeded and its chains are on disk, so the run continues and
# the arm is simply not promoted to "accepted".
SL_ALL_ARM_STATUS = Dict{String,String}()
SL_ALL_ARM_PATH = Dict{String,String}()
SL_ALL_ARM_FAILED_GATES = Dict{String,Vector{String}}()
if !SL_ALL_RUN_GRIDS
    println("Preflight complete; no chains sampled. Set SL_RUN_GRIDS=true after each arm passes v01 Gates 0--5.")
else
    # QueuedNUTS flattens fold × chain work; do not add another scheduler.
    for adapter in SL_ALL_ADAPTERS
        name = sl_model_name(adapter)
        if !SL_ALL_PREFLIGHT_OK[name]
            println("\nSkipping ", name, " — failed preflight in section 4, not sampled.")
            SL_ALL_ARM_STATUS[name] = "skipped (preflight failed)"
            continue
        end
        SL_ALL_ARM_STATUS[name] = try
            results, path = sl_run_experiment(SL_ALL_DS, adapter, SL_ALL_CONTRACT; smoke=false)
            SL_ALL_ARM_PATH[name] = path
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
# A 3am scrollback should show the state of every arm in one place.  Labels are
# built with `map` rather than accumulated in a top-level loop: assigning to a
# global from inside a top-level `for` is a soft-scope error in a script.
# `sampled` and `accepted` are deliberately separate.  An arm that sampled 20/20
# folds cleanly but tripped one advisory gate has produced a usable artifact; the
# run did NOT fail.  Only a run that produced no chains at all is a failed run.
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
println("SCOTTISH LOWER POISSON GRID LAUNCHER — ARM SUMMARY")
println("=" ^ 74)
for s in SL_ALL_SUMMARY
    println("  ", rpad(s.name, 30), s.label)
end
println("=" ^ 74)

println("  ", count(s -> s.accepted, SL_ALL_SUMMARY), " accepted / ",
        count(s -> s.sampled, SL_ALL_SUMMARY), " sampled / ", length(SL_ALL_SUMMARY), " arms")
println("=" ^ 74)

# Only a run that produced NO chains at all is a failed run.  Gate verdicts are
# recorded above; they do not decide whether the launcher itself failed.
if SL_ALL_RUN_GRIDS && count(s -> s.sampled, SL_ALL_SUMMARY) == 0
    error("Scottish Lower grid run: no arm produced chains — see the summary above.")
elseif !SL_ALL_RUN_GRIDS && count(s -> SL_ALL_PREFLIGHT_OK[s.name], SL_ALL_SUMMARY) == 0
    error("Scottish Lower preflight: no arm passed — see the summary above.")
end
