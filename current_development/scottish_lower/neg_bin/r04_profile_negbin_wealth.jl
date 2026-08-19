# current_development/scottish_lower/neg_bin/r04_profile_negbin_wealth.jl
#
# RUNNER: ReverseDiff Gradient & AD Tape Profiler for Scottish NegBin + Wealth Models

using Revise
using BayesianFootball
using Turing
using DynamicPPL
using LogDensityProblems
using ReverseDiff
using BenchmarkTools
using Printf
using Dates
using DataFrames
using Statistics

const Data     = BayesianFootball.Data
const Features = BayesianFootball.Features
const PreGame  = BayesianFootball.Models.PreGame
const ROOT     = pkgdir(BayesianFootball)

include("l02_negbin_wealth_engines.jl")

println("==================================================================")
println(" SCOTTISH LOWER NEGBIN + WEALTH REVERSEDIFF GRADIENT PROFILER")
println("==================================================================")

# 1. Load DataStore
println("\n[1/3] Loading Scottish Lower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
println("✓ Loaded ScottishLower DataStore: ", nrow(ds.matches), " matches")

# 2. Define Model Candidates
dyn = PreGame.TimeDecayDynamics(days_half_life = 365.0)

models = [
    ("Model 1: Goals NegBin + Wealth", TeamGoalsNegBinWealthModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        apm_on = true,
        league_ha_on = true,
        name = "goals_negbin_wealth"
    )),
    ("Model 2: Proxy xG + RAPM + NegBin + Wealth", TeamPxGGoalsAPMNegBinWealthModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        apm_on = true,
        league_ha_on = true,
        name = "pxg_apm_negbin_wealth"
    )),
    ("Model 3: Funnel Proxy xG + RAPM + NegBin + Wealth", TeamFunnelPxGGoalsAPMNegBinWealthModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        apm_on = true,
        team_quality_on = true,
        league_ha_on = true,
        name = "funnel_pxg_apm_negbin_wealth"
    ))
]

# 3. Profile Each Model
println("\n[2/3] Benchmarking Tape Compilation & Gradient Latency...")

results_df = DataFrame(
    Model = String[],
    Params = Int[],
    TapeNodes = Int[],
    CompileTime_s = Float64[],
    GradLatency_ms = Float64[],
    Allocs_B = Int[],
    Status = String[]
)

for (label, model_cfg) in models
    println("\n" * "="^85)
    println("▶ PROFILING: $label ($(model_cfg.name))")
    println("="^85)

    # Extract Features
    boundary = Data.SplitBoundary(1, 1, Int.(ds.matches.match_id[1:min(1200, nrow(ds.matches))]), Int[])
    t_feat = @elapsed fs = Features.create_features(boundary, ds, model_cfg, :match_biweek)
    println("  • Feature extraction: ", round(t_feat * 1000, digits=1), " ms")

    # Build Turing Model
    turing_mod = PreGame.build_turing_model(model_cfg, fs)

    # 1. Initialize VarInfo & Parameter Vector
    vi = DynamicPPL.VarInfo(turing_mod)
    turing_mod(vi)
    θ = vi[:]
    n_params = length(θ)
    @printf("  • Parameter Dimension (θ) : %d latent parameters\n", n_params)

    # 2. Wrap Log-Density Function
    lf = DynamicPPL.LogDensityFunction(turing_mod)
    f = x -> LogDensityProblems.logdensity(lf, x)

    # 3. Construct Raw GradientTape
    raw_tape = ReverseDiff.GradientTape(f, θ)
    tape_nodes = length(raw_tape.tape)
    @printf("  • GradientTape Node Count : %d instructions/nodes\n", tape_nodes)

    # 4. Compile Tape
    t_compile = @elapsed compiled_tape = ReverseDiff.compile(raw_tape)
    @printf("  • Tape Compilation Time   : %.3f seconds\n", t_compile)

    # 5. Evaluate Gradient & Check Validity
    g = similar(θ)
    ReverseDiff.gradient!(g, compiled_tape, θ)
    has_nans = any(isnan, g) || any(isinf, g)
    @printf("  • Gradient Validity       : %s\n", has_nans ? "FAIL ❌ (NaNs/Infs)" : "PASS ✅ (All Finite)")
    @assert !has_nans "ERROR: Non-finite gradients detected!"

    # 6. Benchmark Latency & Memory Allocations
    b = @benchmark ReverseDiff.gradient!($g, $compiled_tape, $θ) samples=50 evals=1
    grad_time_ms = median(b.times) / 1e6
    min_time_ms  = minimum(b.times) / 1e6
    alloc_bytes  = b.memory

    @printf("  • Median Gradient Latency : %.3f ms (Min: %.3f ms)\n", grad_time_ms, min_time_ms)
    @printf("  • Gradient Allocations    : %d allocs (%d bytes)\n", b.allocs, alloc_bytes)

    push!(results_df, (
        label,
        n_params,
        tape_nodes,
        round(t_compile, digits=3),
        round(grad_time_ms, digits=3),
        alloc_bytes,
        has_nans ? "FAIL" : "PASS"
    ))
end

println("\n" * "="^95)
println(" SUMMARY: GRADIENT TAPE PROFILING BENCHMARK (SCOTTISH NEGBIN + WEALTH)")
println("="^95)
show(results_df, allrows=true, allcols=true); println()
println("✓ ReverseDiff Gradient Profiling Complete!")
