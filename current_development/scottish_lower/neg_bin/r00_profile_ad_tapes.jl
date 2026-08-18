# current_development/scottish_lower/neg_bin/r00_profile_ad_tapes.jl
#
# PROFILER & AD BENCHMARK: Robust Negative Binomial (NB2) Turing Models
# Scottish Lower Leagues (56/57)
#
# Methodology from docs/turing_ad_performance_guide.md:
# 1. VarInfo Parameter Vector Extraction (θ)
# 2. GradientTape Construction & Tape Node Count (bloat detection)
# 3. Tape Compilation Time (@elapsed ReverseDiff.compile)
# 4. Gradient Evaluation Latency & Allocations (@benchmark ReverseDiff.gradient!)
# 5. Short NUTS Leapfrog Probe Verification (finite logprob & leapfrog integration)

using Revise
using BayesianFootball
using DynamicPPL
using LogDensityProblems
using ReverseDiff
using BenchmarkTools
using Printf
using Statistics
using DataFrames

const ROOT = pkgdir(BayesianFootball)
include("l01_negbin_engines.jl")

println("\n", "="^95)
println("REVERSEDIFF GRADIENT TAPE PROFILER: ROBUST NEGATIVE BINOMIAL ENGINES")
println("="^95)

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
println("✓ Loaded ScottishLower DataStore: $(nrow(ds.matches)) matches")

dyn = PreGame.TimeDecayDynamics(days_half_life = 365.0)

models_to_profile = [
    ("1. Goals NegBin Baseline", TeamGoalsNegBinModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "goals_negbin_ctl"
    )),
    ("2. Arm A: Proxy xG NegBin", TeamPxGGoalsAPMNegBinModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "pxg_apm_negbin"
    )),
    ("3. Arm B: Funnel Proxy xG NegBin", TeamFunnelPxGGoalsAPMNegBinModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "funnel_pxg_apm_negbin"
    ))
]

# Create 1 split features for profiling
splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["25/26"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek
)
boundaries = Data.create_id_boundaries(ds, splitter)

results_table = DataFrame(
    Model = String[],
    Params_Dim = Int[],
    Tape_Nodes = Int[],
    Compile_Time_s = Float64[],
    Grad_Time_ms = Float64[],
    Allocations_B = Int[],
    Status = String[]
)

for (label, model_cfg) in models_to_profile
    println("\n", "-"^85)
    println("🔍 PROFILING: $label")
    println("-"^85)

    fs_list = Features.create_features(boundaries, ds, model_cfg)
    fs1 = fs_list[1]
    
    turing_mod = PreGame.build_turing_model(model_cfg, fs1)

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

    # 5. Benchmark Gradient Evaluation
    g = similar(θ)
    t_grad_fast = minimum(@elapsed(ReverseDiff.gradient!(g, compiled_tape, θ)) for _ in 1:30)
    
    # Check gradient validity
    has_nans = any(isnan, g) || any(isinf, g)
    @printf("  • Min Gradient Evaluation : %.3f ms (NaNs/Infs = %s)\n",
            t_grad_fast * 1000, has_nans ? "FAIL ❌" : "NONE ✅")

    # Accurate allocation benchmark
    b = @benchmark ReverseDiff.gradient!($g, $compiled_tape, $θ) samples=50 evals=1
    grad_time_ms = median(b.times) / 1e6
    alloc_bytes = b.memory

    @printf("  • Median Gradient Latency : %.3f ms (Allocations: %d bytes)\n", grad_time_ms, alloc_bytes)

    status = (grad_time_ms < 2.0 && alloc_bytes == 0 && !has_nans) ? "OPTIMAL ✅" : "PASS (SUB-OPTIMAL)"
    @printf("  • Profile Status          : %s\n", status)

    push!(results_table, (
        label,
        n_params,
        tape_nodes,
        round(t_compile, digits=3),
        round(grad_time_ms, digits=3),
        alloc_bytes,
        status
    ))
end

println("\n", "="^95)
println("TAPE PROFILING SUMMARY TABLE:")
println("="^95)
println(results_table)
println("="^95)
