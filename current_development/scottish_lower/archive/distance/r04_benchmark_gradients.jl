# current_development/scottish_lower/distance/r04_benchmark_gradients.jl
#
# RUNNER: ReverseDiff Gradient Tape Profiling & SIMD Vectorization Benchmark
#         Follows Section 8 of docs/turing_ad_performance_guide.md
#
# Evaluates:
#   1. ReverseDiff.GradientTape compilation time
#   2. Gradient evaluation time (target: < 1.0 ms)
#   3. Allocation count per gradient step (target: 0 allocs)

using Revise
using BayesianFootball
using DynamicPPL
using LogDensityProblems
using ReverseDiff
using BenchmarkTools
using Printf
using DataFrames

const PreGame     = BayesianFootball.Models.PreGame
const Experiments = BayesianFootball.Experiments
const Features    = BayesianFootball.Features
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_distance_features.jl")
include("l02_negbin_distance_engines.jl")
include("l03_negbin_wealth_distance_engines.jl")

println("\n", "="^95)
println("🔬 REVERSEDIFF GRADIENT PROFILING BENCHMARK (docs/turing_ad_performance_guide.md)")
println("="^95)

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)

dyn = PreGame.TimeDecayDynamics(days_half_life = 365.0)

test_models = [
    ("Goals NegBin + Distance", TeamGoalsNegBinDistanceModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "goals_negbin_dist"
    )),
    ("Proxy xG NegBin + Distance", TeamPxGGoalsAPMNegBinDistanceModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "pxg_apm_negbin_dist"
    )),
    ("Grand Champion (PxG + Wealth + Distance)", TeamPxGGoalsAPMNegBinWealthDistanceModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "pxg_apm_negbin_wealth_dist"
    ))
]

# Generate a representative training split
config = Training.GroupedRollingTrainingConfig(
    target_seasons  = ["24/25", "25/26"],
    history_seasons = 2,
    dynamics_col    = :match_biweek
)
splits = Training.generate_splits(config, ds)
split1 = splits[1]

results_df = DataFrame(
    model_name       = String[],
    n_params         = Int[],
    compile_time_s   = Float64[],
    grad_time_ms     = Float64[],
    allocs_per_eval  = Int[],
    bytes_per_eval   = Int[],
    status           = String[]
)

for (mname, model_spec) in test_models
    println("\n--- Benchmarking Model: $mname ---")
    
    # 1. Build features and instantiate Turing model
    f_set = Features.create_features([split1], ds, model_spec, :match_biweek)[1]
    tmodel = PreGame.build_turing_model(model_spec, f_set)
    
    # 2. Extract parameter vector θ
    vi = DynamicPPL.VarInfo(tmodel)
    tmodel(vi)
    θ = DynamicPPL.getparams(vi)
    n_params = length(θ)
    println("  • Number of Parameters θ: $n_params")
    
    # 3. LogDensityFunction wrapper
    lf = DynamicPPL.LogDensityFunction(tmodel)
    f = x -> LogDensityProblems.logdensity(lf, x)
    
    # 4. Compile ReverseDiff Tape
    print("  • Compiling ReverseDiff GradientTape... ")
    t_comp_start = time()
    tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))
    compile_time = time() - t_comp_start
    @printf("Done in %.2f s\n", compile_time)
    
    # 5. Benchmark gradient! execution
    grad_out = similar(θ)
    
    # Warmup
    ReverseDiff.gradient!(grad_out, tape, θ)
    
    # Benchmark
    b_res = @benchmark ReverseDiff.gradient!($grad_out, $tape, $θ) samples=50 evals=5
    
    grad_time_ms = median(b_res.times) / 1e6
    allocs = b_res.allocs
    bytes = b_res.memory
    
    status = (grad_time_ms < 1.0 && allocs == 0) ? "🟢 OPTIMAL (<1ms, 0 allocs)" :
             (grad_time_ms < 3.0) ? "🟡 ACCEPTABLE (<3ms)" : "🔴 SLOW (>3ms)"
    
    @printf("  • Median Gradient Time : %.3f ms\n", grad_time_ms)
    @printf("  • Allocations per Eval  : %d allocs (%d bytes)\n", allocs, bytes)
    @printf("  • Benchmark Status      : %s\n", status)
    
    push!(results_df, (mname, n_params, round(compile_time, digits=2), round(grad_time_ms, digits=3), allocs, bytes, status))
end

println("\n", "="^95)
println("📊 REVERSEDIFF GRADIENT PROFILING SUMMARY TABLE")
println("="^95)
println(results_df)
println("="^95)
