# current_development/scottish_lower/neg_bin/r04_profile_negbin_wealth.jl
#
# RUNNER: ReverseDiff Gradient & AD Tape Profiler for Scottish NegBin + Wealth Models
#
# Profiles all 3 Negative Binomial + Wealth candidate architectures:
# 1. TeamGoalsNegBinWealthModel
# 2. TeamPxGGoalsAPMNegBinWealthModel
# 3. TeamFunnelPxGGoalsAPMNegBinWealthModel
#
# Measures:
# - Tape compilation time
# - ReverseDiff gradient evaluation latency
# - Memory allocations & Tape node footprint
# - Logprob / Gradient finiteness & numerical stability

using Revise
using BayesianFootball
using Turing
using ReverseDiff
using BenchmarkTools
using Printf
using Dates
using DataFrames

const Data     = BayesianFootball.Data
const Features = BayesianFootball.Features
const Samplers = BayesianFootball.Samplers
const ROOT     = pkgdir(BayesianFootball)

include("l02_negbin_wealth_engines.jl")

println("==================================================================")
println(" SCOTTISH LOWER NEGBIN + WEALTH REVERSEDIFF GRADIENT PROFILER")
println("==================================================================")

# 1. Load DataStore & Split
println("\n[1/3] Loading Scottish Lower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
splitter = Data.GroupedCVConfig(
    target_seasons = ["25/26"],
    history_seasons = 2,
    dynamics_col = :match_biweek
)
splits = Data.split_train_test(splitter, ds)
train_df, test_df = splits[1].train_df, splits[1].test_df
println("✓ Train matches: ", nrow(train_df), " | Test matches: ", nrow(test_df))

# 2. Define Model Candidates
models = [
    ("Model 1: Goals NegBin + Wealth", TeamGoalsNegBinWealthModel(
        dynamics_config = PreGame.TimeDecayDynamics(days_half_life = 365.0),
        homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        apm_on = true,
        league_ha_on = true
    )),
    ("Model 2: Proxy xG + RAPM + NegBin + Wealth", TeamPxGGoalsAPMNegBinWealthModel(
        dynamics_config = PreGame.TimeDecayDynamics(days_half_life = 365.0),
        homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        apm_on = true,
        league_ha_on = true
    )),
    ("Model 3: Funnel Proxy xG + RAPM + NegBin + Wealth", TeamFunnelPxGGoalsAPMNegBinWealthModel(
        dynamics_config = PreGame.TimeDecayDynamics(days_half_life = 365.0),
        homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        apm_on = true,
        team_quality_on = true,
        league_ha_on = true
    ))
]

# 3. Profile Each Model
println("\n[2/3] Benchmarking Tape Compilation & Gradient Latency...")

results_df = DataFrame(
    Model = String[],
    CompileTime_s = Float64[],
    GradLatency_ms = Float64[],
    Allocations_B = Int[],
    LogProb = Float64[],
    GradNorm = Float64[]
)

for (label, model) in models
    println("\n------------------------------------------------------------------")
    println("▶ Profiling: ", label)
    println("------------------------------------------------------------------")

    # Extract Features
    t_feat = @elapsed fs = Features.extract_features(model, train_df, ds)
    println("  • Feature extraction: ", round(t_feat * 1000, digits=1), " ms")

    # Build Turing Model
    turing_mod = PreGame.build_turing_model(model, fs)
    vi = Turing.VarInfo(turing_mod)
    theta = vi[Turing.SampleFromPrior()]
    dim = length(theta)
    println("  • Parameter Dimension: ", dim)

    # Compile Gradient Tape
    f(x) = Turing.LogDensityProblems.logdensity(turing_mod, x)
    tape = ReverseDiff.GradientTape(f, theta)
    t_comp = @elapsed compiled_tape = ReverseDiff.compile(tape)
    println("  • Tape Compilation Time: ", round(t_comp, digits=3), " s")

    # Evaluate LogProb & Gradient
    lp = f(theta)
    grad_buf = similar(theta)
    ReverseDiff.gradient!(grad_buf, compiled_tape, theta)
    grad_norm = sqrt(sum(grad_buf .^ 2))

    println("  • Initial LogProb: ", round(lp, digits=2))
    println("  • Gradient L2 Norm: ", round(grad_norm, digits=2))
    @assert isfinite(lp) "ERROR: Non-finite logprob detected!"
    @assert all(isfinite, grad_buf) "ERROR: Non-finite gradients detected!"
    println("  ✓ Logprob & Gradients are fully finite and AD-safe!")

    # Benchmark Gradient Evaluation
    bench = @benchmark ReverseDiff.gradient!($grad_buf, $compiled_tape, $theta) samples=50 evals=1
    mean_lat_ms = mean(bench.times) / 1e6
    min_lat_ms  = minimum(bench.times) / 1e6
    allocs      = bench.allocs

    println("  • Mean Gradient Latency: ", round(mean_lat_ms, digits=3), " ms (Min: ", round(min_lat_ms, digits=3), " ms)")
    println("  • Allocations per eval: ", allocs, " allocs (", bench.memory, " bytes)")

    push!(results_df, (
        label,
        round(t_comp, digits=3),
        round(mean_lat_ms, digits=3),
        bench.memory,
        round(lp, digits=2),
        round(grad_norm, digits=2)
    ))
end

println("\n==================================================================")
println(" SUMMARY: GRADIENT TAPE PROFILING BENCHMARK")
println("==================================================================")
show(results_df, allrows=true, allcols=true); println()
println("✓ ReverseDiff Gradient Profiling Complete!")
