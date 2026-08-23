# ==============================================================================
# r15_profile_grad_pxg.jl
#
# Dedicated ReverseDiff AD Gradient Profiler:
# Comparing Gross Goals, Pure Open Play, Poisson Recombination, Recombination + Wealth,
# and Open-Play Proxy xG (pxG) Recombination + Wealth.
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Statistics
using Printf
using Turing
using DynamicPPL
using ReverseDiff
using BenchmarkTools
using LogDensityProblems

const ROOT = pkgdir(BayesianFootball)
include("l01_open_play_feature.jl")
include("l02_open_play_engines.jl")
include("l03_recombination_models.jl")
include("l04_recomb_wealth_models.jl")
include("l05_recomb_pxg_models.jl")

println("="^95)
println("🔬 REVERSEDIFF AD GRADIENT PROFILER: OPEN-PLAY PROXY xG (pxG) RECOMBINATION")
println("="^95)

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["25/26"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true
)
boundaries = Data.create_id_boundaries(ds, splitter)
bound1 = boundaries[end][1]

println("✓ Test Split (Fold 40): $(length(bound1.history_match_ids)) history matches")

# 2. Benchmark Models
models = [
    ("Gross Goals Poisson (Control)",                     TeamGoalsPoissonModel()),
    ("Pure Open-Play Poisson",                            TeamGoalsPoissonOpenPlayModel()),
    ("recomb_pois_integrated (Base Recomb)",              TeamGoalsRecombIntegratedPoissonModel()),
    ("recomb_pois_wealth_integrated (+ Squad Wealth ΔW)", TeamGoalsRecombIntegratedPoisWealthModel()),
    ("recomb_negbin_integrated (NegBin Recomb)",          TeamGoalsRecombIntegratedNegBinModel()),
    ("recomb_pxg_wealth_integrated (pxG + Wealth ΔW)",   TeamPxGRecombWealthIntegratedModel())
]

println("\n" * "="^110)
@printf("%-52s | %8s | %14s | %14s | %10s\n", "Model Architecture", "# Params", "Tape Compile", "Grad Eval Time", "Status")
println("="^110)

for (label, m) in models
    feats = Features.create_features(bound1, ds, m)
    turing_mod = PreGame.build_turing_model(m, feats)
    
    vi = DynamicPPL.VarInfo(turing_mod)
    turing_mod(vi)
    θ = vi[:]
    n_params = length(θ)
    
    lf = DynamicPPL.LogDensityFunction(turing_mod)
    f = x -> LogDensityProblems.logdensity(lf, x)
    
    t_compile_start = time()
    tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))
    t_compile = round((time() - t_compile_start) * 1000, digits = 1)
    
    grad_buf = similar(θ)
    t_eval = @belapsed ReverseDiff.gradient!($grad_buf, $tape, $θ) samples=200 evals=1
    t_eval_ms = round(t_eval * 1000, digits = 3)
    
    status = t_eval_ms < 1.0 ? "⚡ LIGHTNING (<1.0ms)" : (t_eval_ms < 2.0 ? "✓ EXCELLENT (<2.0ms)" : "⚠️ ACCEPTABLE (<5.0ms)")
    @printf("%-52s | %8d | %11.1f ms | %11.3f ms | %s\n", label, n_params, t_compile, t_eval_ms, status)
end
println("="^110)

println("\n✓ GRADIENT PROFILING COMPLETED SUCCESSFULLY!")
