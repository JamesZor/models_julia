# ==============================================================================
# r12_profile_grad.jl
#
# Dedicated ReverseDiff AD Gradient Profiler for:
# Recombination Poisson Model vs Recombination Poisson + Squad Wealth Model
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Statistics
using Printf
using Turing
using DynamicPPL
using ReverseDiff
using ForwardDiff
using BenchmarkTools
using LogDensityProblems
using LogDensityProblemsAD

const ROOT = pkgdir(BayesianFootball)
include("l01_open_play_feature.jl")
include("l03_recombination_models.jl")
include("l04_recomb_wealth_models.jl")

println("="^95)
println("🔬 REVERSEDIFF AD GRADIENT PROFILER: RECOMBINATION vs RECOMBINATION + WEALTH")
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

# 2. Build Models
m_base = TeamGoalsRecombIntegratedPoissonModel(name="recomb_pois_base")
m_wlth = TeamGoalsRecombIntegratedPoisWealthModel(name="recomb_pois_wealth")

f_base = Features.create_features(bound1, ds, m_base)
f_wlth = Features.create_features(bound1, ds, m_wlth)

tur_base = PreGame.build_turing_model(m_base, f_base)
tur_wlth = PreGame.build_turing_model(m_wlth, f_wlth)

# 3. Create LogDensityProblems AD Backends (ReverseDiff & ForwardDiff)
ad_base_rd = LogDensityProblemsAD.ADgradient(Val(:ReverseDiff), DynamicPPL.LogDensityFunction(tur_base))
ad_wlth_rd = LogDensityProblemsAD.ADgradient(Val(:ReverseDiff), DynamicPPL.LogDensityFunction(tur_wlth))

ad_base_fd = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), DynamicPPL.LogDensityFunction(tur_base))
ad_wlth_fd = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), DynamicPPL.LogDensityFunction(tur_wlth))

dim_base = LogDensityProblems.dimension(ad_base_rd)
dim_wlth = LogDensityProblems.dimension(ad_wlth_rd)

println("\n" * "="^95)
println("📊 1. MODEL DIMENSIONALITY & TOPOLOGY")
println("="^95)
println("  • Base Recomb Model Latent Dims   : $(dim_base) dimensions")
println("  • Wealth Recomb Model Latent Dims : $(dim_wlth) dimensions (+1 for w_wealth)")

# Initial parameter point (zeros in unconstrained space)
θ_base = zeros(Float64, dim_base)
θ_wlth = zeros(Float64, dim_wlth)

# 4. Verify Gradient Accuracy (ReverseDiff vs ForwardDiff)
println("\n" * "="^95)
println("🔍 2. NUMERICAL GRADIENT VERIFICATION (ReverseDiff vs ForwardDiff)")
println("="^95)

val_base_rd, grad_base_rd = LogDensityProblems.logdensity_and_gradient(ad_base_rd, θ_base)
val_base_fd, grad_base_fd = LogDensityProblems.logdensity_and_gradient(ad_base_fd, θ_base)
diff_base = maximum(abs.(grad_base_rd .- grad_base_fd))

val_wlth_rd, grad_wlth_rd = LogDensityProblems.logdensity_and_gradient(ad_wlth_rd, θ_wlth)
val_wlth_fd, grad_wlth_fd = LogDensityProblems.logdensity_and_gradient(ad_wlth_fd, θ_wlth)
diff_wlth = maximum(abs.(grad_wlth_rd .- grad_wlth_fd))

println("  • Base Model Log-Density       : RD = $(round(val_base_rd, digits=4)), FD = $(round(val_base_fd, digits=4))")
println("  • Base Model Max |∇RD - ∇FD|   : $(diff_base) $(diff_base < 1e-6 ? "✅ EXACT" : "⚠️ MISMATCH")")
println("  • Wealth Model Log-Density     : RD = $(round(val_wlth_rd, digits=4)), FD = $(round(val_wlth_fd, digits=4))")
println("  • Wealth Model Max |∇RD - ∇FD| : $(diff_wlth) $(diff_wlth < 1e-6 ? "✅ EXACT" : "⚠️ MISMATCH")")

# 5. Benchmark ReverseDiff Gradient Evaluation
println("\n" * "="^95)
println("⚡ 3. GRADIENT EXECUTION TIMING & MEMORY ALLOCATION (ReverseDiff Tape)")
println("="^95)

println("Benchmarking Base Recombination Model gradient...")
b_base = @benchmark LogDensityProblems.logdensity_and_gradient($ad_base_rd, $θ_base) samples=200 evals=1
t_base_median = median(b_base.times) / 1e6 # ms
m_base_alloc  = b_base.memory

println("Benchmarking Wealth Recombination Model gradient...")
b_wlth = @benchmark LogDensityProblems.logdensity_and_gradient($ad_wlth_rd, $θ_wlth) samples=200 evals=1
t_wlth_median = median(b_wlth.times) / 1e6 # ms
m_wlth_alloc  = b_wlth.memory

println("\n" * "-"^95)
@printf("Model                                | Median Time (ms) | Min Time (ms) | Allocations (Bytes)\n")
println("-"^95)
@printf("recomb_pois_integrated (Base)        | %16.3f | %13.3f | %19d\n", t_base_median, minimum(b_base.times)/1e6, m_base_alloc)
@printf("recomb_pois_wealth_integrated (+ΔW)  | %16.3f | %13.3f | %19d\n", t_wlth_median, minimum(b_wlth.times)/1e6, m_wlth_alloc)
println("-"^95)

delta_time = t_wlth_median - t_base_median
println("\n📌 Summary:")
println("  • Wealth feature overhead per gradient: $(round(delta_time, digits=4)) ms ($(round((delta_time/t_base_median)*100, digits=2))%)")
println("  • Expected NUTS sample rate: ~$(round(1000.0 / (t_wlth_median * 31), digits=1)) draws/sec per CPU core")

println("\n" * "="^95)
println("✓ GRADIENT PROFILING COMPLETED SUCCESSFULLY!")
println("="^95)
