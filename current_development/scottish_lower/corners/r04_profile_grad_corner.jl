# current_development/scottish_lower/corners/r04_profile_grad_corner.jl
#
# Dedicated ReverseDiff AD Gradient Profiler for 4-Way Corner Recombination Model

using ThreadPinning
pinthreads(:cores)

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

include("l01_corner_data.jl")
include("l04_turing_corner_model.jl")

println("="^95)
println("🔬 REVERSEDIFF AD GRADIENT PROFILER: 4-WAY CORNER RECOMBINATION MODEL")
println("="^95)

# 1. Ingest Data & Set Up 1,258-Match Training Split
df_all = fetch_scottish_corner_dataset()
df_lower = filter(r -> r.tournament_id in (56, 57), df_all)
sort!(df_lower, :match_datetime)

split_date = Date(2024, 8, 1)
df_train = filter(r -> Date(r.match_date) < split_date, df_lower)

all_teams = sort(unique(vcat(df_train.home_team, df_train.away_team)))
team_to_idx = Dict(t => i for (i, t) in enumerate(all_teams))
n_teams = length(all_teams)

h_idx = [team_to_idx[t] for t in df_train.home_team]
a_idx = [team_to_idx[t] for t in df_train.away_team]

month_idx = [Dates.month(r.match_date) for r in eachrow(df_train)]
league_map = Dict(56 => 1, 57 => 2)
league_idx = [league_map[t] for t in df_train.tournament_id]
n_leagues = 2

max_train_date = maximum(df_train.match_datetime)
decay_rate = log(2.0) / 365.0
match_weights = [exp(-decay_rate * max(0.0, (max_train_date - r.match_datetime).value / (1000 * 3600 * 24))) for r in eachrow(df_train)]

println("✓ Ingested Dataset: $(nrow(df_train)) matches, $n_teams teams, $n_leagues divisions\n")

# 2. Instantiate Turing Model
config = DynamicCornerRecombModel()
turing_mod = build_corner_recomb_engine(
    h_idx,
    a_idx,
    month_idx,
    league_idx,
    Int.(df_train.open_goals_h),
    Int.(df_train.open_goals_a),
    Int.(df_train.corners_h),
    Int.(df_train.corners_a),
    Int.(df_train.corner_goals_h),
    Int.(df_train.corner_goals_a),
    Float64.(df_train.corners_h .> 0),
    Float64.(df_train.corners_a .> 0),
    match_weights,
    n_teams,
    n_leagues,
    config
)

# 3. Profile Static Graph Compilation & Gradient Tape Evaluation
println("--- ReverseDiff Tape Compilation & Gradient Speed ---")

vi = DynamicPPL.VarInfo(turing_mod)
turing_mod(vi)
θ = vi[:]
n_params = length(θ)
@printf("Total Latent Dimensions (θ): %d parameters\n", n_params)

lf = DynamicPPL.LogDensityFunction(turing_mod)
f = x -> LogDensityProblems.logdensity(lf, x)

# Test logdensity evaluation
log_val = f(θ)
@printf("Initial Log-Joint Density log(p(θ, y)): %.4f\n", log_val)

println("\nCompiling ReverseDiff.GradientTape...")
t_compile_start = time()
tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))
t_compile = (time() - t_compile_start) * 1000
@printf("✓ Tape Compiled in: %.1f ms\n", t_compile)

# Evaluate Gradient Performance
grad_buf = similar(θ)
ReverseDiff.gradient!(grad_buf, tape, θ) # Warmup

println("Benchmarking ReverseDiff.gradient! (200 samples)...")
b_eval = @benchmark ReverseDiff.gradient!($grad_buf, $tape, $θ) samples=200 evals=1

t_eval_median_ms = median(b_eval).time / 1e6
allocs = median(b_eval).allocs
bytes = median(b_eval).memory

println("\n" * "="^110)
@printf("%-45s | %8s | %14s | %14s | %10s | %12s\n", 
        "Model Architecture", "# Params", "Tape Compile", "Grad Median", "Allocs", "Status")
println("="^110)

status = (allocs == 0 && t_eval_median_ms < 1.0) ? "⚡ LIGHTNING (<1.0ms, 0 allocs)" : 
         ((allocs == 0 && t_eval_median_ms < 2.5) ? "✓ EXCELLENT (<2.5ms, 0 allocs)" : "⚠️ ACCEPTABLE")

@printf("%-45s | %8d | %11.1f ms | %11.3f ms | %10d | %s\n", 
        "dynamic_corner_recomb_baseline", n_params, t_compile, t_eval_median_ms, allocs, status)
println("="^110)

if allocs == 0
    println("\n>>> SUCCESS: Model achieves 0 ALLOCATIONS per gradient evaluation! Graph is 100% STATIC. <<<")
else
    @warn "Model allocates $(allocs) times ($(bytes) bytes) per gradient step! Check vectorization."
end
