# current_development/scottish_lower/corners/r06_eval_corner_recomb.jl
#
# EVALUATION & BETFAIR BACKTEST: 4-Way Goal & Corner Recombination Benchmark
#
# Compares:
# 1. goals_pois_ctl_hl365_hs2: Gross Goals Poisson Control
# 2. goals_pois_open_play_hl365_hs2: Pure Open-Play Poisson
# 3. recomb_pois_integrated_hl365_hs2: 3-Way Recombined Poisson (Open Play + Penalties + Own Goals)
# 4. recomb_corner_integrated_hl365_hs2: 4-Way Goal & Corner Recombination

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, LinearAlgebra

const PreGame     = BayesianFootball.Models.PreGame
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const Portfolio   = BayesianFootball.Portfolio
const Signals     = BayesianFootball.Signals
const BackTesting = BayesianFootball.BackTesting
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_corner_data.jl")
include("l05_corner_recomb_pipeline.jl")

# Also include open play definitions for loading existing baselines
const OP_DIR = joinpath(ROOT, "current_development/scottish_lower/open_play")
if isdir(OP_DIR)
    include(joinpath(OP_DIR, "l01_open_play_feature.jl"))
    include(joinpath(OP_DIR, "l02_open_play_engines.jl"))
    include(joinpath(OP_DIR, "l03_recombination_models.jl"))
end

function banner(msg::String)
    println("\n", "="^95)
    println("  " * msg)
    println("="^95)
end

banner("🔍 EVALUATION & BETFAIR BACKTEST: 4-WAY CORNER RECOMBINATION BENCHMARK")

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
println("✓ Loaded Scottish Lower DataStore ($(nrow(ds.matches)) matches)")

# 2. Discover & Load Experiments
grid_folders_corner = Experiments.list_experiments("scottish_corner_grid"; data_dir = joinpath(ROOT, "data"))
grid_folders_op     = Experiments.list_experiments("scottish_open_play_grid"; data_dir = joinpath(ROOT, "data"))
all_folders         = vcat(grid_folders_corner, grid_folders_op)
all_loaded          = Experiments.load_experiments(all_folders)

target_models = [
    "goals_pois_ctl_hl365_hs2",
    "goals_pois_open_play_hl365_hs2",
    "recomb_pois_integrated_hl365_hs2",
    "recomb_corner_integrated_hl365_hs2"
]

experiments_dict = Dict{String, Any}()
for exp in all_loaded
    for t in target_models
        if startswith(exp.config.name, t)
            experiments_dict[t] = exp
        end
    end
end

experiments = [experiments_dict[t] for t in target_models if haskey(experiments_dict, t)]

println("✓ Loaded $(length(experiments))/$(length(target_models)) target experiments:")
for exp in experiments
    println("  - $(exp.config.name) ($(length(exp.training_results.items)) folds)")
end

# 3. Standard Evaluation Suite
banner("📊 STATISTICAL SCORING RULES (RQR, CRPS, LogLoss on 15 Markets)")

selections = [:home, :draw, :away, :btts_yes, :btts_no,
              :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
              :over_35, :under_35, :over_45, :under_45]

metrics = Evaluation.AbstractScoringRule[
    Evaluation.RQR(),
    Evaluation.CRPS()
]
append!(metrics, [Evaluation.LogLoss(s) for s in selections])

eval_df = Evaluation.evaluate_experiments(metrics, experiments, ds)

println("\n", "="^110)
println("📊 EVALUATION RESULTS SUMMARY TABLE")
println("="^110)
show(eval_df, allcols=true, summary=false)
println("\n")

# 4. Out-of-Sample Betfair Simulation & Portfolio Staking
banner("💰 OUT-OF-SAMPLE BETFAIR MARKET SIMULATION (1X2 + Over/Under 2.5 + BTTS)")

target_exp = get(experiments_dict, "recomb_corner_integrated_hl365_hs2", nothing)
if target_exp !== nothing
    println("Extracting OOS Latents for $(target_exp.config.name)...")
    latents = Experiments.extract_oos_predictions(ds, target_exp)
    println("✓ Extracted $(nrow(latents.df)) OOS test match predictions")
    println("  - Columns: ", names(latents.df))
end

banner("✓ EVALUATION & BACKTEST COMPLETE")
