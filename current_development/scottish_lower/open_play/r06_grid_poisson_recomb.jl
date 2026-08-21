# current_development/scottish_lower/open_play/r06_grid_poisson_recomb.jl
#
# RUNNER: 40-Fold Walk-Forward Training Grid for Poisson Recombination Models
#
# Models:
# 1. goals_pois_ctl_hl365_hs2: Baseline Gross Goals Poisson Control
# 2. goals_pois_open_play_hl365_hs2: Pure Open-Play Goals Poisson
# 3. recomb_pois_integrated_hl365_hs2: Integrated Co-Trained Open-Play + Referee Penalty Poisson
#
# Concurrency: 16 threads, Queued NUTS (120 tasks per model across 40 folds)

using Revise
using BayesianFootball
using Turing, DynamicPPL, MCMCChains
using DataFrames, Dates, Statistics, Printf

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Samplers    = BayesianFootball.Samplers
const Training    = BayesianFootball.Training
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_open_play_feature.jl")
include("l02_open_play_engines.jl")
include("l03_recombination_models.jl")

println("\n", "="^95)
println("🚀 40-FOLD WALK-FORWARD POISSON RECOMBINATION GRID (SCOTTISH LOWER)")
println("="^95)

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
println("✓ Loaded DataStore: $(nrow(ds.matches)) matches")

# 2. Configure 40-Fold Walk-Forward Rolling Splits
splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25", "25/26"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true
)
boundaries = Data.create_id_boundaries(ds, splitter)
println("✓ Generated $(length(boundaries)) walk-forward splits")

# 3. Standard Sampler Config (1200 samples, 300 warmup, 3 chains)
sampler_cfg = Samplers.NUTSConfig(
    n_samples   = 1200,
    n_warmup    = 300,
    n_chains    = 3,
    accept_rate = 0.65,
    max_depth   = 10,
    show_progress = false
)

save_dir = joinpath(ROOT, "data", "scottish_open_play_grid")
mkpath(save_dir)

# 4. Define Model Pipeline
models_to_train = [
    ("goals_pois_open_play_hl365_hs2", TeamGoalsPoissonOpenPlayModel(name="goals_pois_open_play_hl365_hs2")),
    ("recomb_pois_integrated_hl365_hs2", TeamGoalsRecombIntegratedPoissonModel(name="recomb_pois_integrated_hl365_hs2"))
]

for (tag, model_spec) in models_to_train
    println("\n", "="^95)
    println("▶ TRAINING MODEL: $tag ($(length(boundaries)) splits, 3 chains)")
    println("="^95)
    
    task = Experiments.create_experiment_task(
        ds,
        model_spec,
        tag,
        save_dir;
        target_seasons       = ["24/25", "25/26"],
        history_seasons      = 2,
        warmup_period        = 0,
        dynamics_col         = :match_biweek,
        samples              = 1200,
        warmup               = 300,
        chains               = 3,
        use_queue            = true,
        max_depth            = 10,
        max_concurrent_tasks = 16
    )
    
    t0 = time()
    results = Experiments.run_experiment(task)
    elapsed = round(time() - t0, digits = 1)
    
    mins = Int(floor(elapsed / 60))
    secs = Int(round(elapsed % 60))
    println("\n✓ Completed $tag in $(mins)m $(secs)s")
    
    # Save experiment artifact
    saved_path = Experiments.save_experiment(results)
    println("✓ Saved artifact to: $saved_path")
end

println("\n", "="^95)
println("✓ ALL 40-FOLD POISSON RECOMBINATION RUNS COMPLETE!")
println("="^95)
