# current_development/scottish_lower/corners/r05_grid_corner_recomb.jl
#
# RUNNER: 40-Fold Walk-Forward Training Grid for 4-Way Corner Recombination Model
#
# Model:
# recomb_corner_integrated_hl365_hs2:
#   y_goals = y_open_play + y_penalties + y_own_goals + y_corner_goals
#
# Concurrency: 16 threads via Queued NUTS across 40 folds on Scottish Lower (56, 57)

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
include("l01_corner_data.jl")
include("l05_corner_recomb_pipeline.jl")

println("\n", "="^95)
println("🚀 40-FOLD WALK-FORWARD CORNER RECOMBINATION GRID (SCOTTISH LOWER)")
println("="^95)

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
println("✓ Loaded Scottish Lower DataStore: $(nrow(ds.matches)) matches")

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

save_dir = joinpath(ROOT, "data", "scottish_corner_grid")
mkpath(save_dir)

# 3. Model Definition
model_spec = TeamGoalsCornerRecombIntegratedModel(
    dynamics_config      = PreGame.TimeDecayDynamics(days_half_life = 365.0),
    interception_config  = PreGame.SingleInterception(),
    homeadvantage_config = PreGame.GlobalHomeAdvantage(),
    name                 = "recomb_corner_integrated_hl365_hs2"
)

println("\n", "="^95)
println("▶ TRAINING MODEL: $(model_spec.name) ($(length(boundaries)) splits, 3 chains)")
println("="^95)

task = Experiments.create_experiment_task(
    ds,
    model_spec,
    model_spec.name,
    save_dir;
    target_seasons       = ["24/25", "25/26"],
    history_seasons      = 2,
    warmup_period        = 0,
    dynamics_col         = :match_biweek,
    samples              = 800,
    warmup               = 250,
    chains               = 3,
    use_queue            = true,
    max_depth            = 8,
    max_concurrent_tasks = 16
)

t0 = time()
results = Experiments.run_experiment(task)
elapsed = round(time() - t0, digits = 1)

mins = Int(floor(elapsed / 60))
secs = Int(round(elapsed % 60))
println("\n✓ Completed $(model_spec.name) in $(mins)m $(secs)s")

# Save experiment artifact (precomputing OOS latents)
saved_path = Experiments.save_experiment(results; ds=ds, compute_oos=true)
println("✓ Saved artifact to: $saved_path")
println("="^95)
