# current_development/scottish_lower/open_play/r09_grid_negbin_recomb.jl
#
# RUNNER: 40-Fold Walk-Forward Training Grid for Integrated Negative Binomial Recombination Model
#
# Model:
# 1. recomb_negbin_integrated_hl365_hs2: Integrated Co-Trained Open-Play NegBin + Whistle Penalty Turing Engine
#
# Concurrency: 16 threads, Queued NUTS (120 tasks across 40 folds)

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
println("🚀 40-FOLD WALK-FORWARD INTEGRATED NEGATIVE BINOMIAL RECOMBINATION (SCOTTISH LOWER)")
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

save_dir = joinpath(ROOT, "data", "scottish_open_play_grid")
mkpath(save_dir)

tag = "recomb_negbin_integrated_hl365_hs2"
model_spec = TeamGoalsRecombIntegratedNegBinModel(name = tag)

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

# Save experiment artifact & precompute OOS latents
saved_path = Experiments.save_experiment(results; ds = ds, compute_oos = true)
println("✓ Saved artifact to: $saved_path")

println("\n", "="^95)
println("✓ INTEGRATED NEGATIVE BINOMIAL RECOMBINATION RUN COMPLETE!")
println("="^95)
