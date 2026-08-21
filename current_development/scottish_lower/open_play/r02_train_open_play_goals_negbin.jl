# current_development/scottish_lower/open_play/r02_train_open_play_goals_negbin.jl
#
# FULL 40-FOLD WALK-FORWARD TRAINING: Open-Play Goals Negative Binomial Model
#
# Target Season: 25/26 (Scottish Championship & League One)
# Benchmark Target: goals_negbin_ctl_hl365_hs2 (All-Goals Negative Binomial Control)

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, MCMCChains

const DD = BayesianFootball.Data
const FF = BayesianFootball.Features
const MM = BayesianFootball.Models
const EE = BayesianFootball.Experiments
const PP = BayesianFootball.Predictions

const ROOT = pkgdir(BayesianFootball)
include("l01_open_play_feature.jl")
include("l02_open_play_engines.jl")

println("\n", "="^95)
println("TRAINING: OPEN-PLAY GOALS NEGATIVE BINOMIAL MODEL (40 FOLDS)")
println("="^95)

# 1. Load DataStore
ds = DD.load_datastore_cached(DD.ScottishLower(), max_age_hours = 10000)
save_dir = joinpath(ROOT, "data/scottish_open_play_grid/"); mkpath(save_dir)
println("✓ Loaded ScottishLower DataStore: $(nrow(ds.matches)) matches")

# 2. Open-Play Goals Model Definition
dyn = MM.PreGame.TimeDecayDynamics(days_half_life = 365.0)

model = TeamGoalsNegBinOpenPlayModel(
    dynamics_config   = dyn,
    dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
    name              = "goals_negbin_open_play_hl365_hs2"
)

using ThreadPinning
pinthreads(:cores)

println("Configuring 40-fold walk-forward task for: $(model.name)...")

exp_task = EE.create_experiment_task(
    ds,
    model,
    model.name,
    save_dir;
    target_seasons       = ["24/25", "25/26"],
    history_seasons      = 2,
    warmup_period        = 20,
    dynamics_col         = :match_biweek,
    samples              = 1200,
    warmup               = 300,
    chains               = 3,
    use_queue            = true,
    max_depth            = 10,
    max_concurrent_tasks = 16
)

println("Launching 40-fold MCMC sampling on mcmc-beast (pinned 32 threads)...")
t0 = time()
results = EE.run_experiment(exp_task)
elapsed = round(time() - t0, digits = 1)
println("✓ MCMC Sampling Completed in $(elapsed)s ($(round(elapsed/60, digits=2)) mins)")

# 3. Save Experiment
EE.save_experiment(results)
println("✓ Experiment saved to disk: $(model.name)")

