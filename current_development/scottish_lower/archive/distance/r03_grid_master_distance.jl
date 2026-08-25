# current_development/scottish_lower/distance/r03_grid_master_distance.jl
#
# RUNNER: Master 40-Fold MCMC Rolling Grid for Scottish NegBin + Distance + Wealth Models
#
# Runs Queued NUTS MCMC (16 concurrent worker tasks, 3 chains x 1200 samples)
# across 40 rolling out-of-sample splits on Scottish Lower leagues (56, 57).
#
# Architectures:
# 1. goals_negbin_dist_hl365_hs2        (Goals NegBin + Distance)
# 2. pxg_apm_negbin_dist_hl365_hs2      (Proxy xG + RAPM + Distance + NegBin)
# 3. goals_negbin_wealth_dist_hl365_hs2 (Goals NegBin + Wealth + Distance)
# 4. pxg_apm_negbin_wealth_dist_hl365_hs2 (Grand Champion: Proxy xG + Wealth + Distance + RAPM + NegBin)

using Revise
using BayesianFootball
using Turing
using Dates
using Printf

const PreGame     = BayesianFootball.Models.PreGame
const Experiments = BayesianFootball.Experiments
const Training    = BayesianFootball.Training
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_distance_features.jl")
include("l02_negbin_distance_engines.jl")
include("l03_negbin_wealth_distance_engines.jl")

# --- Canonical Scottish Lower Spec ---
const HL      = 365.0
const HS      = 2
const TARGETS = ["24/25", "25/26"]
const DYN_COL = :match_biweek
const SAMPLES = 1200
const WARMUP  = 300
const CHAINS  = 3

println("\n", "="^95)
println("STARTING MASTER 40-FOLD MCMC GRID: DISTANCE & WEALTH NEGBIN ENGINES (SCOTTISH LOWER)")
println("="^95)

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)

dyn  = PreGame.TimeDecayDynamics(days_half_life = HL)
_tag = "hl$(Int(HL))_hs$(HS)"

specs = Tuple{String, String, Any}[
    # Distance Only Models
    ("scottish_distance_negbin_grid", "goals_negbin_dist_$(_tag)", TeamGoalsNegBinDistanceModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "goals_negbin_dist_$(_tag)"
    )),
    ("scottish_distance_negbin_grid", "pxg_apm_negbin_dist_$(_tag)", TeamPxGGoalsAPMNegBinDistanceModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "pxg_apm_negbin_dist_$(_tag)"
    )),
    # Wealth + Distance (Grand Champion) Models
    ("scottish_full_champion_grid", "goals_negbin_wealth_dist_$(_tag)", TeamGoalsNegBinWealthDistanceModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "goals_negbin_wealth_dist_$(_tag)"
    )),
    ("scottish_full_champion_grid", "pxg_apm_negbin_wealth_dist_$(_tag)", TeamPxGGoalsAPMNegBinWealthDistanceModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "pxg_apm_negbin_wealth_dist_$(_tag)"
    ))
]

println("[INFO] Total $(length(specs)) model specs queued for 40-fold MCMC sampling...")

for (cell_idx, (folder_name, exp_name, model)) in enumerate(specs)
    save_dir = joinpath(ROOT, "data", folder_name)
    mkpath(save_dir)

    # Check if already trained to enable resuming if needed
    existing = Experiments.list_experiments(folder_name; data_dir = joinpath(ROOT, "data"))
    already_done = any(f -> contains(f, exp_name), existing)
    if already_done
        println("\n>>> [$(cell_idx)/$(length(specs))] SKIPPING (Already Exists on Disk): $exp_name")
        continue
    end

    println("\n" * "="^85)
    println("[$cell_idx/$(length(specs))] STARTING 40-FOLD MCMC SAMPLING: $exp_name")
    println("="^85)

    task = Experiments.create_experiment_task(
        ds, model, exp_name, save_dir;
        target_seasons       = TARGETS,
        history_seasons      = HS,
        dynamics_col         = DYN_COL,
        samples              = SAMPLES,
        warmup               = WARMUP,
        chains               = CHAINS,
        use_queue            = true,
        max_depth            = 10,
        max_concurrent_tasks = 16
    )

    t0 = time()
    res = Experiments.run_experiment(task)
    Experiments.save_experiment(res)
    elapsed_min = round((time() - t0) / 60.0, digits = 1)
    println("✓ Completed and saved $exp_name in $(elapsed_min) min")
end

println("\n", "="^95)
println("✓ ALL 40-FOLD MCMC GRIDS COMPLETE!")
println("="^95)
