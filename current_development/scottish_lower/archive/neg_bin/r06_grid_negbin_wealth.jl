# current_development/scottish_lower/neg_bin/r06_grid_negbin_wealth.jl
#
# RUNNER: 40-Fold MCMC Rolling Grid for Scottish NegBin + Wealth Models
#
# Runs Queued NUTS MCMC (16 concurrent worker tasks, 3 chains x 1200 samples)
# across 40 rolling out-of-sample splits on Scottish Lower leagues (56, 57).
#
# Grid Architectures:
# 1. goals_negbin_wealth_hl365_hs2   (Goals-Only NegBin + Starting XI Wealth)
# 2. pxg_apm_negbin_wealth_hl365_hs2 (Arm A: Proxy xG + RAPM + Wealth + NegBin)
# 3. funnel_pxg_apm_negbin_wealth_hl365_hs2 (Arm B: 3-Layer Funnel + Quality + RAPM + Wealth + NegBin)

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
include("l02_negbin_wealth_engines.jl")

# --- Canonical Scottish Lower Spec ---
const HL      = 365.0
const HS      = 2
const TARGETS = ["24/25", "25/26"]
const DYN_COL = :match_biweek
const SAMPLES = 1200
const WARMUP  = 300
const CHAINS  = 3

println("\n", "="^95)
println("STARTING 40-FOLD MCMC GRID: NEGBIN + SQUAD WEALTH MODELS (SCOTTISH LOWER)")
println("="^95)

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
save_dir = joinpath(ROOT, "data/scottish_negbin_wealth_grid/"); mkpath(save_dir)

dyn  = PreGame.TimeDecayDynamics(days_half_life = HL)
_tag = "hl$(Int(HL))_hs$(HS)"

specs = Tuple{String, Any}[
    ("goals_negbin_wealth_$(_tag)", TeamGoalsNegBinWealthModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "goals_negbin_wealth_$(_tag)"
    )),
    ("pxg_apm_negbin_wealth_$(_tag)", TeamPxGGoalsAPMNegBinWealthModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "pxg_apm_negbin_wealth_$(_tag)"
    ))
]

println("[INFO] Running $(length(specs)) NegBin + Wealth model specs across 40 folds (16 cores)...")

for (cell_idx, (exp_name, model)) in enumerate(specs)
    println("\n" * "="^85)
    println("[$cell_idx/$(length(specs))] STARTING MODEL GRID: $exp_name")
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
    elapsed_hr = round((time() - t0) / 3600.0, digits = 2)
    println("✓ Completed and saved $exp_name in $(elapsed_hr)h")
end

println("\n", "="^95)
println("✓ 40-FOLD NEGBIN + SQUAD WEALTH GRID COMPLETE!")
println("="^95)
