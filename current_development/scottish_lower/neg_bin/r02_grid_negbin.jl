# current_development/scottish_lower/neg_bin/r02_grid_negbin.jl
#
# 40-FOLD MCMC GRID TRAINING: Robust Negative Binomial (NB2) Goals Models
# Scottish Lower Leagues (56/57: League One & League Two)
#
# Comparing:
# 1. goals_negbin_ctl_hl365_hs2        (Baseline Goals-Only NegBin Control)
# 2. pxg_apm_negbin_hl365_hs2          (Arm A: Proxy xG Gamma + RAPM + NegBin Goals)
# 3. funnel_pxg_apm_negbin_hl365_hs2   (Arm B: Shots Volume Poisson + Proxy xG Quality + RAPM + NegBin Goals)
#
# Executes across all 40 folds (24/25 & 25/26 target seasons) with:
# - max_concurrent_tasks = 16 (pinned 1-to-1 to physical cores 0–15 on mcmc-beast)
# - 1200 samples, 300 warmup, 3 chains

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning
using Dates

pinthreads(:cores)

const Experiments = BayesianFootball.Experiments
const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Samplers    = BayesianFootball.Samplers
const Training    = BayesianFootball.Training
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_negbin_engines.jl")

# --- Canonical Scottish Lower Spec ---
const HL      = 365.0
const HS      = 2
const TARGETS = ["24/25", "25/26"]
const DYN_COL = :match_biweek
const SAMPLES = 1200
const WARMUP  = 300
const CHAINS  = 3

println("\n", "="^95)
println("STARTING 40-FOLD MCMC GRID: ROBUST NEGATIVE BINOMIAL MODELS (SCOTTISH LOWER)")
println("="^95)

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
save_dir = joinpath(ROOT, "data/scottish_negbin_grid/"); mkpath(save_dir)

dyn  = PreGame.TimeDecayDynamics(days_half_life = HL)
_tag = "hl$(Int(HL))_hs$(HS)"

specs = Tuple{String, Any}[
    ("goals_negbin_ctl_$(_tag)", TeamGoalsNegBinModel(
        interception_config  = PreGame.MonthlyInterception(μ_base = Normal(0.25, 0.2), σ_month = Normal(0.0, 0.1)),
        dynamics_config      = dyn,
        homeadvantage_config = PreGame.HierarchicalHomeAdvantage(ha_global = Normal(0.25, 0.1), σ_ha = Normal(0.0, 0.1)),
        dispersion_config    = SCOTTISH_HOMEAWAY_DISPERSION,
        player_ratings_feature = Features.XGPlusMinusFeature(days_half_life = HL, position_structure = :outfield_only),
        w_att_prior          = truncated(Normal(0.05, 0.05), lower = 0.0),
        w_def_prior          = truncated(Normal(0.05, 0.05), lower = 0.0),
        name                 = "goals_negbin_ctl_$(_tag)"
    )),
    ("pxg_apm_negbin_$(_tag)", TeamPxGGoalsAPMNegBinModel(
        interception_config  = PreGame.MonthlyInterception(μ_base = Normal(0.25, 0.2), σ_month = Normal(0.0, 0.1)),
        dynamics_config      = dyn,
        homeadvantage_config = PreGame.HierarchicalHomeAdvantage(ha_global = Normal(0.25, 0.1), σ_ha = Normal(0.0, 0.1)),
        kappa_config         = PreGame.GlobalKappa(log_κ = PXG_LOGK_PRIOR),
        dispersion_config    = SCOTTISH_HOMEAWAY_DISPERSION,
        player_ratings_feature = Features.XGPlusMinusFeature(days_half_life = HL, position_structure = :outfield_only),
        pxg_feature          = Features.ScottishProxyXGFeature(),
        w_att_prior          = truncated(Normal(0.05, 0.05), lower = 0.0),
        w_def_prior          = truncated(Normal(0.05, 0.05), lower = 0.0),
        ν_xg_prior           = PXG_NU_PRIOR,
        name                 = "pxg_apm_negbin_$(_tag)"
    )),
    ("funnel_pxg_apm_negbin_$(_tag)", TeamFunnelPxGGoalsAPMNegBinModel(
        interception_config  = PreGame.MonthlyInterception(μ_base = Normal(0.25, 0.2), σ_month = Normal(0.0, 0.1)),
        dynamics_config      = dyn,
        homeadvantage_config = PreGame.HierarchicalHomeAdvantage(ha_global = Normal(0.25, 0.1), σ_ha = Normal(0.0, 0.1)),
        kappa_config         = PreGame.GlobalKappa(log_κ = PXG_LOGK_PRIOR),
        dispersion_config    = SCOTTISH_HOMEAWAY_DISPERSION,
        player_ratings_feature = Features.XGPlusMinusFeature(days_half_life = HL, position_structure = :outfield_only),
        pxg_feature          = Features.ScottishProxyXGFeature(),
        shot_scale           = 2.2,
        q_prior              = PXG_Q_PRIOR,
        σ_q_prior            = PXG_SIGQ_PRIOR,
        ν_q_prior            = PXG_NU_PRIOR,
        w_att_prior          = truncated(Normal(0.05, 0.05), lower = 0.0),
        w_def_prior          = truncated(Normal(0.05, 0.05), lower = 0.0),
        name                 = "funnel_pxg_apm_negbin_$(_tag)"
    ))
]

println("[INFO] Running $(length(specs)) Negative Binomial model specs across 40 folds...")

for (cell_idx, (exp_name, model)) in enumerate(specs)
    println("\n" * "="^85)
    println("[$cell_idx/$(length(specs))] STARTING MODEL GRID: $exp_name")
    println("="^85)

    sampler = Samplers.QueuedNUTSConfig(
        n_samples   = SAMPLES,
        n_warmup    = WARMUP,
        accept_rate = 0.85,
        n_chains    = CHAINS
    )

    splitter = Data.GroupedCVConfig(
        tournament_groups = [[56, 57]],
        target_seasons    = TARGETS,
        history_seasons   = HS,
        dynamics_col      = DYN_COL
    )

    task = Experiments.create_experiment_task(
        ds, model, exp_name, save_dir;
        splitter = splitter,
        sampler  = sampler,
        force    = true
    )

    t0 = time()
    res = Experiments.run_experiment(task; save = true, max_concurrent_tasks = 16)
    elapsed_hr = round((time() - t0) / 3600.0, digits = 2)
    println("✓ Completed and saved $exp_name in $(elapsed_hr)h")
end

println("\n", "="^95)
println("✓ 40-FOLD NEGATIVE BINOMIAL GRID COMPLETE!")
println("="^95)
