#=
RUNNER for l02_split_market_poisson.jl — SplitMarketDoublePoissonModel.

Goal (step 2 of r00's plan): show the double-Poisson {goals + xG + SPLIT market + outfield}
model with the rotated (level/supremacy) market pillar CONVERGES, on the same Ireland split
the standard DC/Poisson engines converged on in r00.

Run after: git push (laptop) -> git pull (server) -> RESTART REPL (Revise does not reliably
re-track new @model macros), then:
    include("current_development/split_market_pillar/r02_run_split_poisson.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using ThreadPinning
using ProgressMeter

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Data        = BayesianFootball.Data
const Signals     = BayesianFootball.Signals

include("l02_split_market_poisson.jl")

# ==========================================
# 1. DATA — Betfair market pillar
# ==========================================
ds = Data.load_datastore_cached(Data.Ireland())

# Swap the Betfair closing-line summary into ds.odds BEFORE building features, so the
# supremacy anchor is inverted from Betfair (not SofaScore 1X2). NOTE: r00 computed this but
# passed the un-swapped `ds` to the task — here we pass `ds_market`.
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds_market = Data.DataStore(
    ds.segment, ds.matches, ds.statistics, odds,
    ds.lineups, ds.incidents, ds.betfair_odds
)

save_dir = "./data/split_market_dev_area/"
mkpath(save_dir)

# ==========================================
# 2. SHARED COMPONENT CONFIG (matches r00)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

tracker_bayes     = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

half_life = 60.0
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=half_life)

# ==========================================
# 3. THE SPLIT-MARKET MODEL (sampled σ; supremacy-only anchor)
# ==========================================
model = SplitMarketDoublePoissonModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_on              = true,
    supremacy_weight       = 1.0,
    level_weight           = 0.0,    # anchor supremacy only; let the model own totals
)

task = Experiments.create_experiment_task(
    ds_market,
    model,
    "split_poisson_r1",
    save_dir;
    target_seasons  = ["2026"],
    history_seasons = 2,
    warmup_period   = 21,
    dynamics_col    = :match_week,
    samples         = 1000,
    warmup          = 500,
    chains          = 4,
    use_queue       = true,
    max_depth       = 10,   # sampled σ removes the stiffness -> no need for the depth-6 cap
)

results = Experiments.run_experiment(task)
Experiments.save_experiment(results)

# ==========================================
# 4. CONVERGENCE DIAGNOSTICS (success = all R-hat <= 1.05, as in r00)
# ==========================================
chains_df_all = Experiments.Diagnostics.extract_chains(ds_market, results)
println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all = Experiments.Diagnostics.check_convergence(chains_df_all)

conv_diag_all.df
# Sanity check the new pillar params: ν_xg, σ_sup (should pull tight ~0.05–0.15), σ_lev (looser).

# ==========================================
# CONTROL VARIANTS (uncomment to A/B "is the split worth it")
# ==========================================
# model_market_off = SplitMarketDoublePoissonModel(
#     interception_config=inter_cfg, player_dynamics_config=dyn_cfg, dispersion_config=disp_cfg,
#     homeadvantage_config=ha_cfg, kappa_config=kap_cfg, player_ratings_feature=feature_cfg_bayes,
#     market_on=false, level_weight=0.0,
# )
# model_sup_and_level = SplitMarketDoublePoissonModel(
#     interception_config=inter_cfg, player_dynamics_config=dyn_cfg, dispersion_config=disp_cfg,
#     homeadvantage_config=ha_cfg, kappa_config=kap_cfg, player_ratings_feature=feature_cfg_bayes,
#     market_on=true, level_weight=1.0,
# )
