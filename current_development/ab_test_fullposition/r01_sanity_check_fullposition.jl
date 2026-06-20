# current_development/ab_test_fullposition/r01_sanity_check_fullposition.jl
#
# SANITY CHECK for the FULL-POSITION (G/D/M/F) time-decay Dixon-Coles player engine.
# Runs the model on a SINGLE target fold and verifies that we can:
#   1. Sample (check rhat / ess look sane).
#   2. Extract all parameters (8 positional weights, interception, HA, kappa, DC rho).
#   3. Generate predictions (PPD) and compare against the market's fair probabilities.

using Revise
using BayesianFootball
using DataFrames
using Turing

using ThreadPinning
pinthreads(:cores)

const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Data = BayesianFootball.Data

# ==========================================
# 1. SETUP & DATA
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())

# ==========================================
# 2. SHARED COMPONENT CONFIGURATION
# ==========================================
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion()              # Included for interface consistency
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
# NOTE: full-position engine requires PositionalPlayerDynamics (8 global weights)
dyn_cfg   = PreGame.PositionalPlayerDynamics(days_half_life=60.0)

tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# ==========================================
# 3. MODEL: FULL-POSITION DIXON-COLES (WITH MARKET)
# ==========================================
println("[INFO] Initializing DynamicDixonColesXGFullPositionPlayerTimeDecayModel...")
model_fp = PreGame.DynamicDixonColesXGFullPositionPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DixonColesMarketFeature(),
    market_weight          = 0.4
)

# ==========================================
# 4. RUN SINGLE-FOLD EXPERIMENT
# ==========================================
println("[INFO] Creating sanity Task (Season 2026, single fold)...")
task_fp = Experiments.create_experiment_task(
    ds,
    model_fp,
    "sanity_dc_fullposition",
    "./tmp_mcmc_checkpoints/";
    target_seasons=["2026"],
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=1000,
    warmup=500,
    chains=16,     # Reduced for sanity check
    use_queue=true,
)

println("[INFO] Running Full-Position Experiment...")
results_fp = Experiments.run_experiment(task_fp)

# ==========================================
# 5. CHECK SAMPLING (rhat / ess)
# ==========================================
println("[INFO] Completed. Summarizing Split 1, Chain 1...")
chains_fp = results_fp.training_results[1][1]
display(describe(chains_fp))

# Quick eyeball of the 8 positional weights specifically
println("\n=== Positional Weights (p_dyn) ===")
weight_names = [Symbol("p_dyn.w_$(p)_$(s)") for s in ("att", "def") for p in ("G", "D", "M", "F")]
display(describe(chains_fp[weight_names]))

# ==========================================
# 6. CHECK EXTRACTION + PREDICTION (PPD)
# ==========================================
println("\n[INFO] Running model inference (extract + predict)...")
mp = Predictions.model_inference(ds, results_fp)

mp_summary = transform(mp.df, :distribution => ByRow(mean) => :model_prob)

# Join the model predictions with the historical odds data
comparison_df = innerjoin(
    mp_summary[!, [:match_id, :selection, :model_prob]],
    ds.odds[!, [:match_id, :is_winner, :selection, :odds_close, :prob_implied_close, :prob_fair_close]],
    on = [:match_id, :selection]
)

# How far off our model is from the market's fair probability
comparison_df.prob_diff = comparison_df.model_prob .- comparison_df.prob_fair_close
comparison_df.model_odds = 1.0 ./ comparison_df.model_prob
sort!(comparison_df, [:match_id, :selection])

println("\n=== over_25 Market Comparison ===")
display(subset(comparison_df, :selection => ByRow(==(:over_25))))

mae = mean(abs.(comparison_df.prob_diff))
println("\nMean Absolute Error vs Market: ", round(mae, digits=4))
display(describe(comparison_df.prob_diff))

println("\n[INFO] Full-Position Sanity Check Complete!")
