# current_development/ab_test_outfield_player/r03_sanity_check_double_poisson.jl

using Revise
using BayesianFootball
using DataFrames
using Turing

using ThreadPinning
pinthreads(:cores)

const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments

# ==========================================
# 1. SETUP & DATA
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

# ==========================================
# 2. SHARED COMPONENT CONFIGURATION
# ==========================================
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================
println("[INFO] Initializing DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel...")
model_dp = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 0.4
)

# ==========================================
# 4. CREATE EXPERIMENT TASK (For Splitting)
# ==========================================
println("[INFO] Creating Experiment Task (Target Season: 2026)...")
task = Experiments.create_experiment_task(
    ds, 
    model_dp, 
    "sanity_check_double_poisson", 
    "./tmp_mcmc_checkpoints/"; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=1000,
    warmup=500,  
    chains=16,
    use_queue=true,
)

println("[INFO] Running Experiment...")
results = Experiments.run_experiment(task)

println("[INFO] Experiment Completed. Summarizing Split 1, Chain 1...")
describe(results.training_results[1][1])



# analysis
using Statistics
using DataFrames

mp =Predictions.model_inference(ds, results)
# 1. Calculate the mean probability of the posterior predictive distribution
mp_summary = transform(mp.df, :distribution => ByRow(mean) => :model_prob)

# 2. Join the model predictions with the historical odds data
comparison_df = innerjoin(
mp_summary[!, [:match_id, :selection, :model_prob]],
ds.odds[!, [:match_id, :is_winner, :selection, :odds_close, :prob_implied_close, :prob_fair_close]],
on = [:match_id, :selection]
)

# 3. Calculate how far off our model is from the market's true fair probability
comparison_df.prob_diff = comparison_df.model_prob .- comparison_df.prob_fair_close

# 4. Add the model's implied fair odds for easy reading
comparison_df.model_odds = 1.0 ./ comparison_df.model_prob

# Sort for readability
sort!(comparison_df, [:match_id, :market_name, :selection])

# Display the 1X2 market predictions as a quick sanity check
println("=== 1X2 Market Comparison ===")
display(subset(comparison_df, :selection => ByRow(==(:over_25))))

# Display the Mean Absolute Error (MAE) across all predicted markets
mae = mean(abs.(comparison_df.prob_diff))
println("\nMean Absolute Error vs Market: ", round(mae, digits=4))

describe(comparison_df.prob_diff)



# ==========================================
# M2. Double poisosn MODEL INITIALIZATION
# ==========================================

model_a_no_market = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel(
  interception_config = inter_cfg,
  player_dynamics_config = dyn_cfg,
  dispersion_config = disp_cfg,
  homeadvantage_config = ha_cfg,
  kappa_config = kap_cfg,
  player_ratings_feature = feature_cfg_bayes,
)

# ==========================================
# 4. CREATE EXPERIMENT TASK (For Splitting)
# ==========================================
println("[INFO] Creating Experiment Task (Target Season: 2026)...")
task = Experiments.create_experiment_task(
    ds, 
    model_a_no_market, 
    "sanity_check_double_poisson", 
    "./tmp_mcmc_checkpoints/"; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=1000,
    warmup=500,  
    chains=16,
    use_queue=true,
)

println("[INFO] Running Experiment...")
results = Experiments.run_experiment(task)

