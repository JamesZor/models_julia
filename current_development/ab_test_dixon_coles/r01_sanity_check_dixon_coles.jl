# current_development/ab_test_dixon_coles/r01_sanity_check_dixon_coles.jl

using Revise
using BayesianFootball
using DataFrames
using Turing

using ThreadPinning
pinthreads(:cores)

const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Data = BayesianFootball.Data

# ==========================================
# 1. SETUP & DATA
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())

# Optionally, you can summarize Betfair market odds here if you want to test PPD MAE against Betfair
# odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
# ds = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

# ==========================================
# 2. SHARED COMPONENT CONFIGURATION
# ==========================================
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() # Included for interface consistency
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# ==========================================
# 3. MODEL 1: NO MARKET
# ==========================================
println("[INFO] Initializing DynamicDixonColesXGOutfieldPlayerTimeDecayNoMarketModel...")
model_dc_no_market = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayNoMarketModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes
)

# ==========================================
# 4. RUN NO MARKET EXPERIMENT
# ==========================================
println("[INFO] Creating Task 1 (No Market, Season 2026)...")
task_no_market = Experiments.create_experiment_task(
    ds, 
    model_dc_no_market, 
    "sanity_dc_no_market", 
    "./tmp_mcmc_checkpoints/"; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=1000,
    warmup=500,  
    chains=16,     # Reduced for sanity check
    use_queue=true,
)

println("[INFO] Running No Market Experiment...")
results_no_market = Experiments.run_experiment(task_no_market)

println("[INFO] No Market Completed. Summarizing Split 1, Chain 1...")
# Extracting chains array to visually check rhats and ess
chains_no_market = results_no_market.training_results[1][1]
display(describe(chains_no_market))


# ==========================================
# 5. MODEL 2: WITH MARKET
# ==========================================
println("\n[INFO] Initializing DynamicDixonColesXGOutfieldPlayerTimeDecayModel (WITH Market)...")
model_dc_market = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
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
# 6. RUN MARKET EXPERIMENT
# ==========================================
println("[INFO] Creating Task 2 (Market, Season 2026)...")
task_market = Experiments.create_experiment_task(
    ds, 
    model_dc_market, 
    "sanity_dc_market", 
    "./tmp_mcmc_checkpoints/"; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=1000,
    warmup=500,  
    chains=8,     # Reduced for sanity check
    use_queue=true,
)

println("[INFO] Running Market Experiment...")
results_market = Experiments.run_experiment(task_market)

println("[INFO] Market Completed. Summarizing Split 1, Chain 1...")
chains_market = results_market.training_results[1][1]
display(describe(chains_market))

println("\n[INFO] Both Sanity Checks Complete!")
