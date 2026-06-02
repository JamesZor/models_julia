# current_development/ab_test_dixon_coles/r04_sanity_check_monthly_interception.jl

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
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# ==========================================
# 3. MODEL 1: GLOBAL INTERCEPTION
# ==========================================
println("[INFO] Initializing DynamicDixonColesXGOutfieldPlayerTimeDecayModel (GLOBAL INTERCEPTION)...")
model_global = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = PreGame.GlobalInterception(),
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    dixon_coles_config     = PreGame.HierarchicalTeamDixonColesConfig(),
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DixonColesMarketFeature(),
    market_weight          = 0.4
)

# ==========================================
# 4. RUN GLOBAL INTERCEPTION EXPERIMENT
# ==========================================
println("[INFO] Creating Task 1 (Global Interception, Season 2026)...")
task_global = Experiments.create_experiment_task(
    ds, 
    model_global, 
    "sanity_dc_global_inter", 
    "./tmp_mcmc_checkpoints/"; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=500,
    warmup=500,  
    chains=4,
    use_queue=true,
)

println("[INFO] Running Global Interception Experiment...")
results_global = Experiments.run_experiment(task_global)

println("[INFO] Global Interception Completed. Summarizing Split 1, Chain 1...")
chains_global = results_global.training_results[1][1]
display(describe(chains_global))


# ==========================================
# 5. MODEL 2: HIERARCHICAL MONTHLY INTERCEPTION
# ==========================================
println("\n[INFO] Initializing DynamicDixonColesXGOutfieldPlayerTimeDecayModel (MONTHLY INTERCEPTION)...")
model_monthly = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = PreGame.HierarchicalMonthlyInterception(),
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    dixon_coles_config     = PreGame.HierarchicalTeamDixonColesConfig(),
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DixonColesMarketFeature(),
    market_weight          = 0.4
)

# ==========================================
# 6. RUN HIERARCHICAL MONTHLY INTERCEPTION EXPERIMENT
# ==========================================
println("[INFO] Creating Task 2 (Monthly Interception, Season 2026)...")
task_monthly = Experiments.create_experiment_task(
    ds, 
    model_monthly, 
    "sanity_dc_monthly_inter", 
    "./tmp_mcmc_checkpoints/"; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=500,
    warmup=500,  
    chains=4,
    use_queue=true,
)

println("[INFO] Running Monthly Interception Experiment...")
results_monthly = Experiments.run_experiment(task_monthly)

println("[INFO] Monthly Interception Completed. Summarizing Split 1, Chain 1...")
chains_monthly = results_monthly.training_results[1][1]
display(describe(chains_monthly))

println("\n[INFO] Both Sanity Checks Complete! Check the trace of `inter.δ_month` to ensure convergence.")
