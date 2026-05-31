# current_development/ab_test_dixon_coles/r02_ab_test_ireland.jl

using Revise
using BayesianFootball
using DataFrames
using Distributions
using ThreadPinning
using ProgressMeter

# Pin threads for maximum performance
pinthreads(:cores)

const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Data = BayesianFootball.Data

# ==========================================
# 1. SETUP & DATA
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())

save_dir::String = "./data/dixon_coles_ab/"
mkpath(save_dir)

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

# Fast experiment parameters matching the Double Poisson tests
samples = 800
warmup  = 300
chains  = 4
target_seasons = ["2025", "2026"]

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================

# Model 1: Double Poisson No Market
model_dp_nm = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel(
  interception_config = inter_cfg,
  player_dynamics_config = dyn_cfg,
  dispersion_config = disp_cfg,
  homeadvantage_config = ha_cfg,
  kappa_config = kap_cfg,
  player_ratings_feature = feature_cfg_bayes,
)

# Model 2: Double Poisson Market
model_dp_m = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 0.4
)

# Model 3: Dixon Coles No Market
model_dc_nm = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayNoMarketModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes
)

# Model 4: Dixon Coles Market
model_dc_m = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
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
# 4. TASK CREATION
# ==========================================

task_dp_nm = Experiments.create_experiment_task(
    ds, model_dp_nm, "DoublePoisson_NoMarket", save_dir; 
    target_seasons=target_seasons, dynamics_col=:match_biweek,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

task_dp_m = Experiments.create_experiment_task(
    ds, model_dp_m, "DoublePoisson_Market", save_dir; 
    target_seasons=target_seasons, dynamics_col=:match_biweek,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

task_dc_nm = Experiments.create_experiment_task(
    ds, model_dc_nm, "DixonColes_NoMarket", save_dir; 
    target_seasons=target_seasons, dynamics_col=:match_biweek,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

task_dc_m = Experiments.create_experiment_task(
    ds, model_dc_m, "DixonColes_Market", save_dir; 
    target_seasons=target_seasons, dynamics_col=:match_biweek,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

# ==========================================
# 5. RUN EXPERIMENTS
# ==========================================

println("--- Running Double Poisson No Market ---")
res_dp_nm = Experiments.run_experiment(task_dp_nm)
Experiments.save_experiment(res_dp_nm)

println("--- Running Double Poisson Market ---")
res_dp_m = Experiments.run_experiment(task_dp_m)
Experiments.save_experiment(res_dp_m)

println("--- Running Dixon Coles No Market ---")
res_dc_nm = Experiments.run_experiment(task_dc_nm)
Experiments.save_experiment(res_dc_nm)

println("--- Running Dixon Coles Market ---")
res_dc_m = Experiments.run_experiment(task_dc_m)
Experiments.save_experiment(res_dc_m)

all_results = [res_dp_nm, res_dp_m, res_dc_nm, res_dc_m]

# ==========================================
# 6. EVALUATION & BACKTESTING
# ==========================================

# Evaluate with Betfair Odds
odds = Data.summarize_betfair_market(
    ds, 
    open_window=(-100000.0, -10.0), 
    close_window=(-20.0, 0.0)
)
ds1 = Data.DataStore(
  ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds
)

println("\n===========================================")
println("📊 GLM Edge Evaluation (Betfair Odds)")
println("===========================================")
eval_glmedge = Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1)
Evaluation.display_summary_metric(eval_glmedge, :glmedge)

println("\n===========================================")
println("📉 LogLoss Evaluation (Betfair Odds)")
println("===========================================")
eval_logloss = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1)
Evaluation.display_summary_metric(eval_logloss, :logloss)


println("\n===========================================")
println("💰 Backtesting Strategy (Kelly)")
println("===========================================")
ledger = BackTesting.run_backtest(
    ds1, 
    all_results, 
    [BayesianFootball.Signals.BayesianKelly()]; 
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BackTesting.generate_tearsheet(ledger)

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct]
show(tearsheet[:, cols_to_show], allrows=true)

println("\nDone! Full A/B test complete.")



model_names = unique(tearsheet.selection)

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub[!, cols_to_show])
end

