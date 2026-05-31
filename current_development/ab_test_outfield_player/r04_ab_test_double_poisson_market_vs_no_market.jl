# current_development/ab_test_outfield_player/r04_ab_test_double_poisson_market_vs_no_market.jl

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


ds = Data.load_datastore_cached(Data.Ireland())

save_dir::String = "./data/dp_poisson_ab/"
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

samples=800
warmup=300
chains = 4




# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================
#
model_a_no_market = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel(
  interception_config = inter_cfg,
  player_dynamics_config = dyn_cfg,
  dispersion_config = disp_cfg,
  homeadvantage_config = ha_cfg,
  kappa_config = kap_cfg,
  player_ratings_feature = feature_cfg_bayes,
)


model_dp = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 0.4
)



task_nm = Experiments.create_experiment_task(
    ds, 
    model_a_no_market, 
    "no_market_model", 
    save_dir; 
    target_seasons=["2025","2026"], 
    dynamics_col=:match_biweek,
    warmup_period = 0,
    samples=samples,
    warmup=warmup,  
    chains=chains,
    use_queue=true,
)


task_dp = Experiments.create_experiment_task(
    ds, 
    model_dp, 
    "market_model", 
    save_dir; 
    target_seasons=["2025","2026"], 
    dynamics_col=:match_biweek,
    warmup_period = 0,
    samples=samples,
    warmup=warmup,  
    chains=chains,
    use_queue=true,
)


results_a = Experiments.run_experiment(task_nm)
Experiments.save_experiment(results_a)

println("--- Running Model B (Market) ---")
results_b = Experiments.run_experiment(task_dp)
Experiments.save_experiment(results_b)


saved_files = Experiments.list_experiments(save_dir, data_dir="")
results_all = Experiments.load_experiment(saved_files, 1)
results_outfield = Experiments.load_experiment(saved_fiels, 2)

ledger = BackTesting.run_backtest(
    ds, 
    [results_a, results_b], 
    [BayesianFootball.Signals.BayesianKelly()]; 
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BackTesting.generate_tearsheet(ledger)

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct]
show(tearsheet[:, cols_to_show], allrows=true)

metrics = [
    Evaluation.RQR(),
    Evaluation.LogLoss(), 
    Evaluation.CRPS(), 
    Evaluation.GLMEdge()
]
master_eval_df = Evaluation.evaluate_experiments(metrics, [results_a, results_b], ds)

Evaluation.display_summary_metric(master_eval_df, :logloss)
Evaluation.display_summary_metric(master_eval_df, :glmedge)
Evaluation.display_summary_metric(master_eval_df, :rqr)

