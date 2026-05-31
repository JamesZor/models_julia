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
Evaluation.display_summary_metric(master_eval_df, :rqr)




master_eval_df = Evaluation.evaluate_experiments(Evaluation.GLMEdge(), [results_a, results_b], ds)
Evaluation.display_summary_metric(master_eval_df, :glmedge)
#=
julia> Evaluation.display_summary_metric(master_eval_df, :glmedge)

--- GLM Edge Summary ---
2×4 DataFrame
 Row │ model            glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value 
     │ String           Float64                 Float64                   Float64                     
─────┼────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ market_model                   -2.80237                  0.772188                  0.000690505
   2 │ no_market_model                -2.80068                  0.667465                  0.00222036
=#

master_eval_df = Evaluation.evaluate_experiments(Evaluation.LogLoss(), [results_a, results_b], ds)
Evaluation.display_summary_metric(master_eval_df, :logloss)


#=
2×5 DataFrame
 Row │ model            logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll  logloss_overall_n_obs 
     │ String           Float64                   Float64                    Float64                  Int64                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ market_model                     0.509354                    0.47488                0.0344746                   6080
   2 │ no_market_model                  0.516673                    0.47488                0.0417938                   6080
=#




#=
# 1. Evaluate ONLY the Over 2.5 Market
    metrics_over_25 = [
	Evaluation.LogLoss(:over_25),
	Evaluation.GLMEdge(:over_25)
    ]

    master_eval_df_over = Evaluation.evaluate_experiments(metrics_over_25, [results_a, results_b], ds)

    # 2. View the isolated metrics
    Evaluation.display_summary_metric(master_eval_df_over, :logloss_over_25)
    Evaluation.display_summary_metric(master_eval_df_over, :glmedge_over_25)
=#


odds =Data.summarize_betfair_market(
    ds, 
    open_window=(-100000.0, -10.0), 
    close_window=(-20.0, 0.0)
)

ds1 = Data.DataStore(
  ds.segment,
  ds.matches,
  ds.statistics,
  odds,
  ds.lineups,
  ds.incidents,
  ds.betfair_odds
  )

master_eval_df = Evaluation.evaluate_experiments(Evaluation.GLMEdge(), [results_a, results_b], ds1)
Evaluation.display_summary_metric(master_eval_df, :glmedge)



#=
julia> Evaluation.display_summary_metric(master_eval_df, :glmedge)                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                            
--- GLM Edge Summary ---                                                                                                                                                                                                                                                                                                    
2×4 DataFrame                                                                                                                                                                                                                                                                                                               
 Row │ model            glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value                                                                                                                                                                                                                       
     │ String           Float64                 Float64                   Float64                                                                                                                                                                                                                                           
─────┼────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                      
   1 │ market_model                   -2.45764                   2.12775                  0.000898592                                                                                                                                                                                                                       
   2 │ no_market_model                -2.45412                   1.53869                  0.00242633
=#


master_eval_df = Evaluation.evaluate_experiments(Evaluation.LogLoss(), [results_a, results_b], ds1)
Evaluation.display_summary_metric(master_eval_df, :logloss)




#=
julia> Evaluation.display_summary_metric(master_eval_df, :logloss)

--- LogLoss Summary (Lower Diff is Better) ---
2×4 DataFrame
 Row │ model            logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll 
     │ String           Float64                   Float64                    Float64                 
─────┼───────────────────────────────────────────────────────────────────────────────────────────────
   1 │ market_model                     0.553619                    0.58959               -0.0359714
   2 │ no_market_model                  0.559696                    0.58959               -0.0298938
=#

