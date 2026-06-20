# current_development/ab_test_dixon_coles/r05_ab_test_ireland_monthly_interception.jl

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

save_dir::String = "./data/dixon_coles_monthly_inter_ab/"
mkpath(save_dir)

# ==========================================
# 2. SHARED COMPONENT CONFIGURATION
# ==========================================
# CHANGED: Now using HierarchicalMonthlyInterception!
inter_cfg = PreGame.HierarchicalMonthlyInterception()

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
dynamics_col=:match_biweek

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

# Model 5: Dixon Coles Market (Hierarchical Rho)
model_dc_hm = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
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
# 4. TASK CREATION
# ==========================================

# NOTE: Set dynamics_col to :match_month since we are using Monthly Interception
task_dp_nm = Experiments.create_experiment_task(
    ds, model_dp_nm, "DoublePoisson_NoMarket", save_dir; 
    target_seasons=target_seasons, dynamics_col=dynamics_col,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

task_dp_m = Experiments.create_experiment_task(
    ds, model_dp_m, "DoublePoisson_Market", save_dir; 
    target_seasons=target_seasons, dynamics_col=dynamics_col,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

task_dc_nm = Experiments.create_experiment_task(
    ds, model_dc_nm, "DixonColes_NoMarket", save_dir; 
    target_seasons=target_seasons, dynamics_col=dynamics_col,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

task_dc_m = Experiments.create_experiment_task(
    ds, model_dc_m, "DixonColes_Market", save_dir; 
    target_seasons=target_seasons, dynamics_col=dynamics_col,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

task_dc_hm = Experiments.create_experiment_task(
    ds, model_dc_hm, "DixonColes_Market_Hierarchical", save_dir; 
    target_seasons=target_seasons, dynamics_col=dynamics_col,
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

# NOTE: If you have already ran the above models and have them loaded in your REPL,
# you can comment out the run_experiment() calls above to save time.

println("--- Running Dixon Coles Market (Hierarchical) ---")
res_dc_hm = Experiments.run_experiment(task_dc_hm)
Experiments.save_experiment(res_dc_hm)


# load

saved_files = Experiments.list_experiments(save_dir, data_dir="")
res_dp_nm = Experiments.load_experiment(saved_files, 5)
res_dp_m = Experiments.load_experiment(saved_files, 4)
res_dc_nm = Experiments.load_experiment(saved_files, 3)
res_dc_m = Experiments.load_experiment(saved_files, 2)
res_dc_hm = Experiments.load_experiment(saved_files, 1)


all_results = [res_dp_nm, res_dp_m, res_dc_nm, res_dc_m, res_dc_hm]

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



#=
julia> Evaluation.display_summary_metric(eval_glmedge, :glmedge)

--- GLM Edge Summary ---
5×4 DataFrame
 Row │ model                           glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value 
     │ String                          Float64                 Float64                   Float64                     
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DixonColes_Market                             -2.43451                   1.2658                   0.0198314
   2 │ DixonColes_Market_Hierarchical                -2.4759                    2.49408                  0.000102481
   3 │ DixonColes_NoMarket                           -2.46135                   1.52107                  0.000896477
   4 │ DoublePoisson_Market                          -2.45938                   2.09581                  0.00101046
   5 │ DoublePoisson_NoMarket                        -2.47303                   1.58695                  0.000638487
=#


println("\n===========================================")
println("📉 LogLoss Evaluation (Betfair Odds)")
println("===========================================")
eval_logloss = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1)
Evaluation.display_summary_metric(eval_logloss, :logloss)



#=
julia> Evaluation.display_summary_metric(eval_logloss, :logloss)

--- LogLoss Summary (Lower Diff is Better) ---
5×4 DataFrame
 Row │ model                           logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll 
     │ String                          Float64                   Float64                    Float64                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DixonColes_Market                               0.559031                    0.58959               -0.0305586
   2 │ DixonColes_Market_Hierarchical                  0.552316                    0.58959               -0.0372741
   3 │ DixonColes_NoMarket                             0.561917                    0.58959               -0.0276727
   4 │ DoublePoisson_Market                            0.553858                    0.58959               -0.0357318
   5 │ DoublePoisson_NoMarket                          0.560942                    0.58959               -0.028648
=#


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

#=
"hurdle_G_emp"
 "hurdle_scale"
 "hurdle_n_bets"
 "hurdle_shape"
 "hurdle_avg_stake"
 "hurdle_E_R"
 "hurdle_sharpe"
 "hurdle_p"
 "hurdle_G"
=#


println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct, :hurdle_G_emp, :hurdle_scale, :hurdle_shape, :hurdle_n_bets, :hurdle_avg_stake, :hurdle_E_R, :hurdle_sharpe, :hurdle_p, :hurdle_G]
show(tearsheet[:, cols_to_show], allrows=true)

println("\nDone! Full A/B test complete.")

model_names = unique(tearsheet.selection)

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub[!, cols_to_show]; truncate=0)
end



#=
3. Deploy DCMH on:  over_25 ,  under_25 ,  under_35 ,  btts_yes ,  under_15
=#

