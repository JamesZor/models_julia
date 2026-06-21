# current_development/ab_test_dixon_coles/r03_bigchance_ab_ireland.jl
#
# A/B test: does a Big-Chances-Created pillar help the Double-Poisson outfield model?
#
#   M1  {goals, market, xG}            — DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel
#   M2  {goals, market, bigChance}     — BigChance model, xg_weight = 0.0
#   M3  {goals, market, bigChance, xG} — BigChance model, xg_weight = 1.0
#
# bigChance pillar: bigChance_side ~ RobustNegativeBinomial(r_bc, c·rate_side),
# NB2 (EDA: eda/ireland_validation/bigchancecreated_eda.md). Judge OUT-OF-SAMPLE
# (LogLoss diff vs market, GLM edge, Kelly backtest) — NOT in-sample likelihood.

using Revise
using BayesianFootball
using DataFrames
using Distributions
using ThreadPinning
using ProgressMeter

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

save_dir::String = "./data/bigchance_ab/"
mkpath(save_dir)

# ==========================================
# 2. SHARED COMPONENT CONFIGURATION (identical to r02 A/B)
# ==========================================
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

samples = 800
warmup  = 300
chains  = 4
target_seasons = ["2025", "2026"]

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================

# M1: {goals, market, xG} — baseline (existing engine)
model_xg = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 0.4,
)

# M2: {goals, market, bigChance} — xG pillar OFF (xg_weight = 0.0)
model_bc = PreGame.DynamicDoublePoissonBigChanceOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 0.4,
    xg_weight              = 0.0,
    bigchance_weight       = 1.0,
)

# M3: {goals, market, bigChance, xG} — both auxiliary pillars ON
model_bc_xg = PreGame.DynamicDoublePoissonBigChanceOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 0.4,
    xg_weight              = 1.0,
    bigchance_weight       = 1.0,
)

# ==========================================
# 4. TASK CREATION
# ==========================================
task_xg = Experiments.create_experiment_task(
    ds, model_xg, "DP_Goals_Market_XG", save_dir;
    target_seasons=target_seasons, dynamics_col=:match_biweek,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

task_bc = Experiments.create_experiment_task(
    ds, model_bc, "DP_Goals_Market_BigChance", save_dir;
    target_seasons=target_seasons, dynamics_col=:match_biweek,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

task_bc_xg = Experiments.create_experiment_task(
    ds, model_bc_xg, "DP_Goals_Market_BigChance_XG", save_dir;
    target_seasons=target_seasons, dynamics_col=:match_biweek,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

# ==========================================
# 5. RUN EXPERIMENTS
# ==========================================
println("--- Running M1: {goals, market, xG} ---")
res_xg = Experiments.run_experiment(task_xg)
Experiments.save_experiment(res_xg)

println("--- Running M2: {goals, market, bigChance} ---")
res_bc = Experiments.run_experiment(task_bc)
Experiments.save_experiment(res_bc)

println("--- Running M3: {goals, market, bigChance, xG} ---")
res_bc_xg = Experiments.run_experiment(task_bc_xg)
Experiments.save_experiment(res_bc_xg)

all_results = [res_xg, res_bc, res_bc_xg]

# ==========================================
# 6. EVALUATION & BACKTESTING (Betfair odds)
# ==========================================
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
--- GLM Edge Summary ---   (spread_fair_coef >0 & low p = edge; all non-significant here)
 model                         intercept   spread_fair_coef   p_value
 DP_Goals_Market_BigChance      -2.01345         0.145087      0.8995
 DP_Goals_Market_BigChance_XG   -1.95593        -0.251633      0.8229
 DP_Goals_Market_XG             -1.91256        -0.563816      0.6153
→ none show a significant edge; M2 (bigChance) least-bad (slightly +ve coef).
=#

println("\n===========================================")
println("📉 LogLoss Evaluation (Betfair Odds)")
println("===========================================")
eval_logloss = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1)
Evaluation.display_summary_metric(eval_logloss, :logloss)

#=
--- LogLoss Summary (diff = model - market; lower = closer to market) ---
 model                          model_ll   market_ll   diff_ll
 DP_Goals_Market_BigChance      0.613986    0.602248    +0.011738   ← best (closest to market)
 DP_Goals_Market_BigChance_XG   0.615738    0.602248    +0.013490
 DP_Goals_Market_XG             0.616819    0.602248    +0.014570   ← worst
→ Ranking M2 (bigChance) < M3 (both) < M1 (xG). bigChance is a BETTER attacking
  pillar than xG here; adding xG on top of bigChance (M3) slightly HURTS (redundancy).
  CAVEAT: all three LOSE to market (diff > 0) on this 281-match 2025-26 window —
  this is "bigChance ≥ xG as a signal", NOT found alpha.
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

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct]
show(tearsheet[:, cols_to_show], allrows=true)

#=
--- Backtest (BayesianKelly, Betfair, 281 matches) ---
 model                         selection  bets  turnover  profit  roi_pct  win%
 DP_Goals_Market_XG            away        154    12.96    4.96    38.28   19.5
 DP_Goals_Market_XG            home        147    14.98   -1.53   -10.22   29.9
 DP_Goals_Market_XG            draw        108     3.79    1.12    29.46   22.2
 DP_Goals_Market_BigChance     away        157    13.06    4.21    32.22   19.7
 DP_Goals_Market_BigChance     home        142    13.85   -0.36    -2.57   28.2
 DP_Goals_Market_BigChance     draw        111     3.80    1.40    36.77   26.1
 DP_Goals_Market_BigChance_XG  away        155    12.81    5.06    39.52   20.0
 DP_Goals_Market_BigChance_XG  home        148    14.33   -1.67   -11.62   30.4
 DP_Goals_Market_BigChance_XG  draw        109     3.74    1.22    32.69   22.0

 Aggregated:  M1 (XG)        profit +4.55 / turnover 31.73 → ROI ≈ +14.3%
              M2 (BigChance) profit +5.25 / turnover 30.71 → ROI ≈ +17.1%   ← best
              M3 (Both)      profit +4.61 / turnover 30.88 → ROI ≈ +14.9%
→ Same ordering as LogLoss: M2 > M3 ≳ M1. Away bets carry the profit for all three;
  home bets lose for all. Gaps are within noise at n=281 (≲1 unit profit spread) —
  directionally bigChance ≥ xG, combining both not worth it. Confirm on more data/leagues.
=#

println("\nDone! bigChance A/B test complete.")
