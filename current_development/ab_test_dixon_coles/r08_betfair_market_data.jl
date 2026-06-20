# current_development/ab_test_dixon_coles/r08_betfair_market_data.jl
#
# A/B test: does the SOURCE of the market-anchoring data matter?
#
# The market pillar (outfield_xg_dixon_coles.jl:139-149) ties the model's log-λ /
# ρ toward a MARKET-IMPLIED log-λ / ρ. Those market-implied params are fitted by
# `add_feature!(::AbstractMarketFeatureConfig, ...)` (market_extractors.jl) which
# reads `ds.odds.prob_fair_close`. So whatever odds live in the DataStore at TRAIN
# time are what the model anchors to.
#
# Up to r07 we trained against the default `ds.odds` = SofaScrape / Bet365 prices,
# which carry a fat bookmaker overround (vig). De-vigging a high-margin book leaves
# residual bias. Betfair Exchange close is the sharpest, lowest-margin signal we
# have. This script trains the SAME model (Dixon-Coles xG, market_weight = 0.5,
# half_life = 60) twice, changing ONLY the market data source:
#
#   Model A  "DCMH_Bet365_MktW_50"   <- anchor to Bet365 fair odds  (baseline, vig)
#   Model B  "DCMH_Betfair_MktW_50"  <- anchor to Betfair fair odds (sharp)
#
# Both are then evaluated against the SAME held-out truth priced off the Betfair
# close (vig-free benchmark), through our established lens: curated Tier-1 GLM-edge
# + LogLoss (btts_yes, over_25, under_25, under_15) and the Kelly backtest.
#
# market_weight = 0.5 is the value triangulated in r07 across four metrics
# (all-mkt logloss, all-mkt GLM edge, curated GLM edge, curated logloss).

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

# ==========================================
# 1. DATA — build BOTH market-data variants
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())

# Betfair Exchange close -> the sharp, low-vig market signal.
# Same window settings used for evaluation in r07.
println("[INFO] Summarising Betfair market...")
bf_odds = Data.summarize_betfair_market(
    ds,
    open_window  = (-100000.0, -10.0),
    close_window = (-20.0, 0.0),
)

# DataStore whose `odds` slot is the Betfair-derived fair odds. Anything that reads
# `ds.odds` (the market feature extractor, the backtester, the evaluators) now sees
# Betfair instead of Bet365.
ds_bf = Data.DataStore(
    ds.segment, ds.matches, ds.statistics, bf_odds, ds.lineups, ds.incidents, ds.betfair_odds
)

save_dir::String = "./data/dixon_coles_betfair_market/"
mkpath(save_dir)

# ==========================================
# 2. SHARED MODEL CONFIG (identical for A/B)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

tracker_bayes     = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

samples        = 800
warmup         = 300
chains         = 4
target_seasons = ["2025", "2026"]
dynamics_col   = :match_biweek
const HALF_LIFE    = 60.0
const MARKET_WEIGHT = 0.5

dyn_cfg = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = HALF_LIFE)

# A single factory so the two models are guaranteed identical except for the
# DataStore they are trained on.
function make_model()
    return PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
        interception_config    = inter_cfg,
        player_dynamics_config = dyn_cfg,
        dispersion_config      = disp_cfg,
        homeadvantage_config   = ha_cfg,
        kappa_config           = kap_cfg,
        dixon_coles_config     = PreGame.HierarchicalTeamDixonColesConfig(),
        player_ratings_feature = feature_cfg_bayes,
        market_feature_config  = Features.DixonColesMarketFeature(),
        market_weight          = MARKET_WEIGHT,
    )
end

# ==========================================
# 3. TASKS — same model, different market data
# ==========================================
# NOTE: the ONLY difference is the `ds` argument => which odds the market feature
# is fitted against. Model B trains on ds_bf (Betfair), Model A on ds (Bet365).
task_bet365 = Experiments.create_experiment_task(
    ds, make_model(), "DCMH_Bet365_MktW_50", save_dir;
    target_seasons=target_seasons, dynamics_col=dynamics_col,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

task_betfair = Experiments.create_experiment_task(
    ds_bf, make_model(), "DCMH_Betfair_MktW_50", save_dir;
    target_seasons=target_seasons, dynamics_col=dynamics_col,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
)

tasks = [task_bet365, task_betfair]

# ==========================================
# 4. RUN (or load if already cached)
# ==========================================
all_results = []
for task in tasks
    println("\n--- Running Experiment: $(task.config.name) ---")
    res = Experiments.run_experiment(task)
    Experiments.save_experiment(res)
    push!(all_results, res)
end

# --- Load logic (use instead of the run loop if already trained) ---
saved_files = Experiments.list_experiments(save_dir, data_dir="")
all_results = [Experiments.load_experiment(saved_files, i) for i in 1:length(tasks)]

# ==========================================
# 5. EVALUATION — both judged against the Betfair close (vig-free benchmark)
# ==========================================
# We evaluate BOTH models against ds_bf so the truth/edge baseline is the sharp
# market for both. (Model A still gets a fair shake: it was just anchored to a
# noisier source, but is scored on the same yardstick.)

println("\n=== GLM Edge (all selections, Betfair benchmark) ===")
eval_glm = Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds_bf)
Evaluation.display_summary_metric(eval_glm, :glmedge)

#=
--- GLM Edge Summary ---
2×4 DataFrame
 Row │ model                 glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value 
     │ String                Float64                 Float64                   Float64                     
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_Bet365_MktW_50                 -2.49559                   2.96028                   2.24032e-6
   2 │ DCMH_Betfair_MktW_50                -2.45817                   1.93425                   0.00143856
=#


println("\n=== LogLoss (all selections, Betfair benchmark) ===")
eval_ll = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds_bf)
Evaluation.display_summary_metric(eval_ll, :logloss)

#=
julia> Evaluation.display_summary_metric(eval_ll, :logloss)

--- LogLoss Summary (Lower Diff is Better) ---
2×4 DataFrame
 Row │ model                 logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll 
     │ String                Float64                   Float64                    Float64                 
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_Bet365_MktW_50                   0.550612                    0.58959               -0.0389783
   2 │ DCMH_Betfair_MktW_50                  0.554742                    0.58959               -0.0348483
=#



eval_rqr = Evaluation.evaluate_experiments(Evaluation.RQR(), all_results, ds_bf)
Evaluation.display_summary_metric(eval_rqr, :rqr)

#=
julia> Evaluation.display_summary_metric(eval_rqr, :rqr)

--- RQR Summary ---
2×7 DataFrame
 Row │ model                 rqr_all_mean  rqr_all_std  rqr_all_skewness  rqr_all_kurtosis  rqr_all_shapiro_w  rqr_all_shapiro_p 
     │ String                Float64       Float64      Float64           Float64           Float64            Float64           
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_Bet365_MktW_50    -0.00840051     0.959568        -0.173078        -0.0473616            0.994953          0.0676758
   2 │ DCMH_Betfair_MktW_50    0.0155894      0.931582        -0.0537595       -0.00540494           0.998269          0.862211
=#




# --- Curated Tier-1 lens (the r07 betting book) -----------------------------
# Pooled across the four goal/BTTS lines that carry the real edge.
const CURATED = [:btts_yes, :over_25, :under_25, :under_15]

println("\n=== Curated Tier-1 GLM Edge + LogLoss (Betfair benchmark) ===")
cur_rows = NamedTuple[]
for exp in all_results
    latents = Experiments.extract_oos_predictions(ds_bf, exp)
    g = Evaluation.compute_metric(Evaluation.GLMEdge(CURATED),  exp, ds_bf, latents)
    l = Evaluation.compute_metric(Evaluation.LogLoss(CURATED), exp, ds_bf, latents).overall
    push!(cur_rows, (
        model        = exp.config.name,
        edge_coef    = round(g.spread_fair.coef, digits=3),
        edge_z       = round(g.spread_fair.z_score, digits=2),
        edge_p       = round(g.spread_fair.p_value, digits=4),
        ll_diff      = round(l.diff_ll, digits=4),   # negative = beats Betfair
        n            = g.n_obs,
    ))
end
curated_df = DataFrame(cur_rows)
show(curated_df, allrows=true, allcols=true); println()

#=
julia> show(curated_df, allrows=true, allcols=true); println()
2×6 DataFrame
 Row │ model                 edge_coef  edge_z   edge_p   ll_diff  n     
     │ String                Float64    Float64  Float64  Float64  Int64 
─────┼───────────────────────────────────────────────────────────────────
   1 │ DCMH_Betfair_MktW_50      3.355      2.9   0.0037  -0.0045    915
   2 │ DCMH_Bet365_MktW_50       4.723      4.0   0.0001  -0.0089    915
=#


# ==========================================
# 6. BACKTEST (Kelly) — priced on the Betfair close
# ==========================================
println("\n=== Backtest (BayesianKelly, Betfair) ===")
ledger = BackTesting.run_backtest(
    ds_bf,
    all_results,
    [BayesianFootball.Signals.BayesianKelly()];
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG,
)
tearsheet = BackTesting.generate_tearsheet(ledger)

cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed,
                :turnover, :profit, :roi_pct, :win_rate_pct, :hurdle_E_R, :hurdle_sharpe, :hurdle_G]

# Focus the tearsheet on the curated book.
curated_sels = string.(CURATED)
bt_curated = subset(tearsheet, :selection => ByRow(s -> string(s) in curated_sels))
println("\n--- Curated-book tearsheet (Bet365 vs Betfair anchoring) ---")
show(sort(bt_curated[:, cols_to_show], [:selection, :model_name]); allrows=true, truncate=0)
println("\n\nDone! Betfair-vs-Bet365 market-data A/B complete.")




#=
julia> show(sort(bt_curated[:, cols_to_show], [:selection, :model_name]); allrows=true, truncate=0)
8×12 DataFrame
 Row │ model_name            selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_E_R  hurdle_sharpe  hurdle_G 
     │ String                Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64     Float64        Float64  
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_Bet365_MktW_50   btts_yes             201          52.2          105      4.55     1.21    26.7           55.2      0.1835         0.1714  0.006867
   2 │ DCMH_Betfair_MktW_50  btts_yes             201          48.3           97      4.22     0.96    22.88          55.7      0.2039         0.189   0.007761
   3 │ DCMH_Bet365_MktW_50   over_25              246          51.6          127      8.0      1.62    20.29          49.6      0.1891         0.1554  0.009008
   4 │ DCMH_Betfair_MktW_50  over_25              246          48.0          118      7.08     0.67     9.53          47.5      0.1557         0.1264  0.006647
   5 │ DCMH_Bet365_MktW_50   under_15             222          43.7           97      3.51     1.3     37.04          30.9      0.2203         0.1184  0.0058
   6 │ DCMH_Betfair_MktW_50  under_15             222          46.8          104      3.79     1.62    42.66          30.8      0.2102         0.1138  0.005488
   7 │ DCMH_Bet365_MktW_50   under_25             246          48.0          118      8.42     1.66    19.71          56.8      0.1371         0.1367  0.007205
   8 │ DCMH_Betfair_MktW_50  under_25             246          50.8          125      8.97     1.7     18.94          56.0      0.1214         0.121   0.006108
=#

