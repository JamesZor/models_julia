# current_development/ab_test_dixon_coles/r07_grid_search_market_weight.jl
#
# Grid search on the `market_weight` hyperparameter, holding the half-life fixed
# at the "best" value found in r06 (half_life = 60 days).
#
# `market_weight` (see outfield_xg_dixon_coles.jl:149) is a scalar multiplier on
# the MARKET log-likelihood pillar only (the Normal tying model log-λ to the
# market-implied log-λ). The goals + xG pillars are always full weight.
#   - 0.0  -> market term off (pure structural model; baseline)
#   - <1.0 -> market is a soft nudge (r06 used 0.4)
#   - 1.0  -> market counts as ~one pseudo-observation per match
#   - >1.0 -> hard-anchor λ toward the market. Past ~2.0 the model just becomes
#            the market and the betting edge collapses, so 2.0 is the upper bound.

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

save_dir::String = "./data/dixon_coles_market_weight_grid/"
mkpath(save_dir)

# ==========================================
# 2. SHARED COMPONENT CONFIGURATION
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# Fast experiment parameters matching the Double Poisson tests
samples = 800
warmup  = 300
chains  = 4
target_seasons = ["2025", "2026"]
dynamics_col = :match_biweek

# Fixed "best" half-life from r06
const HALF_LIFE = 60.0

# ==========================================
# 3. GRID SEARCH SETUP
# ==========================================
# Dense in [0.1, 1.0] where the action should be, sparser above 1.0.
# 0.0 included as the "market-off" baseline.
market_weights = [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
tasks = []
all_results = []

# Shared dynamics config (half-life is fixed for this sweep)
dyn_cfg = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=HALF_LIFE)

for mw in market_weights
    println("\n[INFO] Creating Model Task for Market Weight: $(mw)")

    # Dixon Coles Market (Hierarchical Rho), sweeping market_weight
    model_dc_hm = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
        interception_config    = inter_cfg,
        player_dynamics_config = dyn_cfg,
        dispersion_config      = disp_cfg,
        homeadvantage_config   = ha_cfg,
        kappa_config           = kap_cfg,
        dixon_coles_config     = PreGame.HierarchicalTeamDixonColesConfig(),
        player_ratings_feature = feature_cfg_bayes,
        market_feature_config  = Features.DixonColesMarketFeature(),
        market_weight          = mw
    )

    # Encode weight in the name (×100 to keep it integer/filename-safe)
    model_name = "DCMH_MktW_$(Int(round(mw * 100)))"

    task = Experiments.create_experiment_task(
        ds, model_dc_hm, model_name, save_dir;
        target_seasons=target_seasons, dynamics_col=dynamics_col,
        warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
    )
    push!(tasks, task)
end

# ==========================================
# 4. RUN EXPERIMENTS
# ==========================================
# for task in tasks
#     println("\n--- Running Experiment: $(task.config.name) ---")
#     res = Experiments.run_experiment(task)
#     Experiments.save_experiment(res)
#     push!(all_results, res)
# end
#
# NOTE: If you have already run the above models and have them loaded in your REPL,
# you can comment out the run loop and use the load logic below instead:
saved_files = Experiments.list_experiments(save_dir, data_dir="")
all_results = [Experiments.load_experiment(saved_files, i) for i in 1:length(market_weights)]

# ==========================================
# 5. EVALUATION & BACKTESTING
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
10×4 DataFrame
 Row │ model          glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value 
     │ String         Float64                 Float64                   Float64                     
─────┼──────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_MktW_0                  -2.46176                   1.51366                  0.00086111
   2 │ DCMH_MktW_10                 -2.46812                   1.98228                  0.0005849
   3 │ DCMH_MktW_100                -2.45158                   1.89848                  0.00224484
   4 │ DCMH_MktW_125                -2.44053                   1.62778                  0.00513117
   5 │ DCMH_MktW_150                -2.44263                   1.77771                  0.00325729
   6 │ DCMH_MktW_200                -2.44184                   1.72743                  0.00325757
   7 │ DCMH_MktW_25                 -2.4715                    2.26675                  0.000279297
   8 │ DCMH_MktW_40                 -2.46645                   2.26849                  0.00038914
   9 │ DCMH_MktW_60                 -2.47314                   2.05788                  0.000137763
  10 │ DCMH_MktW_80                 -2.46494                   2.22396                  0.000299264
=#


println("\n===========================================")
println("📉 LogLoss Evaluation (Betfair Odds)")
println("===========================================")
eval_logloss = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1)
Evaluation.display_summary_metric(eval_logloss, :logloss)

#=
julia> Evaluation.display_summary_metric(eval_logloss, :logloss)

--- LogLoss Summary (Lower Diff is Better) ---
10×4 DataFrame
 Row │ model          logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll 
     │ String         Float64                   Float64                    Float64                 
─────┼─────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_MktW_0                    0.562167                    0.58959               -0.027423
   2 │ DCMH_MktW_10                   0.555464                    0.58959               -0.0341256
   3 │ DCMH_MktW_100                  0.554932                    0.58959               -0.0346582
   4 │ DCMH_MktW_125                  0.555993                    0.58959               -0.0335969
   5 │ DCMH_MktW_150                  0.55561                     0.58959               -0.0339805
   6 │ DCMH_MktW_200                  0.555552                    0.58959               -0.0340383
   7 │ DCMH_MktW_25                   0.553358                    0.58959               -0.0362321
   8 │ DCMH_MktW_40                   0.553055                    0.58959               -0.0365346
   9 │ DCMH_MktW_60                   0.554877                    0.58959               -0.0347128
  10 │ DCMH_MktW_80                   0.553411                    0.58959               -0.0361791
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
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct, :hurdle_G_emp, :hurdle_scale, :hurdle_shape, :hurdle_n_bets, :hurdle_avg_stake, :hurdle_E_R, :hurdle_sharpe, :hurdle_p, :hurdle_G]
show(tearsheet[:, cols_to_show], allrows=true)

println("\nDone! Market-weight Grid Search complete.")

model_names = unique(tearsheet.selection)

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub[!, cols_to_show]; truncate=0)
end




#=
julia> for m_name in model_names                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
           println("\nStats for: $m_name")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
           sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
           show(sub[!, cols_to_show]; truncate=0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
       end                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
Stats for: over_15                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
10×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                                                                                                                                                                                                                                                                                                                                                            
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                                                                                                                                                                                                                                                                                                                                             
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                                                          
   1 │ DCMH_MktW_200  over_15              222          44.6           99     12.83    -0.39    -3.04          66.7     -0.013679        0.018        27.1068             99            0.1296     -0.0075        -0.0107    0.6667  -0.005371                                                                                                                                                                                                                                                                                                                                                                                                           
   2 │ DCMH_MktW_150  over_15              222          43.7           97     11.68    -0.12    -1.06          67.0     -0.009662        0.0186       26.2311             97            0.1205     -0.0037        -0.0053    0.6701  -0.004207                                                                                                                                                                                                                                                                                                                                                                                                           
   3 │ DCMH_MktW_125  over_15              222          44.1           98     12.05     0.01     0.06          67.3     -0.008436        0.0191       25.3782             98            0.123       0.0            0.0001    0.6735  -0.003888                                                                                                                                                                                                                                                                                                                                                                                                           
   4 │ DCMH_MktW_100  over_15              222          44.6           99     11.26    -0.41    -3.61          66.7     -0.01172         0.0188       25.9267             99            0.1137     -0.0091        -0.0129    0.6667  -0.004391                                                                                                                                                                                                                                                                                                                                                                                                           
   5 │ DCMH_MktW_80   over_15              222          45.5          101     10.49    -0.03    -0.26          68.3     -0.006547        0.0187       25.4986            101            0.1039      0.0091         0.0132    0.6832  -0.001743                                                                                                                                                                                                                                                                                                                                                                                                           
   6 │ DCMH_MktW_60   over_15              222          44.1           98     10.06     0.55     5.43          68.4      0.000141        0.0211       22.5053             98            0.1027      0.0083         0.012     0.6837  -0.001767                                                                                                                                                                                                                                                                                                                                                                                                           
   7 │ DCMH_MktW_40   over_15              222          56.8          126     11.66     0.21     1.8           69.8     -0.003257        0.019        24.527             126            0.0926      0.0241         0.0356    0.6984   0.000187                                                                                                                                                                                                                                                                                                                                                                                                           
   8 │ DCMH_MktW_25   over_15              222          61.7          137     13.59     0.38     2.8           69.3     -0.002478        0.0181       25.518             137            0.0992      0.0137         0.0203    0.6934  -0.000996                                                                                                                                                                                                                                                                                                                                                                                                           
   9 │ DCMH_MktW_10   over_15              222          73.0          162     18.23     0.6      3.31          69.1     -0.002528        0.0194       23.0242            162            0.1125     -0.0003        -0.0004    0.6914  -0.00303                                                                                                                                                                                                                                                                                                                                                                                                            
  10 │ DCMH_MktW_0    over_15              222          77.0          171     22.94     0.55     2.38          70.8     -0.004863        0.02         21.6392            171            0.1342      0.0133         0.0203    0.7076  -0.002313                                                                                                                                                                                                                                                                                                                                                                                                           
Stats for: under_15                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
10×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                                                                                                                                                                                                                                                                                                                                                            
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                                                                                                                                                                                                                                                                                                                                             
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                                                          
   1 │ DCMH_MktW_200  under_15             222          54.1          120      7.31     1.74    23.77          27.5      0.004889        0.1312       22.2415            120            0.0609      0.0777         0.0437    0.275   -0.000712                                                                                                                                                                                                                                                                                                                                                                                                           
   2 │ DCMH_MktW_150  under_15             222          55.4          123      6.85     1.87    27.26          28.5      0.006729        0.1338       21.5048            123            0.0557      0.1033         0.058     0.2846   0.001156                                                                                                                                                                                                                                                                                                                                                                                                           
   3 │ DCMH_MktW_125  under_15             222          55.4          123      6.69     1.58    23.6           27.6      0.005023        0.1521       18.9328            123            0.0544      0.0723         0.0409    0.2764  -0.000395                                                                                                                                                                                                                                                                                                                                                                                                           
   4 │ DCMH_MktW_100  under_15             222          53.2          118      5.6      1.63    29.18          28.0      0.006992        0.1333       21.7624            118            0.0475      0.0907         0.0509    0.2797   0.000933                                                                                                                                                                                                                                                                                                                                                                                                           
   5 │ DCMH_MktW_80   under_15             222          52.3          116      5.16     1.81    35.17          29.3      0.00955         0.1319       21.903             116            0.0445      0.1398         0.0776    0.2931   0.003174                                                                                                                                                                                                                                                                                                                                                                                                           
   6 │ DCMH_MktW_60   under_15             222          48.2          107      4.88     1.7     34.87          29.9      0.009599        0.1659       17.3857            107            0.0456      0.1618         0.089     0.2991   0.004124                                                                                                                                                                                                                                                                                                                                                                                                           
   7 │ DCMH_MktW_40   under_15             222          42.3           94      2.98     1.17    39.19          29.8      0.008303        0.1125       26.9637             94            0.0317      0.2012         0.1075    0.2979   0.004679                                                                                                                                                                                                                                                                                                                                                                                                           
   8 │ DCMH_MktW_25   under_15             222          36.9           82      2.04     0.92    45.2           29.3      0.008149        0.1215       25.4029             82            0.0249      0.1962         0.1039    0.2927   0.003811                                                                                                                                                                                                                                                                                                                                                                                                           
   9 │ DCMH_MktW_10   under_15             222          25.2           56      1.08     0.55    50.86          26.8      0.007899        0.1027       33.0456             56            0.0193      0.1771         0.0899    0.2679   0.002715                                                                                                                                                                                                                                                                                                                                                                                                           
  10 │ DCMH_MktW_0    under_15             222          21.6           48      2.13     1.62    75.98          33.3      0.022698        0.1925       16.3784             48            0.0443      0.3843         0.1913    0.3333   0.013302                                                                                                                                                                                                                                                                                                                                                                                                           
Stats for: btts_no                                                                                                                                                                                                                                                                                                          
10×18 DataFrame                                                                                                                                                                                                                                                                                                             
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                               
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                             
   1 │ DCMH_MktW_200  btts_no              201          55.2          111      8.54    -0.25    -2.9           40.5     -0.008621        0.021        51.3783            111            0.0769     -0.1576        -0.1538    0.4054  -0.015211                                                                              
   2 │ DCMH_MktW_150  btts_no              201          56.2          113      8.02    -0.02    -0.25          43.4     -0.00578         0.0211       50.5942            113            0.071      -0.104         -0.1011    0.4336  -0.010041                                                                              
   3 │ DCMH_MktW_125  btts_no              201          56.2          113      7.58    -0.02    -0.25          43.4     -0.005288        0.0284       37.2855            113            0.0671     -0.1076        -0.1048    0.4336  -0.009576                                                                              
   4 │ DCMH_MktW_100  btts_no              201          53.2          107      6.37     0.19     2.93          42.1     -0.002705        0.0188       57.7149            107            0.0595     -0.1232        -0.1192    0.4206  -0.009212                                                                              
   5 │ DCMH_MktW_80   btts_no              201          51.7          104      5.77     0.42     7.24          43.3      0.000212        0.0191       56.9666            104            0.0555     -0.0977        -0.0942    0.4327  -0.007081                                                                              
   6 │ DCMH_MktW_60   btts_no              201          47.3           95      5.17     0.55    10.71          41.1      0.002385        0.023        47.4702             95            0.0545     -0.1407        -0.136     0.4105  -0.009247                                                                              
   7 │ DCMH_MktW_40   btts_no              201          41.8           84      3.23     0.62    19.07          39.3      0.00523         0.0169       67.042              84            0.0384     -0.1631        -0.1562    0.3929  -0.007064                                                                              
   8 │ DCMH_MktW_25   btts_no              201          33.3           67      2.12     0.6     28.33          43.3      0.00747         0.0156       73.7999             67            0.0317     -0.0689        -0.0644    0.4328  -0.002754                                                                              
   9 │ DCMH_MktW_10   btts_no              201          24.4           49      1.01     0.35    34.53          46.9      0.006324        0.0132       90.3056             49            0.0205      0.0271         0.0248    0.4694   0.000303                                                                              
  10 │ DCMH_MktW_0    btts_no              201          16.9           34      1.62     1.14    70.56          58.8      0.030543        0.0267       43.6285             34            0.0476      0.2726         0.254     0.5882   0.011655                                                                              
Stats for: btts_yes                                                                                                                                                                                                                                                                                                         
10×18 DataFrame                                                                                                                                                                                                                                                                                                             
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                               
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                              
   1 │ DCMH_MktW_200  btts_yes             201          43.3           87      5.25     0.64    12.17          56.3      0.003535        0.0182       63.946              87            0.0603      0.2172         0.2016    0.5632  0.010984                                                                               
   2 │ DCMH_MktW_150  btts_yes             201          41.8           84      4.91     1.0     20.42          58.3      0.008309        0.0182       63.946              84            0.0584      0.2607         0.2434    0.5833  0.013269                                                                               
   3 │ DCMH_MktW_125  btts_yes             201          41.8           84      4.58     0.92    20.07          57.1      0.007686        0.0182       63.7653             84            0.0545      0.2333         0.2173    0.5714  0.010995                                                                               
   4 │ DCMH_MktW_100  btts_yes             201          43.3           87      4.63     0.58    12.45          55.2      0.003567        0.017        68.5527             87            0.0532      0.1963         0.1812    0.5517  0.008776                                                                               
   5 │ DCMH_MktW_80   btts_yes             201          44.8           90      4.58     0.95    20.7           56.7      0.007559        0.0171       67.9501             90            0.0509      0.225          0.209     0.5667  0.009944                                                                               
   6 │ DCMH_MktW_60   btts_yes             201          45.3           91      4.1      0.82    19.95          54.9      0.006409        0.0179       63.7691             91            0.0451      0.1755         0.1641    0.5495  0.006751                                                                               
   7 │ DCMH_MktW_40   btts_yes             201          55.2          111      5.06     1.14    22.42          54.1      0.007604        0.017        67.4064            111            0.0456      0.1583         0.1475    0.5405  0.00602                                                                                
   8 │ DCMH_MktW_25   btts_yes             201          62.7          126      6.13     1.42    23.14          57.9      0.00831         0.0163       68.947             126            0.0487      0.2298         0.2182    0.5794  0.009867                                                                               
   9 │ DCMH_MktW_10   btts_yes             201          73.6          148      8.66     2.06    23.74          58.1      0.010024        0.0192       56.8941            148            0.0585      0.2162         0.2082    0.5811  0.010799                                                                               
  10 │ DCMH_MktW_0    btts_yes             201          79.6          160     11.52     2.85    24.77          60.0      0.012546        0.0212       50.6937            160            0.072       0.2442         0.2388    0.6     0.014852                                                                               
Stats for: over_35                                                                                                                                                                                                                                                                                                          
10×18 DataFrame                                                                                                                                                                                                                                                                                                             
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                               
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                             
   1 │ DCMH_MktW_200  over_35              205          43.4           89      4.8      0.79    16.42          23.6     -0.002127        0.2574       15.7099             89            0.054       0.1899         0.0864    0.236    0.003936                                                                              
   2 │ DCMH_MktW_150  over_35              205          42.0           86      4.23    -0.31    -7.33          23.3     -0.009332        0.2079       19.8991             86            0.0491      0.1948         0.0879    0.2326   0.004204                                                                              
   3 │ DCMH_MktW_125  over_35              205          42.0           86      4.38     0.59    13.5           22.1     -0.002876        0.2126       19.3111             86            0.051       0.128          0.0592    0.2209   0.001039                                                                              
   4 │ DCMH_MktW_100  over_35              205          47.3           97      4.31     0.69    15.99          22.7     -0.001079        0.2697       14.7525             97            0.0444      0.1291         0.0603    0.2268   0.001592                                                                              
   5 │ DCMH_MktW_80   over_35              205          45.9           94      3.86    -0.43   -11.22          22.3     -0.008581        0.2278       17.8489             94            0.0411      0.1316         0.061     0.2234   0.001789                                                                              
   6 │ DCMH_MktW_60   over_35              205          47.3           97      3.81     0.62    16.4           22.7     -0.000466        0.2672       14.5692             97            0.0393      0.1096         0.052     0.2268   0.001138                                                                              
   7 │ DCMH_MktW_40   over_35              205          54.6          112      4.52     0.41     9.11          22.3     -0.002706        0.2492       15.576             112            0.0404      0.0895         0.0429    0.2232   0.000339                                                                              
   8 │ DCMH_MktW_25   over_35              205          59.5          122      5.46     0.36     6.54          22.1     -0.003933        0.2664       14.2633            122            0.0447      0.0621         0.0303    0.2213  -0.001072                                                                              
   9 │ DCMH_MktW_10   over_35              205          76.6          157      7.74     0.32     4.18          21.0     -0.005576        0.2907       12.2749            157            0.0493     -0.0397        -0.0207    0.2102  -0.006058                                                                              
  10 │ DCMH_MktW_0    over_35              205          81.0          166     10.87    -0.79    -7.27          21.7     -0.014669        0.2271       14.6143            166            0.0655     -0.0633        -0.0347    0.2169  -0.010575                                                                              
Stats for: under_35                                                                                                                                                                                                                                                                                                         
10×18 DataFrame                                                                                                                                                                                                                                                                                                             
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                               
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                              
   1 │ DCMH_MktW_200  under_35             205          55.1          113     18.92     0.98     5.16          78.8     -0.001525        0.0163       22.8278            113            0.1674      0.0808         0.1428    0.7876  0.008658                                                                               
   2 │ DCMH_MktW_150  under_35             205          57.1          117     17.77     0.71     3.98          77.8     -0.00298         0.0167       22.146             117            0.1519      0.0656         0.1143    0.7778  0.005877                                                                               
   3 │ DCMH_MktW_125  under_35             205          57.1          117     16.63     0.63     3.79          76.1     -0.002793        0.0165       22.612             117            0.1421      0.0437         0.0742    0.7607  0.002457                                                                               
   4 │ DCMH_MktW_100  under_35             205          51.7          106     14.58     0.6      4.14          75.5     -0.002022        0.0153       24.9112            106            0.1375      0.0419         0.07      0.7547  0.002154                                                                               
   5 │ DCMH_MktW_80   under_35             205          54.1          111     13.26     0.55     4.12          77.5     -0.001402        0.0153       24.6184            111            0.1195      0.0667         0.1151    0.7748  0.005428                                                                               
   6 │ DCMH_MktW_60   under_35             205          50.7          104     11.23     0.56     4.94          76.0      4.9e-5          0.0166       22.8142            104            0.108       0.0472         0.0796    0.7596  0.002938                                                                               
   7 │ DCMH_MktW_40   under_35             205          44.4           91      8.05     0.6      7.39          74.7      0.002566        0.0132       29.9329             91            0.0885      0.0419         0.0688    0.7473  0.002193                                                                               
   8 │ DCMH_MktW_25   under_35             205          35.1           72      5.52     0.45     8.14          75.0      0.003048        0.0125       32.591              72            0.0766      0.0563         0.0919    0.75    0.003168                                                                               
   9 │ DCMH_MktW_10   under_35             205          22.4           46      2.81     0.23     8.26          73.9      0.002723        0.0133       32.6442             46            0.061       0.0591         0.0935    0.7391  0.00284                                                                                
  10 │ DCMH_MktW_0    under_35             205          17.6           36      3.05     0.01     0.36          72.2     -0.002867        0.0303       13.7506             36            0.0848      0.0229         0.0356    0.7222  0.000399                                                                               
Stats for: over_25                                                                                                                                                                                                                                                                                                          
10×18 DataFrame                                                                                                                                                                                                                                                                                                             
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                               
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                             
   1 │ DCMH_MktW_200  over_25              246          41.9          103      8.78     1.7     19.31          50.5      0.005656        0.0573       25.3692            103            0.0853      0.239          0.1921    0.5049   0.014853                                                                              
   2 │ DCMH_MktW_150  over_25              246          41.1          101      7.99     0.81    10.15          51.5     -0.000794        0.0534       27.2524            101            0.0791      0.2646         0.2127    0.5149   0.01616                                                                               
   3 │ DCMH_MktW_125  over_25              246          40.7          100      8.22     1.46    17.8           50.0      0.004787        0.052        28.1567            100            0.0822      0.2327         0.1865    0.5      0.013962                                                                              
   4 │ DCMH_MktW_100  over_25              246          43.1          106      7.74     1.78    23.01          49.1      0.008467        0.051        28.9082            106            0.073       0.2139         0.1708    0.4906   0.011505                                                                              
   5 │ DCMH_MktW_80   over_25              246          43.1          106      7.24     0.76    10.45          47.2      0.000341        0.0556       26.3196            106            0.0683      0.1618         0.1299    0.4717   0.007487                                                                              
   6 │ DCMH_MktW_60   over_25              246          45.1          111      7.08     1.51    21.39          50.5      0.006924        0.0585       24.2901            111            0.0638      0.2209         0.18      0.5045   0.011058                                                                              
   7 │ DCMH_MktW_40   over_25              246          52.4          129      8.24     1.36    16.46          46.5      0.004161        0.0582       24.5129            129            0.0639      0.1285         0.1048    0.4651   0.00518                                                                               
   8 │ DCMH_MktW_25   over_25              246          56.9          140      9.79     1.48    15.12          47.1      0.003576        0.0591       23.7717            140            0.0699      0.1342         0.1103    0.4714   0.005813                                                                              
   9 │ DCMH_MktW_10   over_25              246          70.7          174     13.55     2.01    14.83          46.6      0.003235        0.0646       20.7392            174            0.0779      0.0891         0.0753    0.4655   0.002749                                                                              
  10 │ DCMH_MktW_0    over_25              246          78.5          193     18.65     2.03    10.86          46.6     -0.000856        0.0639       20.0004            193            0.0966      0.0625         0.0542    0.4663  -7.0e-5                                                                                
Stats for: under_25                                                                                                                                                                                                                                                                                                         
10×18 DataFrame                                                                                                                                                                                                                                                                                                             
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G 
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64  
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                             
   1 │ DCMH_MktW_200  under_25             246          56.9          140     16.53     2.56    15.47          57.1      0.006053        0.0376       25.9526            140            0.1181      0.1287         0.1303    0.5714  0.008333
   2 │ DCMH_MktW_150  under_25             246          58.9          145     15.51     2.26    14.59          57.2      0.00493         0.0377       25.6588            145            0.107       0.1266         0.1286    0.5724  0.007947
   3 │ DCMH_MktW_125  under_25             246          57.7          142     14.73     2.19    14.88          57.0      0.005519        0.0402       24.0683            142            0.1037      0.1228         0.1245    0.5704  0.007462
   4 │ DCMH_MktW_100  under_25             246          53.7          132     12.88     2.05    15.9           56.8      0.006369        0.036        27.3838            132            0.0975      0.1279         0.1288    0.5682  0.007749
   5 │ DCMH_MktW_80   under_25             246          53.3          131     11.8      1.87    15.87          54.2      0.006375        0.033        30.409             131            0.0901      0.0855         0.0849    0.542   0.003567
   6 │ DCMH_MktW_60   under_25             246          51.2          126     10.34     2.1     20.3           56.3      0.009997        0.0415       23.802             126            0.0821      0.1197         0.1201    0.5635  0.006457
   7 │ DCMH_MktW_40   under_25             246          45.9          113      7.36     1.31    17.86          54.9      0.006651        0.0302       33.8889            113            0.0651      0.111          0.1092    0.5487  0.005026
   8 │ DCMH_MktW_25   under_25             246          39.0           96      5.14     0.88    17.17          54.2      0.005381        0.03         35.0563             96            0.0536      0.1109         0.1076    0.5417  0.004407
   9 │ DCMH_MktW_10   under_25             246          27.6           68      2.7      0.34    12.57          55.9      0.002508        0.0275       40.077              68            0.0398      0.1752         0.1665    0.5588  0.006088
  10 │ DCMH_MktW_0    under_25             246          18.3           45      3.06    -0.02    -0.51          53.3     -0.005637        0.0523       20.909              45            0.0679      0.1162         0.1098    0.5333  0.005308

Stats for: away                                                                                                       
10×18 DataFrame                                                                                                       
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                                                                                                                                                                                              
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                                                                                                                                                                               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                            
   1 │ DCMH_MktW_200  away                 255          54.5          139     10.3      2.55    24.77          20.1     -0.003391        1.2309        3.1662            139            0.0741     -0.0135        -0.0061    0.2014  -0.011851                                                                                                                                                                                                                                             
   2 │ DCMH_MktW_150  away                 255          52.9          135     10.05     2.48    24.72          20.0     -0.003496        1.2503        3.1722            135            0.0744     -0.0068        -0.003     0.2     -0.011662                                                                                                                                                                                                                                             
   3 │ DCMH_MktW_125  away                 255          52.9          135      9.9      2.14    21.59          20.7     -0.004402        1.2309        3.1662            135            0.0733      0.0158         0.0071    0.2074  -0.009744                                                                                                                                                                                                                                             
   4 │ DCMH_MktW_100  away                 255          53.7          137     10.26     2.17    21.13          20.4     -0.006546        1.2309        3.1662            137            0.0749      0.0009         0.0004    0.2044  -0.011125                                                                                                                                                                                                                                             
   5 │ DCMH_MktW_80   away                 255          52.5          134     10.5      2.85    27.1           19.4     -0.00475         1.1557        3.5297            134            0.0784     -0.0145        -0.0065    0.194   -0.013425                                                                                                                                                                                                                                             
   6 │ DCMH_MktW_60   away                 255          52.2          133     10.42     2.47    23.71          20.3     -0.005746        1.2494        3.1551            133            0.0783      0.0033         0.0015    0.203   -0.012045                                                                                                                                                                                                                                             
   7 │ DCMH_MktW_40   away                 255          54.9          140     11.53     2.77    24.03          18.6     -0.008621        1.2977        3.045             140            0.0824     -0.0804        -0.0373    0.1857  -0.019254                                                                                                                                                                                                                                             
   8 │ DCMH_MktW_25   away                 255          54.9          140     11.63     2.76    23.7           18.6     -0.009601        1.2069        3.3063            140            0.0831     -0.0732        -0.0339    0.1857  -0.018953                                                                                                                                                                                                                                             
   9 │ DCMH_MktW_10   away                 255          56.1          143     11.68     2.68    22.96          19.6     -0.010552        1.3132        2.8985            143            0.0816     -0.0589        -0.0274    0.1958  -0.017179                                                                                                                                                                                                                                             
  10 │ DCMH_MktW_0    away                 255          56.5          144     11.49     2.67    23.23          19.4     -0.010799        1.3605        2.794             144            0.0798     -0.0664        -0.0309    0.1944  -0.017138                                                                                                                                                                                                                                             
Stats for: draw                                                                                                       
10×18 DataFrame                                                                                                                                               
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_MktW_200  draw                 255          53.3          136      5.35     1.6     29.97          24.3      0.005273        0.1872       17.4991            136            0.0393      0.0376         0.0201    0.2426  -0.001077
   2 │ DCMH_MktW_150  draw                 255          54.5          139      5.41     1.32    24.43          23.7      0.003253        0.1908       17.1498            139            0.0389      0.0141         0.0076    0.2374  -0.001916
   3 │ DCMH_MktW_125  draw                 255          52.9          135      5.56     1.18    21.27          23.7      0.002281        0.1872       17.6306            135            0.0412      0.0194         0.0104    0.237   -0.00198
   4 │ DCMH_MktW_100  draw                 255          51.8          132      4.82     1.57    32.53          24.2      0.005686        0.1951       16.8615            132            0.0365      0.04           0.0213    0.2424  -0.000771
   5 │ DCMH_MktW_80   draw                 255          52.9          135      5.08     1.47    28.86          23.0      0.004729        0.1851       18.0899            135            0.0377     -0.0013        -0.0007    0.2296  -0.002382
   6 │ DCMH_MktW_60   draw                 255          49.0          125      5.25     1.7     32.49          24.0      0.006683        0.1843       18.2237            125            0.042       0.0459         0.0241    0.24    -0.001054
   7 │ DCMH_MktW_40   draw                 255          44.7          114      3.9      1.54    39.46          22.8      0.00741         0.1738       20.0739            114            0.0343      0.0238         0.0124    0.2281  -0.001229
   8 │ DCMH_MktW_25   draw                 255          42.7          109      3.35     1.49    44.41          22.0      0.008163        0.1676       21.2439            109            0.0307      0.0044         0.0023    0.2202  -0.001525
   9 │ DCMH_MktW_10   draw                 255          37.3           95      2.46     1.16    47.33          23.2      0.007974        0.1605       22.6881             95            0.0259      0.075          0.0376    0.2316   0.000666
  10 │ DCMH_MktW_0    draw                 255          27.8           71      3.52     0.3      8.39          23.9     -0.005916        0.1899       19.9739             71            0.0496      0.1477         0.0708    0.2394   0.002426
Stats for: home                                                                                                                                               
10×18 DataFrame                                                                                                                                               
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_MktW_200  home                 255          51.0          130     11.14    -0.19    -1.75          27.7     -0.014051        0.4607        4.6054            130            0.0857     -0.1355        -0.0909    0.2769  -0.01906
   2 │ DCMH_MktW_150  home                 255          51.8          132     11.47    -0.58    -5.07          27.3     -0.017566        0.4567        4.612             132            0.0869     -0.1528        -0.1036    0.2727  -0.020762
   3 │ DCMH_MktW_125  home                 255          51.0          130     11.57     0.5      4.32          27.7     -0.010719        0.4607        4.6054            130            0.089      -0.1355        -0.0909    0.2769  -0.020063
   4 │ DCMH_MktW_100  home                 255          51.8          132     11.67    -0.14    -1.2           27.3     -0.014832        0.4567        4.6848            132            0.0884     -0.1437        -0.0964    0.2727  -0.020599
   5 │ DCMH_MktW_80   home                 255          48.6          124     12.06    -0.47    -3.94          26.6     -0.019913        0.4888        4.4674            124            0.0972     -0.1527        -0.1015    0.2661  -0.024438
   6 │ DCMH_MktW_60   home                 255          48.2          123     11.3      1.38    12.22          28.5     -0.006214        0.5558        3.7231            123            0.0919     -0.1266        -0.0845    0.2846  -0.020187
   7 │ DCMH_MktW_40   home                 255          50.6          129     13.43     0.43     3.18          27.9     -0.016224        0.5402        3.8123            129            0.1041     -0.1462        -0.0987    0.2791  -0.02584
   8 │ DCMH_MktW_25   home                 255          52.2          133     13.83     0.42     3.02          28.6     -0.016805        0.5238        3.8964            133            0.104      -0.1312        -0.0886    0.2857  -0.024265
   9 │ DCMH_MktW_10   home                 255          54.5          139     14.38     0.21     1.47          31.7     -0.018437        0.526         3.7466            139            0.1035     -0.0597        -0.0399    0.3165  -0.016975
  10 │ DCMH_MktW_0    home                 255          55.7          142     15.06    -1.01    -6.74          31.0     -0.026917        0.6065        3.1106            142            0.106      -0.1055        -0.0722    0.3099  -0.021975
Stats for: over_05                                                                                                                                            
10×18 DataFrame                                                                                                                                               
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_MktW_200  over_05              187          36.4           68     10.16     0.74     7.27          91.2      0.009844        0.0046       25.6975             68            0.1495      0.0202         0.0635    0.9118   0.00177
   2 │ DCMH_MktW_150  over_05              187          34.8           65      9.19     0.58     6.29          90.8      0.007894        0.0045       26.2751             65            0.1414      0.0153         0.047     0.9077   0.001002
   3 │ DCMH_MktW_125  over_05              187          35.8           67      9.95     0.8      8.08          91.0      0.011148        0.0047       25.5052             67            0.1486      0.0194         0.0605    0.9104   0.001628
   4 │ DCMH_MktW_100  over_05              187          34.2           64      8.33     0.62     7.44          90.6      0.008936        0.0041       28.7561             64            0.1302      0.0132         0.0404    0.9062   0.000734
   5 │ DCMH_MktW_80   over_05              187          35.8           67      7.94     0.67     8.41          92.5      0.009404        0.0042       28.2491             67            0.1185      0.0354         0.1199    0.9254   0.003528
   6 │ DCMH_MktW_60   over_05              187          35.8           67      8.22     0.71     8.67          91.0      0.010115        0.0046       25.687              67            0.1227      0.019          0.0594    0.9104   0.001498
   7 │ DCMH_MktW_40   over_05              187          42.8           80      8.56     0.7      8.22          90.0      0.008361        0.0043       27.3722             80            0.107       0.0061         0.0183    0.9     -3.7e-5
   8 │ DCMH_MktW_25   over_05              187          48.7           91      9.61     0.64     6.62          90.1      0.006429        0.0046       24.7948             91            0.1056      0.0045         0.0134    0.9011  -0.000194
   9 │ DCMH_MktW_10   over_05              187          56.7          106     12.81     0.47     3.71          89.6      0.00327         0.0049       22.6853            106            0.1209     -0.0042        -0.0123    0.8962  -0.001416
  10 │ DCMH_MktW_0    over_05              187          73.3          137     19.94    -0.0     -0.01          89.1     -0.002882        0.0055       18.7946            137            0.1455     -0.0179        -0.0518    0.8905  -0.00399
Stats for: under_05                                                                                                                                           
10×18 DataFrame                                                                                                                                               
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_MktW_200  under_05             170          64.1          109      2.85    -0.33   -11.41           9.2     -0.006978        0.4764       27.1099            109            0.0262      0.2765         0.0677    0.0917   0.002515
   2 │ DCMH_MktW_150  under_05             170          64.7          110      2.74    -0.32   -11.65           8.2     -0.006554        0.5291       24.5005            110            0.0249      0.1424         0.0365    0.0818  -0.000367
   3 │ DCMH_MktW_125  under_05             170          63.5          108      2.69    -0.38   -14.07           8.3     -0.006996        0.5291       24.5005            108            0.0249      0.1636         0.0416    0.0833   9.4e-5
   4 │ DCMH_MktW_100  under_05             170          64.1          109      2.3     -0.18    -7.98           8.3     -0.004625        0.5291       24.5005            109            0.0211      0.1529         0.039     0.0826   0.000317
   5 │ DCMH_MktW_80   under_05             170          62.9          107      2.29     0.51    22.32           9.3      0.000145        0.4764       27.1099            107            0.0214      0.3004         0.0729    0.0935   0.003113
   6 │ DCMH_MktW_60   under_05             170          61.2          104      2.54     1.18    46.45           8.7      0.002205        0.5291       24.5005            104            0.0244      0.2083         0.0521    0.0865   0.001112
   7 │ DCMH_MktW_40   under_05             170          56.5           96      1.36    -0.01    -0.66           9.4     -0.001593        0.4766       27.3872             96            0.0142      0.3175         0.0762    0.0938   0.002938
   8 │ DCMH_MktW_25   under_05             170          52.9           90      0.97    -0.09    -9.5           10.0     -0.001831        0.4766       27.3872             90            0.0108      0.4053         0.0945    0.1      0.003397
   9 │ DCMH_MktW_10   under_05             170          41.8           71      0.5     -0.26   -52.02          11.3     -0.003782        0.4449       30.258              71            0.007       0.6296         0.1355    0.1127   0.003895
  10 │ DCMH_MktW_0    under_05             170          25.3           43      1.4      2.25   160.45           4.7      0.005188        0.0438      301.995              43            0.0327     -0.3377        -0.1125    0.0465  -0.014845
Stats for: over_45                                                                                                                                            
10×18 DataFrame                                                                                                                                               
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_MktW_200  over_45               94          39.4           37      1.02     0.01     1.46           8.1     -0.004959        1.3143        7.4311             37            0.0276     -0.127         -0.0408    0.0811  -0.006583
   2 │ DCMH_MktW_150  over_45               94          39.4           37      0.93     0.04     4.29           5.4     -0.003641        0.0835      143.666              37            0.0251     -0.2973        -0.1008    0.0541  -0.009785
   3 │ DCMH_MktW_125  over_45               94          39.4           37      0.94    -0.02    -2.1            5.4     -0.004926        0.0835      143.666              37            0.0255     -0.2973        -0.1008    0.0541  -0.00998
   4 │ DCMH_MktW_100  over_45               94          43.6           41      0.93    -0.05    -4.99           7.3     -0.004624        1.3143        7.4311             41            0.0227     -0.2122        -0.0715    0.0732  -0.006757
   5 │ DCMH_MktW_80   over_45               94          43.6           41      0.85    -0.07    -8.14           4.9     -0.004692        0.0835      143.666              41            0.0208     -0.3659        -0.1302    0.0488  -0.009112
   6 │ DCMH_MktW_60   over_45               94          42.6           40      0.84    -0.13   -15.45           5.0     -0.005902        0.0835      143.666              40            0.0211     -0.35          -0.1231    0.05    -0.008925
   7 │ DCMH_MktW_40   over_45               94          47.9           45      0.93    -0.22   -24.01           6.7     -0.007312        1.3143        7.4311             45            0.0207     -0.2822        -0.0994    0.0667  -0.007335
   8 │ DCMH_MktW_25   over_45               94          61.7           58      1.16    -0.4    -34.18           6.9     -0.008849        1.0229        9.0676             58            0.02       -0.2914        -0.1069    0.069   -0.007136
   9 │ DCMH_MktW_10   over_45               94          71.3           67      1.79    -0.64   -35.96           7.5     -0.012616        0.91          9.6263             67            0.0267     -0.2716        -0.1014    0.0746  -0.009479
  10 │ DCMH_MktW_0    over_45               94          80.9           76      2.76    -0.99   -35.87           9.2     -0.01757         0.7427       10.9257             76            0.0364     -0.1605        -0.0586    0.0921  -0.009992
Stats for: under_45                                                                                                                                           
10×18 DataFrame                                                                                                                                               
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_MktW_200  under_45             123          48.0           59     13.28    -0.46    -3.47          86.4     -0.017486        0.0117       12.6677             59            0.2251     -0.0073        -0.0185    0.8644  -0.006206
   2 │ DCMH_MktW_150  under_45             123          49.6           61     12.72    -0.73    -5.77          85.2     -0.021179        0.0113       13.1533             61            0.2085     -0.0209        -0.0512    0.8525  -0.008502
   3 │ DCMH_MktW_125  under_45             123          49.6           61     11.72    -0.68    -5.81          85.2     -0.019199        0.0121       12.1671             61            0.1921     -0.0216        -0.0528    0.8525  -0.007621
   4 │ DCMH_MktW_100  under_45             123          44.7           55     10.46    -0.54    -5.2           85.5     -0.017558        0.0107       14.3411             55            0.1902     -0.0144        -0.0353    0.8545  -0.006132
   5 │ DCMH_MktW_80   under_45             123          45.5           56      9.55    -0.63    -6.59          83.9     -0.01786         0.0104       14.7553             56            0.1705     -0.0318        -0.0748    0.8393  -0.00834
   6 │ DCMH_MktW_60   under_45             123          43.1           53      7.89    -0.45    -5.77          83.0     -0.013487        0.0108       14.3472             53            0.1488     -0.0406        -0.0932    0.8302  -0.008339
   7 │ DCMH_MktW_40   under_45             123          39.0           48      5.8     -0.3     -5.12          83.3     -0.009719        0.0091       17.8286             48            0.1208     -0.0319        -0.0735    0.8333  -0.005337
   8 │ DCMH_MktW_25   under_45             123          29.3           36      3.99    -0.16    -4.01          80.6     -0.007157        0.0073       23.8287             36            0.1108     -0.0537        -0.1151    0.8056  -0.007368
   9 │ DCMH_MktW_10   under_45             123          21.1           26      2.04    -0.08    -3.8           76.9     -0.004598        0.0081       22.7645             26            0.0785     -0.0887        -0.1773    0.7692  -0.007776
  10 │ DCMH_MktW_0    under_45             123          17.1           21      2.31    -0.32   -13.91          81.0     -0.019965        0.0258        6.6261             21            0.1098     -0.0518        -0.1117    0.8095  -0.007075
Stats for: under_55                                                                                                                                           
10×18 DataFrame                                                                                                                                               
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G 
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64  
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_MktW_200  under_55              79          60.8           48     15.95     0.37     2.29          95.8      0.00325         0.0057       10.8768             48            0.3323      0.0177         0.0832    0.9583  0.002682
   2 │ DCMH_MktW_150  under_55              79          62.0           49     15.25     0.34     2.22          95.9      0.002744        0.0057       10.8832             49            0.3111      0.0183         0.0867    0.9592  0.002981
   3 │ DCMH_MktW_125  under_55              79          59.5           47     14.39     0.31     2.16          95.7      0.002484        0.0049       12.8934             47            0.3062      0.0178         0.0826    0.9574  0.002726
   4 │ DCMH_MktW_100  under_55              79          58.2           46     13.38     0.29     2.17          95.7      0.002349        0.005        12.619              46            0.2909      0.0169         0.0775    0.9565  0.002432
   5 │ DCMH_MktW_80   under_55              79          58.2           46     12.37     0.27     2.15          95.7      0.00206         0.005        12.619              46            0.2689      0.0169         0.0775    0.9565  0.002458
   6 │ DCMH_MktW_60   under_55              79          59.5           47     10.63     0.22     2.11          95.7      0.001521        0.0053       11.7308             47            0.2261      0.0169         0.0786    0.9574  0.002432
   7 │ DCMH_MktW_40   under_55              79          49.4           39      8.0      0.14     1.69          94.9      7.3e-5          0.0044       15.1099             39            0.2051      0.0116         0.049     0.9487  0.00102
   8 │ DCMH_MktW_25   under_55              79          45.6           36      5.49     0.04     0.69          94.4     -0.001711        0.0039       17.6954             36            0.1524      0.0089         0.0363    0.9444  0.000582
   9 │ DCMH_MktW_10   under_55              79          29.1           23      2.73    -0.08    -2.78          95.7     -0.005689        0.0038       19.5756             23            0.1185      0.028          0.1276    0.9565  0.002953
  10 │ DCMH_MktW_0    under_55              79          19.0           15      1.69    -0.07    -3.87          93.3     -0.005578        0.01          7.7196             15            0.1126      0.005          0.0186    0.9333  6.6e-5
Stats for: over_55                                                                                                                                            
10×18 DataFrame                                                                                                                                               
 Row │ model_name     selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String         Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_MktW_200  over_55               41          31.7           13      0.12    -0.12  -100.0            0.0     -0.009179      NaN              NaN               13            0.0091     -1.0          NaN         0.0     -0.009126
   2 │ DCMH_MktW_150  over_55               41          36.6           15      0.12    -0.12  -100.0            0.0     -0.008125      NaN              NaN               15            0.0081     -1.0          NaN         0.0     -0.008078
   3 │ DCMH_MktW_125  over_55               41          41.5           17      0.12    -0.12  -100.0            0.0     -0.007228      NaN              NaN               17            0.0072     -1.0          NaN         0.0     -0.007184
   4 │ DCMH_MktW_100  over_55               41          41.5           17      0.12    -0.12  -100.0            0.0     -0.00711       NaN              NaN               17            0.0071     -1.0          NaN         0.0     -0.007067
   5 │ DCMH_MktW_80   over_55               41          41.5           17      0.13    -0.13  -100.0            0.0     -0.007585      NaN              NaN               17            0.0075     -1.0          NaN         0.0     -0.007545
   6 │ DCMH_MktW_60   over_55               41          41.5           17      0.12    -0.12  -100.0            0.0     -0.006986      NaN              NaN               17            0.0069     -1.0          NaN         0.0     -0.006946
   7 │ DCMH_MktW_40   over_55               41          51.2           21      0.16    -0.16  -100.0            0.0     -0.007861      NaN              NaN               21            0.0078     -1.0          NaN         0.0     -0.007827
   8 │ DCMH_MktW_25   over_55               41          61.0           25      0.24    -0.24  -100.0            0.0     -0.009462      NaN              NaN               25            0.0094     -1.0          NaN         0.0     -0.009419
   9 │ DCMH_MktW_10   over_55               41          75.6           31      0.41    -0.39   -95.61           3.2     -0.012646        0.0019       10000.0             31            0.0131     -0.3548        -0.1004    0.0323  -0.005594
  10 │ DCMH_MktW_0    over_55               41          85.4           35      0.67    -0.56   -83.34           2.9     -0.016202        0.0019       10000.0             35            0.0191     -0.4286        -0.1286    0.0286  -0.00981
=#

