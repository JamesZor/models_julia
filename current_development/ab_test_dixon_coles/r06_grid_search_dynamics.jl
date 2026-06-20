# current_development/ab_test_dixon_coles/r06_grid_search_dynamics.jl

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

save_dir::String = "./data/dixon_coles_halflife_grid/"
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
# Using match_month to align with HierarchicalMonthlyInterception
dynamics_col = :match_biweek

# ==========================================
# 3. GRID SEARCH SETUP
# ==========================================
half_lives = [14.0, 30.0, 60.0, 120.0]
tasks = []
all_results = []

for hl in half_lives
    println("\n[INFO] Creating Model Task for Half-Life: $(hl) days")
    
    dyn_cfg = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=hl)
    
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
    
    model_name = "DCMH_HalfLife_$(Int(hl))"
    
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
for task in tasks
    println("\n--- Running Experiment: $(task.config.name) ---")
    res = Experiments.run_experiment(task)
    Experiments.save_experiment(res)
    push!(all_results, res)
end

# NOTE: If you have already run the above models and have them loaded in your REPL,
# you can comment out the run loop and use the load logic below instead:
saved_files = Experiments.list_experiments(save_dir, data_dir="")
all_results = [Experiments.load_experiment(saved_files, i) for i in 1:length(half_lives)]

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
--- GLM Edge Summary ---
4×4 DataFrame
 Row │ model              glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value 
     │ String             Float64                 Float64                   Float64                     
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_HalfLife_120                -2.45029                   1.30357                   0.00480902
   2 │ DCMH_HalfLife_14                 -2.45806                   1.73337                   0.00393565
   3 │ DCMH_HalfLife_30                 -2.46202                   2.03102                   0.00118463
   4 │ DCMH_HalfLife_60                 -2.48446                   2.60232                   3.90856e-5
=#


println("\n===========================================")
println("📉 LogLoss Evaluation (Betfair Odds)")
println("===========================================")
eval_logloss = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1)
Evaluation.display_summary_metric(eval_logloss, :logloss)

#=
julia> Evaluation.display_summary_metric(eval_logloss, :logloss)

--- LogLoss Summary (Lower Diff is Better) ---
4×4 DataFrame
 Row │ model              logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll 
     │ String             Float64                   Float64                    Float64                 
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_HalfLife_120                  0.562613                    0.58959               -0.0269767
   2 │ DCMH_HalfLife_14                   0.555828                    0.58959               -0.0337623
   3 │ DCMH_HalfLife_30                   0.554138                    0.58959               -0.0354515
   4 │ DCMH_HalfLife_60                   0.551919                    0.58959               -0.0376707
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

println("\nDone! Half-life Grid Search complete.")

model_names = unique(tearsheet.selection)

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub[!, cols_to_show]; truncate=0)
end



#=
Stats for: under_15                                                                                                                                                                                                                                                                                          12:06 [75/1907]
4×18 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                           
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                            
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                          
   1 │ DCMH_HalfLife_120  under_15             222          50.0          111      5.03     1.26    24.97          30.6      0.005799        0.1444       19.6484            111            0.0453      0.1752         0.0971    0.3063  0.004771                                                                           
   2 │ DCMH_HalfLife_60   under_15             222          44.6           99      3.15     1.33    42.17          32.3      0.009415        0.1282       22.8821             99            0.0318      0.2714         0.1449    0.3232  0.006915                                                                           
   3 │ DCMH_HalfLife_30   under_15             222          39.6           88      2.23     0.99    44.54          30.7      0.008017        0.1209       24.9683             88            0.0253      0.2332         0.1238    0.3068  0.004798                                                                           
   4 │ DCMH_HalfLife_14   under_15             222          34.7           77      1.53     0.72    46.95          28.6      0.006672        0.1376       22.5243             77            0.0199      0.1711         0.0908    0.2857  0.002713                                                                           
Stats for: btts_no                                                                                                                                                                                                                                                                                                          
4×18 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                           
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                            
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                         
   1 │ DCMH_HalfLife_120  btts_no              201          50.2          101      5.29     0.43     8.1           41.6      0.001074        0.0288       36.4959            101            0.0524     -0.1475        -0.1451    0.4158  -0.009147                                                                          
   2 │ DCMH_HalfLife_60   btts_no              201          43.8           88      3.42     0.78    22.72          42.0      0.006786        0.0206       53.7332             88            0.0388     -0.1146        -0.1098    0.4205  -0.005271                                                                          
   3 │ DCMH_HalfLife_30   btts_no              201          35.8           72      2.31     0.47    20.24          41.7      0.004987        0.016        71.6128             72            0.0321     -0.1061        -0.1       0.4167  -0.003994                                                                          
   4 │ DCMH_HalfLife_14   btts_no              201          29.9           60      1.48     0.19    13.05          41.7      0.002332        0.0164       70.8555             60            0.0246     -0.0995        -0.0931    0.4167  -0.002793                                                                          
Stats for: btts_yes                                                                                                                                                                                                                                                                                                         
4×18 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                           
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                            
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                          
   1 │ DCMH_HalfLife_120  btts_yes             201          39.8           80      4.18     0.86    20.59          56.2      0.007509        0.0181       62.5221             80            0.0522      0.1983         0.1867    0.5625  0.008811                                                                           
   2 │ DCMH_HalfLife_60   btts_yes             201          52.7          106      5.01     1.18    23.46          56.6      0.008364        0.017        67.4064            106            0.0473      0.2129         0.1995    0.566   0.008786                                                                           
   3 │ DCMH_HalfLife_30   btts_yes             201          59.2          119      5.09     0.99    19.53          57.1      0.005968        0.0168       67.26              119            0.0427      0.217          0.2049    0.5714  0.008246                                                                           
   4 │ DCMH_HalfLife_14   btts_yes             201          67.2          135      5.57     1.04    18.73          57.8      0.005427        0.0199       55.4809            135            0.0413      0.2151         0.2059    0.5778  0.007944                                                                           
Stats for: over_35                                                                                                                                                                                                                                                                                                          
4×18 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                           
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                            
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                         
   1 │ DCMH_HalfLife_120  over_35              205          40.0           82      3.47     0.12     3.47          28.0     -0.003391        0.2904       13.0681             82            0.0423      0.3451         0.1551    0.2805   0.0105                                                                            
   2 │ DCMH_HalfLife_60   over_35              205          51.7          106      4.21    -0.24    -5.77          23.6     -0.006688        0.2492       15.576             106            0.0397      0.1512         0.0711    0.2358   0.0027                                                                            
   3 │ DCMH_HalfLife_30   over_35              205          62.4          128      4.71     0.33     7.05          22.7     -0.00375         0.2575       14.541             128            0.0368      0.0749         0.0367    0.2266   0.000121                                                                          
   4 │ DCMH_HalfLife_14   over_35              205          70.7          145      5.17     0.39     7.61          21.4     -0.003548        0.2726       13.4218            145            0.0356     -0.0039        -0.002     0.2138  -0.002436                                                                          
Stats for: under_35                                                                                                                                                                                                                                                                                                         
4×18 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                           
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                            
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                          
   1 │ DCMH_HalfLife_120  under_35             205          49.8          102      9.92     0.51     5.17          78.4      0.000498        0.0205       17.973             102            0.0973      0.0736         0.1295    0.7843  0.005552                                                                           
   2 │ DCMH_HalfLife_60   under_35             205          46.3           95      8.2      0.55     6.71          76.8      0.001943        0.0148       26.1586             95            0.0864      0.066          0.1121    0.7684  0.004352                                                                           
   3 │ DCMH_HalfLife_30   under_35             205          37.1           76      6.01     0.42     6.98          75.0      0.001966        0.0134       30.1298             76            0.0791      0.0533         0.0872    0.75    0.003003                                                                           
   4 │ DCMH_HalfLife_14   under_35             205          27.8           57      3.87     0.3      7.71          73.7      0.002266        0.0157       26.6454             57            0.068       0.0445         0.0708    0.7368  0.002079                                                                           
Stats for: over_25                                                                                                                                                                                                                                                                                                          
4×18 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                           
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                            
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                          
   1 │ DCMH_HalfLife_120  over_25              246          38.6           95      5.95     1.15    19.28          50.5      0.006423        0.0556       25.3097             95            0.0626      0.2159         0.1771    0.5053  0.010634                                                                           
   2 │ DCMH_HalfLife_60   over_25              246          50.4          124      7.76     0.89    11.43          47.6      0.001393        0.0592       24.1167            124            0.0626      0.1547         0.126     0.4758  0.006767                                                                           
   3 │ DCMH_HalfLife_30   over_25              246          55.7          137      8.3      1.21    14.6           47.4      0.002779        0.057        24.6908            137            0.0606      0.1422         0.1168    0.4745  0.005924                                                                           
   4 │ DCMH_HalfLife_14   over_25              246          63.0          155      8.84     1.24    14.08          47.1      0.002394        0.0598       22.8919            155            0.057       0.1156         0.0965    0.471   0.004278                                                                           
Stats for: under_25                                                                                                                                                                                                                                                                                                         
4×18 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                           
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                            
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                          
   1 │ DCMH_HalfLife_120  under_25             246          50.4          124      9.16     0.55     5.96          55.6     -0.001392        0.05         18.9727            124            0.0738      0.0839         0.0855    0.5565  0.003548                                                                           
   2 │ DCMH_HalfLife_60   under_25             246          47.2          116      7.48     1.32    17.67          56.0      0.006516        0.0342       29.5564            116            0.0645      0.1266         0.1257    0.5603  0.006044                                                                           
   3 │ DCMH_HalfLife_30   under_25             246          41.9          103      5.54     0.92    16.65          55.3      0.004978        0.0303       34.3496            103            0.0538      0.1285         0.1257    0.5534  0.005391                                                                           
   4 │ DCMH_HalfLife_14   under_25             246          35.0           86      3.67     0.45    12.35          52.3      0.002128        0.0328       32.5475             86            0.0426      0.0813         0.0781    0.5233  0.002477                                                                           
Stats for: away                                                                                                                                                                                                                                                                                                             
4×18 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                           
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                            
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                         
   1 │ DCMH_HalfLife_120  away                 255          44.3          113      8.2      0.99    12.03          17.7     -0.012708        1.3991        3.1745            113            0.0726     -0.0369        -0.0159    0.177   -0.014062                                                                          
   2 │ DCMH_HalfLife_60   away                 255          54.1          138     11.03     2.84    25.72          18.8     -0.007833        1.211         3.2944            138            0.0799     -0.06          -0.0276    0.1884  -0.016931                                                                          
   3 │ DCMH_HalfLife_30   away                 255          55.7          142     11.53     2.63    22.77          19.0     -0.00991         1.1617        3.4255            142            0.0812     -0.0532        -0.0246    0.1901  -0.016813                                                                          
   4 │ DCMH_HalfLife_14   away                 255          57.3          146     11.64     2.45    21.0           20.5     -0.011569        1.0669        3.6059            146            0.0798     -0.004         -0.0019    0.2055  -0.012504

Stats for: draw
4×18 DataFrame
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_HalfLife_120  draw                 255          52.5          134      5.99     1.65    27.58          26.1      0.005209        0.2314       13.3839            134            0.0447      0.07           0.0378    0.2612  -8.1e-5
   2 │ DCMH_HalfLife_60   draw                 255          48.2          123      4.09     1.62    39.54          22.8      0.007389        0.1783       19.3054            123            0.0333      0.0112         0.0059    0.2276  -0.001522
   3 │ DCMH_HalfLife_30   draw                 255          44.3          113      3.5      1.52    43.53          23.9      0.007827        0.1739       19.8987            113            0.0309      0.0657         0.0339    0.2389   0.000317
   4 │ DCMH_HalfLife_14   draw                 255          45.1          115      2.97     1.16    39.06          23.5      0.005517        0.1739       19.8987            115            0.0258      0.0471         0.0245    0.2348   2.7e-5
Stats for: home
4×18 DataFrame
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_HalfLife_120  home                 255          43.1          110      9.24    -0.49    -5.28          31.8     -0.017083        0.4972        4.1621            110            0.084      -0.0234        -0.0152    0.3182  -0.009624
   2 │ DCMH_HalfLife_60   home                 255          49.8          127     13.03    -0.48    -3.69          28.3     -0.022006        0.5402        3.8123            127            0.1026     -0.1328        -0.0892    0.2835  -0.02408
   3 │ DCMH_HalfLife_30   home                 255          51.0          130     13.2      0.55     4.16          27.7     -0.015321        0.5295        3.8657            130            0.1015     -0.1563        -0.1063    0.2769  -0.025854
   4 │ DCMH_HalfLife_14   home                 255          51.4          131     12.57     0.88     6.98          28.2     -0.011931        0.5057        4.168             131            0.0959     -0.1223        -0.0814    0.2824  -0.02109
Stats for: over_05
4×18 DataFrame
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_HalfLife_120  over_05              187          27.8           52      5.91     0.48     8.08          90.4      0.00876         0.0037       30.7049             52            0.1136      0.0063         0.0192    0.9038  -3.5e-5
   2 │ DCMH_HalfLife_60   over_05              187          40.1           75      7.76     0.64     8.22          92.0      0.008091        0.0042       27.9185             75            0.1034      0.0275         0.0906    0.92     0.002316
   3 │ DCMH_HalfLife_30   over_05              187          43.9           82      8.24     0.68     8.27          89.0      0.007915        0.0047       24.7105             82            0.1004     -0.0057        -0.0163    0.8902  -0.001233
   4 │ DCMH_HalfLife_14   over_05              187          47.6           89      8.08     0.56     6.9           89.9      0.005836        0.005        22.5892             89            0.0908      0.0012         0.0035    0.8989  -0.000389
Stats for: under_05
4×18 DataFrame
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G 
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64  
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_HalfLife_120  under_05             170          64.7          110      3.14     2.05    65.19           9.1      0.004304        0.5501       23.3218            110            0.0285      0.2572         0.0634    0.0909  0.001896
   2 │ DCMH_HalfLife_60   under_05             170          59.4          101      1.49     0.54    36.41           9.9      0.002514        0.4296       30.2473            101            0.0148      0.3857         0.0908    0.099   0.003939
   3 │ DCMH_HalfLife_30   under_05             170          58.2           99      1.09    -0.05    -4.55          10.1     -0.001406        0.4296       30.2473             99            0.011       0.4137         0.0966    0.101   0.00353
   4 │ DCMH_HalfLife_14   under_05             170          54.7           93      0.85    -0.11   -13.32           9.7     -0.001787        0.4088       32.5167             93            0.0091      0.3832         0.0894    0.0968  0.002776
Stats for: over_45
4×18 DataFrame
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_HalfLife_120  over_45               94          46.8           44      0.87    -0.19   -22.19           6.8     -0.00688         0.7548       13.5575             44            0.0199     -0.2341        -0.0801    0.0682  -0.006136
   2 │ DCMH_HalfLife_60   over_45               94          47.9           45      0.91    -0.2    -22.52           6.7     -0.006888        1.3143        7.4311             45            0.0202     -0.2822        -0.0994    0.0667  -0.007135
   3 │ DCMH_HalfLife_30   over_45               94          60.6           57      0.98    -0.33   -34.23           7.0     -0.007451        0.9942        9.4798             57            0.0171     -0.2684        -0.0964    0.0702  -0.005628
   4 │ DCMH_HalfLife_14   over_45               94          68.1           64      1.16    -0.21   -18.05           9.4     -0.005667        0.7573       11.4882             64            0.018      -0.0906        -0.0309    0.0938  -0.002908
Stats for: under_45
4×18 DataFrame
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_HalfLife_120  under_45             123          41.5           51      6.74    -0.55    -8.22          86.3     -0.015546        0.0137       10.7526             51            0.1322     -0.0104        -0.0263    0.8627  -0.002874
   2 │ DCMH_HalfLife_60   under_45             123          39.8           49      5.94    -0.36    -6.1           83.7     -0.010975        0.0098       16.3884             49            0.1213     -0.0295        -0.0686    0.8367  -0.005047
   3 │ DCMH_HalfLife_30   under_45             123          32.5           40      4.05    -0.22    -5.39          80.0     -0.008403        0.0087       19.3761             40            0.1014     -0.0645        -0.1376    0.8     -0.007743
   4 │ DCMH_HalfLife_14   under_45             123          23.6           29      2.51    -0.1     -4.01          82.8     -0.005601        0.0077       23.413              29            0.0866     -0.0234        -0.0523    0.8276  -0.002818
Stats for: under_55
4×18 DataFrame
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G 
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64  
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_HalfLife_120  under_55              79          51.9           41      8.84     0.15     1.72          95.1      0.000148        0.0056       11.1054             41            0.2156      0.0106         0.0463    0.9512  0.000865
   2 │ DCMH_HalfLife_60   under_55              79          49.4           39      8.17     0.14     1.72          94.9      0.000177        0.0044       15.1099             39            0.2094      0.0116         0.049     0.9487  0.001008
   3 │ DCMH_HalfLife_30   under_55              79          44.3           35      5.82     0.01     0.18          94.3     -0.00318         0.004        17.3248             35            0.1663      0.0075         0.03      0.9429  0.000281
   4 │ DCMH_HalfLife_14   under_55              79          34.2           27      3.72    -0.05    -1.32          96.3     -0.004787        0.0044       16.1631             27            0.1379      0.0317         0.1559    0.963   0.003932
Stats for: over_55
4×18 DataFrame
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DCMH_HalfLife_120  over_55               41          53.7           22      0.19    -0.19   -100.0           0.0     -0.00854            NaN           NaN             22            0.0085        -1.0            NaN       0.0  -0.008487
   2 │ DCMH_HalfLife_60   over_55               41          53.7           22      0.16    -0.16   -100.0           0.0     -0.007468           NaN           NaN             22            0.0074        -1.0            NaN       0.0  -0.007434
   3 │ DCMH_HalfLife_30   over_55               41          58.5           24      0.19    -0.19   -100.0           0.0     -0.008026           NaN           NaN             24            0.008         -1.0            NaN       0.0  -0.00799
   4 │ DCMH_HalfLife_14   over_55               41          68.3           28      0.25    -0.25   -100.0           0.0     -0.009123           NaN           NaN             28            0.0091        -1.0            NaN       0.0  -0.009075
=#

