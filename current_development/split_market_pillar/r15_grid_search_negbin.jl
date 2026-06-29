#=
OVERNIGHT GRID — NegBin market-pillar baseline (mirrors r05's double-Poisson grid for the `nb` row).

Question (= r05, dispersion row): does the SPLIT (level/supremacy) market pillar buy anything over
(a) NO market and (b) the OLD isotropic market pillar — now with goals ~ RobustNegativeBinomial(r,λ)
so the engine carries a structural DISPERSION the market's independent-Poisson template ignores
(the lever that should move BTTS / correct-score / tails).

Canonical naming (NOTES.md → "Canonical naming"): cells = `<pillar>_<disp>_<knob>`, disp = `nb`.
7 cells:
  - none_nb                     : no market pillar (baseline)        [SplitMarketDoubleNegBinModel, market_on=false]
  - iso_nb_mw{50,100}           : OLD isotropic market, market_weight 0.5 / 1.0   [src DynamicDoubleNegBinXG…]
  - split_nb_lw{0,25,50,100}    : split market, supremacy_weight=1.0 FIXED, level_weight 0/.25/.5/1  [l04]

LEAGUE CHOICE (important): NegBin's dispersion is INERT on the near-Poisson top flight Ireland-79
(V/M 0.94 → r08 fitted log_r≈32 ⇒ r→∞ ⇒ ≈ double-Poisson). The dispersion only BITES on Ireland
FIRST DIVISION 718 (V/M≈1.14, [[first-division-718-signature]]). So:
  • SEGMENT = Ireland()           → directly comparable to the saved dp/smile grids (Betfair eval,
                                    same config) → drop these cells into r13/r14; expect ≈ the dp row.
  • SEGMENT = IrelandFirstDivision() → where NegBin SHOULD differ from Poisson (the interesting baseline),
                                    but there is no dp grid on 718 to compare against yet.
Run BOTH; saves are keyed by segment so they don't collide.

Train pillar from SofaScore ds.odds (= r05); eval CLV vs Betfair (ds1) ⇒ no CLV leakage.

Run after git pull + REPL restart:
    include("current_development/split_market_pillar/r15_grid_search_negbin.jl")
=#

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
const Signals     = BayesianFootball.Signals

include("current_development/split_market_pillar/l04_split_market_negbin.jl")

# ==========================================
# 1. DATA  (train pillar = SofaScore ds.odds)
# ==========================================
SEGMENT = Data.Ireland()                     # ← swap to Data.IrelandFirstDivision() for the NB-active league
seg_tag = lowercase(string(nameof(typeof(SEGMENT))))
println("[INFO] Loading $(seg_tag) DataStore...")
ds = Data.load_datastore_cached(SEGMENT)

save_dir = "./data/negbin_market_grid_$(seg_tag)/"
mkpath(save_dir)

# ==========================================
# 2. SHARED CONFIG (identical to r05 for comparability)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()        # NegBin r (now USED)
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

samples        = 800
warmup         = 300
chains         = 4
target_seasons = ["2025", "2026"]
dynamics_col   = :match_biweek

# ==========================================
# 3. MODEL SPECS  (name => model)   [canonical cell names]
# ==========================================
# no-market = the split NegBin engine with the pillar gated off (same goals likelihood as the split cells).
_nomarket() = SplitMarketDoubleNegBinModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_on              = false,
)

# iso = the OLD isotropic market pillar on the src NegBin engine; market_weight sweeps its strength.
_iso(mw) = PreGame.DynamicDoubleNegBinXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_weight          = mw,
)

# split = anisotropic (sup+level); supremacy fixed at the natural 1.0, sweep level_weight only.
_split(lw) = SplitMarketDoubleNegBinModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_on              = true,
    supremacy_weight       = 1.0,
    level_weight           = lw,
)

iso_market_weights  = [0.5, 1.0]
split_level_weights = [0.0, 0.25, 0.5, 1.0]

specs = Tuple{String, Any}[]
push!(specs, ("none_nb", _nomarket()))
for mw in iso_market_weights
    push!(specs, ("iso_nb_mw$(Int(round(mw*100)))", _iso(mw)))
end
for lw in split_level_weights
    push!(specs, ("split_nb_lw$(Int(round(lw*100)))", _split(lw)))
end
println("[INFO] NegBin grid ($seg_tag): $(length(specs)) cells -> ", join(first.(specs), ", "))

# ==========================================
# 4. PHASE 1 — RUN + SAVE (robust; each persisted)
# ==========================================
all_results = Any[]
failures = String[]
t0 = time()
for (name, model) in specs
    println("\n", "#"^70, "\n# RUN: $name\n", "#"^70)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = target_seasons,
            history_seasons = 2,
            warmup_period   = 0,
            dynamics_col    = dynamics_col,
            samples         = samples,
            warmup          = warmup,
            chains          = chains,
            use_queue       = true,
            max_depth       = 10,
        )
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)
        push!(all_results, res)
    catch e
        @error "FAILED: $name" exception=(e, catch_backtrace())
        push!(failures, name)
    end
end
println("\n[INFO] Phase 1 done in $(round((time()-t0)/60, digits=1)) min. " *
        "OK: $(length(all_results))/$(length(specs)). Failures: $(isempty(failures) ? "none" : join(failures, ", "))")

# Re-eval from disk without re-running (uncomment):
saved = Experiments.list_experiments(save_dir, data_dir="")
all_results = [Experiments.load_experiment(saved, i) for i in 1:length(saved)]

# ==========================================
# 5. PHASE 2 — EVAL (CLV vs Betfair) — identical frame to r05/r10
# ==========================================
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1 = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

println("\n", "="^60, "\n📊 GLM Edge (Betfair) — NegBin grid\n", "="^60)
Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1), :glmedge)

#=
--- GLM Edge Summary ---
7×4 DataFrame
 Row │ model           glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value 
     │ String          Float64                 Float64                   Float64                     
─────┼───────────────────────────────────────────────────────────────────────────────────────────────
   1 │ iso_nb_mw100                  -2.32703                  1.07178                     0.088107
   2 │ iso_nb_mw50                   -2.34329                  1.40688                     0.0268384
   3 │ none_nb                       -2.3732                   1.62479                     0.0025396
   4 │ split_nb_lw0                  -2.32544                  0.771185                    0.0683184
   5 │ split_nb_lw100                -2.32016                  0.900527                    0.152346
   6 │ split_nb_lw25                 -2.3177                   0.711159                    0.131246
   7 │ split_nb_lw50                 -2.30923                  0.576601                    0.277444
=#


println("\n", "="^60, "\n📉 LogLoss (Betfair) — NegBin grid\n", "="^60)
Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1), :logloss)

#=
--- LogLoss Summary (Lower Diff is Better) ---
7×4 DataFrame
 Row │ model           logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll 
     │ String          Float64                   Float64                    Float64                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────
   1 │ iso_nb_mw100                    0.569006                   0.591852               -0.0228456
   2 │ iso_nb_mw50                     0.567554                   0.591852               -0.0242978
   3 │ none_nb                         0.569676                   0.591852               -0.022176
   4 │ split_nb_lw0                    0.581282                   0.591852               -0.0105699
   5 │ split_nb_lw100                  0.569222                   0.591852               -0.0226298
   6 │ split_nb_lw25                   0.577651                   0.591852               -0.0142012
   7 │ split_nb_lw50                   0.574729                   0.591852               -0.0171227
=#


println("\n", "="^60, "\n💰 Backtest (BayesianKelly) — NegBin grid\n", "="^60)
ledger = BackTesting.run_backtest(ds1, all_results, [Signals.BayesianKelly()];
                                  market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
tearsheet = BackTesting.generate_tearsheet(ledger)
cols = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover,
        :profit, :roi_pct, :win_rate_pct, :hurdle_G_emp, :hurdle_n_bets, :hurdle_avg_stake,
        :hurdle_E_R, :hurdle_sharpe, :hurdle_p, :hurdle_G]
println("\n>>> Backtest by market selection (hurdle_G higher = better):")
for m_name in unique(tearsheet.selection)
    println("\nSelection: $m_name")
    show(subset(tearsheet, :selection => ByRow(isequal(m_name)))[!, cols]; truncate=0)
    println()
end



#=

Selection: btts_no                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
7×16 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                                                                                                                                                                                                                                                                                                                                                                                       
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                                                                                                                                                                                                                                                                                                                                                                        
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                                                                                     
   1 │ split_nb_lw100  btts_no              202          73.3          148      9.85    -0.91    -9.2           39.9     -0.010088            148            0.0665     -0.1795        -0.1773    0.3986  -0.014199                                                                                                                                                                                                                                                                                                                                                                                                                                      
   2 │ split_nb_lw50   btts_no              202          61.9          125     11.27    -0.76    -6.79          40.8     -0.016478            125            0.0901     -0.1564        -0.1531    0.408   -0.018304                                                                                                                                                                                                                                                                                                                                                                                                                                      
   3 │ split_nb_lw25   btts_no              202          53.5          108     11.48    -0.72    -6.29          40.7     -0.020248            108            0.1063     -0.1544        -0.1505    0.4074  -0.022311                                                                                                                                                                                                                                                                                                                                                                                                                                      
   4 │ split_nb_lw0    btts_no              202          42.6           86      9.48    -0.23    -2.46          44.2     -0.016857             86            0.1103     -0.0794        -0.0763    0.4419  -0.015286                                                                                                                                                                                                                                                                                                                                                                                                                                      
   5 │ iso_nb_mw100    btts_no              202          66.8          135      9.97    -0.63    -6.3           39.3     -0.010049            135            0.0739     -0.1871        -0.1843    0.3926  -0.016619                                                                                                                                                                                                                                                                                                                                                                                                                                      
   6 │ iso_nb_mw50     btts_no              202          65.3          132      7.76    -0.04    -0.54          40.2     -0.004143            132            0.0588     -0.1673        -0.1639    0.4015  -0.01163                                                                                                                                                                                                                                                                                                                                                                                                                                       
   7 │ none_nb         btts_no              202          32.2           65      1.6      0.5     31.35          52.3      0.006889             65            0.0246      0.1125         0.1053    0.5231   0.002421                                                                                                                                                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
Selection: btts_yes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
7×16 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                                                                                                                                                                                                                                                                                                                                                                                       
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                                                                                                                                                                                                                                                                                                                                                                        
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                                                                                      
   1 │ split_nb_lw100  btts_yes             202          22.8           46      1.34     0.38    28.75          54.3      0.00707              46            0.0291      0.2288         0.2024    0.5435  0.006108                                                                                                                                                                                                                                                                                                                                                                                                                                       
   2 │ split_nb_lw50   btts_yes             202          34.7           70      3.91     0.22     5.51          57.1     -0.001                70            0.0559      0.2455         0.2263    0.5714  0.011874                                                                                                                                                                                                                                                                                                                                                                                                                                       
   3 │ split_nb_lw25   btts_yes             202          45.0           91      6.62     0.87    13.12          57.1      0.003547             91            0.0728      0.2209         0.2076    0.5714  0.013075                                                                                                                                                                                                                                                                                                                                                                                                                                       
   4 │ split_nb_lw0    btts_yes             202          54.5          110     10.33     1.94    18.77          60.0      0.008377            110            0.0939      0.2499         0.2427    0.6     0.018766                                                                                                                                                                                                                                                                                                                                                                                                                                       
   5 │ iso_nb_mw100    btts_yes             202          29.7           60      2.17     0.44    20.21          56.7      0.005614             60            0.0362      0.2562         0.2322    0.5667  0.008485                                                                                                                                                                                                                                                                                                                                                                                                                                       
   6 │ iso_nb_mw50     btts_yes             202          31.2           63      1.98     0.52    26.24          58.7      0.006759             63            0.0315      0.3015         0.2753    0.5873  0.008892                                                                                                                                                                                                                                                                                                                                                                                                                                       
   7 │ none_nb         btts_yes             202          65.3          132      7.26     1.97    27.21          63.6      0.011695            132            0.055       0.3207         0.3183    0.6364  0.016083                                                                                                                                                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
Selection: away                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
7×16 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                                                                                                                                                                                                                                                                                                                                                                                       
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                                                                                                                                                                                                                                                                                                                                                                        
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                                                                                     
   1 │ split_nb_lw100  away                 273          57.5          157     15.66     4.62    29.52          19.7     -0.012631            157            0.0997      0.0108         0.0047    0.1975  -0.018899                                                                                                                                                                                                                                                                                                                                                                                                                                      
   2 │ split_nb_lw50   away                 273          60.1          164     11.19     4.96    44.35          20.7      0.00589             164            0.0682      0.0259         0.0113    0.2073  -0.008247                                                                                                                                                                                                                                                                                                                                                                                                                                      
   3 │ split_nb_lw25   away                 273          61.2          167      9.67     4.38    45.33          21.0      0.010095            167            0.0579      0.0178         0.0078    0.2096  -0.006272                                                                                                                                                                                                                                                                                                                                                                                                                                      
   4 │ split_nb_lw0    away                 273          63.7          174      9.6      3.94    41.05          23.6      0.009027            174            0.0552      0.0667         0.0296    0.2356  -0.002873                                                                                                                                                                                                                                                                                                                                                                                                                                      
   5 │ iso_nb_mw100    away                 273          57.1          156     13.47     4.44    32.94          19.2     -0.00446             156            0.0864     -0.0068        -0.003     0.1923  -0.016029                                                                                                                                                                                                                                                                                                                                                                                                                                      
   6 │ iso_nb_mw50     away                 273          59.0          161     13.9      4.81    34.6           20.5     -0.006014            161            0.0863      0.0454         0.0195    0.205   -0.011889                                                                                                                                                                                                                                                                                                                                                                                                                                      
   7 │ none_nb         away                 273          58.6          160     13.36     5.01    37.47          20.0     -0.006136            160            0.0835      0.0127         0.0055    0.2     -0.013664                                                                                                                                                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
Selection: draw                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
7×16 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                                                                                                                                                                                                                                                                                                                                                                                       
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                                                                                                                                                                                                                                                                                                                                                                        
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                                                                                      
   1 │ split_nb_lw100  draw                 273          37.7          103      3.86     1.19    30.74          23.3      0.005194            103            0.0375      0.0728         0.0368    0.233   0.000149                                                                                                                                                                                                                                                                                                                                                                                                                                       
   2 │ split_nb_lw50   draw                 273          37.4          102      3.45     1.13    32.62          25.5      0.006008            102            0.0338      0.1391         0.07      0.2549  0.00256                                                                                                                                                                                                                                                                                                                                                                                                                                        
   3 │ split_nb_lw25   draw                 273          37.7          103      2.97     1.16    38.87          27.2      0.006997            103            0.0289      0.1984         0.0989    0.2718  0.004122                                                                                                                                                                                                                                                                                                                                                                                                                                       
   4 │ split_nb_lw0    draw                 273          32.6           89      2.34     1.02    43.48          28.1      0.007521             89            0.0263      0.2535         0.1234    0.2809  0.00526                                                                                                                                                                                                                                                                                                                                                                                                                                        
   5 │ iso_nb_mw100    draw                 273          36.6          100      3.64     1.18    32.41          24.0      0.00588             100            0.0364      0.1018         0.0511    0.24    0.001227                                                                                                                                                                                                                                                                                                                                                                                                                                       
   6 │ iso_nb_mw50     draw                 273          36.3           99      3.33     1.24    37.21          24.2      0.006815             99            0.0337      0.1129         0.0564    0.2424  0.001655                                                                                                                                                                                                                                                                                                                                                                                                                                       
   7 │ none_nb         draw                 273          22.3           61      1.75     0.89    51.04          26.2      0.009981             61            0.0287      0.2843         0.13      0.2623  0.00628                                                                                                                                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
Selection: home                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
7×16 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                                                                                                                                                                                                                                                                                                                                                                                       
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                                                                                                                                                                                                                                                                                                                                                                        
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                                                                                     
   1 │ split_nb_lw100  home                 273          52.4          143     16.71    -0.4     -2.41          28.0     -0.026041            143            0.1169     -0.124         -0.0831    0.2797  -0.027996                                                                                                                                                                                                                                                                                                                                                                                                                                      
   2 │ split_nb_lw50   home                 273          54.9          150     11.44    -1.6    -13.95          28.7     -0.02131             150            0.0763     -0.1358        -0.0931    0.2867  -0.016094                                                                                                                                                                                                                                                                                                                                                                                                                                      
   3 │ split_nb_lw25   home                 273          57.5          157     10.01    -1.35   -13.47          31.2     -0.015915            157            0.0637     -0.0909        -0.063     0.3121  -0.009796                                                                                                                                                                                                                                                                                                                                                                                                                                      
   4 │ split_nb_lw0    home                 273          57.1          156     10.21    -1.45   -14.2           32.1     -0.016148            156            0.0654     -0.0854        -0.0596    0.3205  -0.009744                                                                                                                                                                                                                                                                                                                                                                                                                                      
   5 │ iso_nb_mw100    home                 273          52.7          144     14.28    -0.61    -4.28          27.8     -0.021306            144            0.0992     -0.1417        -0.0964    0.2778  -0.023638                                                                                                                                                                                                                                                                                                                                                                                                                                      
   6 │ iso_nb_mw50     home                 273          54.9          150     15.18    -0.59    -3.87          30.0     -0.022544            150            0.1012     -0.0826        -0.0552    0.3     -0.018741                                                                                                                                                                                                                                                                                                                                                                                                                                      
   7 │ none_nb         home                 273          60.4          165     17.25    -0.99    -5.76          32.1     -0.02565             165            0.1045     -0.0672        -0.046     0.3212  -0.017584                                                                                                                                                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
Selection: over_15                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
7×16 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                                                                                                                                                                                                                                                                                                                                                                                       
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                                                                                                                                                                                                                                                                                                                                                                                        
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                                                                                     
   1 │ split_nb_lw100  over_15              212          30.7           65      4.25    -0.57   -13.31          64.6     -0.011598             65            0.0653     -0.0183        -0.025     0.6462  -0.002359                                                                                                                                                                                                                                                                                                                                                                                                                                      
   2 │ split_nb_lw50   over_15              212          42.0           89     12.04    -0.48    -3.98          67.4     -0.016594             89            0.1353      0.008          0.0113    0.6742  -0.003698                                                                                                                                                                                                                                                                                                                                                                                                                                      
   3 │ split_nb_lw25   over_15              212          52.4          111     18.56    -0.58    -3.13          67.6     -0.022373            111            0.1672     -0.0069        -0.01      0.6757  -0.008311                                                                                                                                                                                                                                                                                                                                                                                                                                      
   4 │ split_nb_lw0    over_15              212          58.0          123     25.21    -0.54    -2.13          68.3     -0.028477            123            0.2049     -0.0065        -0.0095    0.6829  -0.011922                                                                                                                                                                                                                                                                                                                                                                                                                                      
   5 │ iso_nb_mw100    over_15              212          35.8           76      7.34    -0.43    -5.92          64.5     -0.011058             76            0.0966     -0.0259        -0.0357    0.6447  -0.005053                                                                                                                                                                                                                                                                                                                                                                                                                                      
   6 │ iso_nb_mw50     over_15              212          38.2           81      6.51    -0.24    -3.67          64.2     -0.006889             81            0.0804     -0.0343        -0.0473    0.642   -0.00451                                                                                                                                                                                                                                                                                                                                                                                                                                       
   7 │ none_nb         over_15              212          67.0          142     16.45    -0.11    -0.67          70.4     -0.007635            142            0.1158      0.0178         0.0268    0.7042  -0.00105     

Selection: under_15
7×16 DataFrame
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ split_nb_lw100  under_15             212          66.0          140      7.09     2.56    36.14          26.4      0.010046            140            0.0507      0.0363         0.0207    0.2643  -0.001875
   2 │ split_nb_lw50   under_15             212          57.1          121      8.06     0.93    11.48          27.3     -0.004899            121            0.0666      0.0859         0.0477    0.2727  -0.000886
   3 │ split_nb_lw25   under_15             212          48.1          102      8.37     0.03     0.34          26.5     -0.015676            102            0.082       0.0666         0.0369    0.2647  -0.004401
   4 │ split_nb_lw0    under_15             212          42.5           90      7.09    -0.61    -8.6           27.8     -0.021295             90            0.0788      0.1187         0.0647    0.2778  -9.9e-5
   5 │ iso_nb_mw100    under_15             212          63.2          134      7.14     1.92    26.89          26.1      0.006109            134            0.0533      0.0328         0.0186    0.2612  -0.002378
   6 │ iso_nb_mw50     under_15             212          59.0          125      5.59     2.02    36.1           26.4      0.009183            125            0.0447      0.0552         0.0309    0.264   -0.00055
   7 │ none_nb         under_15             212          31.1           66      1.29     0.42    32.68          30.3      0.004765             66            0.0196      0.2803         0.1418    0.303    0.004753

Selection: over_05
7×16 DataFrame
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ split_nb_lw100  over_05              179          29.6           53      5.34    -0.04    -0.83          88.7     -0.001978             53            0.1007     -0.0051        -0.0143    0.8868  -0.0012
   2 │ split_nb_lw50   over_05              179          40.8           73     13.44     0.27     1.98          87.7      7.6e-5               73            0.1841     -0.0213        -0.0578    0.8767  -0.006495
   3 │ split_nb_lw25   over_05              179          48.6           87     19.31    -0.13    -0.66          88.5     -0.008071             87            0.2219     -0.0161        -0.0454    0.8851  -0.007172
   4 │ split_nb_lw0    over_05              179          55.9          100     25.2     -0.79    -3.14          89.0     -0.019061            100            0.252      -0.0141        -0.0407    0.89    -0.008095
   5 │ iso_nb_mw100    over_05              179          36.3           65      8.71     0.21     2.35          87.7      0.001201             65            0.1341     -0.0187        -0.0508    0.8769  -0.003838
   6 │ iso_nb_mw50     over_05              179          37.4           67      7.81     0.11     1.42          89.6     -2.0e-6               67            0.1166      0.0012         0.0035    0.8955  -0.000723
   7 │ none_nb         over_05              179          59.2          106     16.18    -0.39    -2.43          91.5     -0.007902            106            0.1526      0.0114         0.037     0.9151   0.000513

Selection: under_05
7×16 DataFrame
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G 
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64  
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ split_nb_lw100  under_05             159          68.6          109      2.02     0.66    32.81           8.3      0.00143             109            0.0185      0.1603         0.0408    0.0826  0.000667
   2 │ split_nb_lw50   under_05             159          58.5           93      2.54    -0.71   -27.89           8.6     -0.01197              93            0.0273      0.2094         0.0523    0.086   0.000829
   3 │ split_nb_lw25   under_05             159          49.7           79      2.64    -0.71   -26.72           8.9     -0.014481             79            0.0334      0.2592         0.0631    0.0886  0.001266
   4 │ split_nb_lw0    under_05             159          44.0           70      2.26    -0.45   -19.81           8.6     -0.012158             70            0.0323      0.2361         0.0574    0.0857  0.00069
   5 │ iso_nb_mw100    under_05             159          64.8          103      2.1      0.14     6.54           8.7     -0.001845            103            0.0204      0.249          0.0608    0.0874  0.002093
   6 │ iso_nb_mw50     under_05             159          64.8          103      1.7      0.37    22.07           9.7      0.000912            103            0.0165      0.3588         0.0852    0.0971  0.003788
   7 │ none_nb         under_05             159          39.0           62      0.42    -0.09   -22.77          12.9     -0.001732             62            0.0067      0.8066         0.1686    0.129   0.004923

Selection: over_25
7×16 DataFrame
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G 
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64  
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ split_nb_lw100  over_25              247          37.2           92      4.4      0.23     5.24          46.7     -0.000934             92            0.0478      0.1641         0.1305    0.4674  0.006055
   2 │ split_nb_lw50   over_25              247          44.5          110     10.23     0.91     8.89          49.1     -0.005391            110            0.093       0.2001         0.162     0.4909  0.012137
   3 │ split_nb_lw25   over_25              247          53.8          133     15.99     2.38    14.89          49.6     -0.002788            133            0.1202      0.1748         0.1455    0.4962  0.010812
   4 │ split_nb_lw0    over_25              247          59.9          148     22.3      2.87    12.86          48.6     -0.008845            148            0.1507      0.1326         0.112     0.4865  0.004486
   5 │ iso_nb_mw100    over_25              247          38.9           96      6.53     0.91    13.95          49.0      0.002541             96            0.0681      0.2097         0.1676    0.4896  0.010698
   6 │ iso_nb_mw50     over_25              247          44.9          111      6.14     0.78    12.78          47.7      0.001949            111            0.0553      0.1581         0.1287    0.4775  0.006459
   7 │ none_nb         over_25              247          75.7          187     16.02     1.95    12.15          48.1      0.0014              187            0.0857      0.0939         0.0813    0.4813  0.003211

Selection: under_25
7×16 DataFrame
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ split_nb_lw100  under_25             247          60.3          149     12.95     1.2      9.29          52.3      0.001451            149            0.0869      0.042          0.0418    0.5235  -0.000169
   2 │ split_nb_lw50   under_25             247          52.2          129     15.31     1.84    12.04          52.7     -0.000625            129            0.1187      0.0626         0.0616    0.5271   0.000143
   3 │ split_nb_lw25   under_25             247          44.9          111     15.67     1.82    11.62          55.0     -0.003092            111            0.1411      0.104          0.103     0.5495   0.004464
   4 │ split_nb_lw0    under_25             247          39.7           98     12.75     1.89    14.8           56.1      0.001323             98            0.1301      0.1309         0.1296    0.5612   0.008339
   5 │ iso_nb_mw100    under_25             247          59.9          148     13.35     1.33     9.97          54.1      0.001266            148            0.0902      0.0741         0.0741    0.5405   0.002597
   6 │ iso_nb_mw50     under_25             247          52.2          129     10.31     1.25    12.08          52.7      0.003253            129            0.0799      0.0634         0.0624    0.5271   0.001763
   7 │ none_nb         under_25             247          23.1           57      2.0      0.12     6.04          54.4      0.000433             57            0.0352      0.1324         0.1266    0.5439   0.003975

Selection: over_35
7×16 DataFrame
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ split_nb_lw100  over_35              203          42.4           86      2.77     0.29    10.5           24.4     -0.000421             86            0.0323      0.2532         0.1128    0.2442   0.005705
   2 │ split_nb_lw50   over_35              203          50.7          103      6.6      0.22     3.29          21.4     -0.011839            103            0.0641      0.0658         0.0314    0.2136  -0.003735
   3 │ split_nb_lw25   over_35              203          57.1          116      9.99     1.45    14.52          22.4     -0.010957            116            0.0861      0.0718         0.0349    0.2241  -0.007191
   4 │ split_nb_lw0    over_35              203          64.5          131     13.95     1.58    11.31          22.1     -0.018262            131            0.1065      0.033          0.0165    0.2214  -0.015289
   5 │ iso_nb_mw100    over_35              203          45.3           92      4.14     0.66    16.06          21.7     -0.000798             92            0.045       0.1169         0.0541    0.2174   0.000941
   6 │ iso_nb_mw50     over_35              203          51.7          105      3.98     0.66    16.57          24.8     -8.4e-5              105            0.0379      0.1979         0.0923    0.2476   0.004426
   7 │ none_nb         over_35              203          79.3          161     10.35     1.2     11.57          25.5     -0.003513            161            0.0643      0.1217         0.0615    0.2547   0.000541

Selection: under_35
7×16 DataFrame
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ split_nb_lw100  under_35             203          56.7          115     12.56     0.56     4.44          73.9      6.0e-5              115            0.1092      0.0237         0.0387    0.7391   0.000237
   2 │ split_nb_lw50   under_35             203          49.8          101     15.43    -0.11    -0.69          71.3     -0.013087            101            0.1528     -0.0069        -0.0109    0.7129  -0.00607
   3 │ split_nb_lw25   under_35             203          42.4           86     15.34    -0.27    -1.77          70.9     -0.01812              86            0.1783     -0.0122        -0.0193    0.7093  -0.009135
   4 │ split_nb_lw0    under_35             203          35.0           71     11.91     0.01     0.11          69.0     -0.013239             71            0.1678     -0.0367        -0.0566    0.6901  -0.012509
   5 │ iso_nb_mw100    under_35             203          55.2          112     13.41     0.17     1.28          73.2     -0.004457            112            0.1197      0.0153         0.0248    0.7321  -0.001052
   6 │ iso_nb_mw50     under_35             203          48.8           99     10.06     0.5      5.0           73.7      0.000508             99            0.1017      0.0284         0.0461    0.7374   0.000828
   7 │ none_nb         under_35             203          18.2           37      1.48     0.22    14.59          73.0      0.004966             37            0.0401      0.0478         0.0745    0.7297   0.001577

Selection: over_45
7×16 DataFrame
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ split_nb_lw100  over_45               90          43.3           39      0.67     0.34    51.3            7.7      0.004965             39            0.0172     -0.0923        -0.0289    0.0769  -0.002944
   2 │ split_nb_lw50   over_45               90          50.0           45      1.97    -0.81   -41.28           8.9     -0.024206             45            0.0438     -0.0967        -0.0317    0.0889  -0.011223
   3 │ split_nb_lw25   over_45               90          54.4           49      2.87    -0.84   -29.31           8.2     -0.030459             49            0.0586     -0.1704        -0.058     0.0816  -0.020808
   4 │ split_nb_lw0    over_45               90          65.6           59      4.07    -1.09   -26.87          10.2     -0.0383               59            0.069      -0.1009        -0.0355    0.1017  -0.02095
   5 │ iso_nb_mw100    over_45               90          44.4           40      1.0     -0.06    -5.53           5.0     -0.005744             40            0.025      -0.35          -0.1231    0.05    -0.010862
   6 │ iso_nb_mw50     over_45               90          48.9           44      0.91    -0.05    -5.81           9.1     -0.004636             44            0.0207     -0.0091        -0.0028    0.0909  -0.002146
   7 │ none_nb         over_45               90          84.4           76      2.65    -0.26    -9.9           10.5     -0.011602             76            0.0349     -0.0724        -0.0257    0.1053  -0.006615

Selection: under_45
7×16 DataFrame
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ split_nb_lw100  under_45             116          44.8           52      7.97    -0.32    -3.96          84.6     -0.011961             52            0.1533     -0.0195        -0.0464    0.8462  -0.005263
   2 │ split_nb_lw50   under_45             116          37.9           44      9.49    -0.65    -6.82          84.1     -0.028623             44            0.2157     -0.0208        -0.0488    0.8409  -0.009346
   3 │ split_nb_lw25   under_45             116          33.6           39      9.9     -0.9     -9.09          82.1     -0.044599             39            0.2538     -0.0443        -0.0987    0.8205  -0.018802
   4 │ split_nb_lw0    under_45             116          25.9           30      8.28    -0.78    -9.36          83.3     -0.051848             30            0.2761     -0.0302        -0.0695    0.8333  -0.016923
   5 │ iso_nb_mw100    under_45             116          44.0           51      8.14    -0.46    -5.63          82.4     -0.014548             51            0.1597     -0.043         -0.0968    0.8235  -0.009638
   6 │ iso_nb_mw50     under_45             116          40.5           47      5.97    -0.22    -3.72          85.1     -0.008822             47            0.1271     -0.0089        -0.0214    0.8511  -0.002644
   7 │ none_nb         under_45             116          12.1           14      0.86     0.02     2.2           78.6      0.000852             14            0.0613     -0.0497        -0.1       0.7857  -0.003528

Selection: under_55
7×16 DataFrame
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ split_nb_lw100  under_55              78          52.6           41      8.86     0.02     0.2           95.1     -0.003068             41            0.2161      0.014          0.0609    0.9512   0.001587
   2 │ split_nb_lw50   under_55              78          44.9           35     10.44     0.27     2.54          97.1      0.002971             35            0.2984      0.0361         0.2024    0.9714   0.009008
   3 │ split_nb_lw25   under_55              78          38.5           30     10.77     0.15     1.4           96.7     -0.003822             30            0.359       0.0325         0.1686    0.9667   0.008522
   4 │ split_nb_lw0    under_55              78          32.1           25      8.67     0.07     0.81          96.0     -0.004737             25            0.3467      0.0246         0.1171    0.96     0.005089
   5 │ iso_nb_mw100    under_55              78          50.0           39      9.39     0.16     1.74          94.9      0.000371             39            0.2408      0.0118         0.0501    0.9487   0.000933
   6 │ iso_nb_mw50     under_55              78          44.9           35      6.97     0.05     0.72          94.3     -0.002684             35            0.1992      0.0082         0.0328    0.9429   0.000218
   7 │ none_nb         under_55              78          14.1           11      0.66    -0.03    -4.51          90.9     -0.003064             11            0.0597     -0.0136        -0.0435    0.9091  -0.000993

Selection: over_55
7×16 DataFrame
 Row │ model_name      selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G  
     │ String          Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64   
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ split_nb_lw100  over_55               36          36.1           13      0.1     -0.1   -100.0            0.0     -0.007782             13            0.0078     -1.0          NaN         0.0     -0.007753
   2 │ split_nb_lw50   over_55               36          55.6           20      0.43    -0.43   -99.38          10.0     -0.021669             20            0.0215      0.925          0.16      0.1      0.013642
   3 │ split_nb_lw25   over_55               36          58.3           21      0.57    -0.44   -76.11           9.5     -0.021574             21            0.0273      0.8333         0.1474    0.0952   0.013534
   4 │ split_nb_lw0    over_55               36          72.2           26      0.87    -0.33   -37.57           7.7     -0.016656             26            0.0333      0.4808         0.0936    0.0769   0.005266
   5 │ iso_nb_mw100    over_55               36          44.4           16      0.15    -0.15  -100.0            0.0     -0.009419             16            0.0094     -1.0          NaN         0.0     -0.009372
   6 │ iso_nb_mw50     over_55               36          52.8           19      0.16    -0.16  -100.0            0.0     -0.008682             19            0.0086     -1.0          NaN         0.0     -0.008646
   7 │ none_nb         over_55               36          83.3           30      0.64    -0.08   -12.36           6.7     -0.006013             30            0.0213      0.2833         0.059     0.0667   0.001794
=#



println("""

[INFO] r15 NegBin grid complete ($seg_tag).
 • On Ireland-79 expect the dispersion r→large (≈Poisson) ⇒ the nb cells ≈ the dp grid; the value is
   the apples-to-apples baseline + dropping them into r13/r14 (per-line GLMEdge / pooled totals).
 • On IrelandFirstDivision-718 the dispersion bites — look for the edge on BTTS / correct-score / tails
   (where NB reshapes P(0) and the tail), judged per-line GLMEdge, NOT grouped P/L.
 • Next: point r13_per_line_bias_edge_smile.jl / r14_pooled_totals_edge.jl at this save_dir to score the
   nb row the same way as the pois row.
""")
