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




saved_files = Experiments.list_experiments(save_dir, data_dir="")
expr_result = Experiments.load_experiment(saved_files, 2)




mp =Predictions.model_inference(ds, expr_result)

#=
Running Inference on 276 matches...                                                                                                                                                                                                                                            
8280×5 DataFrame
  Row │ match_id  market_name   market_line  selection  distribution                      
      │ Int64     String        Float64      Symbol     Array…                            
──────┼───────────────────────────────────────────────────────────────────────────────────
    1 │ 13250679  1X2                   0.0  away       [0.208083, 0.237522, 0.258151, 0…
    2 │ 13250679  1X2                   0.0  home       [0.533038, 0.482014, 0.500117, 0…
    3 │ 13250679  1X2                   0.0  draw       [0.258879, 0.280463, 0.241732, 0…
    4 │ 13250679  BTTS                  0.0  btts_yes   [0.483969, 0.54777, 0.482429, 0.…
    5 │ 13250679  BTTS                  0.0  btts_no    [0.516031, 0.45223, 0.517571, 0.…
    6 │ 13250679  DoubleChance          0.0  DC_12      [0.741121, 0.719537, 0.758268, 0…
    7 │ 13250679  DoubleChance          0.0  DC_1X      [0.791917, 0.762477, 0.741849, 0…
    8 │ 13250679  DoubleChance          0.0  DC_X2      [0.466962, 0.517986, 0.499883, 0…
    9 │ 13250679  OverUnder             0.5  under_05   [0.0850369, 0.0834343, 0.0730613…
   10 │ 13250679  OverUnder             0.5  over_05    [0.914963, 0.916565, 0.926939, 0…
   11 │ 13250679  OverUnder             1.5  over_15    [0.72255, 0.769093, 0.705037, 0.…
   12 │ 13250679  OverUnder             1.5  under_15   [0.27745, 0.230907, 0.294963, 0.…
   13 │ 13250679  OverUnder             2.5  over_25    [0.462382, 0.507879, 0.457401, 0…
   14 │ 13250679  OverUnder             2.5  under_25   [0.537618, 0.492121, 0.542599, 0
=#


#=
Stats for: under_05
4×18 DataFrame
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G 
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64  
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   2 │ DCMH_HalfLife_60   under_05             170          59.4          101      1.49     0.54    36.41           9.9      0.002514        0.4296       30.2473            101            0.0148      0.3857         0.0908    0.099   0.003939



Stats for: under_15                                                                                                                                                                                                                                                                                          12:06 [75/1907]
4×18 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name         selection  opportunities  activity_pct  bets_placed  turnover  profit   roi_pct  win_rate_pct  hurdle_G_emp  hurdle_scale  hurdle_shape  hurdle_n_bets  hurdle_avg_stake  hurdle_E_R  hurdle_sharpe  hurdle_p  hurdle_G                                                                           
     │ String             Symbol     Int64          Float64       Int64        Float64   Float64  Float64  Float64       Float64       Float64       Float64       Int64          Float64           Float64     Float64        Float64   Float64                                                                            
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                          
   2 │ DCMH_HalfLife_60   under_15             222          44.6           99      3.15     1.33    42.17          32.3      0.009415        0.1282       22.8821             99            0.0318      0.2714         0.1449    0.3232  0.006915                                                                           
=#

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

