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
const Predictions = BayesianFootball.Predictions



# ==========================================
# 1. SETUP & DATA
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())

# Here we load one of the models that we found the be the best on the backtest of the grid search
save_dir::String = "./data/dixon_coles_halflife_grid/"

#=
# Details regarding some of the setting / config of the test of the model
target_seasons = ["2025", "2026"]
# Using match_month to align with HierarchicalMonthlyInterception
dynamics_col = :match_biweek
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
=#


saved_files = Experiments.list_experiments(save_dir, data_dir="")

#=
Experiments in: ./data/dixon_coles_halflife_grid/
=============================================================================================================================
IDX  | NAME                      | MODEL                | SPLITTER           | SAMPLER         | TIME       | PATH ID
-----------------------------------------------------------------------------------------------------------------------------
[1]  | DCMH_HalfLife_120         | DynamicDixonColesX.. | GroupedCVConfig    | QueuedNUTSCon.. | 3h 15m     | DCMH_HalfLife_120_20260604_085753
[2]  | DCMH_HalfLife_60          | DynamicDixonColesX.. | GroupedCVConfig    | QueuedNUTSCon.. | 2h 39m     | DCMH_HalfLife_60_20260604_054201
[3]  | DCMH_HalfLife_30          | DynamicDixonColesX.. | GroupedCVConfig    | QueuedNUTSCon.. | 2h 19m     | DCMH_HalfLife_30_20260604_030252
[4]  | DCMH_HalfLife_14          | DynamicDixonColesX.. | GroupedCVConfig    | QueuedNUTSCon.. | 2h 17m     | DCMH_HalfLife_14_20260604_004247
=============================================================================================================================
=#

results_model = Experiments.load_experiment(saved_files, 2)
#=
julia> results_model = Experiments.load_experiment(saved_files, 2)
Loading: DCMH_HalfLife_60_20260604_054201
BayesianFootball.Experiments.ExperimentResults
=#


# With the modle results over all the splits we can generate the posterior predictive distribution 
# For each match and each market selection e.g :home,:away, btts_yes, :under_25... etc
posterior_predictive_distrubitons =Predictions.model_inference(ds, results_model)

#=
Running Inference on 271 matches...                                                                                                                                                                                                                                            
8130×5 DataFrame
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
=#

unique(posterior_predictive_distrubitons.df.selection)

#=
julia> unique(posterior_predictive_distrubitons.df.selection)
30-element Vector{Symbol}:
 :away
 :home
 :draw
 :btts_yes
 :btts_no
 :DC_12
 :DC_1X
 :DC_X2
 :under_05
 :over_05
 :over_15
 :under_15
 :over_25
 :under_25
 :under_35
 :over_35
 :under_45
 :over_45
 :under_55
 :over_55
 :under_65
 :over_65
 :over_75
 :under_75
 :under_85
 :over_85
 :under_95
 :over_95
 :over_105
 :under_105
=#




# looking for away to use the Betfair market odds dataset 
ds.betfair

#=
julia> ds.betfair_odds                                                                                                                                                                                                                                                         
1065450×7 DataFrame
     Row │ match_id  market_name  market_line  selection  timestamp                minutes_to_kickoff  traded_price 
         │ Int32     String       Float64      Symbol     DateTime                 Float64             Float64      
─────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
       1 │  9378193  OverUnder            3.5  over_35    2021-11-19T13:16:03.695          -388.938            3.4
       2 │  9378193  OverUnder            3.5  over_35    2021-11-19T17:28:04.191          -136.93             3.35
       3 │  9378193  OverUnder            3.5  over_35    2021-11-19T19:41:02.754            -3.9541           3.25
       4 │  9378193  OverUnder            3.5  over_35    2021-11-19T19:49:04.177             4.06962          3.5
       5 │  9378193  OverUnder            3.5  over_35    2021-11-19T19:52:03.193             7.05322          3.4
       6 │  9378193  OverUnder            3.5  over_35    2021-11-19T19:54:03.944             9.06573          3.6
       7 │  9378193  OverUnder            3.5  over_35    2021-11-19T20:00:04.081            15.068            4.1
       8 │  9378193  OverUnder            3.5  over_35    2021-11-19T20:09:04.161            24.0694           2.36
       9 │  9378193  OverUnder            3.5  over_35    2021-11-19T20:10:04.152            25.0692           2.4
      10 │  9378193  OverUnder            3.5  over_35    2021-11-19T20:11:04.117            26.0686           2.44
A small extraction so you understand the columns and data, tho this only show the under_35, but is for more markets.
Note the timestamp and time till kick off, 
=#

unique(ds.betfair_odds.selection)

#=
julia> unique(ds.betfair_odds.selection)
36-element Vector{Symbol}:
 :over_35
 :under_35
 :btts_no
 :btts_yes
 :over_15
 :under_15
 :over_05
 :under_05
 :away
 :draw
 :home
 :over_55
 :under_55
 :cs_00
 :cs_01
 :cs_02
 :cs_03
 :cs_10
 :cs_11
 :cs_12
 :cs_13
 :cs_20
 :cs_21
 :cs_22
 :cs_23
 :cs_30
 :cs_31
 :cs_32
 :cs_33
 :cs_any_other_away
 :cs_any_other_home
 :over_45
 :under_45
 :over_25
 :under_25
 :cs_any_other_draw
=#


# Evaluate with Betfair Odds
odds = Data.summarize_betfair_market(
    ds, 
    open_window=(-100000.0, -10.0), 
    close_window=(-20.0, 0.0)
)
