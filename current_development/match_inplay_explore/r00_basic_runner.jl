#=
Here we use a basic double poisson model for the pre game since we are using a double poisson in game. 
=#

# imports
using Revise
using BayesianFootball
using DataFrames
using Distributions
using ThreadPinning
using ProgressMeter

# Pin threads for maximum performance
pinthreads(:cores)

# Set const paths
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
:
ds = Data.load_datastore_cached(Data.Ireland())

save_dir::String = "./data/dixon_coles_ab/"


saved_files = Experiments.list_experiments(save_dir, data_dir="")
res_pre_game = Experiments.load_experiment(saved_files, 1)



pre_game_latents = Experiments.extract_oos_predictions(ds, res_pre_game)

#=
julia> names(pre_game_latents.df)
9-element Vector{String}:
 "match_id"
 "true_xg_a"
 "true_xg_h"
 "θ_1"
 "θ_2"
 "θ_3"
 "λ_a"
 "λ_h"
 "ρ"
=#

# I think we use the \lambda_{h,a} for this as this is what score_compute does.


#=
julia> first(pre_game_latents.df, 5)
5×9 DataFrame
 Row │ match_id  true_xg_a                          true_xg_h                          θ_1                                θ_2                                θ_3                                λ_a                                λ_h                                ρ                                 
     │ Any       Any                                Any                                Any                                Any                                Any                                Any                                Any                                Any                               
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 13250679  [0.850007, 1.00067, 0.734656, 0.…  [1.39676, 1.72031, 1.17719, 1.57…  [0.404352, 0.467377, 0.417996, 0…  [0.0613903, -0.0973691, 0.087197…  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0…  [1.06331, 0.907221, 1.09111, 0.9…  [1.49833, 1.5958, 1.51891, 1.775…  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0…
   2 │ 13250678  [1.39648, 1.83047, 1.26201, 1.40…  [0.915143, 0.934644, 0.786419, 0…  [0.120899, -0.160937, 0.140828, …  [0.496724, 0.529543, 0.452583, 0…  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0…  [1.64333, 1.69816, 1.57237, 1.51…  [1.12851, 0.851345, 1.15123, 0.9…  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0…
   3 │ 13250683  [0.90845, 1.09159, 0.805455, 0.8…  [1.26253, 1.62154, 1.0693, 1.388…  [0.384678, 0.39477, 0.408175, 0.…  [0.098164, 0.00245907, 0.206135,…  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0…  [1.10314, 1.00246, 1.22892, 1.01…  [1.46914, 1.48404, 1.50407, 1.54…  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0…
   4 │ 13250686  [1.24041, 1.72828, 1.10035, 1.31…  [0.86946, 1.01238, 0.733988, 0.9…  [0.0228848, -0.0627281, -0.08938…  [0.359065, 0.467629, 0.382095, 0…  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0…  [1.43199, 1.59621, 1.46535, 1.47…  [1.02315, 0.939199, 0.914495, 0.…  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0…
   5 │ 13250680  [0.645078, 0.739316, 0.551278, 0…  [1.64865, 2.31443, 1.36937, 2.00…  [0.643576, 0.759666, 0.600814, 0…  [-0.213378, -0.404283, -0.191759…  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0…  [0.80785, 0.667455, 0.825506, 0.…  [1.90328, 2.13756, 1.8236, 2.244…  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0…
=#




### -------- helper sections 
# 1. get the details from the DataStore for a given match
# goal times, 
# card times, 
# added time 

test_match_id = 13250679  

#=
julia> names(ds.incidents)
17-element Vector{String}:
 "id"
 "match_id"
 "incident_type"
 "time"
 "is_home"
 "added_time"
 "player_name"
 "player_in_name"
 "player_out_name"
 "assist1_name"
 "assist2_name"
 "incident_class"
 "reason"
 "is_injury"
 "rescinded"
 "period_text"
 "time_seconds"
=#
#=
julia> unique(ds.incidents.incident_type)
6-element Vector{InlineStrings.String31}:
 "injuryTime"
 "period"
 "card"
 "substitution"
 "goal"
 "inGamePenalty"
=#
subset(ds.matches, :match_id => ByRow(isequal(test_match_id)))

subset(ds.incidents,
:match_id => ByRow(isequal(test_match_id)),
:incident_type => ByRow(isequal("goal")),
)

#=
julia> subset(ds.incidents,
       :match_id => ByRow(isequal(test_match_id)),
       :incident_type => ByRow(isequal("goal")),
       )
4×17 DataFrame
 Row │ id     match_id  incident_type  time   is_home  added_time  player_name    player_in_name  player_out_name  assist1_name  assist2_name  incident_class  reason     is_injury  rescinded  period_text  time_seconds 
     │ Int32  Int32     String31       Int32  Bool     Int32?      String?        String?         String?          String?       Missing       String31?       String31?  Bool?      Bool?      String31?    Float64?     
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 32759  13250679  goal              45     true           0  sean-boyd      missing         missing          missing            missing  penalty         missing      missing    missing  missing           missing 
   2 │ 32763  13250679  goal              15     true           0  caffrey-evan   missing         missing          sean-boyd          missing  regular         missing      missing    missing  missing           missing 
   3 │ 32764  13250679  goal              14    false           0  michael-duffy  missing         missing          dom-thomas         missing  regular         missing      missing    missing  missing           missing 
   4 │ 32765  13250679  goal              11     true           0  sean-boyd      missing         missing          missing            missing  regular         missing      missing    missing  missing           missing
=#

subset(ds.incidents,
:match_id => ByRow(isequal(test_match_id)),
:incident_type => ByRow(isequal("card")),
)

#=
julia> subset(ds.incidents,
       :match_id => ByRow(isequal(test_match_id)),
       :incident_type => ByRow(isequal("card")),
       )
5×17 DataFrame
 Row │ id     match_id  incident_type  time   is_home  added_time  player_name      player_in_name  player_out_name  assist1_name  assist2_name  incident_class  reason     is_injury  rescinded  period_text  time_seconds 
     │ Int32  Int32     String31       Int32  Bool     Int32?      String?          String?         String?          String?       Missing       String31?       String31?  Bool?      Bool?      String31?    Float64?     
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 32749  13250679  card              84     true           0  paddy-barrett    missing         missing          missing            missing  yellow          Foul         missing      false  missing           missing 
   2 │ 32750  13250679  card              82    false           0  diallo-sadou     missing         missing          missing            missing  yellow          Foul         missing      false  missing           missing 
   3 │ 32757  13250679  card              53     true           0  sean-boyd        missing         missing          missing            missing  yellow          Foul         missing      false  missing           missing 
   4 │ 32761  13250679  card              38    false           0  carl-winchester  missing         missing          missing            missing  yellow          Foul         missing      false  missing           missing 
   5 │ 32762  13250679  card              22    false           0  adam-oreilly     missing         missing          missing            missing  yellow          Foul         missing      false  missing           missing
=#

subset(ds.incidents,
:match_id => ByRow(isequal(test_match_id)),
)

#=
julia> subset(ds.incidents,
       :match_id => ByRow(isequal(test_match_id)),
       )
22×17 DataFrame
 Row │ id     match_id  incident_type  time   is_home  added_time  player_name      player_in_name   player_out_name  assist1_name  assist2_name  incident_class  reason     is_injury  rescinded  period_text  time_seconds 
     │ Int32  Int32     String31       Int32  Bool     Int32?      String?          String?          String?          String?       Missing       String31?       String31?  Bool?      Bool?      String31?    Float64?     
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 32744  13250679  period            90    false         999  missing          missing          missing          missing            missing  missing         missing      missing    missing  FT                 5400.0
   2 │ 32745  13250679  injuryTime        90    false           0  missing          missing          missing          missing            missing  missing         missing      missing    missing  missing         missing   
   3 │ 32746  13250679  substitution      88     true     missing  missing          martin-john      ademipo-odubeko  missing            missing  regular         missing        false    missing  missing         missing   
   4 │ 32747  13250679  substitution      87     true     missing  missing          rayhaan-tulloch  harry-wood       missing            missing  regular         missing        false    missing  missing         missing   
   5 │ 32748  13250679  substitution      87     true     missing  missing          ryan-okane       caffrey-evan     missing            missing  regular         missing        false    missing  missing         missing   
   6 │ 32749  13250679  card              84     true           0  paddy-barrett    missing          missing          missing            missing  yellow          Foul         missing      false  missing         missing   
   7 │ 32750  13250679  card              82    false           0  diallo-sadou     missing          missing          missing            missing  yellow          Foul         missing      false  missing         missing   
   8 │ 32751  13250679  substitution      81     true     missing  missing          ellis-chapman    sean-boyd        missing            missing  regular         missing        false    missing  missing         missing   
   9 │ 32752  13250679  substitution      78    false     missing  missing          paul-mcmullan    dom-thomas       missing            missing  regular         missing        false    missing  missing         missing   
  10 │ 32753  13250679  substitution      69    false     missing  missing          sean-patton      liam-boyce       missing            missing  regular         missing        false    missing  missing         missing   
  11 │ 32754  13250679  substitution      69    false     missing  missing          gavin-whyte      pat-hoban        missing            missing  regular         missing        false    missing  missing         missing   
  12 │ 32755  13250679  substitution      67     true     missing  missing          jonathan-lunney  mcinroy-kerr     missing            missing  regular         missing        false    missing  missing         missing   
  13 │ 32756  13250679  substitution      61    false     missing  missing          diallo-sadou     carl-winchester  missing            missing  regular         missing        false    missing  missing         missing   
  14 │ 32757  13250679  card              53     true           0  sean-boyd        missing          missing          missing            missing  yellow          Foul         missing      false  missing         missing   
  15 │ 32758  13250679  period            45    false         999  missing          missing          missing          missing            missing  missing         missing      missing    missing  HT                 2700.0
  16 │ 32759  13250679  goal              45     true           0  sean-boyd        missing          missing          missing            missing  penalty         missing      missing    missing  missing         missing   
  17 │ 32760  13250679  injuryTime        45    false           0  missing          missing          missing          missing            missing  missing         missing      missing    missing  missing         missing   
  18 │ 32761  13250679  card              38    false           0  carl-winchester  missing          missing          missing            missing  yellow          Foul         missing      false  missing         missing   
  19 │ 32762  13250679  card              22    false           0  adam-oreilly     missing          missing          missing            missing  yellow          Foul         missing      false  missing         missing   
  20 │ 32763  13250679  goal              15     true           0  caffrey-evan     missing          missing          sean-boyd          missing  regular         missing      missing    missing  missing         missing   
  21 │ 32764  13250679  goal              14    false           0  michael-duffy    missing          missing          dom-thomas         missing  regular         missing      missing    missing  missing         missing   
  22 │ 32765  13250679  goal              11     true           0  sean-boyd        missing          missing          missing            missing  regular         missing      missing    missing  missing         missing
=#


subset(ds.betfair_odds,
:match_id => ByRow(isequal(test_match_id)),
:selection => ByRow(isequal(:over_05)),
:minutes_to_kickoff => ByRow(x -> x>=0.0),
)



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



#=
julia> subset(ds.betfair_odds,
       :match_id => ByRow(isequal(test_match_id)),
       :selection => ByRow(isequal(:under_25)),
       :minutes_to_kickoff => ByRow(x -> x>=0.0),
       )
4×7 DataFrame
 Row │ match_id  market_name  market_line  selection  timestamp                minutes_to_kickoff  traded_price 
     │ Int32     String       Float64      Symbol     DateTime                 Float64             Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 13250679  OverUnder            2.5  under_25   2025-02-14T19:51:55.196             6.91993          1.44
   2 │ 13250679  OverUnder            2.5  under_25   2025-02-14T19:52:54.963             7.91605          1.43
   3 │ 13250679  OverUnder            2.5  under_25   2025-02-14T19:55:55.205            10.9201           1.41
   4 │ 13250679  OverUnder            2.5  under_25   2025-02-14T19:59:54.913            14.9152           4.7
=#


#=
julia> names(ds.betfair_odds)
7-element Vector{String}:
 "match_id"
 "market_name"
 "market_line"
 "selection"
 "timestamp"
 "minutes_to_kickoff"
 "traded_price"
=#


