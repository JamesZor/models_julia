using BayesianFootball
using DataFrames
using Statistics
using Plots
using Distributions
using KernelDensity
using Plots
using StatsPlots
using Turing
### Load the saved data for the basic maher model 
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")
# extract the results and config 
result = r1[1] 
experiment_config = r1[2]


# After running your experiment, you have:
# - result (ExperimentResult)
# - data_store (DataStore)
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)

########
first_split = result.chains_sequence[1]
#model_params = extract_model_parameters(first_split, result.mapping, pred_config) # ? 

# Predict all matches for a round with uncertainty
cv_config = TimeSeriesSplitsConfig(["20/21", "21/22", "22/23", "23/24"], "24/25", :round)
target_matches = filter(row -> row.season == cv_config.target_season, data_store.matches)
round_1_matches = filter(row -> row.round == 1, target_matches)


###### Prob prediction 
#
#
#= 


=#

samples_posterior = extract_posterior_samples(first_split.ft, result.mapping) 


#### round 
#=
round one
home_team            away_team             home_score  away_score 
String31             String31              Int64       Int64      
──────────────────────────────────────────────────────────────────
st-johnstone         aberdeen                       1           2 
heart-of-midlothian  rangers                        0           0 
dundee-united        dundee-fc                      2           2 
celtic               kilmarnock                     4           0 
st-mirren            hibernian                      3           0 
motherwell           ross-county                    0           0 
partick-thistle      greenock-morton                0           0 
livingston           dunfermline-athletic           2           0 
hamilton-academical  ayr-united                     0           2 
falkirk-fc           queens-park-fc                 2           1 
airdrieonians        raith-rovers                   1           0 
=#






##################################################
#### P(home wins) = ∫∫∫ P(home wins | α, β, γ) × P(α, β, γ | data) dα dβ dγ
#
round_chain_split = result.chains_sequence[20]
mapping = result.mapping
round_20_matches = filter(row -> row.round == 20, target_matches)




match_number = 4
home_team = round_20_matches[match_number,:].home_team
away_team = round_20_matches[match_number,:].away_team
match_id = only(round_20_matches[match_number,:].match_id)
#=
julia> match_id = only(round_20_matches[match_number,:].match_id)
12477013
=#
t1 = predict_match_chain(home_team, away_team, round_chain_split.ht, mapping )

# test
round( 1 /mean(t1.under_15), digits=2)
round(1 /median(t1.under_15), digits=2)
round( 1/quantile(t1.under_15, 0.40), digits=2)
round( 1/quantile(t1.under_15, 0.60), digits=2)

round( 1/quantile(t1.away_win_probs, 0.40), digits=2)
round( 1/quantile(t1.away_win_probs, 0.60), digits=2)
##################################


m1 = predict_match_ft_ht_chain(home_team, away_team, round_chain_split, mapping )

1/mean(m1.ht.home)

1/mean(m1.ht.draw)

1/ mean(m1.ht.away)

1/mean(m1.ft.home)

1/mean(m1.ft.draw)

1/mean(m1.ft.away)

odds_m1 = filter(row -> ( row.sofa_match_id==match_id && row.minutes ≥ 0 && row.minutes ≤ 10), data_store.odds);
sort(odds_m1[:, [:minutes, :ht_home, :ht_draw, :ht_away, :home, :draw, :away]], :minutes)

round_20_matches[match_number, [:home_score_ht, :away_score_ht, :home_score, :away_score]]
#= match 2 in round 20 
julia> 1/mean(m1.ht.home)
6.63482014503234

julia> 1/mean(m1.ht.draw)
3.020698035497948

julia> 1/ mean(m1.ht.away)
1.929642714985952

julia> 1/mean(m1.ft.home)
3.912454625322855

julia> 1/mean(m1.ft.draw)
3.901925538388655

julia> round_20_matches[2, [:home_score_ht, :away_score_ht, :home_score, :away_score]]
julia> 1/mean(m1.ft.away)
2.048671002300172

 Row │ minutes  ht_home   ht_draw   ht_away   home      draw      away     
     │ Int64    Float64?  Float64?  Float64?  Float64?  Float64?  Float64? 
─────┼─────────────────────────────────────────────────────────────────────
   1 │       0       4.4      2.32       3.3       3.3       3.5      2.42
   2 │       1       4.4      2.32       3.3       3.3       3.5      2.42

julia> round_20_matches[2, [:home_score_ht, :away_score_ht, :home_score, :away_score]]
DataFrameRow
 Row │ home_score_ht  away_score_ht  home_score  away_score 
     │ Int64          Int64          Int64       Int64      
─────┼──────────────────────────────────────────────────────
   2 │             0              1           2           2

julia> filter(row -> row.match_id==match_id && row.incident_type=="goal", incidents)[:, [:time, :player_name, :assist1_name, :home_score, :away_score]
       ]
4×5 DataFrame
 Row │ time   player_name    assist1_name   home_score  away_score 
     │ Int64  String31?      String31?      Float64?    Float64?   
─────┼─────────────────────────────────────────────────────────────
   1 │    90  Jordan White   missing               2.0         2.0
   2 │    90  Joshua Nisbet  missing               1.0         2.0
   3 │    48  James Wilson   James Penrice         0.0         2.0
   4 │     2  James Wilson   Liam Boyce            0.0         1.0

=# 


# under overs 
1 / mean(m1.ft.under_05)
1 / mean(1 .- m1.ft.under_05)

1 / mean(m1.ft.under_15)
1 / mean(1 .- m1.ft.under_15)

1 / mean(m1.ft.under_25)
1 / mean(1 .- m1.ft.under_25)

1 / mean(m1.ft.under_35)
1 / mean(1 .- m1.ft.under_35)

1 / median(m1.ft.under_05)
1 / median(1 .- m1.ft.under_05)

1 / median(m1.ft.under_15)
1 / median(1 .- m1.ft.under_15)

1 / median(m1.ft.under_25)
1 / median(1 .- m1.ft.under_25)

1 / median(m1.ft.under_35)
1 / median(1 .- m1.ft.under_35)
odds_m1[:, [:minutes, :under_1_5, :over_1_5, :under_2_5, :over_2_5, :under_3_5, :over_3_5]]
#=
julia> 1 / mean(m1.ft.under_05)
12.66621401878406

julia> 1 / mean(1 .- m1.ft.under_05)
1.085717611419598

julia> 1 / mean(m1.ft.under_15)
3.5955375781051306

julia> 1 / mean(1 .- m1.ft.under_15)
1.3852766411226647

julia> 1 / mean(m1.ft.under_25)
1.8845924186930474

julia> 1 / mean(1 .- m1.ft.under_25)
2.130464131127715

julia> 1 / mean(m1.ft.under_35)
1.3421255595601498

julia> 1 / mean(1 .- m1.ft.under_35)
3.9229035132178955

julia> 1 / median(m1.ft.under_15)
3.6084176878842475

julia> 1 / median(1 .- m1.ft.under_15)
1.3833741829941066

julia> 1 / median(m1.ft.under_25)
1.8833233890657175

julia> 1 / median(1 .- m1.ft.under_25)
2.1320882163639863

julia> 1 / median(m1.ft.under_35)
1.3390980317121945

julia> 1 / median(1 .- m1.ft.under_35)
3.948999718313723

 Row │ minutes  under_1_5  over_1_5  under_2_5  over_2_5  under_3_5  over_3_5 
     │ Int64    Float64?   Float64?  Float64?   Float64?  Float64?   Float64? 
─────┼────────────────────────────────────────────────────────────────────────
 100 │       1        3.6      1.37       1.86      2.16       1.34       4.1
=#

1 / mean(m1.ht.under_05)
1 / mean(1 .- m1.ht.under_05)

1 / mean(m1.ht.under_15)
1 / mean(1 .- m1.ht.under_15)

1 / mean(m1.ht.under_25)
1 / mean(1 .- m1.ht.under_25)

1 / mean(m1.ht.under_35)
1 / mean(1 .- m1.ht.under_35)

1 / median(m1.ht.under_05)
1 / median(1 .- m1.ht.under_05)

1 / median(m1.ht.under_15)
1 / median(1 .- m1.ht.under_15)

1 / median(m1.ht.under_25)
1 / median(1 .- m1.ht.under_25)

1 / median(m1.ht.under_35)
1 / median(1 .- m1.ht.under_35)
sort(odds_m1[:, [:minutes, :ht_under_0_5, :ht_over_0_5, :ht_under_1_5, :ht_over_1_5, :ht_under_2_5, :ht_over_2_5]], :minutes)
# correct scores ft match 2 

1 ./ mean(m1.ft.correct_score, dims=1)

reshape(1 ./ mean(m1.ft.correct_score, dims=1), (4,4)

odds_m1[1,  ["0_0", "1_0", "2_0", "3_0", "0_1", "1_1", "2_1", "3_1", "0_2", "1_2", "2_2", "3_2", "0_3", "1_3", "2_3", "3_3"]]
#=
1×16 Matrix{Float64}:
 0-0      1-0     2-0      3-0      0-1      1-1      2-1      3-1      0-2      1-2      2-2      3-2      0-3      1-3      2-3      3-3 
12.6662  12.4795  24.3115  70.237  8.40049  8.27466  16.1151  46.5401  11.0482  10.8803  21.1835  61.1558  21.6134  21.2804  41.4205  119.538

DataFrameRow
0_0       1_0       2_0       3_0       0_1       1_1       2_1       3_1       0_2       1_2       2_2       3_2       0_3       1_3       2_3       3_3      
Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64? 
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 11.5      11.0      20.0      50.0       9.2       7.4      13.0      38.0      14.0      10.5      17.5      55.0      30.0      26.0      44.0     100.0
=# 



#=
1/mean(m1.ht.home)
3.6862132214127077

1/mean(m1.ht.draw)
2.7744829750142963 

1/ mean(m1.ht.away)
2.715241425539145

 1/mean(m1.ft.home)
2.1459996484263986

 1/mean(m1.ft.draw)
3.888273701644767

 1/mean(m1.ft.away)
3.6122955866708932

# under overs 
1 / mean(m1.ft.under_05)
13.18384427606017
1 / mean(1 .- m1.ft.under_05)
1.082075901278949


1 / mean(m1.ft.under_15)
3.701366609535103
 1 / mean(1 .- m1.ft.under_15)
1.3701830016223147

1 / mean(m1.ft.under_25)
1.9217091054726365
 1 / mean(1 .- m1.ft.under_25)
2.084941001518279

1 / mean(m1.ft.under_35)
1.3582558313096726
1 / mean(1 .- m1.ft.under_35)
3.791301390250394

 0-0      1-0     2-0      3-0      0-1      1-1      2-1      3-1      0-2      1-2      2-2      3-2      0-3      1-3      2-3      3-3 
 13.1838  8.9098  11.9209  23.6841  12.1826  8.23114  11.0103  21.8701  22.3098  15.0706  20.1552  40.0271  60.7257  41.0153  54.8448  108.901

DataFrameRow
 Row │ 0_0       1_0       2_0       3_0       0_1       1_1       2_1       3_1       0_2       1_2       2_2       3_2       0_3       1_3       2_3       3_3      
     │ Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64?  Float64? 
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     14.5       9.2      11.5      21.0      15.0       8.2       9.8      19.5      29.0      16.0      17.5      36.0      85.0      50.0      55.0      95.0
=# 
names(data_store.odds)
odds_m1 = filter(row -> ( row.sofa_match_id==match_id && row.minutes ≥ 0 && row.minutes ≤ 5), data_store.odds)

odds_m1[:, [:ht_home, :ht_draw, :ht_away, :home, :draw, :away]]
odds_m1[1, [:under_1_5, :over_1_5, :under_2_5, :over_2_5, :under_3_5, :over_3_5]]
odds_m1[1,  ["0_0", "1_0", "2_0", "3_0", "0_1", "1_1", "2_1", "3_1", "0_2", "1_2", "2_2", "3_2", "0_3", "1_3", "2_3", "3_3"]]
odds_m1[1,  ["0_0", "0_1",  "0_2",  "0_3",  "1_0", "1_1", "1_2", "1_3", "2_0", "2_1", "2_2", "2_3", "3_0", "3_1", "3_2", "3_3"]]
 "any_other_away"
 "any_other_draw"
 "any_other_home"


### for a round 
# TODO: finish this for the rounds
predict_round_chains( round_chain_split, round_20_matches, mapping)


### for target season 
matches_predictions  = Dict()
for (round_idx, round_chains) in enumerate(result.chains_sequence)
  round_matches = filter(row -> row.round==round_idx, target_matches) 
  round_matches_predictions = predict_round_chains( round_chain_split, round_matches, mapping)
end

length(result.chains_sequence)
